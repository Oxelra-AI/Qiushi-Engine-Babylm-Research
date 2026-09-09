#!/usr/bin/env python3
"""research: connectivity probe — thin wrapper over research fast runner.

Adapts the proven research DeBERTa encoder + classification head runner for the
research coordinate-connectivity substrate, which has shared eval directories.

Changes from research:
- Default data_root, conditions, output
- Eval loaded from shared eval/ directory, not per-condition
- EVAL_SUITES adapted for research (no name_permutation_counterfactual)
- --save-train-predictions on by default

No official BabyLM evaluation, upload, leaderboard interaction, or 100M training.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("experiments/archive/representation_and_objectives")
WORKSPACE = PROJECT_ROOT
CHECKPOINT_DEFAULT = WORKSPACE / "training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_80M"
DATA_ROOT_DEFAULT = WORKSPACE / "data/coordinate_connectivity_substrate"
OUT_DEFAULT = WORKSPACE / "data/connectivity_probe"
DEFAULT_CONDITIONS = ["connected", "disconnected"]
DEFAULT_ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]
DEFAULT_SEEDS = [28600, 28601, 28602]
EVAL_SUITES = [
    "heldheld_unseen_edge_closure",
    "mixed_held_seen_orientation",
    "paired_state_conservation",
    "cross_template_state_readout",
]

# ── Shared utilities (identical to research) ────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if s: rows.append(json.loads(s))
    return rows

def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

class ClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size, dropout=0.1):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 2)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.head(self.dropout(out.last_hidden_state[:, 0]))

def serialize_pair(row):
    if row.get("task") == "relation_comparison":
        return str(row.get("event1", "")), str(row.get("event2", ""))
    if row.get("task") == "state_query":
        return str(row.get("premise", "")), str(row.get("hypothesis", ""))
    if "text" in row: return str(row.get("text", "")), ""
    raise ValueError(row.get("task"))

def tokenize_row(row, tokenizer, max_len):
    a, b = serialize_pair(row)
    enc = tokenizer(a, b, max_length=max_len, truncation=True, padding=False)
    enc["label"] = 1 if bool(row.get("label")) else 0
    return enc

def make_batches(encoded, batch_size, pad_id, device, shuffle=False):
    idxs = list(range(len(encoded)))
    if shuffle: random.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        sel = idxs[start:start + batch_size]
        batch = [encoded[i] for i in sel]
        ml = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), ml), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), ml), dtype=torch.long, device=device)
        tids = torch.zeros((len(batch), ml), dtype=torch.long, device=device)
        labels = torch.zeros(len(batch), dtype=torch.long, device=device)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long, device=device)
            mask[i, :L] = torch.tensor(b["attention_mask"], dtype=torch.long, device=device)
            if "token_type_ids" in b:
                tids[i, :L] = torch.tensor(b["token_type_ids"], dtype=torch.long, device=device)
            labels[i] = int(b["label"])
        yield sel, ids, mask, tids, labels

def train_model(model, train_encoded, args, pad_id, device):
    enc_params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.lr_head, "weight_decay": 0.0},
    ])
    spe = max(1, (len(train_encoded) + args.batch_size - 1) // args.batch_size)
    total = args.epochs * spe; warmup = int(total * args.warmup_frac)
    def lr_lambda(step):
        if step < warmup: return step / max(1, warmup)
        return max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    hist = []
    for ep in range(args.epochs):
        model.train(); tloss = 0.0; tc = 0; tn = 0
        for _sel, ids, mask, tids, labels in make_batches(train_encoded, args.batch_size, pad_id, device, shuffle=True):
            logits = model(ids, mask, tids)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tloss += float(loss.item()) * len(labels); tc += int((logits.argmax(1) == labels).sum().item()); tn += len(labels)
        if ep == args.epochs - 1 or (ep + 1) % max(1, args.report_every) == 0:
            hist.append({"epoch": ep + 1, "loss": tloss / max(1, tn), "acc": tc / max(1, tn)})
    return {"last_epoch_loss": hist[-1]["loss"], "last_epoch_acc": hist[-1]["acc"], "history": hist, "total_steps": total}

def compact_pred_row(row, logits, condition, arm, seed, suite, split):
    pred = int(logits[1] > logits[0])
    return {
        "condition": condition, "arm": arm, "seed": seed, "suite": suite, "split": split,
        "row_id": row.get("row_id"), "pair_id": row.get("pair_id"), "task": row.get("task"),
        "label": bool(row.get("label")), "pred": bool(pred),
        "correct": bool(pred == int(bool(row.get("label")))),
        "logit0": float(logits[0]), "logit1": float(logits[1]),
        "margin1_minus_0": float(logits[1] - logits[0]),
        "query_kind": row.get("query_kind"), "initial_pattern": row.get("initial_pattern"),
        "initial_owner": row.get("initial_owner"),
        "static_slot": row.get("static_slot"), "static_owner": row.get("static_owner"),
        "relation": row.get("relation"), "voice": row.get("voice"),
        "candidate": row.get("candidate"), "candidate_slot": row.get("candidate_slot"),
        "correct_slot": row.get("correct_slot"),
        "supervised_changed_owner": row.get("supervised_changed_owner"),
        "relation1": row.get("relation1"), "relation2": row.get("relation2"),
        "component1": row.get("component1"), "component2": row.get("component2"),
        "orientation_dependency": row.get("orientation_dependency"),
    }

@torch.no_grad()
def predict_rows(model, encoded, raw_rows, args, pad_id, device, condition, arm, seed, suite, split):
    model.eval(); out = []
    for sel, ids, mask, tids, labels in make_batches(encoded, args.eval_batch_size, pad_id, device, shuffle=False):
        logits = model(ids, mask, tids).detach().cpu().numpy()
        for idx, logit in zip(sel, logits):
            out.append(compact_pred_row(raw_rows[idx], [float(logit[0]), float(logit[1])], condition, arm, seed, suite, split))
    return out

def mean(xs): return float(sum(xs) / len(xs)) if xs else None
def summarize_binary(rows):
    if not rows: return {"n": 0}
    out = {"n": len(rows), "acc": mean([float(r["correct"]) for r in rows])}
    if all("pred" in r for r in rows): out["pred_true_frac"] = mean([float(r["pred"]) for r in rows])
    if any("margin1_minus_0" in r for r in rows): out["margin_mean"] = mean([float(r.get("margin1_minus_0", 0.0)) for r in rows])
    return out

def exact_choice_rows(pred_rows):
    state = [r for r in pred_rows if r.get("task") == "state_query"]
    by_q = defaultdict(list)
    for r in state:
        by_q[(r.get("condition"), r.get("arm"), r.get("seed"), r.get("suite"), r.get("pair_id"), r.get("query_kind"))].append(r)
    choices = []
    for _k, rr in by_q.items():
        tr = [r for r in rr if bool(r.get("label"))]; fa = [r for r in rr if not bool(r.get("label"))]
        if len(tr) != 1 or len(fa) != 1: continue
        t, f = tr[0], fa[0]; margin = float(t["margin1_minus_0"]) - float(f["margin1_minus_0"])
        choices.append({"condition": t.get("condition"), "arm": t.get("arm"), "seed": t.get("seed"),
            "suite": t.get("suite"), "pair_id": t.get("pair_id"), "query_kind": t.get("query_kind"),
            "choice_correct": bool(margin >= 0.0), "choice_margin_true_minus_false": margin,
            "initial_pattern": t.get("initial_pattern"), "static_slot": t.get("static_slot"),
            "relation": t.get("relation"), "voice": t.get("voice")})
    return choices

def exact_pair_both(choice_rows):
    by_p = defaultdict(dict)
    for r in choice_rows:
        by_p[(r.get("condition"), r.get("arm"), r.get("seed"), r.get("suite"), r.get("pair_id"))][str(r.get("query_kind"))] = r
    out = []
    for _k, d in by_p.items():
        if "changed" not in d or "unchanged" not in d: continue
        c, u = d["changed"], d["unchanged"]
        out.append({"condition": c.get("condition"), "arm": c.get("arm"), "seed": c.get("seed"),
            "suite": c.get("suite"), "pair_id": c.get("pair_id"),
            "correct": bool(c["choice_correct"] and u["choice_correct"]),
            "changed_correct": bool(c["choice_correct"]), "unchanged_correct": bool(u["choice_correct"]),
            "initial_pattern": c.get("initial_pattern"), "static_slot": c.get("static_slot"),
            "relation": c.get("relation"), "voice": c.get("voice")})
    return out

def summarize_choice(rows):
    if not rows: return {"n": 0}
    return {"n": len(rows), "acc": mean([float(r["choice_correct"]) for r in rows]),
            "margin_mean": mean([float(r.get("choice_margin_true_minus_false", 0.0)) for r in rows])}

def group_rows(rows, fields):
    g = defaultdict(list)
    for r in rows:
        g["|".join("None" if r.get(f) is None else str(r.get(f)) for f in fields)].append(r)
    return dict(g)

def summarize_predictions(pred_rows):
    out = {"overall": summarize_binary(pred_rows)}
    comp = [r for r in pred_rows if r.get("task") == "relation_comparison"]
    if comp:
        true_comp = [r for r in comp if bool(r.get("label"))]
        out["comparison"] = {"all": summarize_binary(comp), "true_statement": summarize_binary(true_comp)}
    state = [r for r in pred_rows if r.get("task") == "state_query"]
    if state:
        true_state = [r for r in state if bool(r.get("label"))]
        changed = [r for r in true_state if r.get("query_kind") == "changed"]
        unchanged = [r for r in true_state if r.get("query_kind") == "unchanged"]
        choices = exact_choice_rows(pred_rows)
        chg_ch = [r for r in choices if r.get("query_kind") == "changed"]
        unc_ch = [r for r in choices if r.get("query_kind") == "unchanged"]
        pairs = exact_pair_both(choices)
        out["state"] = {
            "true_changed_statement": summarize_binary(changed),
            "true_unchanged_statement": summarize_binary(unchanged),
            "changed_statement_by_pattern": {k: summarize_binary(v) for k, v in sorted(group_rows(changed, ["initial_pattern"]).items())},
            "exact_changed_choice": summarize_choice(chg_ch),
            "exact_unchanged_choice": summarize_choice(unc_ch),
            "exact_changed_choice_by_pattern": {k: summarize_choice(v) for k, v in sorted(group_rows(chg_ch, ["initial_pattern"]).items())},
            "exact_pair_both": summarize_binary(pairs),
            "exact_pair_both_by_pattern": {k: summarize_binary(v) for k, v in sorted(group_rows(pairs, ["initial_pattern"]).items())},
            "exact_pair_both_by_relation": {k: summarize_binary(v) for k, v in sorted(group_rows(pairs, ["relation"]).items())},
        }
    return out

def flatten_numeric(obj, prefix=()):
    out = defaultdict(list)
    def rec(x, p):
        if isinstance(x, dict):
            for k, v in x.items(): rec(v, p + (str(k),))
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)):
            out[p].append(float(x))
    rec(obj, prefix); return out

def aggregate_results(results):
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "error" in r: continue
        key = (r["condition"], r["arm"])
        for p, vals in flatten_numeric(r.get("eval_metrics", {})).items():
            groups[key][p].extend(vals)
        for p, vals in flatten_numeric(r.get("train_metrics", {})).items():
            groups[key][("TRAIN",) + p].extend(vals)
    out = {}
    for (condition, arm), paths in sorted(groups.items()):
        ck = f"{condition}|{arm}"; out[ck] = {}
        for p, vals in paths.items():
            d = out[ck]
            for part in p[:-1]: d = d.setdefault(part, {})
            arr = np.asarray(vals, dtype=float)
            d[p[-1] + "_mean"] = float(arr.mean()); d[p[-1] + "_std"] = float(arr.std()); d[p[-1] + "_n"] = len(arr)
    return out

def fmt(x):
    if x is None: return "n/a"
    try: return f"{float(x):.3f}"
    except: return str(x)

def path_get(obj, path):
    cur = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur: return None
        cur = cur[p]
    return cur

# ── research-specific: shared eval path ─────────────────────────────────────

def train_one(condition, arm, seed, args, tokenizer, hidden_size, device, head_state):
    from transformers import DebertaV2Model
    t0 = time.time(); set_seed(seed)
    cond_dir = args.data_root / condition
    train_rows = load_jsonl(cond_dir / "common_seen_train.jsonl") + load_jsonl(cond_dir / "arms" / arm / "train_supervised.jsonl")
    train_encoded = [tokenize_row(r, tokenizer, args.max_len) for r in train_rows]

    # research: eval is shared, not per-condition
    eval_dir = args.data_root / "eval"
    eval_raw = {}
    for suite in EVAL_SUITES:
        p = eval_dir / f"{suite}.jsonl"
        if p.exists():
            rows = load_jsonl(p)
            if rows: eval_raw[suite] = rows
    eval_encoded = {s: [tokenize_row(r, tokenizer, args.max_len) for r in rows] for s, rows in eval_raw.items()}

    base = DebertaV2Model.from_pretrained(str(args.checkpoint), local_files_only=True)
    model = ClassificationModel(base, hidden_size, dropout=args.dropout)
    model.head.load_state_dict(head_state); model.to(device)
    pad_id = tokenizer.pad_token_id or 0
    train_stats = train_model(model, train_encoded, args, pad_id, device)

    pred_train = predict_rows(model, train_encoded, train_rows, args, pad_id, device, condition, arm, seed, "train", "train")
    eval_metrics = {}; eval_pred_rows = []
    for suite, rows in eval_raw.items():
        pr = predict_rows(model, eval_encoded[suite], rows, args, pad_id, device, condition, arm, seed, suite, "eval")
        eval_metrics[suite] = summarize_predictions(pr); eval_pred_rows.extend(pr)
    train_metrics = summarize_predictions(pred_train)
    elapsed = time.time() - t0
    model.cpu(); del model, base; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"condition": condition, "arm": arm, "seed": seed, "elapsed_sec": elapsed,
            "train_rows": len(train_rows), "train_stats_last_epoch": train_stats,
            "train_metrics": train_metrics, "eval_metrics": eval_metrics,
            "train_pred_rows": pred_train, "eval_pred_rows": eval_pred_rows}

def write_summary(out, args, results, aggregate):
    lines = ["# research coordinate-connectivity learned probe", "",
             "Tests whether filler-connected constraint paths enable reusable coordinate induction.", ""]
    lines += [f"- data_root: `{args.data_root}`", f"- checkpoint: `{args.checkpoint}`",
              f"- conditions: `{args.conditions}`", f"- arms: `{args.arms}`",
              f"- seeds: `{args.seeds}`", f"- epochs: {args.epochs}; batch_size: {args.batch_size}", ""]
    lines += ["## Central metrics (means over completed seeds)", "",
              "| condition | arm | train acc | psc same changed | psc opp changed | psc same pair-both | psc opp pair-both | mixed true stmt |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for condition in args.conditions:
        for arm in args.arms:
            d = aggregate.get(f"{condition}|{arm}", {})
            def g(p): return fmt(path_get(d, p))
            lines.append(f"| {condition} | {arm} | "
                f"{g(['TRAIN','overall','acc_mean'])} | "
                f"{g(['paired_state_conservation','state','exact_changed_choice_by_pattern','same','acc_mean'])} | "
                f"{g(['paired_state_conservation','state','exact_changed_choice_by_pattern','opposite','acc_mean'])} | "
                f"{g(['paired_state_conservation','state','exact_pair_both_by_pattern','same','acc_mean'])} | "
                f"{g(['paired_state_conservation','state','exact_pair_both_by_pattern','opposite','acc_mean'])} | "
                f"{g(['mixed_held_seen_orientation','comparison','true_statement','acc_mean'])} |")
    lines += ["", "## Files", f"- all_results: `{out / 'all_results.json'}`",
              f"- aggregate: `{out / 'aggregate_summary.json'}`",
              f"- per-row eval: `{out / 'per_row_eval_predictions.jsonl'}`",
              f"- per-row train: `{out / 'per_row_train_predictions.jsonl'}`"]
    (out / "connectivity_probe_summary.md").write_text("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=196)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--report-every", type=int, default=10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for fname in ["per_row_eval_predictions.jsonl", "per_row_train_predictions.jsonl"]:
        p = args.out / fname
        if p.exists(): p.unlink()

    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Data root: {args.data_root}", flush=True)
    print(f"Output: {args.out}", flush=True)

    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
    config = AutoConfig.from_pretrained(str(args.checkpoint), local_files_only=True)
    hidden_size = int(config.hidden_size)
    print(f"Hidden size: {hidden_size}", flush=True)

    results = []
    for seed in args.seeds:
        set_seed(seed)
        ref_head = nn.Linear(hidden_size, 2)
        head_state = copy.deepcopy(ref_head.state_dict()); del ref_head
        for condition in args.conditions:
            for arm in args.arms:
                print(f"\n=== {condition}/{arm}/seed={seed} ===", flush=True)
                try:
                    res = train_one(condition, arm, seed, args, tokenizer, hidden_size, device, head_state)
                    eval_rows = res.pop("eval_pred_rows"); train_rows = res.pop("train_pred_rows")
                    append_jsonl(args.out / "per_row_eval_predictions.jsonl", eval_rows)
                    append_jsonl(args.out / "per_row_train_predictions.jsonl", train_rows)
                    results.append(res)
                    psc = res["eval_metrics"].get("paired_state_conservation", {}).get("state", {})
                    mhs = res["eval_metrics"].get("mixed_held_seen_orientation", {}).get("comparison", {})
                    ta = res["train_metrics"].get("overall", {}).get("acc")
                    print(f"train_acc={fmt(ta)} "
                          f"psc_same_exact={fmt(psc.get('exact_changed_choice_by_pattern',{}).get('same',{}).get('acc'))} "
                          f"psc_opp_exact={fmt(psc.get('exact_changed_choice_by_pattern',{}).get('opposite',{}).get('acc'))} "
                          f"pair_same={fmt(psc.get('exact_pair_both_by_pattern',{}).get('same',{}).get('acc'))} "
                          f"mixed_true={fmt(mhs.get('true_statement',{}).get('acc'))} "
                          f"elapsed={res['elapsed_sec']:.1f}s", flush=True)
                except Exception as e:
                    print(f"ERROR {condition}/{arm}/seed={seed}: {e!r}", flush=True)
                    results.append({"condition": condition, "arm": arm, "seed": seed, "error": repr(e)})
                write_json(args.out / "all_results.partial.json", results)

    aggregate = aggregate_results(results)
    write_json(args.out / "all_results.json", results)
    write_json(args.out / "aggregate_summary.json", aggregate)
    write_summary(args.out, args, results, aggregate)
    print(json.dumps({
        "status": "CONNECTIVITY_PROBE_COMPLETE",
        "summary": str(args.out / "connectivity_probe_summary.md"),
        "all_results": str(args.out / "all_results.json"),
        "per_row_eval_predictions": str(args.out / "per_row_eval_predictions.jsonl"),
        "n_results": len([r for r in results if "error" not in r]),
        "n_errors": len([r for r in results if "error" in r]),
        "no_official_evaluation_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
