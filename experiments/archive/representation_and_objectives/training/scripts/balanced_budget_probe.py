#!/usr/bin/env python3
"""research: per-row learned probe for balanced information-budget substrates.

Prepared but not launched in research.  Use only after the running research probe is
read and the result justifies a balanced confirmation.

Default scientific comparison
-----------------------------
`replace_k00_spread` versus `replace_k16_spread` in
`data/information_budget_substrate/`.

The runner preserves per-row predictions and cell metrics so later analysis can
distinguish:
- anti-copy, copy-initial, and static-slot-switch rules;
- state-local event rules versus mixed held-seen coordinate propagation;
- true changed-object transfer versus unchanged/static-owner preservation.

Uses GPU when executed.  No official BabyLM evaluation, upload, or leaderboard.
"""
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("experiments/archive/representation_and_objectives")
CHECKPOINT_DEFAULT = PROJECT_ROOT / "training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_80M"
DATA_ROOT_DEFAULT = PROJECT_ROOT / "data/information_budget_substrate"
OUT_DEFAULT = PROJECT_ROOT / "data/balanced_budget_probe"

DEFAULT_CONDITIONS = ["replace_k00_spread", "replace_k16_spread"]
DEFAULT_ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]
DEFAULT_SEEDS = [28300, 28301, 28302]


# ---------------- IO and datasets ----------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


class NLIDataset(Dataset):
    """Serializes only intended text fields and the label; metadata stay out."""
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer: Any, max_len: int = 128):
        self.rows = list(rows)
        self.encodings: List[Dict[str, torch.Tensor]] = []
        self.labels: List[int] = []
        self.kept_indices: List[int] = []
        for i, r in enumerate(self.rows):
            if "premise" in r and "hypothesis" in r:
                enc = tokenizer(
                    r["premise"], r["hypothesis"],
                    max_length=max_len, padding="max_length", truncation=True,
                    return_tensors="pt",
                )
            elif "text" in r:
                enc = tokenizer(
                    r["text"], max_length=max_len, padding="max_length", truncation=True,
                    return_tensors="pt",
                )
            else:
                continue
            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
            self.labels.append(int(bool(r["label"])))
            self.kept_indices.append(i)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {k: v for k, v in self.encodings[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["row_index"] = torch.tensor(self.kept_indices[idx], dtype=torch.long)
        return item


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    return out


# ---------------- Metrics ----------------

def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def safe_key(*xs: Any) -> str:
    return "|".join("None" if x is None else str(x) for x in xs)


def summarize_binary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "acc": mean([float(r["correct"]) for r in rows]),
        "label_true_frac": mean([float(r["label"]) for r in rows]),
        "pred_true_frac": mean([float(r["pred"]) for r in rows]),
    }


def summarize_predictions(pred_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"overall": summarize_binary(pred_rows)}
    state = [r for r in pred_rows if r.get("task") == "state_query"]
    comp = [r for r in pred_rows if r.get("task") == "relation_comparison"]

    if comp:
        true_comp = [r for r in comp if r.get("label") is True]
        out["comparison"] = {
            "all": summarize_binary(comp),
            "true_statement": summarize_binary(true_comp),
            "true_by_pair_component": {
                k: summarize_binary(v) for k, v in group_rows(true_comp, ["component1", "component2"]).items()
            },
            "true_by_relation_pair": {
                k: summarize_binary(v) for k, v in group_rows(true_comp, ["relation1", "relation2"]).items()
            },
        }

    if state:
        true_state = [r for r in state if r.get("label") is True]
        changed_true = [r for r in true_state if r.get("query_kind") == "changed"]
        unchanged_true = [r for r in true_state if r.get("query_kind") == "unchanged"]
        out["state"] = {
            "all_rows": summarize_binary(state),
            "true_choice_all": summarize_binary(true_state),
            "true_changed_all": summarize_binary(changed_true),
            "true_unchanged_all": summarize_binary(unchanged_true),
            "true_changed_by_pattern": {
                k: summarize_binary(v) for k, v in group_rows(changed_true, ["initial_pattern"]).items()
            },
            "true_changed_by_pattern_static": {
                k: summarize_binary(v) for k, v in group_rows(changed_true, ["initial_pattern", "static_slot"]).items()
            },
            "true_changed_by_pattern_static_relation_voice": {
                k: summarize_binary(v) for k, v in group_rows(changed_true, ["initial_pattern", "static_slot", "relation", "voice"]).items()
            },
            "true_unchanged_by_pattern_static": {
                k: summarize_binary(v) for k, v in group_rows(unchanged_true, ["initial_pattern", "static_slot"]).items()
            },
            "pair_both": pair_both_metrics(true_state),
        }
    return out


def group_rows(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[safe_key(*(r.get(f) for f in fields))].append(r)
    return dict(g)


def pair_both_metrics(true_state_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in true_state_rows:
        by_pair[str(r.get("pair_id"))][str(r.get("query_kind"))] = r
    pair_rows: List[Dict[str, Any]] = []
    for pid, d in by_pair.items():
        if "changed" not in d or "unchanged" not in d:
            continue
        c, u = d["changed"], d["unchanged"]
        pair_rows.append({
            "pair_id": pid,
            "correct": bool(c["correct"] and u["correct"]),
            "changed_correct": bool(c["correct"]),
            "unchanged_correct": bool(u["correct"]),
            "initial_pattern": c.get("initial_pattern"),
            "static_slot": c.get("static_slot"),
            "relation": c.get("relation"),
            "voice": c.get("voice"),
        })
    return {
        "all": summarize_binary(pair_rows),
        "by_pattern": {k: summarize_binary(v) for k, v in group_rows(pair_rows, ["initial_pattern"]).items()},
        "by_pattern_static": {k: summarize_binary(v) for k, v in group_rows(pair_rows, ["initial_pattern", "static_slot"]).items()},
        "by_pattern_static_relation_voice": {k: summarize_binary(v) for k, v in group_rows(pair_rows, ["initial_pattern", "static_slot", "relation", "voice"]).items()},
    }


def compact_pred_row(row: Dict[str, Any], pred: int, logits: Sequence[float], suite: str,
                     condition: str, arm: str, seed: int, split: str) -> Dict[str, Any]:
    return {
        "condition": condition,
        "arm": arm,
        "seed": seed,
        "split": split,
        "suite": suite,
        "row_id": row.get("row_id"),
        "pair_id": row.get("pair_id"),
        "task": row.get("task"),
        "label": bool(row.get("label")),
        "pred": bool(pred),
        "correct": bool(pred == int(bool(row.get("label")))),
        "logit0": float(logits[0]),
        "logit1": float(logits[1]),
        "margin1_minus_0": float(logits[1] - logits[0]),
        "query_kind": row.get("query_kind"),
        "initial_pattern": row.get("initial_pattern"),
        "initial_changed_owner": row.get("initial_changed_owner"),
        "static_slot": row.get("static_slot"),
        "static_owner": row.get("static_owner"),
        "relation": row.get("relation"),
        "voice": row.get("voice"),
        "candidate": row.get("candidate"),
        "candidate_slot": row.get("candidate_slot"),
        "correct_slot": row.get("correct_slot"),
        "supervised_changed_owner": row.get("supervised_changed_owner"),
        "relation1": row.get("relation1"),
        "relation2": row.get("relation2"),
        "component1": row.get("component1"),
        "component2": row.get("component2"),
        "orientation_dependency": row.get("orientation_dependency"),
    }


# ---------------- Training and evaluation ----------------

def evaluate_rows(model: nn.Module, rows: Sequence[Dict[str, Any]], tokenizer: Any,
                  device: torch.device, max_len: int, batch_size: int,
                  suite: str, condition: str, arm: str, seed: int, split: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ds = NLIDataset(rows, tokenizer, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    pred_rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=mask)
            logits = outputs.logits.detach().cpu().numpy()
            preds = logits.argmax(axis=-1).tolist()
            idxs = batch["row_index"].cpu().tolist()
            for idx, pred, logit in zip(idxs, preds, logits):
                pred_rows.append(compact_pred_row(rows[idx], int(pred), [float(logit[0]), float(logit[1])], suite, condition, arm, seed, split))
    return summarize_predictions(pred_rows), pred_rows


def train_one(condition: str, arm: str, seed: int, args: argparse.Namespace, tokenizer: Any, device: torch.device) -> Dict[str, Any]:
    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cond_dir = args.data_root / condition
    train_rows = load_jsonl(cond_dir / "common_seen_train.jsonl")
    train_rows.extend(load_jsonl(cond_dir / "arms" / arm / "train_supervised.jsonl"))

    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.checkpoint), num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    train_ds = NLIDataset(train_rows, tokenizer, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(args.epochs):
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            outputs.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    train_metrics, train_pred_rows = evaluate_rows(
        model, train_rows, tokenizer, device, args.max_len, args.eval_batch_size,
        "train", condition, arm, seed, "train",
    )
    eval_metrics: Dict[str, Any] = {}
    eval_pred_all: List[Dict[str, Any]] = []
    for ef in sorted((cond_dir / "eval").glob("*.jsonl")):
        rows = load_jsonl(ef)
        metrics, pred_rows = evaluate_rows(
            model, rows, tokenizer, device, args.max_len, args.eval_batch_size,
            ef.stem, condition, arm, seed, "eval",
        )
        eval_metrics[ef.stem] = metrics
        eval_pred_all.extend(pred_rows)

    elapsed = time.time() - t0
    del model, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "condition": condition,
        "arm": arm,
        "seed": seed,
        "elapsed_sec": round(elapsed, 2),
        "train_rows": len(train_rows),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "train_pred_rows": train_pred_rows if args.save_train_predictions else [],
        "eval_pred_rows": eval_pred_all,
    }


def flatten_metric(metrics: Dict[str, Any], path: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        key = f"{path}/{k}" if path else k
        if isinstance(v, dict):
            out.update(flatten_metric(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[key] = float(v)
    return out


def format_mean_std(vals: Sequence[float]) -> str:
    if not vals:
        return "n/a"
    arr = np.array(vals, dtype=float)
    return f"{arr.mean():.3f}±{arr.std():.3f}" if len(arr) > 1 and arr.std() > 0.0005 else f"{arr.mean():.3f}"


def write_summary(out: Path, results: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    groups: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "error" in r:
            continue
        key = (r["condition"], r["arm"])
        # Central eval metrics.
        flat = flatten_metric(r["eval_metrics"])
        for mk, mv in flat.items():
            groups[key][mk].append(mv)
        train_flat = flatten_metric(r["train_metrics"])
        for mk, mv in train_flat.items():
            groups[key]["TRAIN/" + mk].append(mv)

    lines: List[str] = []
    lines.append("# research balanced budget probe summary")
    lines.append("")
    lines.append("Prepared runner output.  Central interpretation requires per-row cell metrics, not only aggregate state accuracy.")
    lines.append("")
    lines.append("## Run settings")
    lines.append("")
    lines.append(f"- data_root: `{args.data_root}`")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- conditions: `{args.conditions}`")
    lines.append(f"- arms: `{args.arms}`")
    lines.append(f"- seeds: `{args.seeds}`")
    lines.append(f"- epochs: `{args.epochs}`, batch_size: `{args.batch_size}`, lr: `{args.lr}`")
    lines.append("")
    lines.append("## Central metrics")
    lines.append("")
    lines.append("| condition | arm | train acc | mixed true | psc changed same | psc changed opp | psc pair-both same | psc pair-both opp | offdiag same/st0 changed | offdiag opp/st1 changed |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for condition in args.conditions:
        for arm in args.arms:
            d = groups.get((condition, arm), {})
            def g(metric: str) -> str:
                return format_mean_std(d.get(metric, []))
            lines.append(
                f"| {condition} | {arm} | "
                f"{g('TRAIN/overall/acc')} | "
                f"{g('mixed_held_seen_orientation/comparison/true_statement/acc')} | "
                f"{g('paired_state_conservation/state/true_changed_by_pattern/same/acc')} | "
                f"{g('paired_state_conservation/state/true_changed_by_pattern/opposite/acc')} | "
                f"{g('paired_state_conservation/state/pair_both/by_pattern/same/acc')} | "
                f"{g('paired_state_conservation/state/pair_both/by_pattern/opposite/acc')} | "
                f"{g('paired_state_conservation/state/true_changed_by_pattern_static/same|0/acc')} | "
                f"{g('paired_state_conservation/state/true_changed_by_pattern_static/opposite|1/acc')} |"
            )
    lines.append("")
    lines.append("## Files")
    lines.append(f"- all_results: `{out / 'all_results.json'}`")
    lines.append(f"- per-row eval predictions: `{out / 'per_row_eval_predictions.jsonl'}`")
    if args.save_train_predictions:
        lines.append(f"- per-row train predictions: `{out / 'per_row_train_predictions.jsonl'}`")
    lines.append("")
    (out / "balanced_budget_probe_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------- Entrypoint ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--save-train-predictions", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Only check that requested files exist and print row counts.")
    return ap.parse_args()


def dry_run(args: argparse.Namespace) -> None:
    checks: Dict[str, Any] = {"status": "DRY_RUN", "conditions": {}}
    for cond in args.conditions:
        cdir = args.data_root / cond
        checks["conditions"][cond] = {"exists": cdir.exists(), "arms": {}}
        for arm in args.arms:
            sup = cdir / "arms" / arm / "train_supervised.jsonl"
            common = cdir / "common_seen_train.jsonl"
            ev = sorted((cdir / "eval").glob("*.jsonl")) if (cdir / "eval").exists() else []
            checks["conditions"][cond]["arms"][arm] = {
                "supervised_rows": len(load_jsonl(sup)) if sup.exists() else None,
                "common_seen_rows": len(load_jsonl(common)) if common.exists() else None,
                "eval_suites": {p.stem: len(load_jsonl(p)) for p in ev},
            }
    print(json.dumps(checks, indent=2, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Data root: {args.data_root}", flush=True)
    print(f"Output: {args.out}", flush=True)

    # Ensure stale per-row files from old runs are not silently appended.
    for fname in ["per_row_eval_predictions.jsonl", "per_row_train_predictions.jsonl"]:
        p = args.out / fname
        if p.exists():
            p.unlink()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))

    all_results: List[Dict[str, Any]] = []
    for condition in args.conditions:
        for arm in args.arms:
            for seed in args.seeds:
                tag = f"{condition}/{arm}/seed={seed}"
                print(f"\n=== {tag} ===", flush=True)
                try:
                    result = train_one(condition, arm, seed, args, tokenizer, device)
                    eval_pred_rows = result.pop("eval_pred_rows")
                    train_pred_rows = result.pop("train_pred_rows")
                    append_jsonl(args.out / "per_row_eval_predictions.jsonl", eval_pred_rows)
                    if args.save_train_predictions:
                        append_jsonl(args.out / "per_row_train_predictions.jsonl", train_pred_rows)
                    all_results.append(result)
                    psc = result["eval_metrics"].get("paired_state_conservation", {})
                    mhs = result["eval_metrics"].get("mixed_held_seen_orientation", {})
                    print(f"  train_acc={result['train_metrics']['overall']['acc']:.3f}", flush=True)
                    print(f"  mixed_true={mhs.get('comparison',{}).get('true_statement',{}).get('acc')}", flush=True)
                    print(f"  psc same_changed={psc.get('state',{}).get('true_changed_by_pattern',{}).get('same',{}).get('acc')}", flush=True)
                    print(f"  psc opp_changed={psc.get('state',{}).get('true_changed_by_pattern',{}).get('opposite',{}).get('acc')}", flush=True)
                    print(f"  elapsed={result['elapsed_sec']}s", flush=True)
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    all_results.append({"condition": condition, "arm": arm, "seed": seed, "error": repr(e)})
    write_json(args.out / "all_results.json", all_results)
    write_summary(args.out, all_results, args)
    status = {
        "status": "BALANCED_BUDGET_PROBE_COMPLETE",
        "out": str(args.out),
        "summary": str(args.out / "balanced_budget_probe_summary.md"),
        "n_results": len([r for r in all_results if "error" not in r]),
        "n_errors": len([r for r in all_results if "error" in r]),
        "no_official_evaluation_upload_or_leaderboard": True,
    }
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
