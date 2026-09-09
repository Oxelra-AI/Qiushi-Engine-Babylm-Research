#!/usr/bin/env python3
"""research: context-separation control for temporal bottleneck.

The slot marker test showed the model cannot fit changed rows in any format
when both temporal states appear in context.  This control tests two things:

1. "separated": Before queries see ONLY initial context; after queries see
   ONLY later context.  If this works, the model CAN learn rankings from
   sparse data — the bottleneck is temporal selection, not ranking learning.

2. "distinct_tokens": Use existing pretrained tokens that ARE representationally
   distinct (e.g., "In round one" vs "In round two", or "First fact" vs
   "Second fact") as temporal markers.

3. "position_explicit": Use sentence position with explicit reference:
   "Sentence 1: ... Sentence 2: ... The question refers to Sentence 1/2."

Together with the slot marker test, this establishes whether:
- The model has ranking-learning capacity (separated should work)
- Sufficiently distinct existing tokens enable temporal selection
- The bottleneck is specifically representational collapse of temporal cues
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
ATP_DIR = WS / "data/upset_balanced_state_substrate"

PATH = WS / "training/scripts/contrastive_wording_cross_probe.py"
spec = importlib.util.spec_from_file_location("s262", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
ALIAS_POOL = m.ALIAS_POOL
DEFAULT_MODEL = m.DEFAULT_MODEL

def _si(s): return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)
def als(key, n): return random.Random(_si(f"a266c|{key}")).sample(ALIAS_POOL, n)


def load_changed_worlds():
    changed_tr, changed_he = [], []
    for sp, out_list in [("train", changed_tr), ("held", changed_he)]:
        p = ATP_DIR / f"families_{sp}.jsonl"
        for line in p.read_text("utf-8").splitlines():
            if not line.strip(): continue
            f = json.loads(line)
            a, b = f["participant_a"], f["participant_b"]
            for wk in ["context1", "context2"]:
                c = f[wk]
                if not c.get("state_changed_between_snapshots"): continue
                w, l = c["winner"], c["loser"]
                hi0, lo0 = (w, l) if c["winner_higher_at_time"] else (l, w)
                hi1, lo1 = (w, l) if c["winner_higher_later"] else (l, w)
                out_list.append(dict(
                    family_id=f["family_id"], world_id=f"{f['family_id']}::{wk}",
                    a=a, b=b, hi0=hi0, lo0=lo0, hi1=hi1, lo1=lo1,
                ))
    return changed_tr, changed_he


FORMATS = {
    # Control: before sees only initial, after sees only later
    "separated": {
        "before_ctx": "{H} was ranked above {Lo} in the ATP.",
        "after_ctx": "{H} was ranked above {Lo} in the ATP.",
        "hyp_before": "{X} held the higher ranking than {Y}.",
        "hyp_after": "{X} held the higher ranking than {Y}.",
    },
    # Use explicitly different ordinal tokens
    "ordinal": {
        "init": "First ranking: {H} was ranked above {Lo}.",
        "later": "Second ranking: {H} was ranked above {Lo}.",
        "hyp_before": "In the first ranking, {X} was ranked higher than {Y}.",
        "hyp_after": "In the second ranking, {X} was ranked higher than {Y}.",
    },
    # Use sentence numbering with explicit reference
    "numbered": {
        "init": "Statement one: {H} was ranked above {Lo}.",
        "later": "Statement two: {H} was ranked above {Lo}.",
        "hyp_before": "According to statement one, {X} was ranked higher than {Y}.",
        "hyp_after": "According to statement two, {X} was ranked higher than {Y}.",
    },
    # Use paragraph/section structure
    "section": {
        "init": "Background: {H} was ranked above {Lo}.",
        "later": "Update: {H} was ranked above {Lo}.",
        "hyp_before": "Based on the background, {X} was ranked higher than {Y}.",
        "hyp_after": "Based on the update, {X} was ranked higher than {Y}.",
    },
    # Baseline: full natural (same as slot marker test, for reference)
    "natural": {
        "init": "In the earlier ATP ranking, {H} was above {Lo}.",
        "later": "In the later ATP ranking, {H} was above {Lo}.",
        "hyp_before": "Before the later ranking update, {X} held the higher ranking than {Y}.",
        "hyp_after": "After the later ranking update, {X} held the higher ranking than {Y}.",
    },
}


def build_rows(worlds, fmt_name, ns, split):
    fmt = FORMATS[fmt_name]
    rows = []
    for i, w in enumerate(worlds):
        al = als(f"{ns}|{w['world_id']}", 2)
        am = {w["a"]: al[0], w["b"]: al[1]}
        
        for hd in ["AB", "BA"]:
            x = am[w["a"]] if hd == "AB" else am[w["b"]]
            y = am[w["b"]] if hd == "AB" else am[w["a"]]
            hi0_is_a = (w["hi0"] == w["a"])
            hi1_is_a = (w["hi1"] == w["a"])
            lab_before = int((hd == "AB" and hi0_is_a) or (hd == "BA" and not hi0_is_a))
            lab_after = int((hd == "AB" and hi1_is_a) or (hd == "BA" and not hi1_is_a))
            
            if fmt_name == "separated":
                # Before query: context = initial state only
                ctx_b = fmt["before_ctx"].format(H=am[w["hi0"]], Lo=am[w["lo0"]])
                hyp_b = fmt["hyp_before"].format(X=x, Y=y)
                # After query: context = later state only
                ctx_a = fmt["after_ctx"].format(H=am[w["hi1"]], Lo=am[w["lo1"]])
                hyp_a = fmt["hyp_after"].format(X=x, Y=y)
                
                rows.append({"text": f"{ctx_b} [SEP] {hyp_b}", "label": lab_before,
                             "query": "before", "hd": hd, "fmt": fmt_name, "world": w["world_id"], "split": split})
                rows.append({"text": f"{ctx_a} [SEP] {hyp_a}", "label": lab_after,
                             "query": "after", "hd": hd, "fmt": fmt_name, "world": w["world_id"], "split": split})
            else:
                ctx_init = fmt["init"].format(H=am[w["hi0"]], Lo=am[w["lo0"]])
                ctx_later = fmt["later"].format(H=am[w["hi1"]], Lo=am[w["lo1"]])
                context = f"{ctx_init} {ctx_later}"
                hyp_b = fmt["hyp_before"].format(X=x, Y=y)
                hyp_a = fmt["hyp_after"].format(X=x, Y=y)
                
                rows.append({"text": f"{context} [SEP] {hyp_b}", "label": lab_before,
                             "query": "before", "hd": hd, "fmt": fmt_name, "world": w["world_id"], "split": split})
                rows.append({"text": f"{context} [SEP] {hyp_a}", "label": lab_after,
                             "query": "after", "hd": hd, "fmt": fmt_name, "world": w["world_id"], "split": split})
    return rows


def train_and_eval(train_rows, eval_rows, model_path, device, seed, epochs=12, lr=1e-4, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    hidden = model.config.hidden_size
    head = torch.nn.Linear(hidden, 2).to(device)
    
    def encode(rows):
        texts = [r["text"] for r in rows]
        labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=160, return_tensors="pt")
        return enc, labels
    
    train_enc, train_lab = encode(train_rows)
    
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=1e-3)
    
    history = []
    for ep in range(1, epochs + 1):
        model.train(); head.train()
        perm = torch.randperm(len(train_rows))
        total_loss, correct, total = 0.0, 0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start+batch_size]
            batch = {k: v[idx].to(device) for k, v in train_enc.items()}
            labs = train_lab[idx].to(device)
            out = model(**batch).last_hidden_state[:, 0, :]
            logits = head(out)
            loss = F.cross_entropy(logits, labs)
            loss.backward()
            opt.step(); opt.zero_grad()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(-1) == labs).sum().item()
            total += len(idx)
        acc = correct / total
        history.append({"ep": ep, "loss": total_loss / total, "acc": acc})
        if acc >= 0.999 and ep >= 3:
            break
    
    model.eval(); head.eval()
    by_query_train = collections.defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for start in range(0, len(train_rows), 32):
            end = min(start + 32, len(train_rows))
            batch = {k: v[start:end].to(device) for k, v in train_enc.items()}
            labs = train_lab[start:end].to(device)
            out = model(**batch).last_hidden_state[:, 0, :]
            preds = head(out).argmax(-1)
            for j in range(end - start):
                q = train_rows[start + j]["query"]
                by_query_train[q][0] += int(preds[j] == labs[j])
                by_query_train[q][1] += 1
    
    train_by_query = {q: v[0] / v[1] for q, v in by_query_train.items()}
    
    eval_enc, eval_lab = encode(eval_rows)
    eval_by_query = collections.defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for start in range(0, len(eval_rows), 32):
            end = min(start + 32, len(eval_rows))
            batch = {k: v[start:end].to(device) for k, v in eval_enc.items()}
            labs = eval_lab[start:end].to(device)
            out = model(**batch).last_hidden_state[:, 0, :]
            preds = head(out).argmax(-1)
            for j in range(end - start):
                q = eval_rows[start + j]["query"]
                eval_by_query[q][0] += int(preds[j] == labs[j])
                eval_by_query[q][1] += 1
    
    eval_by_q = {q: v[0] / v[1] for q, v in eval_by_query.items()}
    
    del model, head
    torch.cuda.empty_cache()
    
    return {
        "final_train_acc": history[-1]["acc"],
        "train_by_query": dict(train_by_query),
        "eval_by_query": dict(eval_by_q),
        "epochs_used": len(history),
    }


def similarity_check(model, tokenizer, rows_before, rows_after, device):
    """Paired CLS cosine between before/after queries."""
    n = min(len(rows_before), len(rows_after), 12)
    with torch.no_grad():
        be = tokenizer([r["text"] for r in rows_before[:n]], padding=True, truncation=True,
                      max_length=160, return_tensors="pt").to(device)
        ae = tokenizer([r["text"] for r in rows_after[:n]], padding=True, truncation=True,
                      max_length=160, return_tensors="pt").to(device)
        b_cls = model(**be).last_hidden_state[:, 0, :].float()
        a_cls = model(**ae).last_hidden_state[:, 0, :].float()
    cos = F.cosine_similarity(b_cls, a_cls).cpu().numpy()
    return float(np.mean(cos)), float(np.std(cos))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--out_dir", default=str(WS / "data/context_separation_control"))
    parser.add_argument("--n_train", type=int, default=40)
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=26600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_build", action="store_true")
    args = parser.parse_args()
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    changed_tr, changed_he = load_changed_worlds()
    rng = random.Random(args.seed)
    rng.shuffle(changed_tr); rng.shuffle(changed_he)
    train_worlds = changed_tr[:args.n_train]
    eval_worlds = changed_he[:args.n_eval]
    
    fmt_names = ["separated", "ordinal", "numbered", "section", "natural"]
    
    for fn in fmt_names:
        rows = build_rows(train_worlds, fn, f"train_{fn}", "train")
        labs = collections.Counter(r["label"] for r in rows)
        print(json.dumps({"fmt": fn, "n": len(rows), "labels": dict(labs)}), flush=True)
    
    if args.dry_build:
        print(json.dumps({"status": "DRY_DONE"}), flush=True)
        return
    
    # Similarity check
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(args.device).eval()
    
    sim_results = {}
    for fn in fmt_names:
        rows = build_rows(train_worlds[:5], fn, f"sim_{fn}", "test")
        before_rows = [r for r in rows if r["query"] == "before"]
        after_rows = [r for r in rows if r["query"] == "after"]
        cos_mean, cos_std = similarity_check(model, tokenizer, before_rows, after_rows, args.device)
        sim_results[fn] = {"cos_mean": cos_mean, "cos_std": cos_std}
        print(json.dumps({"fmt": fn, "pretrained_ba_cos": cos_mean, "cos_std": cos_std}), flush=True)
    
    del model; torch.cuda.empty_cache()
    
    # Train and evaluate each format
    results = {}
    for fn in fmt_names:
        train_rows = build_rows(train_worlds, fn, f"train_{fn}", "train")
        eval_rows = build_rows(eval_worlds, fn, f"eval_{fn}", "held")
        
        res = train_and_eval(train_rows, eval_rows, args.model_path, args.device,
                             args.seed, epochs=args.epochs)
        results[fn] = res
        
        print(json.dumps({"fmt": fn,
                          "train_acc": res["final_train_acc"],
                          "train_before": res["train_by_query"].get("before"),
                          "train_after": res["train_by_query"].get("after"),
                          "eval_before": res["eval_by_query"].get("before"),
                          "eval_after": res["eval_by_query"].get("after")}), flush=True)
    
    # Summary
    summary = {"status": "SEPARATION_DONE",
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "n_train_worlds": len(train_worlds), "n_eval_worlds": len(eval_worlds),
               "formats": {}}
    for fn in fmt_names:
        r = results[fn]
        summary["formats"][fn] = {
            "train_acc": r["final_train_acc"],
            "train_before": r["train_by_query"].get("before"),
            "train_after": r["train_by_query"].get("after"),
            "eval_before": r["eval_by_query"].get("before"),
            "eval_after": r["eval_by_query"].get("after"),
            "pretrained_ba_cos": sim_results[fn]["cos_mean"],
        }
    
    (out / "context_separation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    md = ["# research context separation control\n"]
    md.append("| Format | Train | Before | After | Eval before | Eval after | BA cos |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for fn in fmt_names:
        s = summary["formats"][fn]
        md.append(f"| {fn} | {s['train_acc']:.3f} | {s['train_before']:.3f} | "
                  f"{s['train_after']:.3f} | {s['eval_before']:.3f} | {s['eval_after']:.3f} | "
                  f"{s['pretrained_ba_cos']:.4f} |")
    md.append("\n**separated** removes temporal selection (each query sees only its relevant context).")
    md.append("If separated works while others fail, the bottleneck is temporal selection,")
    md.append("not ranking learning capacity.\n")
    
    (out / "context_separation_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "SEPARATION_DONE",
                      "summary_json": str(out / "context_separation_summary.json")}), flush=True)


if __name__ == "__main__":
    main()
