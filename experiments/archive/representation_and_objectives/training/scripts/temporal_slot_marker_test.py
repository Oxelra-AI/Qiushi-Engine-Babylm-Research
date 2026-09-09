#!/usr/bin/env python3
"""research: explicit temporal slot markers to break representational collapse.

Scientific question
-------------------
research diagnostic showed context temporal cues are collapsed (cosine 0.992):
"In the earlier ATP ranking..." ≈ "In the later ATP ranking..." in pretrained
DeBERTa.  This explains why changed-focal rows cannot be fit: the model sees
identical context for before/after queries and encounters irreconcilable labels.

This test replaces natural temporal language with explicit structural markers:
  [TIME_1] A ranked above B.  [TIME_2] B ranked above A.
and paired hypothesis:
  At [TIME_1], who was higher?  /  At [TIME_2], who was higher?

If the model CAN fit changed rows with explicit markers but CANNOT with natural
temporal language, the bottleneck is pretrained representational grounding of
temporal cues, not an architectural limitation.

The test uses the same research worlds but three context formats:
1. "natural": original INIT/LATER wording (expected to fail)
2. "structural": [TIME_1]/[TIME_2] markers (tests architectural capacity)
3. "hybrid": "[TIME_1] In the earlier ranking..." (tests whether marker + cue helps)

Only changed-focal worlds in training (no stable base to create prior dominance).
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

# Import alias pool from research
PATH = WS / "training/scripts/contrastive_wording_cross_probe.py"
spec = importlib.util.spec_from_file_location("s262", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
ALIAS_POOL = m.ALIAS_POOL
DEFAULT_MODEL = m.DEFAULT_MODEL


def _si(s): return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)
def aliases(key, n): return random.Random(_si(f"a266|{key}")).sample(ALIAS_POOL, n)


def load_changed_worlds():
    """Load only changed temporal worlds from ATP substrate."""
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


CTX_FORMATS = {
    "natural": {
        "init": [
            "In the earlier ATP ranking, {H} was above {Lo}.",
            "At the first ranking date, {H} stood higher than {Lo}.",
        ],
        "later": [
            "In the later ATP ranking, {H} was above {Lo}.",
            "At the later ranking date, {H} stood higher than {Lo}.",
        ],
        "hyp_before": "Before the later ranking update, {X} held the higher ranking than {Y}.",
        "hyp_after": "After the later ranking update, {X} held the higher ranking than {Y}.",
    },
    "structural": {
        "init": [
            "[TIME_1] {H} was ranked above {Lo}.",
            "[TIME_1] {H} had a higher ranking than {Lo}.",
        ],
        "later": [
            "[TIME_2] {H} was ranked above {Lo}.",
            "[TIME_2] {H} had a higher ranking than {Lo}.",
        ],
        "hyp_before": "At [TIME_1], {X} was ranked higher than {Y}.",
        "hyp_after": "At [TIME_2], {X} was ranked higher than {Y}.",
    },
    "hybrid": {
        "init": [
            "[TIME_1] In the earlier ranking, {H} was above {Lo}.",
            "[TIME_1] At the first ranking date, {H} stood higher than {Lo}.",
        ],
        "later": [
            "[TIME_2] In the later ranking, {H} was above {Lo}.",
            "[TIME_2] At the later ranking date, {H} stood higher than {Lo}.",
        ],
        "hyp_before": "At [TIME_1], before the update, {X} held the higher ranking than {Y}.",
        "hyp_after": "At [TIME_2], after the update, {X} held the higher ranking than {Y}.",
    },
}


def build_rows(worlds, fmt_name, ns, split, n_tpl_vars=2):
    fmt = CTX_FORMATS[fmt_name]
    rows = []
    for i, w in enumerate(worlds):
        al = aliases(f"{ns}|{w['world_id']}", 2)
        am = {w["a"]: al[0], w["b"]: al[1]}
        for vi in range(min(n_tpl_vars, len(fmt["init"]))):
            ctx_init = fmt["init"][vi].format(H=am[w["hi0"]], Lo=am[w["lo0"]])
            ctx_later = fmt["later"][vi].format(H=am[w["hi1"]], Lo=am[w["lo1"]])
            context = f"{ctx_init} {ctx_later}"
            
            for hd in ["AB", "BA"]:
                x = am[w["a"]] if hd == "AB" else am[w["b"]]
                y = am[w["b"]] if hd == "AB" else am[w["a"]]
                
                # before label: who was higher initially
                hi0_is_a = (w["hi0"] == w["a"])
                lab_before = int((hd == "AB" and hi0_is_a) or (hd == "BA" and not hi0_is_a))
                # after label: who was higher later (different for changed worlds!)
                hi1_is_a = (w["hi1"] == w["a"])
                lab_after = int((hd == "AB" and hi1_is_a) or (hd == "BA" and not hi1_is_a))
                
                hyp_b = fmt["hyp_before"].format(X=x, Y=y)
                hyp_a = fmt["hyp_after"].format(X=x, Y=y)
                
                rows.append({"text": f"{context} [SEP] {hyp_b}", "label": lab_before,
                             "query": "before", "hd": hd, "fmt": fmt_name,
                             "world": w["world_id"], "split": split, "vi": vi})
                rows.append({"text": f"{context} [SEP] {hyp_a}", "label": lab_after,
                             "query": "after", "hd": hd, "fmt": fmt_name,
                             "world": w["world_id"], "split": split, "vi": vi})
    return rows


def train_and_eval(train_rows, eval_rows, model_path, device, seed, epochs=10, lr=1e-4, batch_size=8):
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
    
    # Per-query train accuracy
    model.eval(); head.eval()
    by_query_train = collections.defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for start in range(0, len(train_rows), 32):
            end = min(start + 32, len(train_rows))
            batch = {k: v[start:end].to(device) for k, v in train_enc.items()}
            labs = train_lab[start:end].to(device)
            out = model(**batch).last_hidden_state[:, 0, :]
            logits = head(out)
            preds = logits.argmax(-1)
            for j in range(end - start):
                q = train_rows[start + j]["query"]
                by_query_train[q][0] += int(preds[j] == labs[j])
                by_query_train[q][1] += 1
    
    train_by_query = {q: v[0] / v[1] for q, v in by_query_train.items()}
    
    # Eval
    eval_enc, eval_lab = encode(eval_rows)
    eval_results = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    with torch.no_grad():
        for start in range(0, len(eval_rows), 32):
            end = min(start + 32, len(eval_rows))
            batch = {k: v[start:end].to(device) for k, v in eval_enc.items()}
            labs = eval_lab[start:end].to(device)
            out = model(**batch).last_hidden_state[:, 0, :]
            logits = head(out)
            preds = logits.argmax(-1)
            for j in range(end - start):
                q = eval_rows[start + j]["query"]
                eval_results[q]["correct"][0] += int(preds[j] == labs[j])
                eval_results[q]["correct"][1] += 1
    
    eval_by_query = {}
    for q in eval_results:
        c = eval_results[q]["correct"]
        eval_by_query[q] = c[0] / c[1] if c[1] > 0 else None
    
    del model, head
    torch.cuda.empty_cache()
    
    return {
        "final_train_acc": history[-1]["acc"],
        "train_by_query": dict(train_by_query),
        "eval_by_query": dict(eval_by_query),
        "epochs_used": len(history),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--out_dir", default=str(WS / "data/temporal_slot_marker_test"))
    parser.add_argument("--n_train", type=int, default=40)
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=26600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_build", action="store_true")
    args = parser.parse_args()
    
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print(json.dumps({"status": "SLOT_MARKER_START", "args": {k: str(v) for k, v in vars(args).items()}}), flush=True)
    
    changed_tr, changed_he = load_changed_worlds()
    print(json.dumps({"changed_train": len(changed_tr), "changed_held": len(changed_he)}), flush=True)
    
    rng = random.Random(args.seed)
    rng.shuffle(changed_tr)
    rng.shuffle(changed_he)
    train_worlds = changed_tr[:args.n_train]
    eval_worlds = changed_he[:args.n_eval]
    
    # Check label balance
    for fmt_name in CTX_FORMATS:
        rows = build_rows(train_worlds, fmt_name, f"train_{fmt_name}", "train")
        labs = collections.Counter(r["label"] for r in rows)
        queries = collections.Counter(r["query"] for r in rows)
        print(json.dumps({"fmt": fmt_name, "n_train_rows": len(rows),
                          "labels": dict(labs), "queries": dict(queries)}), flush=True)
    
    if args.dry_build:
        print(json.dumps({"status": "DRY_DONE"}), flush=True)
        return
    
    results = {}
    for fmt_name in ["natural", "structural", "hybrid"]:
        train_rows = build_rows(train_worlds, fmt_name, f"train_{fmt_name}", "train")
        eval_rows = build_rows(eval_worlds, fmt_name, f"eval_{fmt_name}", "held")
        
        res = train_and_eval(train_rows, eval_rows, args.model_path, args.device,
                             args.seed, epochs=args.epochs)
        results[fmt_name] = res
        
        print(json.dumps({"fmt": fmt_name,
                          "train_acc": res["final_train_acc"],
                          "train_before": res["train_by_query"].get("before"),
                          "train_after": res["train_by_query"].get("after"),
                          "eval_before": res["eval_by_query"].get("before"),
                          "eval_after": res["eval_by_query"].get("after"),
                          "epochs": res["epochs_used"]}), flush=True)
    
    # Representational similarity check for each format
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(args.device).eval()
    
    for fmt_name in ["natural", "structural", "hybrid"]:
        # Pick a few rows and compare before/after embeddings
        test_rows = build_rows(train_worlds[:3], fmt_name, f"sim_{fmt_name}", "test")
        before_rows = [r for r in test_rows if r["query"] == "before"][:6]
        after_rows = [r for r in test_rows if r["query"] == "after"][:6]
        
        with torch.no_grad():
            be = tokenizer([r["text"] for r in before_rows], padding=True, truncation=True,
                          max_length=160, return_tensors="pt").to(args.device)
            ae = tokenizer([r["text"] for r in after_rows], padding=True, truncation=True,
                          max_length=160, return_tensors="pt").to(args.device)
            b_cls = model(**be).last_hidden_state[:, 0, :].float()
            a_cls = model(**ae).last_hidden_state[:, 0, :].float()
        
        # Paired cosine
        cos_vals = F.cosine_similarity(b_cls, a_cls).cpu().numpy()
        results[f"{fmt_name}_before_after_cos"] = float(np.mean(cos_vals))
        print(json.dumps({"fmt": fmt_name, "pretrained_before_after_cos": float(np.mean(cos_vals)),
                          "cos_std": float(np.std(cos_vals))}), flush=True)
    
    del model
    torch.cuda.empty_cache()
    
    # Summary
    summary = {
        "status": "SLOT_MARKER_DONE",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_train_worlds": len(train_worlds),
        "n_eval_worlds": len(eval_worlds),
        "formats": {},
    }
    for fmt_name in ["natural", "structural", "hybrid"]:
        r = results[fmt_name]
        summary["formats"][fmt_name] = {
            "train_acc": r["final_train_acc"],
            "train_before": r["train_by_query"].get("before"),
            "train_after": r["train_by_query"].get("after"),
            "eval_before": r["eval_by_query"].get("before"),
            "eval_after": r["eval_by_query"].get("after"),
            "pretrained_before_after_cos": results.get(f"{fmt_name}_before_after_cos"),
            "epochs": r["epochs_used"],
        }
    
    (out / "slot_marker_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    md = ["# research temporal slot marker test\n"]
    md.append("| Format | Train acc | Train before | Train after | Eval before | Eval after | Before/After cos |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for fmt_name in ["natural", "structural", "hybrid"]:
        s = summary["formats"][fmt_name]
        md.append(f"| {fmt_name} | {s['train_acc']:.3f} | {s['train_before']:.3f} | "
                  f"{s['train_after']:.3f} | {s['eval_before']:.3f} | {s['eval_after']:.3f} | "
                  f"{s.get('pretrained_before_after_cos', 0):.4f} |")
    md.append("\nIf structural/hybrid can fit changed rows while natural cannot, the bottleneck is")
    md.append("pretrained representational grounding of temporal cues, not architectural capacity.\n")
    
    (out / "slot_marker_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "SLOT_MARKER_DONE",
                      "summary_json": str(out / "slot_marker_summary.json")}), flush=True)


if __name__ == "__main__":
    main()
