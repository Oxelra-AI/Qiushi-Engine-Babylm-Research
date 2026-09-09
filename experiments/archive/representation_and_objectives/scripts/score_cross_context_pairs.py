#!/usr/bin/env python3
"""research: Score corpus-derived cross-context pairs on existing MLM checkpoints.

For each pair (Ca,a,b; Cb,b,a), computes:
  margin_a = logit(a|Ca) - logit(b|Ca)
  margin_b = logit(b|Cb) - logit(a|Cb)
  delta = margin_a + margin_b
This is the paired correct-assignment vs swapped-assignment logit difference;
target-specific global priors cancel exactly.

Also reports crossed success (both margins > 0), mean delta, robust quantiles,
and performance by pair family. No official evaluation labels are used.
"""

import argparse, json, math, os, re, statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def mask_state(context: str, start: int, end: int, mask_token: str) -> str:
    # Verify span is nonempty; replacement by character offsets preserved from extraction.
    assert 0 <= start < end <= len(context), (start, end, len(context))
    return context[:start] + mask_token + context[end:]


def token_id(tok, word: str):
    # Byte-BPE tokenizers encode in-sentence words with leading space.
    ids = tok.encode(" " + word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    ids2 = tok.encode(word, add_special_tokens=False)
    if len(ids2) == 1:
        return ids2[0]
    return None


def score_batch(model, tok, texts, candidates, device, max_length=256):
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    mask_id = tok.mask_token_id
    mask_pos = []
    valid = []
    for i, ids in enumerate(enc["input_ids"]):
        pos = (ids == mask_id).nonzero(as_tuple=False).flatten()
        if len(pos) == 1:
            mask_pos.append(int(pos[0]))
            valid.append(True)
        else:
            mask_pos.append(0)
            valid.append(False)
    with torch.no_grad():
        logits = model(**enc).logits
    out = []
    for i, ((target_id, foil_id), ok) in enumerate(zip(candidates, valid)):
        if not ok or target_id is None or foil_id is None:
            out.append(None)
            continue
        p = mask_pos[i]
        margin = float((logits[i, p, target_id] - logits[i, p, foil_id]).cpu())
        out.append(margin)
    return out


def summarize(records):
    margins_a = [r["margin_a"] for r in records]
    margins_b = [r["margin_b"] for r in records]
    deltas = [r["delta"] for r in records]
    crossed = [r["margin_a"] > 0 and r["margin_b"] > 0 for r in records]
    summary = {
        "n": len(records),
        "crossed_success": sum(crossed) / len(crossed) if crossed else None,
        "mean_margin_a": statistics.mean(margins_a) if margins_a else None,
        "mean_margin_b": statistics.mean(margins_b) if margins_b else None,
        "mean_delta": statistics.mean(deltas) if deltas else None,
        "median_delta": statistics.median(deltas) if deltas else None,
        "delta_q10": float(np.quantile(deltas, 0.1)) if deltas else None,
        "delta_q25": float(np.quantile(deltas, 0.25)) if deltas else None,
        "delta_positive_frac": sum(d > 0 for d in deltas) / len(deltas) if deltas else None,
        "worst_context_accuracy": sum(min(r["margin_a"], r["margin_b"]) > 0 for r in records) / len(records) if records else None,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "load_model", "model": args.model, "device": str(device)}), flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device).eval()

    pairs = []
    with open(args.pairs) as f:
        for line in f:
            pairs.append(json.loads(line))
            if args.max_pairs and len(pairs) >= args.max_pairs:
                break

    records = []
    skipped = defaultdict(int)
    for base in range(0, len(pairs), args.batch_size):
        chunk = pairs[base:base + args.batch_size]
        texts = []
        candidates = []
        meta = []
        for p in chunk:
            ta, tb = p["target_a"], p["target_b"]
            ida, idb = token_id(tok, ta), token_id(tok, tb)
            if ida is None or idb is None:
                skipped["multitoken"] += 1
                continue
            try:
                ca = mask_state(p["context_a"], p["state_start_a"], p["state_end_a"], tok.mask_token)
                cb = mask_state(p["context_b"], p["state_start_b"], p["state_end_b"], tok.mask_token)
            except AssertionError:
                skipped["bad_span"] += 1
                continue
            # Sanity: each generated context has exactly one mask string.
            if ca.count(tok.mask_token) != 1 or cb.count(tok.mask_token) != 1:
                skipped["mask_count"] += 1
                continue
            texts.extend([ca, cb])
            candidates.extend([(ida, idb), (idb, ida)])
            meta.append(p)
        if not texts:
            continue
        margins = score_batch(model, tok, texts, candidates, device)
        for i, p in enumerate(meta):
            ma, mb = margins[2*i], margins[2*i+1]
            if ma is None or mb is None:
                skipped["truncated_or_mask_missing"] += 1
                continue
            family = p.get("antonym_pair", p.get("entity", "unknown"))
            records.append({
                "pair_type": p.get("pair_type"),
                "family": family,
                "target_a": p["target_a"], "target_b": p["target_b"],
                "row_a": p["row_a"], "row_b": p["row_b"],
                "margin_a": ma, "margin_b": mb,
                "delta": ma + mb,
                "crossed_success": ma > 0 and mb > 0,
            })
        if base % (args.batch_size * 10) == 0:
            print(json.dumps({"event": "progress", "done": min(base + args.batch_size, len(pairs)), "scored": len(records)}), flush=True)

    # Aggregate by pair type and family.
    by_type = defaultdict(list)
    by_family = defaultdict(list)
    for r in records:
        by_type[r["pair_type"]].append(r)
        by_family[r["family"]].append(r)

    out = {
        "status": "CROSS_CONTEXT_SCORE",
        "model": args.model,
        "pairs": args.pairs,
        "overall": summarize(records),
        "by_type": {k: summarize(v) for k, v in sorted(by_type.items())},
        "by_family": {k: summarize(v) for k, v in sorted(by_family.items())},
        "skipped": dict(skipped),
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"status": out["status"], "overall": out["overall"], "skipped": out["skipped"], "output": str(out_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
