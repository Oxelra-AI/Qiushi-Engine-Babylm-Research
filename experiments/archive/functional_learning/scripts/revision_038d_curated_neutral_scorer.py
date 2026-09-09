#!/usr/bin/env python3
"""Step038d: no-update neutral scoring for the curated source-grounded probe.

For each curated pair, score target_source_answer vs shared_new_answer in a
context with the raw source and the same target use frame but no update sentence.
This distinguishes a general use-frame/new-answer bias from the specific damage
caused by a distractor update.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import sys
from typing import Any

import torch

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))

from corruption_vs_loss_factorial import locate_span_token_positions, load_private_model  # noqa: E402


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_candidate(model, tokenizer, source: str, use_frame: str, candidate: str, device: torch.device, seq_length: int) -> dict[str, Any]:
    before, after = use_frame.split("{STATE}")
    use = before + candidate + after
    prefix = source.rstrip() + " "
    full = prefix + use
    cand_start = len(prefix) + len(before)
    cand_end = cand_start + len(candidate)
    enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True, return_tensors="pt", max_length=seq_length, truncation=True)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
    positions = locate_span_token_positions(offsets, cand_start, cand_end)
    if not positions:
        raise ValueError(f"candidate span not tokenized/truncated: {candidate!r}")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    target_ids = input_ids[0, positions].detach().clone()
    masked = input_ids.clone()
    masked[0, positions] = int(tokenizer.mask_token_id)
    with torch.no_grad():
        logits = model(input_ids=masked, attention_mask=attention_mask).logits[0]
        logp = torch.log_softmax(logits[positions], dim=-1)
        vals = logp[torch.arange(len(positions), device=device), target_ids].detach().cpu().tolist()
    return {
        "full_text": full,
        "use_sentence": use,
        "candidate": candidate,
        "n_tokens": len(positions),
        "token_logps": vals,
        "sum_logp": float(sum(vals)),
        "mean_logp": float(sum(vals) / len(vals)),
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="experiments/archive/functional_learning/data/curated_source_grounded_probe/curated_source_grounded_pairs.jsonl")
    ap.add_argument("--paired-scores", default="experiments/archive/functional_learning/data/curated_source_grounded_probe/scorer_coherent86_cpu/pair_scores.jsonl")
    ap.add_argument("--model-path", default="models/frontier")
    ap.add_argument("--out-dir", default="experiments/archive/functional_learning/data/curated_source_grounded_probe/neutral_scorer_coherent86_cpu")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--private-bottleneck", type=int, default=128)
    ap.add_argument("--private-scale", type=float, default=0.75)
    args = ap.parse_args()

    pairs = read_jsonl(pathlib.Path(args.pairs))
    paired = {r["pair_id"]: r for r in read_jsonl(pathlib.Path(args.paired_scores))}
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    model, load_info = load_private_model(pathlib.Path(args.model_path), args.private_bottleneck, args.private_scale, device)
    model.eval()

    row_scores = []
    for p in pairs:
        source_score = score_candidate(model, tokenizer, p["source_sentence"], p["use_sentence_frame"], p["target_source_answer"], device, args.seq_length)
        new_score = score_candidate(model, tokenizer, p["source_sentence"], p["use_sentence_frame"], p["shared_new_answer"], device, args.seq_length)
        neutral_source_minus_new = source_score["mean_logp"] - new_score["mean_logp"]
        neutral_source_minus_new_sum = source_score["sum_logp"] - new_score["sum_logp"]
        ps = paired.get(p["pair_id"], {})
        U = float(ps.get("U_mean_new_minus_source", float("nan")))
        R = float(ps.get("R_mean_source_minus_new", float("nan")))
        row_scores.append({
            "pair_id": p["pair_id"],
            "target_entity": p["target_entity"],
            "distractor_entity": p["distractor_entity"],
            "target_source_answer": p["target_source_answer"],
            "shared_new_answer": p["shared_new_answer"],
            "neutral_source_minus_new_mean": neutral_source_minus_new,
            "neutral_source_minus_new_sum": neutral_source_minus_new_sum,
            "neutral_correct_mean": neutral_source_minus_new > 0,
            "neutral_correct_sum": neutral_source_minus_new_sum > 0,
            "source_answer_n_tokens": source_score["n_tokens"],
            "new_answer_n_tokens": new_score["n_tokens"],
            "paired_U_new_minus_source": U,
            "paired_R_source_minus_new": R,
            "retain_drop_vs_neutral": R - neutral_source_minus_new if not math.isnan(R) else float("nan"),
            "update_gain_vs_neutral_new_minus_source": U + neutral_source_minus_new if not math.isnan(U) else float("nan"),
            "source_score": source_score,
            "new_score": new_score,
        })
    with (out_dir / "neutral_row_scores.jsonl").open("w", encoding="utf-8") as f:
        for r in row_scores:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "status": "STEP038D_CURATED_NEUTRAL_SCORER",
        "pairs": len(row_scores),
        "model_path": args.model_path,
        "device": str(device),
        "neutral_correct_mean": sum(r["neutral_correct_mean"] for r in row_scores),
        "neutral_correct_sum": sum(r["neutral_correct_sum"] for r in row_scores),
        "mean_neutral_source_minus_new": mean([r["neutral_source_minus_new_mean"] for r in row_scores]),
        "mean_retain_R_after_distractor_update": mean([r["paired_R_source_minus_new"] for r in row_scores]),
        "mean_retain_drop_vs_neutral": mean([r["retain_drop_vs_neutral"] for r in row_scores]),
        "mean_paired_U": mean([r["paired_U_new_minus_source"] for r in row_scores]),
        "load_info": {"missing": list(load_info.get("missing", []))[:20], "unexpected": list(load_info.get("unexpected", []))[:20]},
    }
    (out_dir / "neutral_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Step038d curated no-update neutral scorer\n\n"]
    md.append("No-update context: raw source + same target use frame, with no update sentence. Positive neutral margin means coherent86 prefers the original source answer when no entity has been updated.\n\n")
    md.append(f"Neutral correct: {summary['neutral_correct_mean']}/{summary['pairs']} by mean-token score; mean neutral source-new={summary['mean_neutral_source_minus_new']:+.4f}.\n\n")
    md.append(f"After distractor update, mean RETAIN R={summary['mean_retain_R_after_distractor_update']:+.4f}; mean R-neutral drop={summary['mean_retain_drop_vs_neutral']:+.4f}.\n\n")
    md.append("| pair | neutral source-new | R after distractor update | drop | U target update | neutral correct |\n")
    md.append("|---|---:|---:|---:|---:|---|\n")
    for r in row_scores:
        md.append(f"| {r['pair_id']} | {r['neutral_source_minus_new_mean']:+.3f} | {r['paired_R_source_minus_new']:+.3f} | {r['retain_drop_vs_neutral']:+.3f} | {r['paired_U_new_minus_source']:+.3f} | {r['neutral_correct_mean']} |\n")
    (out_dir / "neutral_summary.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "pairs", "neutral_correct_mean", "mean_neutral_source_minus_new", "mean_retain_R_after_distractor_update", "mean_retain_drop_vs_neutral", "mean_paired_U"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
