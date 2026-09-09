#!/usr/bin/env python3
"""Probe evidence-visible masking token-budget behavior before H100 training.

Uses the clean-Qwen tokenizer and a sampled set of qwen_aligned_10M chunks, but no
model forward/backward pass. The purpose is to verify that the repaired
`--mask_budget token_count` mode keeps the number of loss-bearing subword tokens
near the uniform WWM target while changing which whole words are selected.

No downstream BabyLM examples, labels, AoA/CDI words, child curves, or leaderboard
scores are read.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import random
import statistics
import sys
import time

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
WORKSPACE = _public_path('experiments/archive/compact_experience')
sys.path.insert(0, str(SCRIPT_DIR))
import evidence_visible_continuation_trainer as ev  # noqa: E402

DEFAULT_TOKENIZER = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_80M')
DEFAULT_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/evidence_visible_mask_probe/mask_budget_probe.json')
MODES = ["uniform", "evidence_visible", "random_priority", "inverse_priority"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_examples(path: pathlib.Path, max_rows: int) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if len(out) >= max_rows:
                    break
    return out


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    xs = sorted(values)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        i = p * (len(xs) - 1)
        lo = int(i)
        hi = min(lo + 1, len(xs) - 1)
        frac = i - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p05": q(0.05),
        "p95": q(0.95),
        "min": xs[0],
        "max": xs[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max_rows", type=int, default=4096)
    ap.add_argument("--sample_chunks", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=44044)
    args = ap.parse_args()

    tokenizer_path = pathlib.Path(args.tokenizer)
    pool_path = pathlib.Path(args.pool)
    out_path = pathlib.Path(args.out)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)
    if not pool_path.exists():
        raise FileNotFoundError(pool_path)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    examples = load_examples(pool_path, args.max_rows)
    dataset = ev.ContinuationDataset(examples, tokenizer, ev.SEQ_LENGTH)
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:min(args.sample_chunks, len(indices))]
    chunks = [dataset[i] for i in indices]

    mode_rows = {}
    for mode in MODES:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.seed)
        ratios = []
        masked_words = []
        masked_tokens = []
        target_tokens = []
        high_fracs = []
        for start in range(0, len(chunks), args.batch_size):
            batch = torch.stack(chunks[start:start + args.batch_size])
            _masked, _attn, _labels, stats = ev.priority_wwm_mask_batch(
                batch, tokenizer, ev.MASK_PROB, gen, torch.device("cpu"), mode, "token_count"
            )
            ratios.append(float(stats["masked_token_budget_ratio"]))
            masked_words.append(float(stats["masked_words"]))
            masked_tokens.append(float(stats["masked_tokens"]))
            target_tokens.append(float(stats["target_masked_tokens"]))
            high_fracs.append(float(stats["high_priority_selected"]) / max(float(stats["masked_words"]), 1.0))
        mode_rows[mode] = {
            "masked_token_budget_ratio": summarize(ratios),
            "masked_words_per_batch": summarize(masked_words),
            "masked_tokens_per_batch": summarize(masked_tokens),
            "target_tokens_per_batch": summarize(target_tokens),
            "high_priority_fraction_selected": summarize(high_fracs),
            "totals": {
                "masked_words": sum(masked_words),
                "masked_tokens": sum(masked_tokens),
                "target_masked_tokens": sum(target_tokens),
                "masked_token_budget_ratio": sum(masked_tokens) / max(sum(target_tokens), 1.0),
            },
        }

    payload = {
        "status": "EVIDENCE_VISIBLE_MASK_BUDGET_PROBE",
        "created_utc": now(),
        "tokenizer": str(tokenizer_path),
        "pool": str(pool_path),
        "max_rows_loaded": args.max_rows,
        "dataset_chunks_loaded": len(dataset),
        "sample_chunks": len(chunks),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "modes": mode_rows,
        "interpretation": "Training should only proceed if token_count mode keeps masked_token_budget_ratio close to 1.0 while evidence_visible changes high-priority selection relative to uniform/random/inverse controls.",
        "non_leakage_statement": "Reads only the training pool and tokenizer; no downstream evaluation labels/items and no AoA/CDI material.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "sample_chunks": len(chunks), "mode_totals": {k: v["totals"] for k, v in mode_rows.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
