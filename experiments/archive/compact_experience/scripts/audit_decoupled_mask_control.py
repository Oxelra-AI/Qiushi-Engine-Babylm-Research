#!/usr/bin/env python3
"""CPU audit for research decoupled masking controls.

This does not load a model and does not train. It tokenizes a small prefix of the
clean Qwen-aligned pool, exercises the new decoupled masking sampler, and writes
facts needed before any full GPU control run:

- token-budget matching for each mode;
- inverse vs length_matched_random selected wordpiece-length histogram agreement;
- base-priority/high-word differences between inverse and length-matched random;
- deterministic corruption decisions on overlapping selected token positions.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
from collections import Counter
from typing import Any

import torch
from transformers import AutoTokenizer

import evidence_visible_continuation_trainer as old
import decoupled_mask_control_trainer as new

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
DEFAULT_TOKENIZER = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_80M')
DEFAULT_OUT = _public_path('data/external/decoupled_mask_audit.json')
MODES = ["uniform", "inverse_priority", "length_matched_random", "evidence_visible", "random_priority"]


def load_examples(path: pathlib.Path, n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if len(out) >= n:
                    break
    return out


def selected_positions(labels: torch.Tensor) -> set[int]:
    return {int(i) for i in torch.nonzero(labels[0] != -100, as_tuple=False).flatten().tolist()}


def analyze_one_sequence(x: torch.Tensor, tokenizer, seed: int, seq_index: int) -> dict[str, Any]:
    # Use batch size 1 to make length-template exactness easy to inspect.
    input_ids = x.unsqueeze(0)
    rows: dict[str, Any] = {}
    labels_by_mode: dict[str, torch.Tensor] = {}
    masked_by_mode: dict[str, torch.Tensor] = {}
    for mode in MODES:
        masked, _attn, labels, stats = new.decoupled_mask_batch(
            input_ids, tokenizer, old.MASK_PROB, torch.device("cpu"), mode, "token_count", seed, global_step=0,
            batch_indices=[seq_index],
        )
        rows[mode] = dict(stats)
        labels_by_mode[mode] = labels.cpu()
        masked_by_mode[mode] = masked.cpu()
    inv_pos = selected_positions(labels_by_mode["inverse_priority"])
    lm_pos = selected_positions(labels_by_mode["length_matched_random"])
    overlap = sorted(inv_pos & lm_pos)
    corruption_mismatches = 0
    for pos in overlap:
        if int(masked_by_mode["inverse_priority"][0, pos].item()) != int(masked_by_mode["length_matched_random"][0, pos].item()):
            corruption_mismatches += 1
    # Directly compare selected wordpiece-length multiset by reconstructing selected word groups.
    word_groups = old.build_word_groups(x, tokenizer)
    token_strs = [str(tokenizer.convert_ids_to_tokens(int(tid))) for tid in x.tolist()]
    base_prior = torch.tensor([old.priority_for_group(g, token_strs, j, word_groups) for j, g in enumerate(word_groups)], dtype=torch.float32)
    base_prior = torch.clamp(base_prior, min=0.05)
    n_mask_uniform = max(1, int(len(word_groups) * old.MASK_PROB))
    g_budget = new.make_gen(seed, "budget", 0, seq_index, 0)
    uniform_target_words = torch.randperm(len(word_groups), generator=g_budget)[:n_mask_uniform].tolist()
    target_tokens = sum(len(word_groups[wi]) for wi in uniform_target_words)
    inv_chosen, _p, _r = new.choose_words_decoupled(word_groups, base_prior, "inverse_priority", "token_count", target_tokens, seed, 0, seq_index, 0)
    lm_chosen, _p2, _r2 = new.choose_words_decoupled(word_groups, base_prior, "length_matched_random", "token_count", target_tokens, seed, 0, seq_index, 0)
    return {
        "sequence_index": seq_index,
        "mode_stats": rows,
        "inverse_selected_tokens": len(inv_pos),
        "length_matched_selected_tokens": len(lm_pos),
        "overlap_selected_tokens": len(overlap),
        "corruption_mismatches_on_overlap": corruption_mismatches,
        "inverse_selected_word_lengths": sorted(len(word_groups[j]) for j in inv_chosen),
        "length_matched_selected_word_lengths": sorted(len(word_groups[j]) for j in lm_chosen),
        "length_histograms_equal": Counter(len(word_groups[j]) for j in inv_chosen) == Counter(len(word_groups[j]) for j in lm_chosen),
        "inverse_selected_word_count": len(inv_chosen),
        "length_matched_selected_word_count": len(lm_chosen),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--examples", type=int, default=256)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=48044)
    args = ap.parse_args()

    pool = pathlib.Path(args.pool)
    tok_root = pathlib.Path(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(str(tok_root), use_fast=True)
    examples = load_examples(pool, args.examples)
    dataset = old.ContinuationDataset(examples, tokenizer, old.SEQ_LENGTH)
    n = min(args.chunks, len(dataset))
    totals: dict[str, Counter] = {m: Counter() for m in MODES}
    sequences = []
    length_ok = True
    corruption_mismatches = 0
    overlap_total = 0
    for i in range(n):
        rec = analyze_one_sequence(dataset[i], tokenizer, args.seed, i)
        sequences.append(rec)
        length_ok = length_ok and bool(rec["length_histograms_equal"])
        corruption_mismatches += int(rec["corruption_mismatches_on_overlap"])
        overlap_total += int(rec["overlap_selected_tokens"])
        for m in MODES:
            for k, v in rec["mode_stats"][m].items():
                if isinstance(v, (int, float)):
                    totals[m][k] += v
    summary = {}
    for m in MODES:
        t = totals[m]
        summary[m] = {
            "masked_words": int(t["masked_words"]),
            "masked_tokens": int(t["masked_tokens"]),
            "target_masked_tokens": int(t["target_masked_tokens"]),
            "token_budget_ratio": float(t["masked_tokens"] / max(t["target_masked_tokens"], 1)),
            "base_high_fraction_among_selected_words": float(t["selected_base_high_words"] / max(t["masked_words"], 1)),
            "selected_wordpiece_len_mean": float(t["selected_wordpiece_len_sum"] / max(t["masked_words"], 1)),
            "selected_base_priority_mean": float((t["selected_base_priority_sum_x1000"] / 1000.0) / max(t["masked_words"], 1)),
            "length_template_match_failures": int(t["length_template_match_failures"]),
        }
    payload = {
        "status": "DECOUPLED_MASK_CONTROL_AUDIT",
        "pool": str(pool),
        "tokenizer": str(tok_root),
        "examples_loaded": len(examples),
        "chunks_audited": n,
        "seed": args.seed,
        "summary_by_mode": summary,
        "all_inverse_length_histograms_matched_by_control": length_ok,
        "overlap_selected_tokens_inverse_vs_length_matched": overlap_total,
        "corruption_mismatches_on_overlap_inverse_vs_length_matched": corruption_mismatches,
        "sequences": sequences[:5],
        "interpretation": "length_matched_random must match inverse selected wordpiece-length histograms while shifting base-priority identity; corruption mismatch on overlapping positions should be zero because corruption RNG is indexed by sequence/token, not selection-loop state.",
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "chunks_audited": n,
        "all_inverse_length_histograms_matched_by_control": length_ok,
        "corruption_mismatches_on_overlap": corruption_mismatches,
        "summary_by_mode": summary,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
