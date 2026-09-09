#!/usr/bin/env python3
"""research CPU audits for restart/masking interpretation.

1. Verify that the clean-tail WWM trainer and the evidence-visible trainer's
   `uniform + token_count` mode produce identical first-batch masks under the same
   seed, batch order, tokenizer, and data chunks. This makes `mask_uniform_control`
   a true duplicate/noise reference for the clean-tail restart implementation.
2. Measure actual base-priority strata and token-budget behavior for uniform,
   evidence_visible, random_priority, and inverse_priority on a sample of chunks,
   using the original (untransformed) surface priority for all arms.
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
import sys
import time
from typing import Any

import torch
from transformers import AutoTokenizer

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import untouched_restart_ladder_trainer as tail  # noqa: E402
import evidence_visible_continuation_trainer as evm  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
DEFAULT_INIT = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_80M')
DEFAULT_POOL = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/mask_sampler_audit/masking_equivalence_and_sampler_audit.json')
MODES = ["uniform", "evidence_visible", "random_priority", "inverse_priority"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_examples(path: pathlib.Path) -> tuple[list[dict], int]:
    examples = []
    words = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                examples.append(obj)
                words += int(obj.get("words", len(str(obj["text"]).split())))
    return examples, words


def first_batch_indices(n_chunks: int, seed: int, batch_size: int) -> list[int]:
    indices = list(range(n_chunks))
    random.seed(seed)
    random.shuffle(indices)
    return indices[:batch_size]


def apply_tail_mask(batch: torch.Tensor, tokenizer, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    masked, _att, labels, _stats = tail.wwm_mask_batch(batch, tokenizer, tail.MASK_PROB, gen, torch.device("cpu"))
    return masked.cpu(), labels.cpu()


def apply_evm_uniform(batch: torch.Tensor, tokenizer, seed: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    masked, _att, labels, stats = evm.priority_wwm_mask_batch(
        batch, tokenizer, evm.MASK_PROB, gen, torch.device("cpu"), "uniform", "token_count"
    )
    return masked.cpu(), labels.cpu(), stats


def base_priority_stats_for_chunk(ids: torch.Tensor, tokenizer, mode: str, seed: int) -> dict[str, Any]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    word_groups = evm.build_word_groups(ids, tokenizer)
    token_strs = [str(tokenizer.convert_ids_to_tokens(int(tid))) for tid in ids.tolist()]
    if not word_groups:
        return {}
    base = torch.tensor([evm.priority_for_group(g, token_strs, i, word_groups) for i, g in enumerate(word_groups)], dtype=torch.float32)
    priorities = evm.make_priorities(word_groups, token_strs, mode, gen)
    n_mask_uniform = max(1, int(len(word_groups) * evm.MASK_PROB))
    n_mask_uniform = min(n_mask_uniform, len(word_groups))
    uniform_perm = torch.randperm(len(word_groups), generator=gen)
    uniform_target_words = uniform_perm[:n_mask_uniform].tolist()
    target_tokens = sum(len(word_groups[wi]) for wi in uniform_target_words)
    if mode == "uniform":
        chosen = uniform_target_words
    else:
        chosen = evm.choose_priority_words(word_groups, priorities, target_tokens, gen, mode=mode, budget_mode="token_count")
    high_cut = torch.quantile(base, 0.75).item() if base.numel() >= 4 else base.max().item()
    selected_base_priorities = [float(base[wi].item()) for wi in chosen]
    selected_lengths = [len(word_groups[wi]) for wi in chosen]
    available_high = int((base >= high_cut).sum().item())
    selected_high = sum(1 for wi in chosen if float(base[wi].item()) >= high_cut)
    selected_tokens = sum(selected_lengths)
    return {
        "n_words": len(word_groups),
        "target_tokens": target_tokens,
        "selected_tokens": selected_tokens,
        "token_budget_ratio": selected_tokens / max(target_tokens, 1),
        "selected_words": len(chosen),
        "uniform_target_words": n_mask_uniform,
        "available_base_high_words": available_high,
        "selected_base_high_words": selected_high,
        "selected_base_high_fraction": selected_high / max(len(chosen), 1),
        "selected_base_priority_mean": sum(selected_base_priorities) / max(len(selected_base_priorities), 1),
        "selected_wordpiece_len_mean": sum(selected_lengths) / max(len(selected_lengths), 1),
    }


def summarize_sampler(dataset, tokenizer, sample_chunks: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + 991)
    ids = list(range(len(dataset)))
    rng.shuffle(ids)
    ids = ids[:min(sample_chunks, len(ids))]
    out = {}
    for mode in MODES:
        totals = {
            "chunks": 0,
            "target_tokens": 0,
            "selected_tokens": 0,
            "selected_words": 0,
            "uniform_target_words": 0,
            "available_base_high_words": 0,
            "selected_base_high_words": 0,
            "selected_base_priority_sum": 0.0,
            "selected_wordpiece_len_sum": 0.0,
        }
        for k, idx in enumerate(ids):
            stats = base_priority_stats_for_chunk(dataset[idx], tokenizer, mode, seed + 17 * k + 100003)
            if not stats:
                continue
            totals["chunks"] += 1
            for key in ["target_tokens", "selected_tokens", "selected_words", "uniform_target_words", "available_base_high_words", "selected_base_high_words"]:
                totals[key] += stats[key]
            totals["selected_base_priority_sum"] += stats["selected_base_priority_mean"] * stats["selected_words"]
            totals["selected_wordpiece_len_sum"] += stats["selected_wordpiece_len_mean"] * stats["selected_words"]
        out[mode] = {
            **totals,
            "token_budget_ratio": totals["selected_tokens"] / max(totals["target_tokens"], 1),
            "selected_words_over_uniform_word_count": totals["selected_words"] / max(totals["uniform_target_words"], 1),
            "base_high_fraction_among_selected_words": totals["selected_base_high_words"] / max(totals["selected_words"], 1),
            "base_high_selection_rate": totals["selected_base_high_words"] / max(totals["available_base_high_words"], 1),
            "selected_base_priority_mean": totals["selected_base_priority_sum"] / max(totals["selected_words"], 1),
            "selected_wordpiece_len_mean": totals["selected_wordpiece_len_sum"] / max(totals["selected_words"], 1),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(DEFAULT_INIT))
    ap.add_argument("--pool", default=str(DEFAULT_POOL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=43044)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--sample_chunks", type=int, default=2048)
    args = ap.parse_args()

    init = pathlib.Path(args.init)
    pool = pathlib.Path(args.pool)
    tokenizer = AutoTokenizer.from_pretrained(str(init), use_fast=True)
    examples, pool_words = load_examples(pool)
    dataset = evm.ContinuationDataset(examples, tokenizer, evm.SEQ_LENGTH)
    batch_idxs = first_batch_indices(len(dataset), args.seed, args.batch_size)
    batch = torch.stack([dataset[i] for i in batch_idxs])

    tail_masked, tail_labels = apply_tail_mask(batch, tokenizer, args.seed)
    evm_masked, evm_labels, evm_stats = apply_evm_uniform(batch, tokenizer, args.seed)
    equivalence = {
        "masked_inputs_equal": bool(torch.equal(tail_masked, evm_masked)),
        "labels_equal": bool(torch.equal(tail_labels, evm_labels)),
        "label_disagreements": int((tail_labels != evm_labels).sum().item()),
        "masked_input_disagreements": int((tail_masked != evm_masked).sum().item()),
        "tail_masked_tokens": int((tail_labels != -100).sum().item()),
        "evm_masked_tokens": int((evm_labels != -100).sum().item()),
        "evm_uniform_stats": evm_stats,
        "first_batch_indices_sha1": __import__("hashlib").sha1(json.dumps(batch_idxs).encode()).hexdigest(),
    }
    sampler = summarize_sampler(dataset, tokenizer, args.sample_chunks, args.seed)
    payload = {
        "status": "MASKING_EQUIVALENCE_AND_SAMPLER_AUDIT",
        "created_utc": now(),
        "init": str(init),
        "pool": str(pool),
        "pool_words": pool_words,
        "dataset_chunks": len(dataset),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sample_chunks": min(args.sample_chunks, len(dataset)),
        "clean_tail_vs_mask_uniform_first_batch": equivalence,
        "sampler_base_priority_summary": sampler,
        "interpretation": "Uniform equivalence should be exact for the first batch. Sampler summaries use original base surface priority for all modes, avoiding the arm-internal high-priority ambiguity in training metrics.",
        "non_leakage_statement": "Uses only training pool text/tokenization and masking code; no downstream labels/items and no AoA/CDI material.",
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "equivalence": equivalence, "sampler_modes": sampler}, indent=2), flush=True)


if __name__ == "__main__":
    main()
