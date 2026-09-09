#!/usr/bin/env python3
"""research/044: held-out cluster diagnostic for the natural-cluster mechanism.

For each continuation arm checkpoint, compute masked-language-model loss on:
  1. Held-out sentences from selected clusters (anchor-bearing, not in the packed rows)
  2. Seen cluster sentences (one training-side sentence per held-out cluster)

A useful natural-cluster mechanism should reduce held-out cluster loss for E1
relative to matched controls, not merely reduce loss on repeated/seen text.
This diagnostic uses only training-corpus cluster material. It does not use
BabyLM downstream labels/items, AoA/CDI words, child curves, or leaderboard scores.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import os
import pathlib
import time
from typing import Iterable

import torch
from transformers import AutoTokenizer, DebertaV2ForMaskedLM

ROOT = _public_path('experiments/archive/compact_experience')
HELDOUT = _public_path('experiments/archive/compact_experience/data/cluster_mechanism_test/heldout_diagnostic.jsonl')
RUNS = _public_path('experiments/archive/compact_experience/training/runs')
OUT = _public_path('experiments/archive/compact_experience/data/cluster_heldout_diagnostic')
ARMS = ["E1_true_cluster", "E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]
CHECKPOINTS = ["chck_99M"]  # continuation final checkpoint is 80M + ~19.9M exposure
MAX_LENGTH = 256


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def deterministic_mask_flag(text_index: int, pos: int, seed: int, mask_prob: float) -> bool:
    """Stable Bernoulli mask decision independent of batching and model arm."""
    raw = f"{seed}:{text_index}:{pos}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") / float(1 << 64)
    return value < mask_prob


def iter_batches(xs: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for i in range(0, len(xs), batch_size):
        yield i, xs[i:i + batch_size]


def make_masked_batch(tokenizer, texts: list[str], start_index: int, seed: int, mask_prob: float, device: torch.device):
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = torch.full_like(input_ids, -100)
    masked_input = input_ids.clone()
    special_ids = set(tokenizer.all_special_ids)
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    for b in range(input_ids.size(0)):
        valid_positions = []
        for pos in range(input_ids.size(1)):
            tid = int(input_ids[b, pos].item())
            if attention_mask[b, pos].item() == 0:
                continue
            if tid in special_ids or (pad_id is not None and tid == pad_id):
                continue
            valid_positions.append(pos)
        chosen = [pos for pos in valid_positions if deterministic_mask_flag(start_index + b, pos, seed, mask_prob)]
        if not chosen and valid_positions:
            # Ensure every usable sentence contributes at least one token, deterministically.
            raw = f"fallback:{seed}:{start_index + b}".encode("utf-8")
            k = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") % len(valid_positions)
            chosen = [valid_positions[k]]
        for pos in chosen:
            labels[b, pos] = input_ids[b, pos]
            masked_input[b, pos] = mask_id

    return masked_input.to(device), attention_mask.to(device), labels.to(device)


def compute_masked_loss(model, tokenizer, texts: list[str], device: torch.device, *,
                        mask_prob: float = 0.15, seed: int = 12345, batch_size: int = 32) -> dict:
    """Compute average MLM loss with deterministic token masking."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_texts = 0
    total_batches = 0

    with torch.no_grad():
        for start, batch_texts in iter_batches(texts, batch_size):
            masked_input, attention_mask, labels = make_masked_batch(
                tokenizer, batch_texts, start, seed, mask_prob, device
            )
            n_masked = int((labels != -100).sum().item())
            if n_masked == 0:
                continue
            out = model(input_ids=masked_input, attention_mask=attention_mask, labels=labels)
            total_loss += float(out.loss.item()) * n_masked
            total_tokens += n_masked
            total_texts += len(batch_texts)
            total_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    return {"avg_loss": avg_loss, "total_tokens": total_tokens, "n_texts": total_texts, "batches": total_batches}


def load_texts(limit: int | None) -> tuple[list[str], list[str]]:
    heldout_items: list[dict] = []
    with HELDOUT.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                heldout_items.append(json.loads(line))
    heldout_texts = [item["heldout_text"] for item in heldout_items if item.get("heldout_text")]
    seen_texts = []
    for item in heldout_items:
        if item.get("cluster_texts"):
            seen_texts.append(item["cluster_texts"][0])
    n = min(len(heldout_texts), len(seen_texts))
    heldout_texts = heldout_texts[:n]
    seen_texts = seen_texts[:n]
    if limit is not None:
        heldout_texts = heldout_texts[:limit]
        seen_texts = seen_texts[:limit]
    return heldout_texts, seen_texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0, help="Optional text limit for smoke tests; 0 uses all heldout texts.")
    ap.add_argument("--out", default=str(_public_path('experiments/archive/compact_experience/data/cluster_heldout_diagnostic/heldout_diagnostic_results.json')))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    heldout_texts, seen_texts = load_texts(args.limit if args.limit > 0 else None)
    print(json.dumps({
        "event": "loaded_diagnostic_texts",
        "heldout_texts": len(heldout_texts),
        "seen_texts": len(seen_texts),
        "checkpoints": args.checkpoints,
        "arms": args.arms,
        "batch_size": args.batch_size,
        "device": str(device),
        "started_utc": now(),
    }), flush=True)

    results: list[dict] = []
    for arm in args.arms:
        for ckpt in args.checkpoints:
            ckpt_path = RUNS / f"continuation_{arm}" / "hf_model" / ckpt
            if not ckpt_path.exists():
                print(json.dumps({"event": "skip_missing_checkpoint", "arm": arm, "checkpoint": ckpt, "path": str(ckpt_path)}), flush=True)
                continue

            print(json.dumps({"event": "loading", "arm": arm, "checkpoint": ckpt, "path": str(ckpt_path)}), flush=True)
            model = DebertaV2ForMaskedLM.from_pretrained(str(ckpt_path))
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), use_fast=True)

            heldout_result = compute_masked_loss(
                model, tokenizer, heldout_texts, device, mask_prob=args.mask_prob, seed=12345, batch_size=args.batch_size
            )
            seen_result = compute_masked_loss(
                model, tokenizer, seen_texts, device, mask_prob=args.mask_prob, seed=54321, batch_size=args.batch_size
            )
            entry = {
                "arm": arm,
                "checkpoint": ckpt,
                "heldout_loss": round(heldout_result["avg_loss"], 6),
                "heldout_tokens": heldout_result["total_tokens"],
                "heldout_texts": heldout_result["n_texts"],
                "seen_loss": round(seen_result["avg_loss"], 6),
                "seen_tokens": seen_result["total_tokens"],
                "seen_texts": seen_result["n_texts"],
                "generalization_gap": round(heldout_result["avg_loss"] - seen_result["avg_loss"], 6),
            }
            results.append(entry)
            print(json.dumps({"event": "diagnostic_done", **entry}), flush=True)
            del model
            torch.cuda.empty_cache()

    diagnostic = {
        "status": "HELDOUT_DIAGNOSTIC",
        "created_utc": now(),
        "mask_prob": args.mask_prob,
        "batch_size": args.batch_size,
        "n_heldout_sentences": len(heldout_texts),
        "n_seen_sentences": len(seen_texts),
        "results": results,
        "interpretation": {
            "E1_lower_heldout_than_E2": "Complementary same-block evidence helps beyond anchor exposure/topic continuity.",
            "E1_lower_heldout_than_E3": "Diverse evidence helps beyond mere repetition.",
            "E1_lower_heldout_than_E4": "Cluster packing teaches recoverable entity/relation knowledge beyond untouched continuation.",
        },
        "non_leakage_statement": "Uses only training-corpus cluster heldout/seen sentences; no BabyLM downstream labels/items, AoA/CDI words, child curves, or leaderboard scores.",
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Held-out diagnostic summary ===")
    print(f"{'Arm':<25} {'Ckpt':<12} {'Heldout':>10} {'Seen':>10} {'Gap':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r['arm']:<25} {r['checkpoint']:<12} {r['heldout_loss']:>10.4f} {r['seen_loss']:>10.4f} {r['generalization_gap']:>+10.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
