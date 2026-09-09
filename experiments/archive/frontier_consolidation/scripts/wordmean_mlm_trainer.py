#!/usr/bin/env python3
"""research trainer: whole-word-unit-normalized MLM credit.

This script keeps the legal research tokenizer, compact-view-reinvest training stream,
model family, WWM sampling, seeds, optimizer, schedule, and checkpoint cadence identical
to the inherited compliant trainer, but replaces token-mean MLM loss with a selected-
word-group mean.  For a selected whole-word group split into k BPE pieces, the group's
credit is the mean of its k token cross-entropies, and the batch loss is the mean over
selected groups.  This tests whether part of the legal-tokenizer weakness is an
optimization/credit-allocation effect induced by variable BPE fragmentation rather than a
failure of the compact-view data mechanism.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
BASE_TRAINER_PATH = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"


def load_base_trainer():
    if not BASE_TRAINER_PATH.exists():
        raise FileNotFoundError(BASE_TRAINER_PATH)
    spec = importlib.util.spec_from_file_location("compact_experience_masking_curriculum_trainer_base", BASE_TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base trainer from {BASE_TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_base_trainer()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def word_mean_mlm_loss(logits: torch.Tensor, labels: torch.Tensor, word_group: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mean CE over selected WWM groups, with token CE averaged within each group."""
    selected = labels != -100
    n_selected_tokens = int(selected.sum().detach().cpu().item())
    if n_selected_tokens == 0:
        # The masking code should force at least one selected token when candidates exist.
        loss = logits.sum() * 0.0
        return loss, {"selected_tokens": 0, "selected_word_groups": 0, "mean_tokens_per_selected_group": 0.0}

    selected_logits = logits[selected]
    selected_labels = labels[selected]
    token_ce = F.cross_entropy(selected_logits, selected_labels, reduction="none")

    bsz, seq = labels.shape
    batch_ids = torch.arange(bsz, device=labels.device).view(bsz, 1).expand(bsz, seq)
    selected_groups = word_group[selected]
    if (selected_groups < 0).any():
        bad = int((selected_groups < 0).sum().detach().cpu().item())
        raise RuntimeError(f"word_mean_mlm_loss saw {bad} selected tokens with no WWM group")
    keys = batch_ids[selected].to(torch.long) * (seq + 1) + selected_groups.to(torch.long)
    unique_keys, inverse, counts = torch.unique(keys, sorted=False, return_inverse=True, return_counts=True)
    group_token_counts = counts[inverse].to(token_ce.dtype)
    weighted = token_ce / group_token_counts
    loss = weighted.sum() / unique_keys.numel()
    stats = {
        "selected_tokens": n_selected_tokens,
        "selected_word_groups": int(unique_keys.numel()),
        "mean_tokens_per_selected_group": float(n_selected_tokens / max(1, int(unique_keys.numel()))),
    }
    return loss, stats


def token_mean_mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    selected = labels != -100
    n_selected_tokens = int(selected.sum().detach().cpu().item())
    if n_selected_tokens == 0:
        loss = logits.sum() * 0.0
    else:
        loss = F.cross_entropy(logits[selected], labels[selected], reduction="mean")
    return loss, {"selected_tokens": n_selected_tokens, "selected_word_groups": None, "mean_tokens_per_selected_group": None}


def finite_mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whole-word-unit-normalized MLM trainer")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=80_000_000)
    p.add_argument("--example_pool_words", type=int, default=10_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--example_jsonl_label", default="")
    p.add_argument("--example_jsonl_meta", default="")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--tokenizer_label", default="compliant16k_reinvest10M")
    p.add_argument("--loss_normalization", choices=["word_mean", "token_mean"], default="word_mean")

    p.add_argument("--masking_curriculum", choices=base.MASKING_CURRICULA, default="wwm_fixed")
    p.add_argument("--mask_prob_start", type=float, default=0.15)
    p.add_argument("--mask_prob_end", type=float, default=0.15)
    p.add_argument("--switch_frac", type=float, default=0.7)
    p.add_argument("--amlm_window", type=int, default=10)
    p.add_argument("--amlm_lambda", type=float, default=0.2)

    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--seq_len_schedule", default="")

    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--dynamics_trace_every", type=int, default=200)
    p.add_argument("--tokenization_summary_limit", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = build_args()
    start_time = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    selected_words = int(args.max_word_exposure)
    examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(pathlib.Path(args.example_jsonl), selected_words)
    actual_words = sum(ex.words for ex in examples)
    if actual_words != selected_words:
        raise RuntimeError(f"word selection mismatch {actual_words} vs {selected_words}")
    if selected_words > 100_000_000:
        raise RuntimeError(f"requested word exposure exceeds Strict-Small ten-pass cap: {selected_words}")

    example_jsonl_meta: dict[str, Any] = {}
    if args.example_jsonl_meta:
        meta_path = pathlib.Path(args.example_jsonl_meta)
        example_jsonl_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    manifest_files = [{
        "path": str(pathlib.Path(args.example_jsonl)),
        "name": pathlib.Path(args.example_jsonl).name,
        "bytes": pathlib.Path(args.example_jsonl).stat().st_size,
        "sha256": base.sha256_file(pathlib.Path(args.example_jsonl)),
        "whitespace_words": jsonl_total_words,
        "rows": jsonl_total_rows,
    }]
    example_selection_metadata = {
        "data_source_type": "example_jsonl",
        "example_jsonl": str(pathlib.Path(args.example_jsonl)),
        "example_jsonl_label": args.example_jsonl_label,
        "sample_rows_without_text": sample_rows,
    }

    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=base.collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    total_steps = len(loader)
    token_freq_ranks = base.compute_token_frequency_ranks(examples, tokenizer, args.max_seq_length)

    curriculum_state = base.MaskingCurriculumState(
        curriculum=args.masking_curriculum,
        mask_prob_start=args.mask_prob_start,
        mask_prob_end=args.mask_prob_end,
        switch_frac=args.switch_frac,
        amlm_window=args.amlm_window,
        amlm_lambda=args.amlm_lambda,
    )
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)

    def reset_all_rng(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = base.build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)

    seq_schedule: list[tuple[float, int]] = []
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(","):
            t, length = part.split(":")
            seq_schedule.append((float(t), int(length)))
        seq_schedule.sort()

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words

    (out / "example_order_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "selected_for_training_words": actual_words,
        "num_consumed_examples": len(examples),
        "masking_curriculum": args.masking_curriculum,
        "mask_prob_start": args.mask_prob_start,
        "mask_prob_end": args.mask_prob_end,
        "switch_frac": args.switch_frac,
        "loss_normalization": args.loss_normalization,
        "base_trainer_path": str(BASE_TRAINER_PATH),
        "source_words_consumed": source_words,
        **example_selection_metadata,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log_path = out / "training_log.jsonl"
    dynamics_path = out / "dynamics_traces.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    selected_group_means: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    dynamics_traces: list[dict[str, Any]] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            frac = (step - 1) / max(1, schedule_total)
            cur_len = base.seq_length_for_progress(frac, seq_schedule, args.seq_length) if seq_schedule else args.seq_length
            cur_len = min(cur_len, args.max_seq_length)
            input_ids = input_ids[:, :cur_len].contiguous()
            attention_mask = attention_mask[:, :cur_len].contiguous()
            word_group = word_group[:, :cur_len].contiguous()

            curriculum_state.current_step = step - 1
            masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, curriculum_state, gen)

            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask)
            if args.loss_normalization == "word_mean":
                loss, loss_stats = word_mean_mlm_loss(out_model.logits, labels, word_group)
            else:
                loss, loss_stats = token_mean_mlm_loss(out_model.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            if loss_stats.get("mean_tokens_per_selected_group") is not None:
                selected_group_means.append(float(loss_stats["mean_tokens_per_selected_group"]))
            n_pred = int((labels != -100).sum().item())
            special_ids_tensor = torch.tensor(sorted(tokenizer.all_special_ids), device=input_ids.device)
            candidate_for_rate = attention_mask.bool() & ~torch.isin(input_ids, special_ids_tensor)
            n_candidate = int(candidate_for_rate.sum().item())
            effective_mask_rate = n_pred / max(1, n_candidate)

            if curriculum_state.uses_amlm:
                with torch.no_grad():
                    curriculum_state.record_batch_accuracy(input_ids, labels, out_model.logits)
                curriculum_state.maybe_update_amlm_weights()

            if step % args.dynamics_trace_every == 0:
                with torch.no_grad():
                    curriculum_state.record_dynamics_trace(
                        input_ids, labels, out_model.logits, loss_float, token_freq_ranks, effective_mask_rate
                    )

            rec = {
                "step": step,
                "loss": loss_float,
                "lr": float(sched.get_last_lr()[0]),
                "batch_words": words,
                "cumulative_word_exposure": cumulative_words,
                "seq_len": cur_len,
                "masked_tokens": n_pred,
                "selected_word_groups": loss_stats.get("selected_word_groups"),
                "mean_tokens_per_selected_group": loss_stats.get("mean_tokens_per_selected_group"),
                "effective_mask_rate": round(effective_mask_rate, 4),
                "mask_mode": curriculum_state.get_current_mask_mode(),
                "mask_prob_nominal": round(curriculum_state.get_current_mask_prob(), 4),
                "loss_normalization": args.loss_normalization,
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)

            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                if next_ckpt < 1_000_000:
                    name = "chck_1M"
                elif next_ckpt % 1_000_000 == 0:
                    name = f"chck_{next_ckpt // 1_000_000}M"
                else:
                    name = f"chck_{next_ckpt}w"
                cp = out / "hf_model" / name
                base.save_hf_checkpoint(model, tokenizer, cp)
                trace = curriculum_state.flush_dynamics_trace(name)
                trace["loss_normalization"] = args.loss_normalization
                dynamics_traces.append(trace)
                saved_checkpoints.append({
                    "name": name,
                    "target_word_exposure": next_ckpt,
                    "actual_cumulative_word_exposure": cumulative_words,
                    "path": str(cp),
                })
                print(json.dumps({
                    "event": "checkpoint_saved",
                    "name": name,
                    "cum_words": cumulative_words,
                    "mask_mode": curriculum_state.get_current_mask_mode(),
                    "mask_prob": round(curriculum_state.get_current_mask_prob(), 4),
                    "loss_normalization": args.loss_normalization,
                }), flush=True)
                next_ckpt += args.checkpoint_words

    base.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    with dynamics_path.open("w", encoding="utf-8") as f:
        for trace in dynamics_traces:
            f.write(json.dumps(trace) + "\n")

    metrics = {
        "variant": f"masking_curriculum_{args.masking_curriculum}_{args.loss_normalization}",
        "backend": "mlm",
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "word_exposure": cumulative_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "masking_curriculum": args.masking_curriculum,
        "mask_prob_start": args.mask_prob_start,
        "mask_prob_end": args.mask_prob_end,
        "switch_frac": args.switch_frac,
        "loss_normalization": args.loss_normalization,
        "mean_tokens_per_selected_group_trace_mean": finite_mean(selected_group_means),
        "n_amlm_updates": curriculum_state.n_amlm_updates,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "seq_len_schedule": args.seq_len_schedule,
        "saved_checkpoints": saved_checkpoints,
        "dynamics_traces_file": str(dynamics_path),
        "source_words_consumed": source_words,
        "manifest_files": manifest_files,
        "base_trainer_path": str(BASE_TRAINER_PATH),
        **example_selection_metadata,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "event": "done",
        "param_count": param_count,
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "word_exposure": cumulative_words,
        "masking_curriculum": args.masking_curriculum,
        "loss_normalization": args.loss_normalization,
        "checkpoints": [c["name"] for c in saved_checkpoints],
    }), flush=True)


if __name__ == "__main__":
    main()
