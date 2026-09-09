#!/usr/bin/env python3
"""Memory-safe gradient-accumulation variant of the COMPACT_EXPERIENCE masking-curriculum trainer.

Purpose in representation_and_objectives research: the legal-40k compact_view_reinvest run OOMed at
batch_size=256 because the 40k MLM logits/activations do not fit on the H100
under the original trainer. This local trainer preserves the intended effective
256-row optimizer batch and LR/checkpoint schedule while splitting each effective
batch into 64-row forward/backward microbatches.

Important scientific note: this is not bit-identical to a hypothetical full
256-row legal-40k batch because dropout/forward execution occurs per microbatch.
It does preserve the data order, effective row grouping, full-effective-batch
mask sampling, loss weighting by masked-token count, optimizer-step count, LR
schedule, checkpoint word-exposure positions, architecture, tokenizer, and RNG
seed identities. It is therefore the minimal memory repair after the OOM, not a
new data/architecture/curriculum route.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

USER_ROOT = Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Accumulated masking-curriculum trainer")
    # Data
    p.add_argument("--dataset_id", default="BabyLM-community/BabyLM-2026-Strict-Small")
    p.add_argument("--dataset_revision", default="c92ab16b4f08858304b0815706065b3354d8fc0a")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=4_000_000)
    p.add_argument("--example_pool_words", type=int, default=10_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--example_jsonl", default="")
    p.add_argument("--example_jsonl_label", default="")
    p.add_argument("--example_jsonl_meta", default="")
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--tokenizer_label", default="baseline16k")
    p.add_argument("--tokenization_summary_limit", type=int, default=500)

    # Masking curriculum
    p.add_argument("--masking_curriculum", choices=base.MASKING_CURRICULA, default="wwm_fixed")
    p.add_argument("--mask_prob_start", type=float, default=0.15)
    p.add_argument("--mask_prob_end", type=float, default=0.15)
    p.add_argument("--switch_frac", type=float, default=0.7)
    p.add_argument("--amlm_window", type=int, default=10)
    p.add_argument("--amlm_lambda", type=float, default=0.2)

    # Sequence length
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--seq_len_schedule", default="")

    # Model
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")

    # Optimization. batch_size is the effective optimizer batch; micro_batch_size
    # is the memory chunk used for forward/backward accumulation.
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)

    # Reproducibility
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=50)

    # Dynamics trace. This accumulated trainer records lightweight mask-rate
    # traces only; full per-token-logit trace is intentionally disabled to avoid
    # reintroducing the large-logit memory failure.
    p.add_argument("--dynamics_trace_every", type=int, default=200)
    return p.parse_args()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def combine_microbatches(micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.cat([b["input_ids"] for b in micro_batches], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in micro_batches], dim=0),
        "word_group": torch.cat([b["word_group"] for b in micro_batches], dim=0),
        "words": torch.cat([b["words"] for b in micro_batches], dim=0),
    }


def effective_batch_iterator(loader: DataLoader, accum_steps: int):
    buf: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        buf.append(batch)
        if len(buf) == accum_steps:
            yield combine_microbatches(buf)
            buf = []
    if buf:
        yield combine_microbatches(buf)


def load_examples_and_metadata(args: argparse.Namespace) -> tuple[list[Any], int, int, dict[str, Any], list[dict], int]:
    """Load only the example_jsonl path used by the compact-view experiments."""
    selected_words = args.max_word_exposure
    if not args.example_jsonl:
        raise RuntimeError("This accumulated repair trainer is intended for example_jsonl runs only.")
    jsonl_path = Path(args.example_jsonl)
    examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(jsonl_path, selected_words)
    actual_words = sum(ex.words for ex in examples)
    manifest_files = [{
        "path": str(jsonl_path),
        "name": jsonl_path.name,
        "bytes": jsonl_path.stat().st_size,
        "sha256": base.sha256_file(jsonl_path),
        "whitespace_words": jsonl_total_words,
        "rows": jsonl_total_rows,
    }]
    example_jsonl_meta = {}
    if args.example_jsonl_meta:
        meta_path = Path(args.example_jsonl_meta)
        example_jsonl_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    example_selection_metadata = {
        "data_source_type": "example_jsonl",
        "example_jsonl": str(jsonl_path),
        "example_jsonl_label": args.example_jsonl_label,
        "example_jsonl_meta": example_jsonl_meta,
        "manifest_files": manifest_files,
        "sample_rows": sample_rows,
    }
    if actual_words != selected_words:
        raise RuntimeError(f"word selection mismatch {actual_words} vs {selected_words}")
    return examples, actual_words, jsonl_total_words, example_selection_metadata, manifest_files, jsonl_total_rows


def main() -> None:
    args = build_args()
    if args.batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch_size and micro_batch_size must be positive")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("for exact row grouping, batch_size must be divisible by micro_batch_size")
    accum_steps = args.batch_size // args.micro_batch_size

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    examples, actual_words, total_words, example_selection_metadata, manifest_files, jsonl_total_rows = load_examples_and_metadata(args)

    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        collate_fn=base.collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    total_steps = math.ceil(len(dataset) / args.batch_size)

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

    reset_all_rng = lambda s: (random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None)
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
            t, L = part.split(":")
            seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    log_path = out / "training_log.jsonl"
    dynamics_path = out / "dynamics_traces.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    dynamics_traces: list[dict[str, Any]] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None

    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    (out / "example_order_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "selected_for_training_words": actual_words,
        "num_consumed_examples": len(examples),
        "masking_curriculum": args.masking_curriculum,
        "mask_prob_start": args.mask_prob_start,
        "mask_prob_end": args.mask_prob_end,
        "switch_frac": args.switch_frac,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "source_words_consumed": source_words,
        **example_selection_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "event": "accum_trainer_start",
        "created_utc": now_utc(),
        "output_dir": str(out),
        "device": str(device),
        "vocab_size": len(tokenizer),
        "param_count": param_count,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "total_effective_steps": total_steps,
        "word_exposure_target": args.max_word_exposure,
        "tokenizer_label": args.tokenizer_label,
    }), flush=True)

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(effective_batch_iterator(loader, accum_steps), 1):
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
            n_pred_total_t = (labels != -100).sum()
            n_pred_total = int(n_pred_total_t.item())
            n_candidate = int(attention_mask.bool().sum().item())
            effective_mask_rate = n_pred_total / max(1, n_candidate)
            if n_pred_total <= 0:
                raise RuntimeError(f"no masked tokens in effective batch at step {step}")

            optim.zero_grad(set_to_none=True)
            weighted_loss_sum = 0.0
            active_microbatches = 0
            batch_rows = input_ids.shape[0]
            for start in range(0, batch_rows, args.micro_batch_size):
                end = min(start + args.micro_batch_size, batch_rows)
                sl_labels = labels[start:end]
                n_pred_i = int((sl_labels != -100).sum().item())
                if n_pred_i <= 0:
                    continue
                out_model = model(
                    input_ids=masked_inputs[start:end],
                    attention_mask=attention_mask[start:end],
                    labels=sl_labels,
                )
                loss_i = out_model.loss
                if loss_i is None:
                    raise RuntimeError("model returned no loss")
                scale = n_pred_i / n_pred_total
                (loss_i * scale).backward()
                weighted_loss_sum += float(loss_i.detach().cpu()) * scale
                active_microbatches += 1
                del out_model, loss_i

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cumulative_words += words
            loss_float = float(weighted_loss_sum)
            loss_values.append(loss_float)

            # wwm_fixed is used in the 40k route, so AMLM updates are normally off.
            # Keep compatibility for any future use by updating on the full effective labels.
            if curriculum_state.uses_amlm:
                # Full-logit AMLM accuracy is deliberately not implemented because it
                # would reintroduce the large-logit memory path. This script is not
                # intended for AMLM routes.
                raise RuntimeError("accumulated trainer does not support AMLM curricula")

            if step % args.dynamics_trace_every == 0:
                curriculum_state.trace_mask_rates.append(effective_mask_rate)

            rec = {
                "step": step,
                "loss": loss_float,
                "lr": float(sched.get_last_lr()[0]),
                "batch_words": words,
                "cumulative_word_exposure": cumulative_words,
                "seq_len": cur_len,
                "masked_tokens": n_pred_total,
                "effective_mask_rate": round(effective_mask_rate, 4),
                "mask_mode": curriculum_state.get_current_mask_mode(),
                "mask_prob_nominal": round(curriculum_state.get_current_mask_prob(), 4),
                "active_microbatches": active_microbatches,
                "micro_batch_size": args.micro_batch_size,
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
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
                }), flush=True)
                next_ckpt += args.checkpoint_words

            del input_ids, attention_mask, word_group, masked_inputs, labels

    base.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    if not saved_checkpoints:
        cp = out / "hf_model" / "chck_1M"
        base.save_hf_checkpoint(model, tokenizer, cp)
        saved_checkpoints.append({
            "name": "chck_1M",
            "target_word_exposure": args.checkpoint_words,
            "actual_cumulative_word_exposure": cumulative_words,
            "path": str(cp),
        })

    with dynamics_path.open("w", encoding="utf-8") as f:
        for trace in dynamics_traces:
            f.write(json.dumps(trace) + "\n")

    metrics = {
        "variant": f"masking_curriculum_{args.masking_curriculum}_accumulated",
        "backend": "mlm",
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "word_exposure": cumulative_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "memory_repair_note": (
            "OOM repair for legal40k: split each effective 256-row optimizer batch into "
            "64-row microbatches. Full effective-batch masking and masked-token-weighted "
            "loss accumulation preserve the update schedule but are not bit-identical to a "
            "hypothetical full-batch forward because dropout execution is microbatched."
        ),
        "masking_curriculum": args.masking_curriculum,
        "mask_prob_start": args.mask_prob_start,
        "mask_prob_end": args.mask_prob_end,
        "switch_frac": args.switch_frac,
        "amlm_window": None,
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
        **example_selection_metadata,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "event": "done",
        "param_count": param_count,
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "word_exposure": cumulative_words,
        "masking_curriculum": args.masking_curriculum,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "checkpoints": [c["name"] for c in saved_checkpoints],
    }), flush=True)


if __name__ == "__main__":
    main()
