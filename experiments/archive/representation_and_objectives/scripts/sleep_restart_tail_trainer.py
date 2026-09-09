#!/usr/bin/env python3
"""research: continue an existing legal40k checkpoint with a fresh AdamW tail schedule.

Scientific purpose: test whether a late training-state sleep (weights preserved,
optimizer and LR schedule restarted, remaining legal stream only) can improve the
fixed-256 legal40k compact-view coordinate without spending a new 100M endpoint.
The run starts from a checkpoint that already consumed a recorded number of words
and trains only the still-unseen suffix up to the same 100M total exposure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

USER_ROOT = Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fresh-AdamW tail continuation from a recorded checkpoint")
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--source_model_path", required=True)
    p.add_argument("--source_metrics_path", required=True)
    p.add_argument("--source_checkpoint_name", default="chck_80M")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_total_word_exposure", type=int, default=100_000_000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--checkpoint_interval_words", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--train_rng_seed", type=int, default=53023)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_source_checkpoint(metrics_path: Path, checkpoint_name: str) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for rec in metrics.get("saved_checkpoints", []):
        if rec.get("name") == checkpoint_name:
            return dict(rec)
    raise RuntimeError(f"checkpoint {checkpoint_name!r} not found in {metrics_path}")


def select_tail_examples(examples: list[Any], consumed_words: int, total_target: int) -> tuple[list[Any], int, int]:
    cumulative = 0
    start_idx = None
    for i, ex in enumerate(examples):
        cumulative += int(ex.words)
        if cumulative == consumed_words:
            start_idx = i + 1
            break
        if cumulative > consumed_words:
            raise RuntimeError(
                "source checkpoint word count falls inside an example: "
                f"consumed={consumed_words} crossed_at_row={i} cumulative={cumulative}"
            )
    if start_idx is None:
        raise RuntimeError(f"could not match consumed_words={consumed_words} in {len(examples)} examples")

    tail: list[Any] = []
    running = consumed_words
    for ex in examples[start_idx:]:
        w = int(ex.words)
        if running + w > total_target:
            raise RuntimeError(
                "tail selection would require a partial example: "
                f"running={running} next_words={w} target={total_target}"
            )
        tail.append(ex)
        running += w
        if running == total_target:
            break
    if running != total_target:
        raise RuntimeError(f"tail ended at {running}, target {total_target}")
    return tail, start_idx, running - consumed_words


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


def main() -> None:
    args = build_args()
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    accum_steps = args.batch_size // args.micro_batch_size

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_rec = read_source_checkpoint(Path(args.source_metrics_path), args.source_checkpoint_name)
    consumed_words = int(source_rec["actual_cumulative_word_exposure"])
    if consumed_words >= args.max_total_word_exposure:
        raise RuntimeError(f"source checkpoint already at {consumed_words} words")

    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    all_examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(
        Path(args.example_jsonl), args.max_total_word_exposure
    )
    tail_examples, tail_start_idx, continuation_words = select_tail_examples(
        all_examples, consumed_words, args.max_total_word_exposure
    )

    dataset = base.MaskedChunkDataset(tail_examples, tokenizer, args.max_seq_length)
    total_steps = math.ceil(len(dataset) / args.batch_size)
    if total_steps <= 0:
        raise RuntimeError("empty tail dataset")

    manifest = {
        "status": "SLEEP_RESTART_TAIL_READY",
        "created_utc": now_utc(),
        "purpose": "fresh AdamW/LR-schedule continuation from a legal40k 80M checkpoint over only the remaining stream words",
        "source_checkpoint": {
            "name": args.source_checkpoint_name,
            "path": str(args.source_model_path),
            "metrics_path": str(args.source_metrics_path),
            "record": source_rec,
            "consumed_words": consumed_words,
        },
        "stream": {
            "example_jsonl": str(args.example_jsonl),
            "jsonl_total_words": jsonl_total_words,
            "jsonl_total_rows": jsonl_total_rows,
            "loaded_words": args.max_total_word_exposure,
            "tail_start_row_index": tail_start_idx,
            "tail_examples": len(tail_examples),
            "continuation_words": continuation_words,
            "max_total_word_exposure": args.max_total_word_exposure,
            "sample_rows": sample_rows[:3],
        },
        "method": {
            "model_family": "DebertaV2ForMaskedLM",
            "tokenizer_path": str(args.tokenizer_path),
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": accum_steps,
            "optimizer": "AdamW_fresh_state",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_fraction": args.warmup_fraction,
            "mask_prob": args.mask_prob,
            "seed": args.seed,
            "train_rng_seed": args.train_rng_seed,
            "total_steps": total_steps,
        },
    }
    (out / "sleep_restart_tail_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "tail_manifest", **manifest["stream"], "total_steps": total_steps}), flush=True)

    if args.dry_run:
        print(json.dumps({"event": "dry_run_done", "manifest": str(out / "sleep_restart_tail_manifest.json")}), flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    reset_all_rng(args.seed)
    model = DebertaV2ForMaskedLM.from_pretrained(args.source_model_path, local_files_only=True)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())

    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        collate_fn=base.collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    curriculum_state = base.MaskingCurriculumState(
        curriculum="wwm_fixed",
        mask_prob_start=args.mask_prob,
        mask_prob_end=args.mask_prob,
        switch_frac=1.0,
    )
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    warmup = max(1, int(total_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)

    print(json.dumps({
        "event": "training_start",
        "device": str(device),
        "params": param_count,
        "consumed_words_start": consumed_words,
        "continuation_words": continuation_words,
        "total_steps": total_steps,
        "warmup_steps": warmup,
        "lr": args.learning_rate,
    }), flush=True)

    log_path = out / "training_log.jsonl"
    saved_checkpoints: list[dict[str, Any]] = []
    loss_values: list[float] = []
    cumulative_tail_words = 0
    cumulative_total_words = consumed_words
    if args.checkpoint_interval_words > 0:
        next_ckpt_total = ((consumed_words // args.checkpoint_interval_words) + 1) * args.checkpoint_interval_words
    else:
        next_ckpt_total = None

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(effective_batch_iterator(loader, accum_steps), 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            curriculum_state.current_step = step - 1
            masked_inputs, labels = base.apply_masking_curriculum(
                input_ids, attention_mask, word_group, tokenizer, curriculum_state, gen
            )
            n_pred_total_t = (labels != -100).sum()
            n_pred_total = int(n_pred_total_t.item())
            if n_pred_total <= 0:
                raise RuntimeError(f"no masked tokens at step {step}")

            optim.zero_grad(set_to_none=True)
            weighted_loss_sum = 0.0
            batch_rows = input_ids.shape[0]
            for mb_start in range(0, batch_rows, args.micro_batch_size):
                mb_end = min(mb_start + args.micro_batch_size, batch_rows)
                sl_labels = labels[mb_start:mb_end]
                n_pred_i = int((sl_labels != -100).sum().item())
                if n_pred_i <= 0:
                    continue
                out_model = model(
                    input_ids=masked_inputs[mb_start:mb_end],
                    attention_mask=attention_mask[mb_start:mb_end],
                    labels=sl_labels,
                )
                loss_i = out_model.loss
                if loss_i is None:
                    raise RuntimeError("model returned no loss")
                scale = n_pred_i / n_pred_total
                (loss_i * scale).backward()
                weighted_loss_sum += float(loss_i.detach().cpu()) * scale
                del out_model, loss_i

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cumulative_tail_words += words
            cumulative_total_words += words
            n_candidate = int(attention_mask.bool().sum().item())
            effective_mask_rate = n_pred_total / max(1, n_candidate)
            loss_float = float(weighted_loss_sum)
            loss_values.append(loss_float)

            rec = {
                "step": step,
                "loss": loss_float,
                "lr": float(sched.get_last_lr()[0]),
                "batch_words": words,
                "tail_word_exposure": cumulative_tail_words,
                "total_word_exposure": cumulative_total_words,
                "masked_tokens": n_pred_total,
                "effective_mask_rate": round(effective_mask_rate, 4),
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                wps = cumulative_tail_words / max(1, time.time() - start_time)
                print(json.dumps({"event": "train", **rec, "tail_wps": round(wps, 0)}), flush=True)

            while next_ckpt_total is not None and cumulative_total_words >= next_ckpt_total and next_ckpt_total <= args.max_total_word_exposure:
                name = f"chck_{next_ckpt_total // 1_000_000}M" if next_ckpt_total % 1_000_000 == 0 else f"chck_{next_ckpt_total}w"
                cp = out / "hf_model" / name
                base.save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({
                    "name": name,
                    "target_total_word_exposure": next_ckpt_total,
                    "actual_total_word_exposure": cumulative_total_words,
                    "tail_word_exposure": cumulative_tail_words,
                    "step": step,
                    "path": str(cp),
                })
                print(json.dumps({"event": "checkpoint", "name": name, "total_words": cumulative_total_words, "step": step}), flush=True)
                next_ckpt_total += args.checkpoint_interval_words

            del input_ids, attention_mask, word_group, masked_inputs, labels

    final_dir = out / "hf_model" / "chck_final"
    base.save_hf_checkpoint(model, tokenizer, final_dir)
    if not any(c.get("name") == "chck_100M" for c in saved_checkpoints) and cumulative_total_words == args.max_total_word_exposure:
        cp = out / "hf_model" / "chck_100M"
        base.save_hf_checkpoint(model, tokenizer, cp)
        saved_checkpoints.append({
            "name": "chck_100M",
            "target_total_word_exposure": args.max_total_word_exposure,
            "actual_total_word_exposure": cumulative_total_words,
            "tail_word_exposure": cumulative_tail_words,
            "step": total_steps,
            "path": str(cp),
        })

    metrics = {
        "variant": "sleep_restart_adamw_tail",
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "source_checkpoint_name": args.source_checkpoint_name,
        "source_model_path": str(args.source_model_path),
        "source_consumed_words": consumed_words,
        "tail_examples": len(tail_examples),
        "tail_word_exposure": cumulative_tail_words,
        "word_exposure": cumulative_total_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "optimizer": "AdamW_fresh_state",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_fraction": args.warmup_fraction,
        "warmup_steps": warmup,
        "masking_curriculum": "wwm_fixed",
        "mask_prob": args.mask_prob,
        "seed": args.seed,
        "train_rng_seed": args.train_rng_seed,
        "max_seq_length": args.max_seq_length,
        "saved_checkpoints": saved_checkpoints,
        "final_checkpoint": str(final_dir),
        "manifest": str(out / "sleep_restart_tail_manifest.json"),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "event": "done",
        "word_exposure": cumulative_total_words,
        "tail_word_exposure": cumulative_tail_words,
        "total_steps": total_steps,
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "checkpoints": [c["name"] for c in saved_checkpoints],
        "metrics": str(out / "scientific_metrics.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
