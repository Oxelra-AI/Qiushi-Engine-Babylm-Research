#!/usr/bin/env python3
"""research: factor-separated 80M->100M tail continuations from legal40k baseline.

Scientific purpose
------------------
Do not treat a late "sleep" as one opaque optimizer/schedule package.  The two
intended continuation arms both start from the exact same recorded legal40k
80M checkpoint, consume the exact same remaining stream rows, use fresh AdamW
optimizer moments, and replay the same WWM/dropout random stream between arms.
They differ only in the LR trajectory:

  * residual_low_lr: fresh AdamW moments but the original baseline's residual
    low-LR cosine trajectory from the 80M checkpoint step to 100M.
  * reheat_cosine: fresh AdamW moments with a restarted cosine schedule over
    only the 80M->100M tail.

The uninterrupted legal40k 100M endpoint remains the anchor.  These arms are
not a new baseline and should only be launched after the corrected curriculum
comparison is read and misses the leader region.
"""
from __future__ import annotations

import argparse
import hashlib
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
from transformers import DebertaV2ForMaskedLM

USER_ROOT = Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired fresh-moment legal40k tail continuation")
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--source_model_path", required=True)
    p.add_argument("--source_metrics_path", required=True)
    p.add_argument("--source_training_log", required=True)
    p.add_argument("--source_checkpoint_name", default="chck_80M")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", required=True, choices=["residual_low_lr", "reheat_cosine"])

    p.add_argument("--max_total_word_exposure", type=int, default=100_000_000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.98)

    # Original baseline schedule used by residual_low_lr.
    p.add_argument("--original_base_lr", type=float, default=0.001)
    p.add_argument("--original_total_steps", type=int, default=0,
                   help="0 = infer from source_training_log length")
    p.add_argument("--original_warmup_fraction", type=float, default=0.06)

    # Tail reheating schedule used by reheat_cosine.
    p.add_argument("--reheat_peak_lr", type=float, default=3e-4)
    p.add_argument("--reheat_warmup_fraction", type=float, default=0.06)

    p.add_argument("--checkpoint_interval_words", type=int, default=10_000_000)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--train_rng_seed", type=int, default=53023)
    p.add_argument("--gpu", type=int, default=0)
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


def read_source_checkpoint(metrics_path: Path, checkpoint_name: str) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for rec in metrics.get("saved_checkpoints", []):
        if rec.get("name") == checkpoint_name:
            return dict(rec)
    raise RuntimeError(f"checkpoint {checkpoint_name!r} not found in {metrics_path}")


def read_training_log(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty training log: {log_path}")
    return rows


def find_source_step(training_rows: list[dict[str, Any]], consumed_words: int) -> dict[str, Any]:
    exact = [r for r in training_rows if int(r.get("cumulative_word_exposure", -1)) == consumed_words]
    if not exact:
        near = min(training_rows, key=lambda r: abs(int(r.get("cumulative_word_exposure", 0)) - consumed_words))
        raise RuntimeError(
            "source checkpoint word count is not an exact logged effective-batch boundary: "
            f"consumed={consumed_words}, nearest_step={near.get('step')}, "
            f"nearest_words={near.get('cumulative_word_exposure')}"
        )
    return dict(exact[0])


def hf_cosine_lr(base_lr: float, warmup_steps: int, total_steps: int, completed_scheduler_steps: int) -> float:
    """LR held by HF's get_cosine_schedule_with_warmup after N scheduler steps.

    The original trainers call optim.step() then sched.step(), so update k uses
    the LR value produced after k-1 scheduler steps.  Use completed_scheduler_steps
    for the number of completed steps before the current update.
    """
    s = max(0, int(completed_scheduler_steps))
    if s < warmup_steps:
        return base_lr * (float(s) / float(max(1, warmup_steps)))
    progress = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


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


def set_optimizer_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for group in optim.param_groups:
        group["lr"] = lr


def main() -> None:
    args = build_args()
    if args.batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch_size and micro_batch_size must be positive")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    accum_steps = args.batch_size // args.micro_batch_size

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_metrics_path = Path(args.source_metrics_path)
    source_training_log_path = Path(args.source_training_log)
    source_model_path = Path(args.source_model_path)
    source_rec = read_source_checkpoint(source_metrics_path, args.source_checkpoint_name)
    consumed_words = int(source_rec["actual_cumulative_word_exposure"])
    if consumed_words >= args.max_total_word_exposure:
        raise RuntimeError(f"source checkpoint already at {consumed_words} words")

    training_rows = read_training_log(source_training_log_path)
    source_step_rec = find_source_step(training_rows, consumed_words)
    source_step = int(source_step_rec["step"])
    original_total_steps = args.original_total_steps if args.original_total_steps > 0 else len(training_rows)
    original_warmup_steps = max(1, int(original_total_steps * args.original_warmup_fraction))
    # Baseline's own logged LR at the first continued global step (source_step+1);
    # the residual arm's first update LR must equal this to reconstruct the
    # baseline's residual trajectory.
    next_baseline_rows = [r for r in training_rows if int(r.get("step", -1)) == source_step + 1]
    baseline_next_step_logged_lr = float(next_baseline_rows[0]["lr"]) if next_baseline_rows else None

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
    final_global_step = source_step + total_steps
    if final_global_step != original_total_steps:
        raise RuntimeError(
            "tail step count does not land on original total steps: "
            f"source_step={source_step}, tail_steps={total_steps}, original_total_steps={original_total_steps}"
        )

    reheat_warmup_steps = max(1, int(total_steps * args.reheat_warmup_fraction))

    # Convention verified against the baseline trainer: the log records
    # sched.get_last_lr() after sched.step(), so the LR logged at baseline step k
    # is the optimizer LR prepared for the next update. The actual update at
    # global step source_step+1 therefore uses the value logged after source_step.
    # The residual arm uses that same applied-LR trajectory; reheat_cosine uses
    # the analogous fresh tail schedule with first update at hf(completed=0).
    def lr_for_update(local_step: int) -> float:
        if args.mode == "residual_low_lr":
            completed_before_update = source_step + local_step - 1
            return hf_cosine_lr(args.original_base_lr, original_warmup_steps, original_total_steps, completed_before_update)
        if args.mode == "reheat_cosine":
            completed_before_update = local_step - 1
            return hf_cosine_lr(args.reheat_peak_lr, reheat_warmup_steps, total_steps, completed_before_update)
        raise AssertionError(args.mode)

    def lr_after_update(local_step: int) -> float:
        if args.mode == "residual_low_lr":
            completed_after_update = source_step + local_step
            return hf_cosine_lr(args.original_base_lr, original_warmup_steps, original_total_steps, completed_after_update)
        if args.mode == "reheat_cosine":
            completed_after_update = local_step
            return hf_cosine_lr(args.reheat_peak_lr, reheat_warmup_steps, total_steps, completed_after_update)
        raise AssertionError(args.mode)

    first_update_lr = lr_for_update(1)
    first_after_lr = lr_after_update(1)
    last_update_lr = lr_for_update(total_steps)
    final_after_lr = lr_after_update(total_steps)
    source_logged_lr = float(source_step_rec.get("lr"))

    # CPU-side deterministic first-batch tokenization fingerprint.  Masking and
    # dropout replay hashes during actual training are logged from the train device.
    dry_loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False,
                            collate_fn=base.collate, num_workers=0, pin_memory=False)
    first_effective_batch = next(effective_batch_iterator(dry_loader, accum_steps))
    first_batch_plain_hashes = {
        "input_ids_sha256": tensor_sha(first_effective_batch["input_ids"]),
        "attention_mask_sha256": tensor_sha(first_effective_batch["attention_mask"]),
        "word_group_sha256": tensor_sha(first_effective_batch["word_group"]),
        "words_sha256": tensor_sha(first_effective_batch["words"]),
        "words_sum": int(first_effective_batch["words"].sum().item()),
    }
    first_cpu_curriculum_state = base.MaskingCurriculumState(
        curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=1.0
    )
    first_cpu_curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)
    first_cpu_gen = torch.Generator(device="cpu")
    first_cpu_gen.manual_seed(args.train_rng_seed)
    first_cpu_masked_inputs, first_cpu_labels = base.apply_masking_curriculum(
        first_effective_batch["input_ids"], first_effective_batch["attention_mask"],
        first_effective_batch["word_group"], tokenizer, first_cpu_curriculum_state, first_cpu_gen
    )
    first_batch_cpu_mask_hashes = {
        "masked_inputs_sha256": tensor_sha(first_cpu_masked_inputs),
        "labels_sha256": tensor_sha(first_cpu_labels),
        "selected_positions_sha256": tensor_sha(first_cpu_labels != -100),
        "masked_tokens": int((first_cpu_labels != -100).sum().item()),
    }

    manifest = {
        "status": "PAIRED_TAIL_CONTINUATION_READY" if args.dry_run else "PAIRED_TAIL_CONTINUATION_RUNNING",
        "created_utc": now_utc(),
        "purpose": (
            "factor-separated fresh-AdamW tail continuation: both arms reset optimizer moments "
            "from the same legal40k chck_80M weights and replay the same tail rows/masks/dropout; "
            "only LR trajectory differs"
        ),
        "mode": args.mode,
        "pairing_contract": {
            "same_source_checkpoint": True,
            "same_tail_rows": True,
            "same_mask_rng_seed_between_modes": args.train_rng_seed,
            "same_dropout_global_seed_between_modes": args.train_rng_seed,
            "fresh_adamw_moments_in_both_modes": True,
            "only_intended_difference": "original residual low-LR trajectory vs restarted reheated cosine LR",
            "anchor_endpoint": "uninterrupted legal40k fixed-256 100M baseline, not either continuation arm",
        },
        "source_checkpoint": {
            "name": args.source_checkpoint_name,
            "path": str(source_model_path),
            "metrics_path": str(source_metrics_path),
            "training_log": str(source_training_log_path),
            "record": source_rec,
            "source_step_record": source_step_rec,
            "consumed_words": consumed_words,
            "source_step": source_step,
            "source_logged_lr_after_step": source_logged_lr,
        },
        "stream": {
            "example_jsonl": str(args.example_jsonl),
            "example_jsonl_sha256": sha256_file(Path(args.example_jsonl)),
            "jsonl_total_words": jsonl_total_words,
            "jsonl_total_rows": jsonl_total_rows,
            "loaded_words": args.max_total_word_exposure,
            "tail_start_row_index": tail_start_idx,
            "tail_examples": len(tail_examples),
            "continuation_words": continuation_words,
            "max_total_word_exposure": args.max_total_word_exposure,
            "sample_rows": sample_rows[:3],
            "first_effective_batch_plain_hashes": first_batch_plain_hashes,
            "first_effective_batch_cpu_mask_hashes": first_batch_cpu_mask_hashes,
        },
        "method": {
            "model_family": "DebertaV2ForMaskedLM",
            "tokenizer_path": str(args.tokenizer_path),
            "tokenizer_len": len(tokenizer),
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": accum_steps,
            "optimizer": "AdamW_fresh_state",
            "adam_betas": [args.adam_beta1, args.adam_beta2],
            "weight_decay": args.weight_decay,
            "masking_curriculum": "wwm_fixed",
            "mask_prob": args.mask_prob,
            "seed": args.seed,
            "train_rng_seed": args.train_rng_seed,
            "total_tail_steps": total_steps,
            "source_step": source_step,
            "final_global_step": final_global_step,
        },
        "lr_geometry": {
            "mode": args.mode,
            "original_base_lr": args.original_base_lr,
            "original_total_steps": original_total_steps,
            "original_warmup_fraction": args.original_warmup_fraction,
            "original_warmup_steps": original_warmup_steps,
            "reheat_peak_lr": args.reheat_peak_lr,
            "reheat_warmup_fraction": args.reheat_warmup_fraction,
            "reheat_warmup_steps": reheat_warmup_steps,
            "first_update_lr": first_update_lr,
            "first_after_update_lr": first_after_lr,
            "last_update_lr": last_update_lr,
            "final_after_update_lr": final_after_lr,
            "source_logged_lr_after_step": source_logged_lr,
            "baseline_next_step_logged_lr": baseline_next_step_logged_lr,
            "residual_first_update_matches_source_logged_lr": (
                abs(first_update_lr - source_logged_lr) < 1e-12 if args.mode == "residual_low_lr" else None
            ),
            "residual_first_after_update_matches_baseline_next_step_log": (
                abs(first_after_lr - baseline_next_step_logged_lr) < 1e-12
                if (args.mode == "residual_low_lr" and baseline_next_step_logged_lr is not None) else None
            ),
        },
    }
    manifest_path = out / "paired_tail_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "event": "paired_tail_manifest",
        "mode": args.mode,
        "tail_examples": len(tail_examples),
        "continuation_words": continuation_words,
        "source_step": source_step,
        "total_tail_steps": total_steps,
        "first_update_lr": first_update_lr,
        "last_update_lr": last_update_lr,
        "manifest": str(manifest_path),
    }), flush=True)

    if args.dry_run:
        print(json.dumps({"event": "dry_run_done", "mode": args.mode, "manifest": str(manifest_path)}), flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    reset_all_rng(args.seed)
    model = DebertaV2ForMaskedLM.from_pretrained(source_model_path, local_files_only=True)
    # This seed is intentionally shared by both arms.  base.apply_masking_curriculum
    # uses the explicit generator below, while DeBERTa dropout uses the global CUDA
    # RNG; identical seeds and identical tensor shapes give replay-matched stochastic
    # masks/dropout between residual_low_lr and reheat_cosine.
    reset_all_rng(args.train_rng_seed)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())

    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False,
                        collate_fn=base.collate, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    curriculum_state = base.MaskingCurriculumState(
        curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=1.0
    )
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)
    optim = torch.optim.AdamW(
        model.parameters(), lr=0.0, weight_decay=args.weight_decay, betas=(args.adam_beta1, args.adam_beta2)
    )

    print(json.dumps({
        "event": "training_start",
        "mode": args.mode,
        "device": str(device),
        "params": param_count,
        "consumed_words_start": consumed_words,
        "source_step": source_step,
        "continuation_words": continuation_words,
        "total_tail_steps": total_steps,
        "first_update_lr": first_update_lr,
        "final_after_update_lr": final_after_lr,
    }), flush=True)

    log_path = out / "training_log.jsonl"
    saved_checkpoints: list[dict[str, Any]] = []
    loss_values: list[float] = []
    replay_hashes: dict[str, Any] = {}
    cumulative_tail_words = 0
    cumulative_total_words = consumed_words
    if args.checkpoint_interval_words > 0:
        next_ckpt_total = ((consumed_words // args.checkpoint_interval_words) + 1) * args.checkpoint_interval_words
    else:
        next_ckpt_total = None

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for local_step, batch in enumerate(effective_batch_iterator(loader, accum_steps), 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            curriculum_state.current_step = local_step - 1
            masked_inputs, labels = base.apply_masking_curriculum(
                input_ids, attention_mask, word_group, tokenizer, curriculum_state, gen
            )
            n_pred_total_t = (labels != -100).sum()
            n_pred_total = int(n_pred_total_t.item())
            if n_pred_total <= 0:
                raise RuntimeError(f"no masked tokens at local step {local_step}")

            update_lr = lr_for_update(local_step)
            set_optimizer_lr(optim, update_lr)
            optim.zero_grad(set_to_none=True)
            weighted_loss_sum = 0.0
            active_microbatches = 0
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
                active_microbatches += 1
                del out_model, loss_i

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            cumulative_tail_words += words
            cumulative_total_words += words
            global_step = source_step + local_step
            n_candidate = int(attention_mask.bool().sum().item())
            effective_mask_rate = n_pred_total / max(1, n_candidate)
            loss_float = float(weighted_loss_sum)
            loss_values.append(loss_float)
            next_update_lr = lr_after_update(local_step)

            rec = {
                "local_step": local_step,
                "global_step": global_step,
                "loss": loss_float,
                "update_lr": update_lr,
                "next_update_lr": next_update_lr,
                "batch_words": words,
                "tail_word_exposure": cumulative_tail_words,
                "total_word_exposure": cumulative_total_words,
                "masked_tokens": n_pred_total,
                "effective_mask_rate": round(effective_mask_rate, 4),
                "active_microbatches": active_microbatches,
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            if local_step == 1:
                replay_hashes = {
                    "input_ids_sha256": tensor_sha(input_ids),
                    "attention_mask_sha256": tensor_sha(attention_mask),
                    "word_group_sha256": tensor_sha(word_group),
                    "masked_inputs_sha256": tensor_sha(masked_inputs),
                    "labels_sha256": tensor_sha(labels),
                    "selected_positions_sha256": tensor_sha(labels != -100),
                    "first_step_words": words,
                    "first_step_masked_tokens": n_pred_total,
                    "first_step_update_lr": update_lr,
                }
                rec["replay_hashes"] = replay_hashes

            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if local_step == 1 or local_step % args.log_every == 0 or local_step == total_steps:
                wps = cumulative_tail_words / max(1, time.time() - start_time)
                print(json.dumps({"event": "train", "mode": args.mode, **rec, "tail_wps": round(wps, 0)}), flush=True)

            while next_ckpt_total is not None and cumulative_total_words >= next_ckpt_total and next_ckpt_total <= args.max_total_word_exposure:
                name = f"chck_{next_ckpt_total // 1_000_000}M" if next_ckpt_total % 1_000_000 == 0 else f"chck_{next_ckpt_total}w"
                cp = out / "hf_model" / name
                base.save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({
                    "name": name,
                    "target_total_word_exposure": next_ckpt_total,
                    "actual_total_word_exposure": cumulative_total_words,
                    "tail_word_exposure": cumulative_tail_words,
                    "local_step": local_step,
                    "global_step": global_step,
                    "path": str(cp),
                })
                print(json.dumps({"event": "checkpoint", "mode": args.mode, "name": name,
                                  "total_words": cumulative_total_words, "local_step": local_step,
                                  "global_step": global_step}), flush=True)
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
            "local_step": total_steps,
            "global_step": final_global_step,
            "path": str(cp),
        })

    metrics = {
        "variant": f"paired_tail_freshmom_{args.mode}",
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "mode": args.mode,
        "pairing_contract": manifest["pairing_contract"],
        "source_checkpoint_name": args.source_checkpoint_name,
        "source_model_path": str(source_model_path),
        "source_consumed_words": consumed_words,
        "source_step": source_step,
        "tail_examples": len(tail_examples),
        "tail_word_exposure": cumulative_tail_words,
        "word_exposure": cumulative_total_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "final_global_step": final_global_step,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "optimizer": "AdamW_fresh_state",
        "adam_betas": [args.adam_beta1, args.adam_beta2],
        "weight_decay": args.weight_decay,
        "masking_curriculum": "wwm_fixed",
        "mask_prob": args.mask_prob,
        "seed": args.seed,
        "train_rng_seed": args.train_rng_seed,
        "max_seq_length": args.max_seq_length,
        "lr_geometry": manifest["lr_geometry"],
        "first_step_replay_hashes": replay_hashes,
        "saved_checkpoints": saved_checkpoints,
        "final_checkpoint": str(final_dir),
        "manifest": str(manifest_path),
        "training_log": str(log_path),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "event": "done",
        "mode": args.mode,
        "word_exposure": cumulative_total_words,
        "tail_word_exposure": cumulative_tail_words,
        "total_tail_steps": total_steps,
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "checkpoints": [c["name"] for c in saved_checkpoints],
        "metrics": str(out / "scientific_metrics.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
