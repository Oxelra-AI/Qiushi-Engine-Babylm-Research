#!/usr/bin/env python3
"""research PVDM/control continuation trainer, physically stopped at 80M.

This trainer warm-starts from the compact `chck_70M` checkpoint and trains
only the next segment up to the original compact `chck_80M` row boundary.  It is
explicitly staged: this script does not consume the 80M->100M remainder unless a
later step launches a separate resume using the saved trainer state.

Scientific contrast:
  treatment: true relational pivot remains visible while the dependent target is
             predicted and the matched surrogate anchor is masked.
  control:   the same dependent target is predicted, but the surrogate anchor
             remains visible and the true pivot is masked.

The paired masking library guarantees equal realized masked token mass, identical
dependent targets, identical background masks, and matched 80/10/10 replacement
actions for the swapped anchors on every real tokenized batch.  Events that do
not survive truncation or have unequal pivot/control token lengths are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

USER_ROOT = Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / "scripts"
for p in [str(COMPACT_EXPERIENCE_SCRIPTS), str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import masking_curriculum_trainer as base  # noqa: E402
import pvdm_masking_lib as pvdm  # noqa: E402

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
DEFAULT_TAIL = WS / "data/pvdm_strict_labels/compact_tail_70M_100M.jsonl"
DEFAULT_LABELS = WS / "data/pvdm_strict_labels/pvdm_strict_tail_70M_100M_labels.jsonl"
DEFAULT_TOKENIZER = WS / "data/shared_tokenizer/shared_16k_tokenizer"
DEFAULT_INIT = Path("experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_70M")
EXPECTED = {
    "tail_sha256": "19304f23a57a3809b50af496a99068a03417cb27886796604da5e755bf004675",
    "labels_sha256": "39ed2b8eec610cb9f5e6456695d1b92057d6bb2af9fad0369790f1295589306a",
    "tokenizer_sha256": "e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366",
}
# The archived chck_70M is saved after full batch 1756, not exactly at the
# 70,000,000-word row boundary.  It has already seen 40,037 words from the exact
# 70M tail: tail rows 0..254.  The next unseen row is tail_row_idx=255.
DEFAULT_START_TAIL_ROW = 255
DEFAULT_START_TAIL_WORDS = 40_037
DEFAULT_INIT_ACTUAL_EXPOSURE = 70_040_037
# Original compact chck_80M is after research, actual exposure 80,011,326.
# Training unseen tail rows 255..64510 consumes 9,971,289 words in 251 full
# 256-row batches, exactly matching that stage boundary.
DEFAULT_STAGE_WORDS_TO_80M = 9_971_289
DEFAULT_TOTAL_SCHEDULE_STEPS_70M_TO_100M = 752


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_segment(tail_path: Path, labels_path: Path, *, start_tail_row: int, expected_start_tail_words: int, max_word_exposure: int, max_rows: int = 0) -> tuple[list[base.Example], list[dict[str, Any]], dict[str, Any]]:
    examples: list[base.Example] = []
    labels: list[dict[str, Any]] = []
    skipped_words = 0
    selected_words = 0
    selected_rows = 0
    first_tail_row = None
    last_tail_row = None
    with tail_path.open(encoding="utf-8") as tf, labels_path.open(encoding="utf-8") as lf:
        for idx, (tl, ll) in enumerate(zip(tf, lf)):
            if not tl.strip() or not ll.strip():
                continue
            tobj = json.loads(tl); lobj = json.loads(ll)
            if int(lobj.get("tail_row_idx", -999999)) != idx:
                raise RuntimeError(f"label tail_row_idx mismatch at pair index {idx}: {lobj.get('tail_row_idx')}")
            text = str(tobj["text"])
            words = int(tobj.get("words", len(text.split())))
            if words != len(text.split()) or int(lobj.get("words", words)) != words:
                raise RuntimeError(f"word-count mismatch at tail row {idx}")
            if idx < start_tail_row:
                skipped_words += words
                continue
            if first_tail_row is None:
                first_tail_row = idx
            if max_rows and selected_rows >= max_rows:
                break
            if not max_rows and selected_words + words > max_word_exposure:
                raise RuntimeError(
                    f"stage would end inside row {idx}: selected={selected_words} row_words={words} target={max_word_exposure}"
                )
            examples.append(base.Example(text=text, words=words, example_id=int(tobj.get("example_id", idx)), source=str(tobj.get("source", ""))))
            labels.append(lobj)
            selected_words += words
            selected_rows += 1
            last_tail_row = idx
            if not max_rows and selected_words == max_word_exposure:
                break
    if skipped_words != expected_start_tail_words:
        raise RuntimeError(f"skipped_words {skipped_words} != expected_start_tail_words {expected_start_tail_words}")
    if max_rows:
        if selected_rows != max_rows:
            raise RuntimeError(f"selected_rows {selected_rows} != max_rows {max_rows}")
    elif selected_words != max_word_exposure:
        raise RuntimeError(f"selected_words {selected_words} != max_word_exposure {max_word_exposure}")
    return examples, labels, {
        "tail_path": str(tail_path),
        "labels_path": str(labels_path),
        "start_tail_row": start_tail_row,
        "expected_start_tail_words": expected_start_tail_words,
        "skipped_words": skipped_words,
        "selected_rows": selected_rows,
        "selected_words": selected_words,
        "first_tail_row": first_tail_row,
        "last_tail_row": last_tail_row,
        "max_rows_mode": bool(max_rows),
    }


def collate_with_indices(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return base.collate(batch)


def convert_counter_dict(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, Counter):
            out[k] = dict(v)
        elif isinstance(v, defaultdict):
            out[k] = {kk: dict(vv) if isinstance(vv, Counter) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PVDM staged continuation trainer")
    ap.add_argument("--mode", choices=["treatment", "control", "standard_exact", "standard_legacy"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--init_checkpoint", default=str(DEFAULT_INIT))
    ap.add_argument("--tail_jsonl", default=str(DEFAULT_TAIL))
    ap.add_argument("--labels_jsonl", default=str(DEFAULT_LABELS))
    ap.add_argument("--tokenizer_path", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--start_tail_row", type=int, default=DEFAULT_START_TAIL_ROW)
    ap.add_argument("--expected_start_tail_words", type=int, default=DEFAULT_START_TAIL_WORDS)
    ap.add_argument("--initial_actual_word_exposure", type=int, default=DEFAULT_INIT_ACTUAL_EXPOSURE)
    ap.add_argument("--max_word_exposure", type=int, default=DEFAULT_STAGE_WORDS_TO_80M)
    ap.add_argument("--max_rows", type=int, default=0, help="smoke mode: ignore max_word_exposure and use this many rows")
    ap.add_argument("--checkpoint_name", default="chck_80M")
    ap.add_argument("--total_schedule_steps", type=int, default=DEFAULT_TOTAL_SCHEDULE_STEPS_70M_TO_100M)
    ap.add_argument("--batch_size", type=int, default=256, help="effective optimizer batch")
    ap.add_argument("--micro_batch_size", type=int, default=64, help="forward/backward microbatch preserving full-batch masks")
    ap.add_argument("--seq_length", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_fraction", type=float, default=0.06)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--target_prob", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--train_rng_seed", type=int, default=43023)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_trainer_state", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = build_args()
    if args.batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch_size and micro_batch_size must be positive")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size for exact grouping")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tail_path = Path(args.tail_jsonl); labels_path = Path(args.labels_jsonl); tok_path = Path(args.tokenizer_path); init_ckpt = Path(args.init_checkpoint)
    hashes = {
        "tail_sha256": sha256_file(tail_path),
        "labels_sha256": sha256_file(labels_path),
        "tokenizer_sha256": sha256_file(tok_path / "tokenizer.json"),
    }
    for k, expected in EXPECTED.items():
        if hashes[k] != expected:
            raise RuntimeError(f"{k} mismatch {hashes[k]} != {expected}")
    if not init_ckpt.exists():
        raise RuntimeError(f"init checkpoint not found: {init_ckpt}")
    tokenizer = base.make_portable_tokenizer(str(tok_path))
    examples, label_records, segment = load_segment(
        tail_path, labels_path,
        start_tail_row=args.start_tail_row,
        expected_start_tail_words=args.expected_start_tail_words,
        max_word_exposure=args.max_word_exposure,
        max_rows=args.max_rows,
    )
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_indices, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    stage_steps = math.ceil(len(dataset) / args.batch_size)
    schedule_total = max(args.total_schedule_steps, stage_steps)
    warmup = max(1, int(schedule_total * args.warmup_fraction))

    reset_all_rng(args.seed)
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_ckpt))
    if model.config.vocab_size != len(tokenizer):
        raise RuntimeError(f"vocab mismatch model {model.config.vocab_size} tokenizer {len(tokenizer)}")
    reset_all_rng(args.train_rng_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    param_count = sum(p.numel() for p in model.parameters())
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    legacy_state = base.MaskingCurriculumState(curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=0.7)
    legacy_state.initialize(vocab_size=len(tokenizer), total_steps=schedule_total)
    legacy_gen = torch.Generator(device=device)
    legacy_gen.manual_seed(args.train_rng_seed)

    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    manifest = {
        "status": "PVDM_CONTINUATION_MANIFEST",
        "created_utc": now_utc(),
        "mode": args.mode,
        "output_dir": str(out),
        "init_checkpoint": str(init_ckpt),
        "initial_actual_word_exposure": args.initial_actual_word_exposure,
        "target_stage_stop": args.checkpoint_name,
        "stage_is_physically_stopped_at_end": True,
        "segment": segment,
        "hashes": hashes,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.batch_size // args.micro_batch_size,
        "batch_size": args.batch_size,
        "stage_steps": stage_steps,
        "total_schedule_steps_for_possible_70M_to_100M_resume": schedule_total,
        "optimizer_reset": "AdamW reset symmetrically for treatment/control",
        "learning_rate": args.learning_rate,
        "warmup_fraction": args.warmup_fraction,
        "weight_decay": args.weight_decay,
        "mask_prob": args.mask_prob,
        "target_prob": args.target_prob,
        "seed": args.seed,
        "train_rng_seed": args.train_rng_seed,
        "source_words_consumed": source_words,
        "strict_batch_invariant_preflight": "experiments/archive/representation_and_objectives/data/pvdm_batch_invariants/pvdm_batch_invariant_summary.json",
    }
    (out / "example_order_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "pvdm_continuation_start",
        "mode": args.mode,
        "output_dir": str(out),
        "device": str(device),
        "param_count": param_count,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "stage_rows": len(examples),
        "stage_words": segment["selected_words"],
        "stage_steps": stage_steps,
        "schedule_total": schedule_total,
        "initial_actual_word_exposure": args.initial_actual_word_exposure,
        "expected_total_at_stage_end": args.initial_actual_word_exposure + segment["selected_words"],
    }), flush=True)

    log_path = out / "training_log.jsonl"
    mask_stats_path = out / "pvdm_mask_stats.jsonl"
    loss_values: list[float] = []
    cumulative_words = 0
    agg_stats = Counter()
    cat_stats: dict[str, Counter] = defaultdict(Counter)
    replacement_stats = Counter()
    reject_stats = Counter()
    selected_event_stats = Counter()

    with log_path.open("w", encoding="utf-8") as logf, mask_stats_path.open("w", encoding="utf-8") as maskf:
        for step, batch in enumerate(loader, 1):
            lo = (step - 1) * args.batch_size
            hi = lo + int(batch["input_ids"].shape[0])
            words = int(batch.pop("words").sum().item())
            input_ids_cpu = batch["input_ids"][:, :args.seq_length].contiguous()
            attention_cpu = batch["attention_mask"][:, :args.seq_length].contiguous()
            word_group_cpu = batch["word_group"][:, :args.seq_length].contiguous()
            if args.mode in {"treatment", "control"}:
                masked_cpu, labels_cpu, mask_stats = pvdm.apply_paired_pvdm_masking(
                    input_ids_cpu, attention_cpu, word_group_cpu, label_records[lo:hi], tokenizer,
                    arm=args.mode, seed=args.train_rng_seed, mask_prob=args.mask_prob, target_prob=args.target_prob,
                )
            elif args.mode == "standard_exact":
                row_ids = [int(r["tail_row_idx"]) for r in label_records[lo:hi]]
                masked_cpu, labels_cpu, mask_stats = pvdm.standard_wwm_exact_count_masking(
                    input_ids_cpu, attention_cpu, word_group_cpu, tokenizer,
                    seed=args.train_rng_seed, mask_prob=args.mask_prob, row_tail_indices=row_ids,
                )
            else:
                legacy_state.current_step = step - 1
                masked_dev, labels_dev = base.apply_masking_curriculum(
                    input_ids_cpu.to(device, non_blocking=True), attention_cpu.to(device, non_blocking=True), word_group_cpu.to(device, non_blocking=True), tokenizer, legacy_state, legacy_gen,
                )
                masked_cpu, labels_cpu = masked_dev.cpu(), labels_dev.cpu()
                mask_stats = {"arm": "standard_legacy", "selected_tokens": int((labels_cpu != -100).sum().item())}
                del masked_dev, labels_dev
            n_pred_total = int((labels_cpu != -100).sum().item())
            if n_pred_total <= 0:
                raise RuntimeError(f"no masked tokens at step {step}")
            optim.zero_grad(set_to_none=True)
            weighted_loss_sum = 0.0
            active_microbatches = 0
            batch_rows = int(masked_cpu.shape[0])
            for mb_start in range(0, batch_rows, args.micro_batch_size):
                mb_end = min(mb_start + args.micro_batch_size, batch_rows)
                sl_labels_cpu = labels_cpu[mb_start:mb_end]
                n_pred_i = int((sl_labels_cpu != -100).sum().item())
                if n_pred_i <= 0:
                    continue
                input_ids = masked_cpu[mb_start:mb_end].to(device, non_blocking=True)
                attention_mask = attention_cpu[mb_start:mb_end].to(device, non_blocking=True)
                labels_dev = sl_labels_cpu.to(device, non_blocking=True)
                out_model = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_dev)
                loss = out_model.loss
                if loss is None:
                    raise RuntimeError("model returned no loss")
                scale = n_pred_i / n_pred_total
                (loss * scale).backward()
                weighted_loss_sum += float(loss.detach().cpu()) * scale
                active_microbatches += 1
                del out_model, loss, input_ids, attention_mask, labels_dev
            if active_microbatches <= 0:
                raise RuntimeError(f"no active microbatches at step {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            loss_float = float(weighted_loss_sum)
            loss_values.append(loss_float)
            cumulative_words += words
            effective_mask_rate = n_pred_total / max(1, int(attention_cpu.sum().item()))
            for k in ["rows", "valid_groups", "label_events_total", "events_tokenization_usable", "events_selected", "rows_with_label_events", "rows_with_usable_events", "rows_with_selected_events", "desired_group_count_sum", "treatment_group_count_sum", "control_group_count_sum", "treatment_token_count_sum", "control_token_count_sum", "target_group_count_sum", "background_group_count_sum", "anchor_swap_group_count_sum", "rows_forced_above_desired_k", "selected_tokens"]:
                if k in mask_stats and isinstance(mask_stats[k], int):
                    agg_stats[k] += int(mask_stats[k])
            if isinstance(mask_stats.get("selected_event_categories"), dict):
                selected_event_stats.update({str(k): int(v) for k, v in mask_stats["selected_event_categories"].items()})
            if isinstance(mask_stats.get("reject_reasons"), dict):
                reject_stats.update({str(k): int(v) for k, v in mask_stats["reject_reasons"].items()})
            if isinstance(mask_stats.get("replacement_action_counts"), dict):
                replacement_stats.update({str(k): int(v) for k, v in mask_stats["replacement_action_counts"].items()})

            rec = {
                "step": step,
                "loss": loss_float,
                "lr": float(sched.get_last_lr()[0]),
                "batch_words": words,
                "cumulative_continuation_words": cumulative_words,
                "total_actual_word_exposure": args.initial_actual_word_exposure + cumulative_words,
                "seq_len": args.seq_length,
                "masked_tokens": n_pred_total,
                "effective_mask_rate": round(effective_mask_rate, 6),
                "mode": args.mode,
                "active_microbatches": active_microbatches,
                "micro_batch_size": args.micro_batch_size,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            maskf.write(json.dumps({"step": step, **convert_counter_dict(mask_stats)}, ensure_ascii=False) + "\n"); maskf.flush()
            if step == 1 or step % args.log_every == 0 or step == stage_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            del masked_cpu, labels_cpu, input_ids_cpu, attention_cpu, word_group_cpu

    # Save the staged checkpoint and a parent hf_model copy.  The parent is
    # intentionally the 80M model for easy evaluator loading; it is not a 100M endpoint.
    ckpt_dir = out / "hf_model" / args.checkpoint_name
    base.save_hf_checkpoint(model, tokenizer, ckpt_dir)
    base.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    trainer_state_path = None
    if args.save_trainer_state:
        state_dir = out / "trainer_state" / args.checkpoint_name
        state_dir.mkdir(parents=True, exist_ok=True)
        trainer_state_path = state_dir / "optimizer_scheduler_rng.pt"
        torch.save({
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "completed_stage_steps": stage_steps,
            "cumulative_continuation_words": cumulative_words,
            "total_actual_word_exposure": args.initial_actual_word_exposure + cumulative_words,
            "last_tail_row": segment["last_tail_row"],
            "next_tail_row": int(segment["last_tail_row"]) + 1 if segment["last_tail_row"] is not None else None,
            "schedule_total_steps": schedule_total,
        }, trainer_state_path)
    metrics = {
        "variant": f"pvdm_continuation_{args.mode}",
        "backend": "mlm",
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "word_exposure": args.initial_actual_word_exposure + cumulative_words,
        "continuation_words": cumulative_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": stage_steps,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.batch_size // args.micro_batch_size,
        "stage_stop_name": args.checkpoint_name,
        "saved_checkpoints": [{
            "name": args.checkpoint_name,
            "target_word_exposure": args.initial_actual_word_exposure + args.max_word_exposure,
            "actual_cumulative_word_exposure": args.initial_actual_word_exposure + cumulative_words,
            "path": str(ckpt_dir),
        }],
        "pvdm_pairing": {
            "mode": args.mode,
            "mask_prob": args.mask_prob,
            "target_prob": args.target_prob,
            "selected_event_categories": dict(selected_event_stats),
            "reject_reasons": dict(reject_stats),
            "replacement_action_counts": dict(replacement_stats),
            "aggregate_mask_stats": dict(agg_stats),
        },
        "trainer_state_path": str(trainer_state_path) if trainer_state_path else None,
        "example_order_manifest": str(out / "example_order_manifest.json"),
        "training_log": str(log_path),
        "pvdm_mask_stats": str(mask_stats_path),
        **manifest,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "done",
        "mode": args.mode,
        "output_dir": str(out),
        "checkpoint": str(ckpt_dir),
        "continuation_words": cumulative_words,
        "total_actual_word_exposure": args.initial_actual_word_exposure + cumulative_words,
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "trainer_state_path": metrics["trainer_state_path"],
    }), flush=True)


if __name__ == "__main__":
    main()
