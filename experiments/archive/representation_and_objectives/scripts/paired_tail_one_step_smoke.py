#!/usr/bin/env python3
"""research one-effective-batch smoke for the research paired-tail continuation.

This is not an endpoint and does not save a model.  It verifies that both paired
80M->100M continuation modes can load the same legal40k 80M checkpoint, consume
the first true tail effective batch, apply the same actual device-side WWM mask
stream, run a forward/backward/update, and differ only in the intended LR at the
first update.  The corrected curriculum endpoint remains the primary pending
experiment; this smoke only reduces execution risk for the held fallback.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2ForMaskedLM

USER_ROOT = Path(".").resolve()
SCRIPT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/scripts"
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive/compact_experience/scripts"
for p in (SCRIPT_DIR, COMPACT_EXPERIENCE_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paired_tail_continuation_trainer as tail  # noqa: E402
import masking_curriculum_trainer as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-step actual-device smoke for paired tail continuations")
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--source_model_path", required=True)
    p.add_argument("--source_metrics_path", required=True)
    p.add_argument("--source_training_log", required=True)
    p.add_argument("--source_checkpoint_name", default="chck_80M")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--train_rng_seed", type=int, default=53023)
    p.add_argument("--max_total_word_exposure", type=int, default=100_000_000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.98)
    p.add_argument("--original_base_lr", type=float, default=0.001)
    p.add_argument("--original_total_steps", type=int, default=0)
    p.add_argument("--original_warmup_fraction", type=float, default=0.06)
    p.add_argument("--reheat_peak_lr", type=float, default=3e-4)
    p.add_argument("--reheat_warmup_fraction", type=float, default=0.06)
    return p.parse_args()


def set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for group in optim.param_groups:
        group["lr"] = lr


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    source_metrics_path = Path(args.source_metrics_path)
    source_training_log_path = Path(args.source_training_log)
    source_model_path = Path(args.source_model_path)
    source_rec = tail.read_source_checkpoint(source_metrics_path, args.source_checkpoint_name)
    consumed_words = int(source_rec["actual_cumulative_word_exposure"])
    training_rows = tail.read_training_log(source_training_log_path)
    source_step_rec = tail.find_source_step(training_rows, consumed_words)
    source_step = int(source_step_rec["step"])
    original_total_steps = args.original_total_steps if args.original_total_steps > 0 else len(training_rows)
    original_warmup_steps = max(1, int(original_total_steps * args.original_warmup_fraction))

    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    all_examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(
        Path(args.example_jsonl), args.max_total_word_exposure
    )
    tail_examples, tail_start_idx, continuation_words = tail.select_tail_examples(
        all_examples, consumed_words, args.max_total_word_exposure
    )
    dataset = base.MaskedChunkDataset(tail_examples, tokenizer, args.max_seq_length)
    accum_steps = args.batch_size // args.micro_batch_size
    full_tail_steps = math.ceil(len(dataset) / args.batch_size)
    if source_step + full_tail_steps != original_total_steps:
        raise RuntimeError(
            f"tail steps do not land at original total: source={source_step} tail={full_tail_steps} original={original_total_steps}"
        )
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False,
                        collate_fn=base.collate, num_workers=0, pin_memory=False)
    first_batch = next(tail.effective_batch_iterator(loader, accum_steps))
    first_plain = {
        "input_ids_sha256": tail.tensor_sha(first_batch["input_ids"]),
        "attention_mask_sha256": tail.tensor_sha(first_batch["attention_mask"]),
        "word_group_sha256": tail.tensor_sha(first_batch["word_group"]),
        "words_sha256": tail.tensor_sha(first_batch["words"]),
        "words_sum": int(first_batch["words"].sum().item()),
    }

    reheat_warmup_steps = max(1, int(full_tail_steps * args.reheat_warmup_fraction))

    def lr_for_update(mode: str, local_step: int) -> float:
        if mode == "residual_low_lr":
            return tail.hf_cosine_lr(
                args.original_base_lr, original_warmup_steps, original_total_steps,
                source_step + local_step - 1,
            )
        if mode == "reheat_cosine":
            return tail.hf_cosine_lr(args.reheat_peak_lr, reheat_warmup_steps, full_tail_steps, local_step - 1)
        raise AssertionError(mode)

    def lr_after_update(mode: str, local_step: int) -> float:
        if mode == "residual_low_lr":
            return tail.hf_cosine_lr(
                args.original_base_lr, original_warmup_steps, original_total_steps,
                source_step + local_step,
            )
        if mode == "reheat_cosine":
            return tail.hf_cosine_lr(args.reheat_peak_lr, reheat_warmup_steps, full_tail_steps, local_step)
        raise AssertionError(mode)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    mode_records: dict[str, Any] = {}
    for mode in ["residual_low_lr", "reheat_cosine"]:
        reset_all_rng(args.seed)
        model = DebertaV2ForMaskedLM.from_pretrained(source_model_path, local_files_only=True)
        reset_all_rng(args.train_rng_seed)
        model.to(device)
        model.train()
        param_count = sum(p.numel() for p in model.parameters())
        optim = torch.optim.AdamW(
            model.parameters(), lr=0.0, weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
        )
        curriculum_state = base.MaskingCurriculumState(
            curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob, switch_frac=1.0,
        )
        curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=full_tail_steps)
        gen = torch.Generator(device=device)
        gen.manual_seed(args.train_rng_seed)

        input_ids = first_batch["input_ids"].to(device)
        attention_mask = first_batch["attention_mask"].to(device)
        word_group = first_batch["word_group"].to(device)
        words = int(first_batch["words"].sum().item())
        curriculum_state.current_step = 0
        masked_inputs, labels = base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, curriculum_state, gen)
        n_pred_total = int((labels != -100).sum().item())
        if n_pred_total <= 0:
            raise RuntimeError(f"no masked tokens for mode {mode}")
        update_lr = lr_for_update(mode, 1)
        next_lr = lr_after_update(mode, 1)
        set_lr(optim, update_lr)
        optim.zero_grad(set_to_none=True)
        weighted_loss_sum = 0.0
        batch_rows = input_ids.shape[0]
        active_microbatches = 0
        for mb_start in range(0, batch_rows, args.micro_batch_size):
            mb_end = min(mb_start + args.micro_batch_size, batch_rows)
            sl_labels = labels[mb_start:mb_end]
            n_pred_i = int((sl_labels != -100).sum().item())
            if n_pred_i <= 0:
                continue
            out = model(input_ids=masked_inputs[mb_start:mb_end], attention_mask=attention_mask[mb_start:mb_end], labels=sl_labels)
            loss_i = out.loss
            if loss_i is None:
                raise RuntimeError("model returned no loss")
            scale = n_pred_i / n_pred_total
            (loss_i * scale).backward()
            weighted_loss_sum += float(loss_i.detach().cpu()) * scale
            active_microbatches += 1
            del out, loss_i
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        optim.step()
        rec = {
            "mode": mode,
            "device": str(device),
            "parameter_count": param_count,
            "update_lr": update_lr,
            "next_update_lr": next_lr,
            "loss_before_update_weighted": float(weighted_loss_sum),
            "grad_norm_preclip": grad_norm,
            "words": words,
            "masked_tokens": n_pred_total,
            "effective_mask_rate": n_pred_total / max(1, int(attention_mask.bool().sum().item())),
            "active_microbatches": active_microbatches,
            "hashes": {
                "masked_inputs_sha256": tail.tensor_sha(masked_inputs),
                "labels_sha256": tail.tensor_sha(labels),
                "selected_positions_sha256": tail.tensor_sha(labels != -100),
            },
        }
        mode_records[mode] = rec
        print(json.dumps({"event": "mode_done", **{k: rec[k] for k in ["mode", "device", "update_lr", "loss_before_update_weighted", "grad_norm_preclip", "masked_tokens"]}}), flush=True)
        del optim, model, input_ids, attention_mask, word_group, masked_inputs, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    r = mode_records["residual_low_lr"]
    h = mode_records["reheat_cosine"]
    summary = {
        "status": "PAIRED_TAIL_ONE_STEP_SMOKE_DONE",
        "purpose": "one-effective-batch actual-device validation of research paired-tail fallback; no endpoint and no saved model",
        "script": str(_public_path('experiments/archive/representation_and_objectives/scripts/paired_tail_one_step_smoke.py')),
        "source_checkpoint": {
            "path": str(source_model_path),
            "name": args.source_checkpoint_name,
            "consumed_words": consumed_words,
            "source_step": source_step,
            "source_logged_lr_after_step": float(source_step_rec.get("lr")),
        },
        "stream": {
            "example_jsonl": args.example_jsonl,
            "jsonl_total_words": jsonl_total_words,
            "jsonl_total_rows": jsonl_total_rows,
            "tail_start_row_index": tail_start_idx,
            "tail_examples": len(tail_examples),
            "continuation_words": continuation_words,
            "full_tail_steps": full_tail_steps,
            "first_effective_batch_plain_hashes": first_plain,
            "sample_rows": sample_rows[:3],
        },
        "lr_geometry": {
            "original_total_steps": original_total_steps,
            "original_warmup_steps": original_warmup_steps,
            "reheat_warmup_steps": reheat_warmup_steps,
            "residual_first_update_lr": r["update_lr"],
            "residual_first_after_update_lr": r["next_update_lr"],
            "reheat_first_update_lr": h["update_lr"],
            "reheat_first_after_update_lr": h["next_update_lr"],
        },
        "mode_records": mode_records,
        "paired_checks": {
            "same_actual_device_masked_inputs": r["hashes"]["masked_inputs_sha256"] == h["hashes"]["masked_inputs_sha256"],
            "same_actual_device_labels": r["hashes"]["labels_sha256"] == h["hashes"]["labels_sha256"],
            "same_actual_device_selected_positions": r["hashes"]["selected_positions_sha256"] == h["hashes"]["selected_positions_sha256"],
            "same_loss_before_update": abs(r["loss_before_update_weighted"] - h["loss_before_update_weighted"]) < 1e-10,
            "same_grad_norm_preclip": abs(r["grad_norm_preclip"] - h["grad_norm_preclip"]) < 1e-7,
            "residual_lr_matches_source_logged": abs(r["update_lr"] - float(source_step_rec.get("lr"))) < 1e-12,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = out_dir / "paired_tail_one_step_smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "summary_saved", "path": str(out_path), "paired_checks": summary["paired_checks"], "elapsed_sec": summary["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
