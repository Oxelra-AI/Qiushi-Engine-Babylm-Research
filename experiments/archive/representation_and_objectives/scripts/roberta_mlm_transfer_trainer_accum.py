#!/usr/bin/env python3
"""research repaired RoBERTa MLM trainer with microbatch accumulation.

This is a local repair of the RoBERTa masked-LM trainer after
full 256-row backward passes OOMed under the current shared GPU memory envelope.
It preserves the scientific optimizer-batch geometry: examples are still grouped
into 256-row training updates, LR scheduling still has 2,529 optimizer steps,
and checkpoints are saved at the same cumulative-word milestones.  Only the
within-update forward/backward computation is split into smaller microbatches.

The masking is generated once per full optimizer batch on CPU before splitting,
so changing ``--micro_batch_size`` does not change which WWM labels are selected
inside a 256-row update.  Microbatch losses are weighted by active-label count,
matching the full-batch mean-loss gradient up to dropout/RNG implementation
changes inherent in microbatch execution.
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
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
BASE_TRAINER = ROOT / "experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py"
spec = importlib.util.spec_from_file_location("a02_step209_roberta_trainer", BASE_TRAINER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import base trainer from {BASE_TRAINER}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)  # type: ignore[union-attr]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def aggregate_mask_stats(stats_list: list[dict[str, Any]]) -> dict[str, int]:
    keys = ["masked_tokens", "candidate_tokens", "selected_groups", "total_groups"]
    return {k: int(sum(int(s.get(k, 0)) for s in stats_list)) for k in keys}


def safe_cuda_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": "cpu"}
    idx = torch.cuda.current_device()
    try:
        free, total = torch.cuda.mem_get_info(idx)
        return {
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "free_bytes": int(free),
            "total_bytes": int(total),
            "allocated_bytes": int(torch.cuda.memory_allocated(idx)),
            "reserved_bytes": int(torch.cuda.memory_reserved(idx)),
        }
    except Exception as e:  # pragma: no cover - defensive logging only
        return {"device": str(device), "cuda_mem_info_error": repr(e)}


def save_checkpoint_with_record(model, tokenizer, out: Path, name: str, target_words: int, actual_words: int, saved_checkpoints: list[dict[str, Any]], final_alias: bool = False) -> None:
    cp = out / "hf_model" / name
    base.save_hf_checkpoint(model, tokenizer, cp)
    saved_checkpoints.append({
        "name": name,
        "target_word_exposure": int(target_words),
        "actual_cumulative_word_exposure": int(actual_words),
        "path": str(cp),
        "final_alias": bool(final_alias),
    })
    print(json.dumps({"event": "checkpoint_saved", "name": name, "target_words": target_words, "cum_words": actual_words, "final_alias": final_alias}), flush=True)


def train_accum(args: argparse.Namespace) -> None:
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.micro_batch_size <= 0 or args.micro_batch_size > args.batch_size:
        raise ValueError(f"micro_batch_size must be in [1,batch_size], got {args.micro_batch_size} vs batch_size {args.batch_size}")
    if args.batch_size % args.micro_batch_size != 0:
        print(json.dumps({"event": "warning", "message": "batch_size not divisible by micro_batch_size; final microstep in each batch will be smaller", "batch_size": args.batch_size, "micro_batch_size": args.micro_batch_size}), flush=True)

    command_payload = {
        "created_utc": now_utc(),
        "argv": os.sys.argv,
        "base_trainer": str(BASE_TRAINER),
        "recipe": base.summarize_recipe(args),
        "effective_optimizer_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "mask_generation": "one CPU WWM call per full optimizer batch, then microbatch split",
        "loss_scaling": "microbatch mean losses weighted by active label count over full optimizer batch",
        "save_final_as_chck_100M": bool(args.save_final_as_chck_100M),
        "max_train_steps": args.max_train_steps,
    }
    base.write_json(out / "train_command.json", command_payload)

    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    examples, data_meta = base.load_examples_jsonl(args.example_jsonl, selected_words=args.max_word_exposure, max_rows=None)
    actual_words = sum(ex.words for ex in examples)
    if actual_words != args.max_word_exposure:
        raise RuntimeError(f"actual_words {actual_words} != max_word_exposure {args.max_word_exposure}")

    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    total_steps = len(loader)
    if total_steps != math.ceil(len(examples) / args.batch_size):
        raise RuntimeError(f"unexpected total_steps {total_steps} for rows {len(examples)} batch {args.batch_size}")
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))

    base.reset_rng(args.seed)
    if args.extra_init_seed >= 0:
        base.reset_rng(args.extra_init_seed)
    model = base.build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        base.reset_rng(args.train_rng_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)

    # Use a CPU generator so the full-batch mask is independent of microbatch size
    # and can be generated without allocating the full batch on GPU.
    mask_gen = torch.Generator(device="cpu")
    mask_gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    base.write_json(out / "example_order_manifest.json", {
        "data_source_type": "example_jsonl",
        "example_jsonl": str(args.example_jsonl),
        "example_jsonl_label": args.example_jsonl_label,
        "selected_for_training_words": actual_words,
        "num_consumed_examples": len(examples),
        "source_words_consumed": source_words,
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_json_sha256": base.sha256_file(Path(args.tokenizer_path) / "tokenizer.json"),
        **data_meta,
    })

    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    start = time.time()

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch.pop("words").sum().item())
            _example_ids = batch.pop("example_id")
            input_ids_cpu = batch["input_ids"][:, :args.seq_length].contiguous()
            attention_mask_cpu = batch["attention_mask"][:, :args.seq_length].contiguous()
            word_group_cpu = batch["word_group"][:, :args.seq_length].contiguous()

            masked_inputs_cpu, labels_cpu, mask_stats = base.apply_wwm_masking(
                input_ids_cpu, attention_mask_cpu, word_group_cpu, tokenizer, args.mask_prob, mask_gen
            )
            active_counts: list[int] = []
            ranges: list[tuple[int, int]] = []
            n = int(masked_inputs_cpu.shape[0])
            for s in range(0, n, args.micro_batch_size):
                e = min(n, s + args.micro_batch_size)
                ranges.append((s, e))
                active_counts.append(int((labels_cpu[s:e] != -100).sum().item()))
            total_active = int(sum(active_counts))
            if total_active <= 0:
                raise RuntimeError(f"no active labels at optimizer step {step}")

            optim.zero_grad(set_to_none=True)
            weighted_loss_value = 0.0
            for (s, e), active in zip(ranges, active_counts):
                if active <= 0:
                    continue
                mb_inputs = masked_inputs_cpu[s:e].to(device, non_blocking=True)
                mb_attention = attention_mask_cpu[s:e].to(device, non_blocking=True)
                mb_labels = labels_cpu[s:e].to(device, non_blocking=True)
                out_model = model(input_ids=mb_inputs, attention_mask=mb_attention, labels=mb_labels)
                loss = out_model.loss
                if loss is None:
                    raise RuntimeError("model returned no loss")
                weight = float(active) / float(total_active)
                (loss * weight).backward()
                weighted_loss_value += float(loss.detach().cpu()) * weight
                del mb_inputs, mb_attention, mb_labels, out_model, loss

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()

            cumulative_words += words
            loss_float = float(weighted_loss_value)
            loss_values.append(loss_float)
            rec = {
                "step": step,
                "loss": loss_float,
                "lr": float(sched.get_last_lr()[0]),
                "batch_words": words,
                "cumulative_word_exposure": cumulative_words,
                "seq_len": args.seq_length,
                "masked_tokens": mask_stats["masked_tokens"],
                "candidate_tokens": mask_stats["candidate_tokens"],
                "selected_groups": mask_stats["selected_groups"],
                "total_groups": mask_stats["total_groups"],
                "effective_mask_rate": round(mask_stats["masked_tokens"] / max(1, mask_stats["candidate_tokens"]), 4),
                "active_label_tokens": total_active,
                "micro_batch_size": args.micro_batch_size,
                "micro_steps": len(ranges),
                "effective_optimizer_batch_size": args.batch_size,
                "loss_weighting": "active_label_count",
                "grad_norm_preclip": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
                "mask_mode": "wwm_cpu_full_batch_then_microbatch",
                "mask_prob_nominal": args.mask_prob,
                "elapsed_sec": round(time.time() - start, 1),
            }
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                rec["cuda_memory"] = safe_cuda_memory(device)
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
                save_checkpoint_with_record(model, tokenizer, out, name, int(next_ckpt), cumulative_words, saved_checkpoints, final_alias=False)
                next_ckpt += args.checkpoint_words

            if args.max_train_steps is not None and step >= args.max_train_steps:
                print(json.dumps({"event": "max_train_steps_reached", "step": step, "cum_words": cumulative_words}), flush=True)
                break

    # Save final root.  For the 99,999,910-word factorial streams, also save a
    # chck_100M alias that records the actual exposure, so the existing evaluator
    # can use a stable endpoint name without hiding the 90-word shortfall.
    base.save_hf_checkpoint(model, tokenizer, out / "hf_model")
    if args.save_final_as_chck_100M and (args.max_train_steps is None):
        save_checkpoint_with_record(model, tokenizer, out, "chck_100M", args.max_word_exposure, cumulative_words, saved_checkpoints, final_alias=True)

    metrics = {
        "status": "ROBERTA_ACCUM_TRAINING_DONE" if args.max_train_steps is None else "ROBERTA_ACCUM_PARTIAL_TRAINING_DONE",
        "variant": f"step241_{args.model_family}_wwm_microbatch_accum",
        "backend": "mlm",
        "model_family": f"{args.model_family}_ForMaskedLM",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": str(args.tokenizer_path),
        "word_exposure": cumulative_words,
        "target_word_exposure": args.max_word_exposure,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": step if loss_values else 0,
        "planned_training_steps": total_steps,
        "optimizer_steps": len(loss_values),
        "optimizer": "AdamW betas=(0.9,0.98)",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_fraction": args.warmup_fraction,
        "lr_total_steps": schedule_total,
        "masking_curriculum": "wwm_fixed",
        "mask_generation": "CPU full optimizer batch before microbatch split",
        "mask_prob_start": args.mask_prob,
        "mask_prob_end": args.mask_prob,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "seq_length": args.seq_length,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "effective_optimizer_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "loss_scaling": "active-label weighted microbatch mean",
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "gradient_checkpointing": args.gradient_checkpointing,
        "saved_checkpoints": saved_checkpoints,
        "source_words_consumed": source_words,
        "data_source_type": "example_jsonl",
        "example_jsonl": str(args.example_jsonl),
        "example_jsonl_label": args.example_jsonl_label,
        "elapsed_sec": round(time.time() - start, 1),
        "cuda_memory_final": safe_cuda_memory(device),
        "base_trainer": str(BASE_TRAINER),
        "scientific_note": "Preserves 256-example optimizer-step geometry and 2529-update LR horizon while reducing memory through microbatch accumulation; not bit-equivalent to the failed full-batch GPU-masking trainer.",
    }
    base.write_json(out / "scientific_metrics.json", metrics)
    print(json.dumps({"event": "done", "status": metrics["status"], "param_count": metrics["parameter_count"], "loss_first": metrics["loss_first"], "loss_last": metrics["loss_last"], "word_exposure": cumulative_words, "checkpoints": [c["name"] for c in saved_checkpoints]}, indent=2), flush=True)


def build_argparser() -> argparse.ArgumentParser:
    ap = base.build_argparser()
    ap.add_argument("--micro_batch_size", type=int, default=32, help="Rows per forward/backward microbatch; gradients are accumulated to the requested batch_size.")
    ap.add_argument("--save_final_as_chck_100M", action="store_true", help="Save final endpoint under hf_model/chck_100M while recording actual word exposure.")
    ap.add_argument("--max_train_steps", type=int, default=None, help="Debug/pilot limit on optimizer steps; do not use for final arms.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        base.preflight(args)
        # Add a small sidecar noting accumulation settings.
        p = args.output_dir / "preflight_accum_sidecar.json"
        base.write_json(p, {"status": "ACCUM_PREFLIGHT_SIDECAR", "micro_batch_size": args.micro_batch_size, "effective_optimizer_batch_size": args.batch_size, "save_final_as_chck_100M": args.save_final_as_chck_100M, "base_trainer": str(BASE_TRAINER)})
        return
    if args.smoke_forward_only:
        base.smoke_forward(args)
        return
    train_accum(args)


if __name__ == "__main__":
    main()
