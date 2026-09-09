#!/usr/bin/env python3
"""research: SGCR (Support-Gated Compositional Residual) trainer for DeBERTa-v2.

Wraps the research accumulated masking curriculum trainer with SGCR embedding
modification. Preserves exact legal40k tokenization, WWM labels, data order,
stream, masking, optimizer schedule, and seed identities.

New: component_embeddings (16384×d_comp) + component_proj (d_comp→hidden).

Usage:
  python3 sgcr_trainer.py \
    --example_jsonl <100M stream> \
    --tokenizer_path <legal40k dir> \
    --output_dir <output> \
    --sgcr_d_comp 64 \
    --sgcr_K 50 \
    --sgcr_tok16k_path <legal16k dir> \
    --sgcr_pool_jsonl <10M pool> \
    [all other research args]

For depth architecture: add --n_layer 12 --hidden_size 384 --n_head 12 --intermediate_size 1280
For uniform-gate control: add --sgcr_uniform_gate
"""
from __future__ import annotations

import argparse
import collections
import hashlib
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
REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = USER_ROOT / "experiments/archive" / 'representation_and_objectives' / "scripts"
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPRESENTATION_FRONTIER_STUDIES_SCRIPTS))
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402
import sgcr_module as sgcr_mod     # noqa: E402


def build_args() -> argparse.Namespace:
    """Build argument namespace matching research + SGCR extensions."""
    p = argparse.ArgumentParser(description="SGCR accumulated masking-curriculum trainer")

    # === research inherited arguments ===
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
    p.add_argument("--intermediate_size", type=int, default=0)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--dynamics_trace_every", type=int, default=200)

    # === SGCR-specific arguments ===
    p.add_argument("--sgcr_d_comp", type=int, default=64,
                   help="Component embedding dimension")
    p.add_argument("--sgcr_K", type=float, default=50.0,
                   help="Support gate parameter K: rho_t = n_t/(n_t+K)")
    p.add_argument("--sgcr_tok16k_path", required=True,
                   help="Path to legal16k tokenizer directory")
    p.add_argument("--sgcr_pool_jsonl", required=True,
                   help="Path to 10M training pool JSONL for computing counts")
    p.add_argument("--sgcr_uniform_gate", action="store_true",
                   help="Use uniform gate (parameter-matched control)")
    p.add_argument("--sgcr_init_mode", default="cold", choices=["cold", "warm"],
                   help="Initialization mode for component parameters; cold is exact-preserving but gradient-live")

    # === Dry-run mode ===
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())



def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decomposition_map_sha256(decomp_map: dict[int, list[int]]) -> str:
    payload = json.dumps(
        {str(k): decomp_map[k] for k in sorted(decomp_map)},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def combine_microbatches(micro_batches):
    return {
        "input_ids": torch.cat([b["input_ids"] for b in micro_batches], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in micro_batches], dim=0),
        "word_group": torch.cat([b["word_group"] for b in micro_batches], dim=0),
        "words": torch.cat([b["words"] for b in micro_batches], dim=0),
    }


def effective_batch_iterator(loader, accum_steps):
    buf = []
    for batch in loader:
        buf.append(batch)
        if len(buf) == accum_steps:
            yield combine_microbatches(buf)
            buf = []
    if buf:
        yield combine_microbatches(buf)


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

    # Load tokenizer and examples (same as research)
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)

    if not args.example_jsonl:
        raise RuntimeError("SGCR trainer requires example_jsonl")
    jsonl_path = Path(args.example_jsonl)
    examples, jsonl_total_words, jsonl_total_rows, sample_rows = base.load_examples_jsonl(
        jsonl_path, args.max_word_exposure
    )
    actual_words = sum(ex.words for ex in examples)
    if actual_words != args.max_word_exposure:
        raise RuntimeError(f"word selection mismatch {actual_words} vs {args.max_word_exposure}")

    dataset = base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(
        dataset, batch_size=args.micro_batch_size, shuffle=False,
        collate_fn=base.collate, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    total_steps = math.ceil(len(dataset) / args.batch_size)

    token_freq_ranks = base.compute_token_frequency_ranks(examples, tokenizer, args.max_seq_length)

    curriculum_state = base.MaskingCurriculumState(
        curriculum=args.masking_curriculum,
        mask_prob_start=args.mask_prob_start, mask_prob_end=args.mask_prob_end,
        switch_frac=args.switch_frac, amlm_window=args.amlm_window,
        amlm_lambda=args.amlm_lambda,
    )
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)

    # === Build model (same RNG sequence as research) ===
    reset_all_rng = lambda s: (random.seed(s), torch.manual_seed(s),
                                torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None)
    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)

    # Handle intermediate_size override for depth models
    if args.intermediate_size > 0:
        from transformers import DebertaV2Config, DebertaV2ForMaskedLM
        max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
        pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]
        cfg = DebertaV2Config(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.n_layer,
            num_attention_heads=args.n_head,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=max_pos,
            max_relative_positions=args.max_relative_positions,
            position_buckets=args.position_buckets,
            relative_attention=True,
            pos_att_type=pos_att_type,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = DebertaV2ForMaskedLM(cfg)
    else:
        model = base.build_model(args, tokenizer)

    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    base_param_count = sum(p.numel() for p in model.parameters())

    # === Apply SGCR ===
    print(json.dumps({"event": "sgcr_building", "created_utc": now_utc()}), flush=True)

    # Build decomposition map and pool counts
    decomp_map = sgcr_mod.build_decomposition_map(args.tokenizer_path, args.sgcr_tok16k_path)
    counts_40k = sgcr_mod.build_pool_counts(args.tokenizer_path, args.sgcr_pool_jsonl)

    sgcr_config = sgcr_mod.SGCRConfig(
        vocab_40k=len(tokenizer),
        vocab_16k=16384,
        hidden_size=args.hidden_size,
        d_comp=args.sgcr_d_comp,
        K=args.sgcr_K,
        uniform_gate=args.sgcr_uniform_gate,
        init_mode=args.sgcr_init_mode,
        force_standard_ids=sorted(set(int(x) for x in tokenizer.all_special_ids if x is not None)),
    )
    model, sgcr_emb = sgcr_mod.apply_sgcr_to_model(model, sgcr_config, decomp_map, counts_40k)

    # Apply forward hooks so the standard HF forward pass uses SGCR embeddings
    model = sgcr_mod.sgcr_forward_embedding_hook(model, sgcr_emb)

    sgcr_new_params = sgcr_emb.new_parameter_count
    total_param_count = base_param_count + sgcr_new_params

    # SGCR component initialization intentionally adds random parameters, but it
    # must not shift the inherited training stochasticity (dropout/kernel RNG)
    # relative to the matched depth baseline.  The standard trainer resets the
    # train RNG after model initialization; SGCR construction occurs after that
    # reset, so restore the train RNG here before optimizer/training.
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    # Verify cold-init preservation
    with torch.no_grad():
        zsl = sgcr_mod.verify_zero_sharing_limit(model, sgcr_emb)
    if not zsl["exact_recovery"]:
        raise RuntimeError(f"SGCR zero-sharing limit failed: max_diff={zsl['max_diff']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sgcr_emb.to(device)

    # === Optimizer: include SGCR parameters ===
    # Standard model params + SGCR component params in same group
    all_params = list(model.parameters()) + list(sgcr_emb.component_embeddings.parameters()) + list(sgcr_emb.component_proj.parameters())
    # Deduplicate (word_embeddings.weight is in both model and sgcr_emb)
    seen = set()
    unique_params = []
    for p in all_params:
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique_params.append(p)

    optim = torch.optim.AdamW(unique_params, lr=args.learning_rate,
                               weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup,
                                             num_training_steps=schedule_total)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    decomp_lengths_list = [len(decomp_map[i]) for i in range(len(tokenizer))]
    decomp_length_histogram = dict(sorted(collections.Counter(decomp_lengths_list).items()))
    counts_tensor = torch.tensor([counts_40k.get(i, 0) for i in range(len(tokenizer))], dtype=torch.float32)
    rho_cpu = sgcr_emb.rho.detach().cpu().float()
    special_id_set = set(int(x) for x in tokenizer.all_special_ids if x is not None)
    force_mask = torch.tensor([(i in special_id_set) or decomp_lengths_list[i] <= 0 for i in range(len(tokenizer))], dtype=torch.bool)
    ordinary_used = (counts_tensor > 0) & ~force_mask
    if ordinary_used.any():
        gate_mass_weighted_rho = float((counts_tensor[ordinary_used] * rho_cpu[ordinary_used]).sum() / counts_tensor[ordinary_used].sum())
        gate_mass_weighted_residual = float((counts_tensor[ordinary_used] * (1.0 - rho_cpu[ordinary_used])).sum() / counts_tensor[ordinary_used].sum())
        gate_type_mean_rho = float(rho_cpu[ordinary_used].mean())
    else:
        gate_mass_weighted_rho = None
        gate_mass_weighted_residual = None
        gate_type_mean_rho = None

    sgcr_manifest = {
        "sgcr_enabled": True,
        "sgcr_d_comp": args.sgcr_d_comp,
        "sgcr_K": args.sgcr_K,
        "sgcr_uniform_gate": args.sgcr_uniform_gate,
        "sgcr_init_mode": args.sgcr_init_mode,
        "sgcr_tok16k_path": args.sgcr_tok16k_path,
        "sgcr_pool_jsonl": args.sgcr_pool_jsonl,
        "sgcr_tokenizer40_path": args.tokenizer_path,
        "sgcr_tokenizer40_sha256": sha256_file(Path(args.tokenizer_path) / "tokenizer.json"),
        "sgcr_tokenizer16_sha256": sha256_file(Path(args.sgcr_tok16k_path) / "tokenizer.json"),
        "sgcr_exact_prefix_decomposition": True,
        "sgcr_decomposition_map_sha256": decomposition_map_sha256(decomp_map),
        "sgcr_decomposition_length_histogram": decomp_length_histogram,
        "sgcr_decomposition_max_components": max(decomp_lengths_list),
        "sgcr_decomposition_component_slots": sum(decomp_lengths_list),
        "sgcr_pool_count_total_tokens": int(counts_tensor.sum().item()),
        "sgcr_pool_count_used_types": int((counts_tensor > 0).sum().item()),
        "sgcr_gate_type_mean_rho_ordinary_used": gate_type_mean_rho,
        "sgcr_gate_mass_weighted_rho_ordinary_used": gate_mass_weighted_rho,
        "sgcr_gate_mass_weighted_residual_ordinary_used": gate_mass_weighted_residual,
        "sgcr_force_standard_ids": sorted(special_id_set),
        "sgcr_sidecar_resume_status": "diagnostic_partial_state_only_no_optimizer_scheduler_rng_resume",
        "base_param_count": base_param_count,
        "sgcr_new_params": sgcr_new_params,
        "total_param_count": total_param_count,
        "optimizer_unique_params": len(unique_params),
        "zero_sharing_verified": zsl["exact_recovery"],
    }

    print(json.dumps({
        "event": "sgcr_trainer_start",
        "created_utc": now_utc(),
        "device": str(device),
        "base_params": base_param_count,
        "sgcr_new_params": sgcr_new_params,
        "total_params": total_param_count,
        "sgcr_K": args.sgcr_K,
        "sgcr_d_comp": args.sgcr_d_comp,
        "uniform_gate": args.sgcr_uniform_gate,
        "effective_batch_size": args.batch_size,
        "total_effective_steps": total_steps,
    }, indent=2), flush=True)

    if args.dry_run:
        print(json.dumps({"event": "dry_run_complete", "sgcr_manifest": sgcr_manifest}), flush=True)
        return

    # === Training loop (same structure as research) ===
    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values = []
    saved_checkpoints = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(effective_batch_iterator(loader, accum_steps), 1):
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            frac = (step - 1) / max(1, schedule_total)
            cur_len = args.seq_length
            input_ids = input_ids[:, :cur_len].contiguous()
            attention_mask = attention_mask[:, :cur_len].contiguous()
            word_group = word_group[:, :cur_len].contiguous()

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

            # Match the inherited accumulated trainer's optimization contract:
            # gradient clipping at norm 1.0 before AdamW update.  Clipping the
            # deduplicated parameter list includes both standard model parameters
            # and SGCR component/projection parameters without double counting tied
            # embeddings.
            torch.nn.utils.clip_grad_norm_(unique_params, 1.0)
            optim.step()
            sched.step()

            cumulative_words += words
            loss_values.append(weighted_loss_sum)
            effective_mask_rate = n_pred_total / max(1, int(attention_mask.bool().sum().item()))

            log_entry = {
                "step": step, "loss": weighted_loss_sum,
                "lr": sched.get_last_lr()[0],
                "batch_words": words, "cumulative_word_exposure": cumulative_words,
                "seq_len": cur_len, "masked_tokens": n_pred_total,
                "effective_mask_rate": round(effective_mask_rate, 4),
                "mask_mode": curriculum_state.get_current_mask_mode(),
                "mask_prob_nominal": round(curriculum_state.get_current_mask_prob(), 4),
                "active_microbatches": active_microbatches,
            }

            if step % args.log_every == 0 or step == 1 or step == total_steps:
                logf.write(json.dumps(log_entry) + "\n")
                logf.flush()

            # Checkpoint
            while next_ckpt is not None and cumulative_words >= next_ckpt:
                if next_ckpt < 1_000_000:
                    ckpt_name = "chck_1M"
                elif next_ckpt % 1_000_000 == 0:
                    ckpt_name = f"chck_{next_ckpt // 1_000_000}M"
                else:
                    ckpt_name = f"chck_{next_ckpt}w"
                ckpt_dir = out / "hf_model" / ckpt_name
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # Save baked checkpoint (standard HF model with effective embeddings)
                bake_info = sgcr_mod.save_baked_checkpoint(
                    model, sgcr_emb, tokenizer, ckpt_dir, also_save_sgcr=True
                )
                saved_checkpoints.append({
                    "name": ckpt_name,
                    "target_word_exposure": next_ckpt,
                    "actual_cumulative_word_exposure": cumulative_words,
                    "words": cumulative_words,
                    "step": step,
                    "path": str(ckpt_dir),
                    "bake_info": bake_info,
                })
                next_ckpt += args.checkpoint_words
                if next_ckpt > args.max_word_exposure:
                    next_ckpt = None

    # Save final model (baked for evaluation compatibility)
    final_dir = out / "hf_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    bake_info = sgcr_mod.save_baked_checkpoint(
        model, sgcr_emb, tokenizer, final_dir, also_save_sgcr=True
    )

    # Save metrics
    elapsed = time.time() - start_time
    metrics = {
        "event": "sgcr_training_complete",
        "created_utc": now_utc(),
        "word_exposure": cumulative_words,
        "total_steps": step,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "checkpoints_written": len(saved_checkpoints),
        "saved_checkpoints": saved_checkpoints,
        "elapsed_sec": round(elapsed, 1),
        "gradient_clip_norm": 1.0,
        "train_rng_restored_after_sgcr": bool(args.train_rng_seed >= 0),
        "warmup_fraction": args.warmup_fraction,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "masking_curriculum": args.masking_curriculum,
        "mask_prob_start": args.mask_prob_start,
        "mask_prob_end": args.mask_prob_end,
        **sgcr_manifest,
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    # Final diagnostic
    diag = sgcr_mod.sgcr_diagnostic(sgcr_emb)
    (out / "sgcr_diagnostic.json").write_text(
        json.dumps(diag, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
