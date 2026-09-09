#!/usr/bin/env python3
"""research: MLM-primary same-corruption MNTP auxiliary trainer.

Scientific intervention
-----------------------
Keeps the FULL clean-Qwen WWM-MLM stream intact on every batch.
Adds a same-corruption token_shift MNTP auxiliary loss whose coefficient λ_t
is dynamically calibrated so ||λ_t * ∇aux|| = target_ratio * ||∇MLM||.

This differs from research (whole-batch causal substitution) because:
  1. Every word's MLM reconstruction signal is preserved.
  2. The auxiliary operates on the SAME masked input (no uncorrupted causal batches).
  3. The coefficient derives from running gradient-norm statistics only.

The MNTP auxiliary computes: at position j-1, predict original token x_j, where
position j was selected by WWM masking. The model sees bidirectional attention on
the masked input — position j-1 sees [MASK] at j and surrounding context.

Architecture, tokenizer, initialization, corpus order, word exposure, optimizer,
LR schedule, batch size, and checkpoint policy remain matched to clean-Qwen.
No eval/AoA/CDI information is read or used at any point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

import masking_curriculum_trainer as base


# ─── MNTP label construction (from research probe, verified) ───────────────────

def make_mntp_labels_token_shift(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mlm_labels: torch.Tensor,
) -> torch.Tensor:
    """At position j-1, predict original token x_j where j was masked by MLM.
    
    Returns a label tensor where non-(-100) entries are at positions j-1 for each
    masked position j. Uses only the existing MLM corruption — no new randomness.
    """
    labels = torch.full_like(input_ids, -100)
    target_selected = mlm_labels != -100  # positions that MLM selected
    # Valid prediction pairs: both j-1 and j must be within attention
    valid_pair = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    # Select position j (shifted to j-1 for prediction) where j was masked
    selected_next = target_selected[:, 1:] & valid_pair
    # Place original token x_j at predictor position j-1
    labels[:, :-1] = torch.where(selected_next, input_ids[:, 1:], labels[:, :-1])
    return labels


def loss_from_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss ignoring -100 positions."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


# ─── Gradient norm utility ────────────────────────────────────────────────────

def grad_norm_sq(model: torch.nn.Module) -> float:
    """Sum of squared gradient norms across all parameters with grad."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().norm().item() ** 2)
    return total


# ─── File hash ────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Arguments ────────────────────────────────────────────────────────────────

def build_args():
    p = argparse.ArgumentParser(description="MLM-primary same-corruption MNTP auxiliary trainer")
    # Data — matched to clean-Qwen
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--example_jsonl_meta", required=True)
    p.add_argument("--example_jsonl_label", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tokenizer_path", required=True)
    # Exposure and checkpointing
    p.add_argument("--max_word_exposure", type=int, default=100_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    # Architecture — matched to clean-Qwen DeBERTa-v2 8×480
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    # Optimizer — matched to clean-Qwen
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--mask_prob", type=float, default=0.15)
    # Seeds — matched to clean-Qwen
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    # MNTP auxiliary calibration — the new intervention
    p.add_argument("--aux_target_ratio", type=float, default=0.15,
                   help="Target: ||lambda*grad_aux|| / ||grad_mlm|| = this value")
    p.add_argument("--aux_ema_beta", type=float, default=0.9,
                   help="EMA smoothing for gradient norm tracking (0.9 = 10-step half-life)")
    p.add_argument("--aux_lambda_max", type=float, default=1.0,
                   help="Hard cap on auxiliary coefficient to prevent instability")
    # Misc
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=0, help="Smoke-test cap; 0 = full run")
    p.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# ─── Main training loop ──────────────────────────────────────────────────────

def main():
    args = build_args()
    start = time.time()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output_dir must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    # ── Data loading (matched to clean-Qwen) ──
    data_path = Path(args.example_jsonl)
    meta_path = Path(args.example_jsonl_meta)
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    examples, file_words, file_rows, sample_rows = base.load_examples_jsonl(
        data_path, args.max_word_exposure
    )
    actual_words = sum(x.words for x in examples)
    if actual_words != args.max_word_exposure:
        raise RuntimeError(f"word exposure mismatch {actual_words} != {args.max_word_exposure}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    dataset = base.MaskedChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=base.collate, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())
    planned_steps = len(loader)
    effective_steps = min(planned_steps, args.max_steps) if args.max_steps > 0 else planned_steps

    # ── Model init (matched seed sequence to clean-Qwen) ──
    def reset_rng(s: int):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

    reset_rng(args.extra_init_seed)
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer), hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer, num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=max(512, args.seq_length + 8),
        max_relative_positions=256, position_buckets=256,
        relative_attention=True, pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = DebertaV2ForMaskedLM(cfg)
    init_probe = {
        "word_embedding_sha256": hashlib.sha256(
            model.deberta.embeddings.word_embeddings.weight.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    reset_rng(args.train_rng_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ── Optimizer (matched to clean-Qwen) ──
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, betas=(0.9, 0.98))
    warmup = max(1, int(planned_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, warmup, planned_steps)
    mask_gen = torch.Generator(device=device)
    mask_gen.manual_seed(args.train_rng_seed)
    curriculum = base.MaskingCurriculumState(
        curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob
    )
    curriculum.initialize(len(tokenizer), planned_steps)

    # ── Auxiliary calibration state ──
    # Initialize EMA with probe-informed estimates:
    # From research probe at 100M: aux/mlm norm ratio ≈ 3.0
    # So initial lambda = target_ratio / estimated_ratio = target_ratio / 3.0
    initial_ratio_estimate = 3.0
    ema_mlm_norm = 1.0  # Arbitrary scale; ratio is what matters
    ema_aux_norm = initial_ratio_estimate  # So initial lambda = target/3
    lambda_t = args.aux_target_ratio / initial_ratio_estimate
    beta = args.aux_ema_beta

    # ── Run manifest ──
    data_manifest = {
        "status": "MLM_MNTP_AUXILIARY_MANIFEST",
        "scientific_intervention": "MLM-primary same-corruption token_shift MNTP auxiliary with norm-calibrated coefficient",
        "data": str(data_path), "data_sha256": sha256_file(data_path),
        "metadata": str(meta_path), "metadata_sha256": sha256_file(meta_path),
        "example_jsonl_label": args.example_jsonl_label,
        "file_words": file_words, "file_rows": file_rows,
        "selected_word_exposure": actual_words, "planned_steps": planned_steps,
        "effective_steps": effective_steps, "sample_rows": sample_rows,
        "aux_target_ratio": args.aux_target_ratio,
        "aux_ema_beta": args.aux_ema_beta,
        "aux_lambda_max": args.aux_lambda_max,
        "initial_ratio_estimate": initial_ratio_estimate,
        "initial_lambda": lambda_t,
        "objective_policy": "every_batch_mlm_plus_norm_calibrated_mntp_auxiliary",
        "mntp_variant": "token_shift",
        "mntp_definition": "At position j-1, predict original token x_j where j was masked; same corruption, bidirectional attention",
        "mlm_definition": "15pct whole-word masking, bidirectional attention, full MLM stream retained",
        "word_count_policy": "each consumed row word counted once; auxiliary adds no extra exposure",
        "gradient_norm_policy": "two-backward every step; EMA of individual norms; lambda = target_ratio * ema_mlm / ema_aux",
        "corpus_metadata_status": meta.get("status"),
        **init_probe,
    }
    (out / "run_manifest.json").write_text(json.dumps(data_manifest, indent=2) + "\n")

    # ── Training loop ──
    cumulative_words = 0
    mlm_losses = []
    aux_losses = []
    lambda_history = []
    saved = []
    next_ckpt = args.checkpoint_words
    model.train()

    # Preallocate buffer for storing MLM gradients (saves per-step allocation)
    param_list = [p for p in model.parameters() if p.requires_grad]
    mlm_grad_buffer = [torch.zeros_like(p) for p in param_list]

    with (out / "training_log.jsonl").open("w") as logf:
        for step, batch in enumerate(loader, 1):
            if step > effective_steps:
                break
            words = int(batch.pop("words").sum())
            ids = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)
            wg = batch["word_group"].to(device, non_blocking=True)

            # ── Apply WWM masking (exact same as clean-Qwen) ──
            curriculum.current_step = step - 1
            masked, mlm_labels = base.apply_masking_curriculum(
                ids, am, wg, tokenizer, curriculum, mask_gen
            )

            # ── Forward pass (single, shared for both losses) ──
            optim.zero_grad(set_to_none=True)
            logits = model(input_ids=masked, attention_mask=am).logits

            # ── MLM loss ──
            mlm_loss = loss_from_labels(logits, mlm_labels)
            n_mlm_targets = int((mlm_labels != -100).sum().item())

            # ── MNTP auxiliary loss (same corruption, token_shift) ──
            aux_labels = make_mntp_labels_token_shift(ids, am, mlm_labels)
            n_aux_targets = int((aux_labels != -100).sum().item())

            if n_aux_targets > 0:
                aux_loss = loss_from_labels(logits, aux_labels)

                # ── Two-backward for exact gradient norm measurement ──
                # Phase 1: MLM gradient
                mlm_loss.backward(retain_graph=True)
                mlm_norm = math.sqrt(grad_norm_sq(model))

                # Store MLM grads into buffer
                for buf, p in zip(mlm_grad_buffer, param_list):
                    if p.grad is not None:
                        buf.copy_(p.grad)
                    else:
                        buf.zero_()

                # Phase 2: Aux gradient (graph freed after this)
                optim.zero_grad(set_to_none=True)
                aux_loss.backward()
                aux_norm = math.sqrt(grad_norm_sq(model))

                # ── Update EMA and compute λ_t ──
                if step == 1:
                    # First step: use raw measurements (no EMA bias)
                    ema_mlm_norm = mlm_norm
                    ema_aux_norm = aux_norm
                else:
                    ema_mlm_norm = beta * ema_mlm_norm + (1.0 - beta) * mlm_norm
                    ema_aux_norm = beta * ema_aux_norm + (1.0 - beta) * aux_norm

                lambda_t = args.aux_target_ratio * ema_mlm_norm / max(ema_aux_norm, 1e-8)
                lambda_t = min(lambda_t, args.aux_lambda_max)

                # ── Combine: final grad = mlm_grad + λ_t * aux_grad ──
                # Currently p.grad holds aux_grad; multiply by lambda and add mlm
                for buf, p in zip(mlm_grad_buffer, param_list):
                    if p.grad is not None:
                        p.grad.mul_(lambda_t).add_(buf)
                    else:
                        p.grad = buf.clone()

                # ── Clip and step ──
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()

                mlm_losses.append(float(mlm_loss.detach().cpu()))
                aux_losses.append(float(aux_loss.detach().cpu()))
                lambda_history.append(lambda_t)

                rec = {
                    "step": step, "mlm_loss": mlm_losses[-1], "aux_loss": aux_losses[-1],
                    "lambda_t": lambda_t, "mlm_norm": mlm_norm, "aux_norm": aux_norm,
                    "ema_mlm_norm": ema_mlm_norm, "ema_aux_norm": ema_aux_norm,
                    "effective_aux_ratio": (lambda_t * aux_norm / max(mlm_norm, 1e-8)),
                    "mlm_targets": n_mlm_targets, "aux_targets": n_aux_targets,
                    "batch_words": words, "cumulative_word_exposure": cumulative_words + words,
                    "lr": float(sched.get_last_lr()[0]),
                    "elapsed_sec": round(time.time() - start, 1),
                }
            else:
                # Rare: no valid aux targets (e.g., all masked at position 0)
                mlm_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()

                mlm_losses.append(float(mlm_loss.detach().cpu()))
                aux_losses.append(0.0)
                lambda_history.append(lambda_t)

                rec = {
                    "step": step, "mlm_loss": mlm_losses[-1], "aux_loss": 0.0,
                    "lambda_t": lambda_t, "mlm_norm": 0.0, "aux_norm": 0.0,
                    "ema_mlm_norm": ema_mlm_norm, "ema_aux_norm": ema_aux_norm,
                    "effective_aux_ratio": 0.0,
                    "mlm_targets": n_mlm_targets, "aux_targets": 0,
                    "batch_words": words, "cumulative_word_exposure": cumulative_words + words,
                    "lr": float(sched.get_last_lr()[0]),
                    "elapsed_sec": round(time.time() - start, 1),
                }

            cumulative_words += words
            rec["cumulative_word_exposure"] = cumulative_words
            logf.write(json.dumps(rec) + "\n")

            if not torch.isfinite(torch.tensor(mlm_losses[-1])):
                raise RuntimeError(f"nonfinite MLM loss at step {step}: {mlm_losses[-1]}")

            if step == 1 or step % args.log_every == 0 or step == effective_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)

            # ── Checkpoint saving ──
            if args.save_checkpoints and args.max_steps == 0:
                while cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                    name = f"chck_{next_ckpt // 1_000_000}M"
                    cp = out / "hf_model" / name
                    base.save_hf_checkpoint(model, tokenizer, cp)
                    saved.append({"name": name, "target_word_exposure": next_ckpt,
                                  "actual_cumulative_word_exposure": cumulative_words})
                    print(json.dumps({"event": "checkpoint_saved", "name": name,
                                      "cum_words": cumulative_words, "lambda_t": lambda_t}),
                          flush=True)
                    next_ckpt += args.checkpoint_words

    # ── Final model save ──
    if args.max_steps == 0:
        base.save_hf_checkpoint(model, tokenizer, out / "hf_model")

    # ── Scientific metrics ──
    summary = {
        "status": "MLM_MNTP_AUXILIARY_TRAINING_COMPLETE",
        **data_manifest,
        "actual_word_exposure": cumulative_words,
        "actual_steps": step if step <= effective_steps else effective_steps,
        "mlm_loss_first": mlm_losses[0] if mlm_losses else None,
        "mlm_loss_last": mlm_losses[-1] if mlm_losses else None,
        "aux_loss_first": aux_losses[0] if aux_losses else None,
        "aux_loss_last": aux_losses[-1] if aux_losses else None,
        "lambda_first": lambda_history[0] if lambda_history else None,
        "lambda_last": lambda_history[-1] if lambda_history else None,
        "lambda_mean": float(np.mean(lambda_history)) if lambda_history else None,
        "lambda_std": float(np.std(lambda_history)) if lambda_history else None,
        "ema_mlm_norm_final": ema_mlm_norm,
        "ema_aux_norm_final": ema_aux_norm,
        "final_aux_mlm_ratio": ema_aux_norm / max(ema_mlm_norm, 1e-8),
        "saved_checkpoints": saved,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "event": "done",
        "actual_word_exposure": cumulative_words,
        "actual_steps": summary["actual_steps"],
        "mlm_loss_last": summary["mlm_loss_last"],
        "aux_loss_last": summary["aux_loss_last"],
        "lambda_last": summary["lambda_last"],
        "lambda_mean": summary["lambda_mean"],
        "final_aux_mlm_ratio": summary["final_aux_mlm_ratio"],
        "saved_checkpoints_count": len(saved),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
