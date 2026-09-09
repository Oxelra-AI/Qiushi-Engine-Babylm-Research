#!/usr/bin/env python3
"""research: Context-credit replay trainer for BabyLM Strict-Small.

Translates the synthetic interface-credit mechanism into real pretraining:
instead of equal WWM loss weight on all masked tokens, weight by carrier error
(1 - p_carrier(correct_token)), directing the private adapter's scarce credit
toward tokens the frozen carrier cannot predict from its existing representations.

Architecture (unchanged):
  h_next = slow_output + private_adapter(slow_output)
  where slow_output = base_output + adapter(base_output)  [frozen chck_82M]

Credit modes:
  standard          - all masked tokens weighted equally (research reproduction)
  carrier_residual  - weight_i = (1 - p_carrier(correct_i)), normalized to unit mean

Both modes share: same frozen chck_82M carrier, same legal data tail (82M→86M),
same optimizer/schedule/seed, same neutral KL, inference at alpha=0.75.

Scientific hypothesis: the private adapter should learn more transferable competence
by focusing credit on tokens the carrier cannot already predict, rather than
redundantly encoding carrier-predictable patterns.  This parallels the synthetic
finding that relation-aligned credit preserves broader interface reach while
bag-independent credit narrows it.
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
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

ROOT = _public_path('.')
A01 = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(A02_SCRIPTS))

from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402
import frozen82_fastpath_replay_trainer as base  # noqa: E402

# ── inherited constants ────────────────────────────────────────────
ENDPOINT = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')

INITIAL_CONSUMED_WORDS = 82_012_495
FULL_CAP_WORDS = 100_000_000
MAX_TAIL_CHARGED_WORDS = 3_992_918
SKIP_ROWS = 530_944
MACRO_BATCH = 256
MICRO_BATCH = 32
ACCUM_STEPS = MACRO_BATCH // MICRO_BATCH  # 8
SEQ_LENGTH = 256
LR = 0.001
WARMUP_FRAC = 0.06
SCHEDULE_TOTAL = 455
WEIGHT_DECAY = 0.01
MASK_PROB = 0.15
CHECKPOINT_WORDS = 1_000_000
LOG_EVERY = 25
PRIVATE_BOTTLENECK = 128
PRIVATE_SCALE = 1.0  # training scale; alpha=0.75 applied at inference
MAIN_LAMBDA = 1.0
NEUTRAL_LAMBDA = 1.0
NEUTRAL_SUBSAMPLE = 32
TRAIN_SEED = 43023


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def reset_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at_update(u, total, warmup, peak):
    if u < warmup:
        return peak * u / max(1, warmup)
    p = (u - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * p))


def load_model(endpoint, device, private_bottleneck=128, private_scale=1.0):
    """Load frozen-slow + fresh-private model from chck_82M."""
    from transformers import DebertaV2Config
    cfg = DebertaV2Config.from_pretrained(str(endpoint), local_files_only=True)
    cfg.private_adapter_bottleneck = private_bottleneck
    cfg.private_adapter_scale = private_scale
    cfg.private_adapter_enabled = True
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd = load_file(str(Path(endpoint) / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tied_missing = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    bad_missing = [x for x in missing if "private_adapter" not in x and x not in tied_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"load mismatch: bad_missing={bad_missing[:10]} unexpected={unexpected[:10]}")
    model.tie_weights()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    return model


def compute_carrier_weights(model, masked_inputs, attention_mask, labels, device):
    """Compute per-token loss weight = 1 - p_carrier(correct_token).

    Uses the frozen carrier (private disabled) to identify what it already knows.
    Returns weights tensor [B, S] with 0 for non-target positions and
    normalized (1 - p_correct) for masked positions.
    """
    model.set_private_enabled(False)
    with torch.no_grad():
        carrier_logits = model(input_ids=masked_inputs, attention_mask=attention_mask).logits
    model.set_private_enabled(True)

    # p_carrier for each position's correct token
    carrier_probs = F.softmax(carrier_logits, dim=-1)  # [B, S, V]
    B, S = labels.shape
    target_mask = labels != -100  # [B, S]

    # Gather carrier probability of correct token at masked positions
    safe_labels = labels.clone()
    safe_labels[~target_mask] = 0
    p_correct = carrier_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, S]

    # Weight = 1 - p_correct for masked positions, 0 elsewhere
    weights = (1.0 - p_correct) * target_mask.float()

    # Normalize so mean weight over masked positions = 1.0 (preserves loss magnitude)
    n_targets = target_mask.sum()
    if n_targets > 0:
        mean_w = weights.sum() / n_targets
        if mean_w > 1e-8:
            weights = weights / mean_w

    del carrier_logits, carrier_probs
    return weights


def compute_neutral_kl(model, masked_inputs, attention_mask, n_sub=32):
    """Neutral KL: private-ON should stay close to carrier (private-OFF)."""
    n = min(n_sub, masked_inputs.shape[0])
    if n <= 0:
        return None, 0.0

    ids = masked_inputs[:n]
    att = attention_mask[:n]

    was_training = model.training
    model.eval()
    model.set_private_enabled(False)
    with torch.no_grad():
        slow_logits = model(input_ids=ids, attention_mask=att).logits.detach()
    model.set_private_enabled(True)
    out = model(input_ids=ids, attention_mask=att)
    log_p = F.log_softmax(out.logits, dim=-1)
    p_slow = F.softmax(slow_logits, dim=-1)
    kl = F.kl_div(log_p, p_slow, reduction="none").sum(-1)
    mask_f = att.float()
    loss = (kl * mask_f).sum() / max(1.0, float(mask_f.sum()))
    if was_training:
        model.train()
    return loss, float(loss.detach().cpu())


def save_checkpoint(model, tokenizer, dst, alpha=None):
    """Save model checkpoint.  If alpha is given, set private_adapter_scale."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if alpha is not None:
        model.config.private_adapter_scale = alpha
        for layer in model.deberta.encoder.layer:
            layer.private_adapter.scale = alpha
    model.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    # Copy modeling file for trust_remote_code loading
    src = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_private_modeling.py')
    d = dst / src.name
    if src.exists() and not d.exists():
        shutil.copy2(str(src), str(d))


def train(args):
    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_cache = out / "hf_cache"
    (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_MODULES_CACHE"] = str(hf_cache / "modules")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    credit_mode = args.credit_mode

    print(json.dumps({
        "event": "start",
        "trainer": "CONTEXT_CREDIT_REPLAY",
        "credit_mode": credit_mode,
        "device": str(device),
        "gpu": args.gpu,
        "hypothesis": (
            "carrier_residual directs private adapter credit toward tokens the "
            "frozen carrier cannot predict, paralleling synthetic relation-aligned "
            "credit that preserves interface reach"
        ) if credit_mode == "carrier_residual" else "standard control",
    }), flush=True)

    # ── data ────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ENDPOINT), local_files_only=True)
    examples = base.load_examples_tail(STREAM, SKIP_ROWS, MAX_TAIL_CHARGED_WORDS)
    main_words = sum(e.words for e in examples)
    print(json.dumps({
        "event": "data_loaded",
        "tail_examples": len(examples),
        "main_words": main_words,
        "first_row": examples[0].row_index if examples else None,
        "last_row": examples[-1].row_index if examples else None,
    }), flush=True)

    dataset = base.TailDataset(examples, tokenizer, SEQ_LENGTH)
    loader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=False,
                        collate_fn=base.collate, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    # ── model ───────────────────────────────────────
    reset_all(TRAIN_SEED)
    model = load_model(ENDPOINT, device, PRIVATE_BOTTLENECK, PRIVATE_SCALE)

    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)
    private_up_ids = {id(p) for n, p in model.named_parameters() if ".private_adapter.up." in n}
    private_normal = [p for n, p in model.named_parameters() if n in private_names and id(p) not in private_up_ids]
    private_zero_wd = [p for n, p in model.named_parameters() if n in private_names and id(p) in private_up_ids]
    optimizer = torch.optim.AdamW([
        {"params": private_normal, "weight_decay": WEIGHT_DECAY},
        {"params": private_zero_wd, "weight_decay": 0.0},
    ], lr=LR, betas=(0.9, 0.98), eps=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    private_params = sum(p.numel() for n, p in model.named_parameters() if n in private_names)
    print(json.dumps({
        "event": "model_loaded",
        "total_params": total_params,
        "private_params": private_params,
        "frozen_params": total_params - private_params,
    }), flush=True)

    warmup = max(1, int(SCHEDULE_TOTAL * WARMUP_FRAC))
    gen = torch.Generator(device=device)
    gen.manual_seed(TRAIN_SEED)

    # ── training config ─────────────────────────────
    config = {
        "status": "CONTEXT_CREDIT_CONFIG",
        "credit_mode": credit_mode,
        "endpoint": rel(ENDPOINT),
        "data_stream": rel(STREAM),
        "initial_consumed_words": INITIAL_CONSUMED_WORDS,
        "max_tail_charged_words": MAX_TAIL_CHARGED_WORDS,
        "skip_rows": SKIP_ROWS,
        "schedule_total": SCHEDULE_TOTAL,
        "warmup": warmup,
        "macro_batch_size": MACRO_BATCH,
        "micro_batch_size": MICRO_BATCH,
        "accum_steps": ACCUM_STEPS,
        "seq_length": SEQ_LENGTH,
        "lr": LR,
        "mask_prob": MASK_PROB,
        "main_lambda": MAIN_LAMBDA,
        "neutral_lambda": NEUTRAL_LAMBDA,
        "private_bottleneck": PRIVATE_BOTTLENECK,
        "private_training_scale": PRIVATE_SCALE,
        "inference_alpha": 0.75,
        "total_params": total_params,
        "private_params": private_params,
        "seed": TRAIN_SEED,
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2))

    # ── training loop ───────────────────────────────
    losses_main = []
    losses_neutral = []
    cum_main = 0
    updates = 0
    micro_in_macro = 0  # micro-batch counter within macro-batch
    accum_main_loss = 0.0
    accum_neutral_loss = 0.0
    accum_targets = 0
    accum_words = 0
    accum_weight_info = {"sum_raw_weight": 0.0, "n_raw": 0, "min_w": 1e9,
                         "max_w": -1e9, "sum_easy": 0, "n_easy": 0}
    next_ckpt_tail = CHECKPOINT_WORDS
    log_path = out / "training_log.jsonl"

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for loader_step, batch in enumerate(loader, 1):
            words = int(batch["words"].sum().item())
            tail_next = cum_main + words
            total_next = INITIAL_CONSUMED_WORDS + tail_next
            if tail_next > MAX_TAIL_CHARGED_WORDS or total_next > FULL_CAP_WORDS:
                print(json.dumps({"event": "stop_at_cap", "tail_words": cum_main,
                                  "total": INITIAL_CONSUMED_WORDS + cum_main}), flush=True)
                break

            # Start new macro-batch: set LR and zero grads
            if micro_in_macro == 0:
                lr = lr_at_update(updates, SCHEDULE_TOTAL, warmup, LR)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                accum_main_loss = 0.0
                accum_neutral_loss = 0.0
                accum_targets = 0
                accum_words = 0
                accum_weight_info = {"sum_raw_weight": 0.0, "n_raw": 0, "min_w": 1e9,
                                     "max_w": -1e9, "sum_easy": 0, "n_easy": 0}

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            masked_inputs, labels, _select, _selected_groups = base.apply_wwm(
                input_ids, attention_mask, word_group, tokenizer, MASK_PROB, gen)
            n_targets = int((labels != -100).sum().item())

            # ── carrier-residual credit weighting ───
            if credit_mode == "carrier_residual" and n_targets > 0:
                token_weights = compute_carrier_weights(
                    model, masked_inputs, attention_mask, labels, device)
                mask_w = token_weights[labels != -100]
                accum_weight_info["sum_raw_weight"] += float(mask_w.sum().item())
                accum_weight_info["n_raw"] += int(mask_w.numel())
                accum_weight_info["min_w"] = min(accum_weight_info["min_w"], float(mask_w.min().item()))
                accum_weight_info["max_w"] = max(accum_weight_info["max_w"], float(mask_w.max().item()))
                accum_weight_info["sum_easy"] += int((mask_w < 0.5).sum().item())
                accum_weight_info["n_easy"] += int(mask_w.numel())
            else:
                token_weights = None

            # ── main MLM loss (scaled by 1/ACCUM_STEPS for gradient accumulation) ──
            model.train()
            model.set_private_enabled(True)
            out_main = model(input_ids=masked_inputs, attention_mask=attention_mask)
            vocab = out_main.logits.shape[-1]

            if token_weights is not None:
                ce_per_token = F.cross_entropy(
                    out_main.logits.reshape(-1, vocab), labels.reshape(-1),
                    ignore_index=-100, reduction="none")
                flat_weights = token_weights.reshape(-1)
                micro_main = (ce_per_token * flat_weights).sum() / max(1, n_targets)
                del ce_per_token, flat_weights
            else:
                micro_main_sum = F.cross_entropy(
                    out_main.logits.reshape(-1, vocab), labels.reshape(-1),
                    ignore_index=-100, reduction="sum")
                micro_main = micro_main_sum / max(1, n_targets)
                del micro_main_sum

            micro_main_val = float(micro_main.detach().cpu())
            (MAIN_LAMBDA * micro_main / ACCUM_STEPS).backward()
            del out_main, micro_main

            # ── neutral KL (for subsample of this micro-batch) ──
            micro_neutral_val = 0.0
            neutral_sub = max(1, NEUTRAL_SUBSAMPLE // ACCUM_STEPS)
            if NEUTRAL_LAMBDA > 0:
                nl, micro_neutral_val = compute_neutral_kl(
                    model, masked_inputs, attention_mask, neutral_sub)
                if nl is not None:
                    (NEUTRAL_LAMBDA * nl / ACCUM_STEPS).backward()
                    del nl

            accum_main_loss += micro_main_val
            accum_neutral_loss += micro_neutral_val
            accum_targets += n_targets
            accum_words += words
            cum_main += words
            micro_in_macro += 1

            del input_ids, attention_mask, word_group, masked_inputs, labels
            if token_weights is not None:
                del token_weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── optimizer step when macro-batch is complete ──
            if micro_in_macro >= ACCUM_STEPS:
                all_private = private_normal + private_zero_wd
                torch.nn.utils.clip_grad_norm_(all_private, 1.0)
                optimizer.step()
                updates += 1
                micro_in_macro = 0

                avg_main = accum_main_loss / ACCUM_STEPS
                avg_neutral = accum_neutral_loss / ACCUM_STEPS
                losses_main.append(avg_main)
                losses_neutral.append(avg_neutral)

                tail_charged = cum_main
                total_consumed = INITIAL_CONSUMED_WORDS + tail_charged

                rec = {
                    "update": updates,
                    "lr": round(lr, 8),
                    "macro_words": accum_words,
                    "tail_words": cum_main,
                    "total_consumed": total_consumed,
                    "n_targets": accum_targets,
                    "main_loss": round(avg_main, 6),
                    "neutral_loss": round(avg_neutral, 6),
                    "private_rms_max": round(max(model.private_adapter_rms()), 6) if hasattr(model, "private_adapter_rms") else None,
                    "elapsed_sec": round(time.time() - t0, 1),
                }
                if credit_mode == "carrier_residual" and accum_weight_info["n_raw"] > 0:
                    rec["mean_raw_weight"] = round(accum_weight_info["sum_raw_weight"] / accum_weight_info["n_raw"], 4)
                    rec["min_weight"] = round(accum_weight_info["min_w"], 4)
                    rec["max_weight"] = round(accum_weight_info["max_w"], 4)
                    rec["carrier_easy_frac"] = round(accum_weight_info["sum_easy"] / max(1, accum_weight_info["n_easy"]), 4)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if updates == 1 or updates % LOG_EVERY == 0:
                    print(json.dumps({"event": "train", **rec}), flush=True)

                # ── checkpoints ─────────────────────────
                while next_ckpt_tail is not None and tail_charged >= next_ckpt_tail and next_ckpt_tail <= MAX_TAIL_CHARGED_WORDS:
                    approx_total = INITIAL_CONSUMED_WORDS + next_ckpt_tail
                    name = f"chck_{approx_total // 1_000_000}M"
                    save_checkpoint(model, tokenizer, out / "hf_model_scale1p0" / name, alpha=PRIVATE_SCALE)
                    save_checkpoint(model, tokenizer, out / "hf_model_alpha0p75" / name, alpha=0.75)
                    model.config.private_adapter_scale = PRIVATE_SCALE
                    for layer in model.deberta.encoder.layer:
                        layer.private_adapter.scale = PRIVATE_SCALE
                    print(json.dumps({"event": "checkpoint", "name": name,
                                      "tail_words": tail_charged,
                                      "total_consumed": total_consumed}), flush=True)
                    next_ckpt_tail += CHECKPOINT_WORDS

    # ── save final ──────────────────────────────────
    model.set_private_enabled(True)
    save_checkpoint(model, tokenizer, out / "hf_model_scale1p0" / "final", alpha=PRIVATE_SCALE)
    save_checkpoint(model, tokenizer, out / "hf_model_alpha0p75" / "final", alpha=0.75)
    # Restore training scale for RMS report
    model.config.private_adapter_scale = PRIVATE_SCALE
    for layer in model.deberta.encoder.layer:
        layer.private_adapter.scale = PRIVATE_SCALE

    elapsed = round(time.time() - t0, 1)
    metrics = {
        "status": "CONTEXT_CREDIT_COMPLETE",
        "credit_mode": credit_mode,
        "updates": updates,
        "tail_main_words": cum_main,
        "total_consumed_words": INITIAL_CONSUMED_WORDS + cum_main,
        "final_main_loss": round(sum(losses_main[-10:]) / max(1, len(losses_main[-10:])), 6) if losses_main else None,
        "final_neutral_loss": round(sum(losses_neutral[-10:]) / max(1, len(losses_neutral[-10:])), 6) if losses_neutral else None,
        "final_private_rms": model.private_adapter_rms() if hasattr(model, "private_adapter_rms") else None,
        "elapsed_sec": elapsed,
        "out_dir": rel(out),
        "alpha075_final": rel(out / "hf_model_alpha0p75" / "final"),
        "scale1p0_final": rel(out / "hf_model_scale1p0" / "final"),
    }
    (out / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credit_mode", default="standard",
                        choices=["standard", "carrier_residual"],
                        help="standard=equal weight (control), carrier_residual=1-p_carrier")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="2-step smoke test")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(_public_path('experiments/archive/functional_learning/training/runs') / f"step025_{args.credit_mode}_seed{TRAIN_SEED}")

    if args.smoke:
        global MAX_TAIL_CHARGED_WORDS, SCHEDULE_TOTAL, LOG_EVERY, CHECKPOINT_WORDS, ACCUM_STEPS, MICRO_BATCH
        MAX_TAIL_CHARGED_WORDS = 60_000
        SCHEDULE_TOTAL = 4
        LOG_EVERY = 1
        CHECKPOINT_WORDS = 30_000
        ACCUM_STEPS = 2
        MICRO_BATCH = 16

    train(args)


if __name__ == "__main__":
    main()
