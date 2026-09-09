#!/usr/bin/env python3
"""research: continue from the exact coherent86 alpha0.75 BabyLM endpoint.

Motivation
----------
The research/026 82M->86M replay pair showed that raw carrier-error credit is not a
trustworthy improvement when the replay trajectory itself fails to recover the inherited
coherent86 gain.  This script instead starts from the exact scored coherent86 alpha0.75
checkpoint and spends the remaining legal Strict-Small word budget on the same compact-
view-reinvest stream.  It asks whether the already useful private correction can acquire
additional competence without reconstructing the lost 82M->86M trajectory.

Arms
----
standard          : ordinary WWM MLM continuation of the existing private adapters.
carrier_residual  : deterministic carrier-error weighting, using the exact coherent86
                    parent with private adapters temporarily disabled as the carrier.

Both arms default to the same parent, same remaining stream rows, same word accounting,
same trainable parameter set (private_adapter only), same scale during training
(alpha0.75), same cumulative cosine schedule continuation (offset after 101 previous
updates), and gradient accumulation only to reduce memory pressure.  Checkpoints are saved
at legal 1M-word tail increments so later screens can inspect whether the frontier peaks
before the final 100M endpoint.
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
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(A02_SCRIPTS))

from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402
import frozen82_fastpath_replay_trainer as base  # noqa: E402

DEFAULT_PARENT = _public_path('models/frontier')
DEFAULT_STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')

# Exact previous coherent86 accounting from research/153.
DEFAULT_INITIAL_WORDS = 86_005_295
DEFAULT_FULL_CAP_WORDS = 100_000_000
DEFAULT_REMAINING_WORDS = DEFAULT_FULL_CAP_WORDS - DEFAULT_INITIAL_WORDS  # 13,994,705
DEFAULT_SKIP_ROWS = 556_791  # next row after research final row 556790
DEFAULT_SCHEDULE_TOTAL = 455
DEFAULT_SCHEDULE_OFFSET = 101  # next cumulative update index after 101 research updates


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def reset_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at_update(update_index0: int, total: int, warmup: int, peak: float) -> float:
    if update_index0 < warmup:
        return peak * update_index0 / max(1, warmup)
    p = (update_index0 - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * p))


def load_model(endpoint: Path, device: torch.device, private_bottleneck: int, private_scale: float):
    from transformers import DebertaV2Config
    cfg = DebertaV2Config.from_pretrained(str(endpoint), local_files_only=True)
    cfg.private_adapter_bottleneck = int(private_bottleneck)
    cfg.private_adapter_scale = float(private_scale)
    cfg.private_adapter_enabled = True
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
    sd = load_file(str(endpoint / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tied_missing = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    bad_missing = [x for x in missing if "private_adapter" not in x and x not in tied_missing]
    if bad_missing or unexpected:
        raise RuntimeError(f"load mismatch: bad_missing={bad_missing[:10]} unexpected={unexpected[:10]}")
    model.tie_weights()
    model.register_for_auto_class("AutoModelForMaskedLM")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    if torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    return model, missing, unexpected


def private_flags(model) -> list[bool]:
    return [bool(layer.private_adapter.enabled) for layer in model.deberta.encoder.layer]


def restore_private_flags(model, flags: list[bool]) -> None:
    for layer, flag in zip(model.deberta.encoder.layer, flags):
        layer.private_adapter.enabled = bool(flag)
    model.config.private_adapter_enabled = bool(any(flags))


def compute_carrier_weights_deterministic(model, masked_inputs, attention_mask, labels) -> tuple[torch.Tensor, dict[str, float]]:
    """Weight = 1 - p_parent_private_off(correct), normalized over masked positions."""
    was_training = bool(model.training)
    flags = private_flags(model)
    target_mask = labels != -100
    devices: list[int] = []
    if masked_inputs.is_cuda:
        idx = masked_inputs.device.index
        if idx is None:
            idx = torch.cuda.current_device()
        devices = [idx]
    try:
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(990627)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(990627)
            model.eval()
            model.set_private_enabled(False)
            with torch.no_grad():
                carrier_logits = model(input_ids=masked_inputs, attention_mask=attention_mask).logits
                carrier_probs = F.softmax(carrier_logits, dim=-1)
                safe_labels = labels.clone()
                safe_labels[~target_mask] = 0
                p_correct = carrier_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
                raw = (1.0 - p_correct) * target_mask.float()
                n_targets = target_mask.sum()
                weights = raw
                mean_raw = torch.tensor(0.0, device=raw.device)
                if n_targets > 0:
                    mean_raw = raw.sum() / n_targets
                    if mean_raw > 1e-8:
                        weights = raw / mean_raw
                wv = weights[target_mask]
                pc = p_correct[target_mask]
                stats = {
                    "n": float(wv.numel()),
                    "mean_weight": float(wv.mean().detach().cpu()) if wv.numel() else 0.0,
                    "min_weight": float(wv.min().detach().cpu()) if wv.numel() else 0.0,
                    "max_weight": float(wv.max().detach().cpu()) if wv.numel() else 0.0,
                    "mean_raw_error": float(mean_raw.detach().cpu()) if n_targets > 0 else 0.0,
                    "carrier_easy_frac_p_gt_0p5": float((pc > 0.5).float().mean().detach().cpu()) if pc.numel() else 0.0,
                    "carrier_hard_frac_p_lt_0p1": float((pc < 0.1).float().mean().detach().cpu()) if pc.numel() else 0.0,
                }
                weights = weights.detach()
                del carrier_logits, carrier_probs, safe_labels, p_correct, raw, wv, pc
    finally:
        restore_private_flags(model, flags)
        if was_training:
            model.train()
        else:
            model.eval()
    return weights, stats


def compute_neutral_kl(model, masked_inputs, attention_mask, n_sub: int):
    n = min(int(n_sub), masked_inputs.shape[0])
    if n <= 0:
        return None, 0.0
    was_training = bool(model.training)
    ids = masked_inputs[:n]
    att = attention_mask[:n]
    model.eval()
    flags = private_flags(model)
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
    restore_private_flags(model, flags)
    if was_training:
        model.train()
    return loss, float(loss.detach().cpu())


def save_checkpoint(model, tokenizer, dst: Path, private_scale: float) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.config.private_adapter_scale = float(private_scale)
    for layer in model.deberta.encoder.layer:
        layer.private_adapter.scale = float(private_scale)
    model.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    src = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_private_modeling.py')
    d = dst / src.name
    if src.exists() and not d.exists():
        shutil.copy2(str(src), str(d))


def adapter_rms(model) -> list[float]:
    if hasattr(model, "private_adapter_rms"):
        return [float(x) for x in model.private_adapter_rms()]
    vals = []
    for layer in model.deberta.encoder.layer:
        vals.append(float(getattr(layer.private_adapter, "last_rms", 0.0)))
    return vals


def train(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hf_cache = out / "hf_cache"
    (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_MODULES_CACHE"] = str(hf_cache / "modules")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.macro_batch % args.micro_batch != 0:
        raise ValueError("macro_batch must be divisible by micro_batch")
    accum_steps = args.macro_batch // args.micro_batch
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    endpoint = Path(args.endpoint)
    stream = Path(args.example_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(str(endpoint), local_files_only=True)
    examples = base.load_examples_tail(stream, int(args.skip_rows), int(args.max_tail_charged_words))
    main_words = sum(e.words for e in examples)

    print(json.dumps({
        "event": "start",
        "trainer": "COHERENT86_CONTINUATION",
        "credit_mode": args.credit_mode,
        "device": str(device),
        "endpoint": rel(endpoint),
        "private_scale_train_and_eval": args.private_scale,
        "initial_consumed_words": args.initial_consumed_words,
        "full_cap_words": args.full_cap_words,
        "max_tail_charged_words": args.max_tail_charged_words,
        "skip_rows": args.skip_rows,
        "schedule_total": args.schedule_total,
        "schedule_offset": args.schedule_offset,
        "macro_batch": args.macro_batch,
        "micro_batch": args.micro_batch,
        "accum_steps": accum_steps,
    }), flush=True)
    print(json.dumps({
        "event": "data_loaded",
        "tail_examples": len(examples),
        "main_words_available": main_words,
        "first_row_index": examples[0].row_index if examples else None,
        "last_row_index": examples[-1].row_index if examples else None,
    }), flush=True)

    dataset = base.TailDataset(examples, tokenizer, int(args.seq_length))
    loader = DataLoader(dataset, batch_size=int(args.micro_batch), shuffle=False,
                        collate_fn=base.collate, num_workers=0,
                        pin_memory=torch.cuda.is_available())

    reset_all(int(args.train_seed))
    model, missing, unexpected = load_model(endpoint, device, int(args.private_adapter_bottleneck), float(args.private_scale))
    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)
    private_up_ids = {id(p) for n, p in model.named_parameters() if ".private_adapter.up." in n}
    private_normal = [p for n, p in model.named_parameters() if n in private_names and id(p) not in private_up_ids]
    private_zero_wd = [p for n, p in model.named_parameters() if n in private_names and id(p) in private_up_ids]
    optimizer = torch.optim.AdamW([
        {"params": private_normal, "weight_decay": float(args.weight_decay)},
        {"params": private_zero_wd, "weight_decay": 0.0},
    ], lr=float(args.learning_rate), betas=(0.9, 0.98), eps=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    private_params = sum(p.numel() for n, p in model.named_parameters() if n in private_names)
    warmup = max(1, int(int(args.schedule_total) * float(args.warmup_fraction)))
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.train_seed))

    config = {
        "status": "COHERENT86_CONTINUATION_CONFIG",
        "scientific_question": "Can the exact coherent86 alpha0.75 private correction acquire additional useful competence under remaining legal budget from the same parent?",
        "credit_mode": args.credit_mode,
        "endpoint": rel(endpoint),
        "example_jsonl": rel(stream),
        "initial_consumed_words": int(args.initial_consumed_words),
        "full_cap_words": int(args.full_cap_words),
        "max_tail_charged_words": int(args.max_tail_charged_words),
        "skip_rows": int(args.skip_rows),
        "tail_start_row_index_0based": int(args.skip_rows),
        "trainable": "private_adapter_only_existing_weights",
        "parent": "exact_coherent86_alpha0p75",
        "private_scale_train_and_saved": float(args.private_scale),
        "objective": "main_mlm_ce_plus_deterministic_private_on_vs_private_off_kl" if args.credit_mode == "standard" else "carrier_error_weighted_mlm_plus_deterministic_private_on_vs_private_off_kl",
        "schedule_total": int(args.schedule_total),
        "schedule_offset": int(args.schedule_offset),
        "warmup_steps": warmup,
        "macro_batch": int(args.macro_batch),
        "micro_batch": int(args.micro_batch),
        "accum_steps": accum_steps,
        "seq_length": int(args.seq_length),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "mask_prob": float(args.mask_prob),
        "checkpoint_words": int(args.checkpoint_words),
        "main_lambda": float(args.main_lambda),
        "neutral_lambda": float(args.neutral_lambda),
        "neutral_subsample_per_macro": int(args.neutral_subsample),
        "private_adapter_bottleneck": int(args.private_adapter_bottleneck),
        "train_seed": int(args.train_seed),
        "total_params": total_params,
        "private_params": private_params,
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "carrier_weighting_repair": {
            "eval_mode": True,
            "private_disabled_for_carrier": True,
            "rng_isolated": True,
            "weight_formula": "normalized 1 - p_parent_private_off(correct_token)"
        } if args.credit_mode == "carrier_residual" else None,
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "model_loaded", "total_params": total_params, "private_params": private_params,
                      "missing_keys_count": len(missing), "unexpected_keys_count": len(unexpected)}), flush=True)

    log_path = out / "training_log.jsonl"
    losses_main: list[float] = []
    losses_neutral: list[float] = []
    cum_words = 0
    updates = 0
    micro_in_macro = 0
    accum_main = 0.0
    accum_neutral = 0.0
    accum_targets = 0
    accum_words = 0
    accum_wstats = {"n": 0.0, "sum_easy": 0.0, "sum_hard": 0.0, "sum_mean_raw_error": 0.0,
                    "min_weight": 1e9, "max_weight": -1e9, "count_batches": 0.0}
    next_ckpt_tail = int(args.checkpoint_words) if int(args.checkpoint_words) > 0 else None
    stopped_before_cap = False

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for loader_step, batch in enumerate(loader, 1):
            if args.max_updates and updates >= int(args.max_updates):
                print(json.dumps({"event": "stop_after_max_updates", "updates": updates}), flush=True)
                break
            words = int(batch["words"].sum().item())
            tail_next = cum_words + words
            total_next = int(args.initial_consumed_words) + tail_next
            if tail_next > int(args.max_tail_charged_words) or total_next > int(args.full_cap_words):
                stopped_before_cap = True
                print(json.dumps({"event": "stop_before_over_cap", "loader_step": loader_step,
                                  "tail_current_charged": cum_words, "next_words": words,
                                  "total_next": total_next}), flush=True)
                break

            if micro_in_macro == 0:
                sched_index = int(args.schedule_offset) + updates
                lr = lr_at_update(sched_index, int(args.schedule_total), warmup, float(args.learning_rate))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                accum_main = 0.0
                accum_neutral = 0.0
                accum_targets = 0
                accum_words = 0
                accum_wstats = {"n": 0.0, "sum_easy": 0.0, "sum_hard": 0.0, "sum_mean_raw_error": 0.0,
                                "min_weight": 1e9, "max_weight": -1e9, "count_batches": 0.0}

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            masked_inputs, labels, _select, _selected_groups = base.apply_wwm(
                input_ids, attention_mask, word_group, tokenizer, float(args.mask_prob), gen)
            n_targets = int((labels != -100).sum().item())

            if args.credit_mode == "carrier_residual" and n_targets > 0:
                token_weights, wstats = compute_carrier_weights_deterministic(model, masked_inputs, attention_mask, labels)
                accum_wstats["n"] += wstats["n"]
                accum_wstats["sum_easy"] += wstats["carrier_easy_frac_p_gt_0p5"] * wstats["n"]
                accum_wstats["sum_hard"] += wstats["carrier_hard_frac_p_lt_0p1"] * wstats["n"]
                accum_wstats["sum_mean_raw_error"] += wstats["mean_raw_error"]
                accum_wstats["min_weight"] = min(accum_wstats["min_weight"], wstats["min_weight"])
                accum_wstats["max_weight"] = max(accum_wstats["max_weight"], wstats["max_weight"])
                accum_wstats["count_batches"] += 1.0
            else:
                token_weights = None

            model.train()
            model.set_private_enabled(True)
            out_main = model(input_ids=masked_inputs, attention_mask=attention_mask)
            vocab = out_main.logits.shape[-1]
            if token_weights is not None:
                ce_per_token = F.cross_entropy(out_main.logits.reshape(-1, vocab), labels.reshape(-1),
                                               ignore_index=-100, reduction="none")
                micro_main = (ce_per_token * token_weights.reshape(-1)).sum() / max(1, n_targets)
                del ce_per_token
            else:
                main_sum = F.cross_entropy(out_main.logits.reshape(-1, vocab), labels.reshape(-1),
                                           ignore_index=-100, reduction="sum")
                micro_main = main_sum / max(1, n_targets)
                del main_sum
            micro_main_val = float(micro_main.detach().cpu())
            (float(args.main_lambda) * micro_main / accum_steps).backward()
            del out_main, micro_main

            neutral_val = 0.0
            neutral_sub = max(1, int(args.neutral_subsample) // accum_steps)
            if float(args.neutral_lambda) > 0:
                nl, neutral_val = compute_neutral_kl(model, masked_inputs, attention_mask, neutral_sub)
                if nl is not None:
                    (float(args.neutral_lambda) * nl / accum_steps).backward()
                    del nl

            accum_main += micro_main_val
            accum_neutral += neutral_val
            accum_targets += n_targets
            accum_words += words
            cum_words += words
            micro_in_macro += 1

            del input_ids, attention_mask, word_group, masked_inputs, labels
            if token_weights is not None:
                del token_weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if micro_in_macro >= accum_steps:
                all_private = private_normal + private_zero_wd
                torch.nn.utils.clip_grad_norm_(all_private, 1.0)
                optimizer.step()
                updates += 1
                micro_in_macro = 0
                avg_main = accum_main / accum_steps
                avg_neutral = accum_neutral / accum_steps
                losses_main.append(avg_main)
                losses_neutral.append(avg_neutral)
                tail_charged = cum_words
                total_consumed = int(args.initial_consumed_words) + tail_charged
                rec = {
                    "update": updates,
                    "loader_step": loader_step,
                    "schedule_index": int(args.schedule_offset) + updates - 1,
                    "lr": lr,
                    "source_row_start": int(batch["row_index"][0].item()),
                    "source_row_end": int(batch["row_index"][-1].item()),
                    "macro_words": accum_words,
                    "tail_words": tail_charged,
                    "total_consumed_words": total_consumed,
                    "main_targets": accum_targets,
                    "main_loss": avg_main,
                    "neutral_loss": avg_neutral,
                    "private_rms_max": max(adapter_rms(model)),
                    "elapsed_sec": round(time.time() - t0, 1),
                }
                if args.credit_mode == "carrier_residual" and accum_wstats["n"] > 0:
                    rec.update({
                        "carrier_easy_frac_p_gt_0p5": accum_wstats["sum_easy"] / max(1.0, accum_wstats["n"]),
                        "carrier_hard_frac_p_lt_0p1": accum_wstats["sum_hard"] / max(1.0, accum_wstats["n"]),
                        "mean_raw_error_over_microbatches": accum_wstats["sum_mean_raw_error"] / max(1.0, accum_wstats["count_batches"]),
                        "min_weight": accum_wstats["min_weight"],
                        "max_weight": accum_wstats["max_weight"],
                    })
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if updates == 1 or updates % int(args.log_every) == 0:
                    print(json.dumps({"event": "train", **rec}), flush=True)

                while next_ckpt_tail is not None and tail_charged >= next_ckpt_tail and next_ckpt_tail <= int(args.max_tail_charged_words):
                    approx_total = int(args.initial_consumed_words) + next_ckpt_tail
                    name = f"chck_{approx_total // 1_000_000}M" if approx_total % 1_000_000 == 0 else f"chck_total_{approx_total}w"
                    save_checkpoint(model, tokenizer, out / "hf_model" / name, float(args.private_scale))
                    print(json.dumps({"event": "checkpoint", "name": name,
                                      "tail_charged_words": tail_charged,
                                      "total_consumed_words": total_consumed,
                                      "update": updates}), flush=True)
                    next_ckpt_tail += int(args.checkpoint_words)

    if micro_in_macro != 0:
        # The remaining stream length is not guaranteed to be a multiple of the macro-batch
        # size.  Step the final partial macro instead of silently discarding useful/legal
        # exposure.  Gradients were scaled by the full accumulation denominator, so the
        # final update has proportionally smaller mass when the last macro is short.
        all_private = private_normal + private_zero_wd
        torch.nn.utils.clip_grad_norm_(all_private, 1.0)
        optimizer.step()
        updates += 1
        avg_main = accum_main / max(1, micro_in_macro)
        avg_neutral = accum_neutral / max(1, micro_in_macro)
        losses_main.append(avg_main)
        losses_neutral.append(avg_neutral)
        total_consumed = int(args.initial_consumed_words) + cum_words
        rec = {
            "update": updates,
            "loader_step": loader_step if 'loader_step' in locals() else None,
            "schedule_index": int(args.schedule_offset) + updates - 1,
            "lr": lr if 'lr' in locals() else None,
            "partial_macro": True,
            "micro_in_macro": micro_in_macro,
            "accum_steps": accum_steps,
            "macro_words": accum_words,
            "tail_words": cum_words,
            "total_consumed_words": total_consumed,
            "main_targets": accum_targets,
            "main_loss": avg_main,
            "neutral_loss": avg_neutral,
            "private_rms_max": max(adapter_rms(model)),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        if args.credit_mode == "carrier_residual" and accum_wstats["n"] > 0:
            rec.update({
                "carrier_easy_frac_p_gt_0p5": accum_wstats["sum_easy"] / max(1.0, accum_wstats["n"]),
                "carrier_hard_frac_p_lt_0p1": accum_wstats["sum_hard"] / max(1.0, accum_wstats["n"]),
                "mean_raw_error_over_microbatches": accum_wstats["sum_mean_raw_error"] / max(1.0, accum_wstats["count_batches"]),
                "min_weight": accum_wstats["min_weight"],
                "max_weight": accum_wstats["max_weight"],
            })
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(json.dumps(rec) + "\n")
        print(json.dumps({"event": "final_partial_optimizer_step", **rec}), flush=True)

    model.set_private_enabled(True)
    save_checkpoint(model, tokenizer, out / "hf_model" / "final", float(args.private_scale))
    metrics = {
        "status": "COHERENT86_CONTINUATION_COMPLETE",
        "credit_mode": args.credit_mode,
        "endpoint": rel(endpoint),
        "updates": updates,
        "schedule_offset": int(args.schedule_offset),
        "schedule_total": int(args.schedule_total),
        "tail_main_word_exposure": cum_words,
        "tail_charged_words": cum_words,
        "total_consumed_words": int(args.initial_consumed_words) + cum_words,
        "full_cap_words": int(args.full_cap_words),
        "stopped_before_cap": stopped_before_cap,
        "final_main_loss_last10": sum(losses_main[-10:]) / max(1, len(losses_main[-10:])),
        "mean_main_loss": sum(losses_main) / max(1, len(losses_main)),
        "final_neutral_loss_last10": sum(losses_neutral[-10:]) / max(1, len(losses_neutral[-10:])),
        "mean_neutral_loss": sum(losses_neutral) / max(1, len(losses_neutral)),
        "final_private_rms": adapter_rms(model),
        "elapsed_sec": round(time.time() - t0, 2),
        "out_dir": rel(out),
        "final_model": rel(out / "hf_model" / "final"),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics), flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--credit_mode", choices=["standard", "carrier_residual"], required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--endpoint", default=str(DEFAULT_PARENT))
    p.add_argument("--example_jsonl", default=str(DEFAULT_STREAM))
    p.add_argument("--initial_consumed_words", type=int, default=DEFAULT_INITIAL_WORDS)
    p.add_argument("--full_cap_words", type=int, default=DEFAULT_FULL_CAP_WORDS)
    p.add_argument("--max_tail_charged_words", type=int, default=DEFAULT_REMAINING_WORDS)
    p.add_argument("--skip_rows", type=int, default=DEFAULT_SKIP_ROWS)
    p.add_argument("--macro_batch", type=int, default=256)
    p.add_argument("--micro_batch", type=int, default=32)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--schedule_total", type=int, default=DEFAULT_SCHEDULE_TOTAL)
    p.add_argument("--schedule_offset", type=int, default=DEFAULT_SCHEDULE_OFFSET)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--main_lambda", type=float, default=1.0)
    p.add_argument("--neutral_lambda", type=float, default=1.0)
    p.add_argument("--neutral_subsample", type=int, default=32)
    p.add_argument("--private_adapter_bottleneck", type=int, default=128)
    p.add_argument("--private_scale", type=float, default=0.75)
    p.add_argument("--train_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--max_updates", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
