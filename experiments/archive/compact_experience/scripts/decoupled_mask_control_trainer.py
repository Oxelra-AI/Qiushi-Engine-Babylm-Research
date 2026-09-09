#!/usr/bin/env python3
"""Decoupled-RNG and length/count-control masking continuation trainer.

This is a follow-up control for the research inverse-priority masking result.
It keeps the same clean-Qwen parent, data pool, exposure accounting, optimizer,
batch size, sequence length, and fixed masked-token budget, but improves the
experimental isolation in two ways:

1. Mask-selection randomness and corruption/replacement randomness are separated.
   Corruption decisions are deterministic functions of (seed, global step,
   dataset chunk id, token position), so a token selected by two modes receives
   the same 80/10/10 corruption realization. Variable-length multinomial loops in
   non-uniform modes therefore cannot shift later corruption RNG states.

2. A `length_matched_random` mode first draws the same inverse-priority template
   length histogram for each sequence, then chooses random word identities with
   exactly the same wordpiece-length multiset. This tests whether inverse's gain
   is due to masking more/shorter whole words at fixed token budget rather than
   the linguistic identity of low-surface-priority words.

All priority rules are inherited from the legal research trainer and use only the
training text/tokenization surface. No downstream BabyLM labels/items, no AoA/CDI
words, no child curves, and no leaderboard feedback are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

import evidence_visible_continuation_trainer as base

SEQ_LENGTH = base.SEQ_LENGTH
MASK_PROB = base.MASK_PROB
WEIGHT_DECAY = base.WEIGHT_DECAY
CHECKPOINT_INTERVAL = base.CHECKPOINT_INTERVAL
MODE_CHOICES = ["uniform", "evidence_visible", "random_priority", "inverse_priority", "length_matched_random"]


def stable_u64(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    # Stay inside a conservative positive seed range accepted by torch.Generator.
    return int.from_bytes(h.digest(), "little") % (2**63 - 1)


def make_gen(seed: int, *parts: object) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(stable_u64(seed, *parts))
    return g


def hash_u01(seed: int, *parts: object) -> float:
    return stable_u64(seed, *parts) / float(2**63 - 1)


def hash_int(mod: int, seed: int, *parts: object) -> int:
    return stable_u64(seed, *parts) % max(1, mod)


def inverse_priorities(base_priorities: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.clamp(base_priorities, min=0.1)


def choose_random_with_exact_lengths(word_groups: list[list[int]], template: list[int], gen: torch.Generator) -> list[int]:
    """Choose random identities with the exact length multiset of template.

    The control should preserve inverse-priority's target geometry while removing
    its word-identity signal. Therefore it avoids inverse-template identities when
    enough same-length alternatives exist; if a length bin is exhausted it falls
    back to template identities only to preserve the exact length multiset.
    """
    template_set = set(template)
    all_by_len: dict[int, list[int]] = {}
    non_template_by_len: dict[int, list[int]] = {}
    template_by_len: dict[int, list[int]] = {}
    for j, g in enumerate(word_groups):
        L = len(g)
        all_by_len.setdefault(L, []).append(j)
        if j in template_set:
            template_by_len.setdefault(L, []).append(j)
        else:
            non_template_by_len.setdefault(L, []).append(j)
    # Randomize candidate order inside each length bin.
    for mapping in (non_template_by_len, template_by_len, all_by_len):
        for L, vals in list(mapping.items()):
            if len(vals) > 1:
                perm = torch.randperm(len(vals), generator=gen).tolist()
                mapping[L] = [vals[k] for k in perm]
    lengths = [len(word_groups[j]) for j in template]
    if len(lengths) > 1:
        perm = torch.randperm(len(lengths), generator=gen).tolist()
        lengths = [lengths[k] for k in perm]
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for L in lengths:
        bucket = non_template_by_len.get(L, [])
        while bucket and bucket[-1] in chosen_set:
            bucket.pop()
        if bucket:
            wi = bucket.pop()
        else:
            # Preserve the exact length count even if the sequence lacks enough
            # non-template alternatives of this length.
            tbucket = template_by_len.get(L, [])
            while tbucket and tbucket[-1] in chosen_set:
                tbucket.pop()
            if tbucket:
                wi = tbucket.pop()
            else:
                # Robustness fallback for future adaptations; should be unreachable
                # for a template drawn without replacement from the same sequence.
                abucket = all_by_len.get(L, [])
                while abucket and abucket[-1] in chosen_set:
                    abucket.pop()
                if not abucket:
                    break
                wi = abucket.pop()
        chosen.append(wi)
        chosen_set.add(wi)
    return chosen


def choose_words_decoupled(word_groups: list[list[int]], base_prior: torch.Tensor, mode: str, budget_mode: str,
                           target_tokens: int, seed: int, global_step: int, dataset_index: int, row_slot: int) -> tuple[list[int], torch.Tensor, str]:
    """Return chosen word indices, mode priorities, and a selection-role label."""
    if not word_groups:
        return [], base_prior, "empty"
    # The uniform target budget is supplied by the caller and is identical across
    # modes for this sequence. Selection generators are per-sequence, so variable
    # loops cannot affect later sequences or corruption decisions.
    if mode == "uniform":
        n_mask = max(1, int(len(word_groups) * MASK_PROB))
        n_mask = min(n_mask, len(word_groups))
        g_budget = make_gen(seed, "budget", global_step, dataset_index, row_slot)
        return torch.randperm(len(word_groups), generator=g_budget)[:n_mask].tolist(), torch.ones_like(base_prior), "uniform_reference"

    if mode == "evidence_visible":
        pr = base_prior
        g = make_gen(seed, "select", "evidence_visible", global_step, dataset_index, row_slot)
        return base.choose_priority_words(word_groups, pr, target_tokens, g, mode="evidence_visible", budget_mode=budget_mode), pr, "surface_priority_identity"

    if mode == "inverse_priority":
        pr = inverse_priorities(base_prior)
        g = make_gen(seed, "select", "inverse_priority", global_step, dataset_index, row_slot)
        return base.choose_priority_words(word_groups, pr, target_tokens, g, mode="inverse_priority", budget_mode=budget_mode), pr, "inverse_surface_priority_identity"

    if mode == "random_priority":
        g_perm = make_gen(seed, "select", "random_priority_perm", global_step, dataset_index, row_slot)
        pr = base_prior[torch.randperm(base_prior.numel(), generator=g_perm)]
        g = make_gen(seed, "select", "random_priority", global_step, dataset_index, row_slot)
        return base.choose_priority_words(word_groups, pr, target_tokens, g, mode="random_priority", budget_mode=budget_mode), pr, "priority_multiset_randomized"

    if mode == "length_matched_random":
        inv = inverse_priorities(base_prior)
        g_template = make_gen(seed, "select", "inverse_priority", global_step, dataset_index, row_slot)
        template = base.choose_priority_words(word_groups, inv, target_tokens, g_template, mode="inverse_priority", budget_mode=budget_mode)
        g_match = make_gen(seed, "select", "length_matched_random_identity", global_step, dataset_index, row_slot)
        return choose_random_with_exact_lengths(word_groups, template, g_match), torch.ones_like(base_prior), "inverse_length_histogram_random_identity"

    raise ValueError(f"unknown mode: {mode}")


def decoupled_mask_batch(input_ids: torch.Tensor, tokenizer, mask_prob: float, device: torch.device,
                         mode: str, budget_mode: str, seed: int, global_step: int,
                         batch_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    batch_size, _seq_len = input_ids.shape
    mask_id = tokenizer.mask_token_id
    masked_inputs = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    stats: dict[str, Any] = {
        "masked_words": 0,
        "masked_tokens": 0,
        "target_masked_tokens": 0,
        "selected_base_high_words": 0,
        "available_base_high_words": 0,
        "selected_base_priority_sum_x1000": 0,
        "selected_wordpiece_len_sum": 0,
        "length_template_match_failures": 0,
    }

    for i in range(batch_size):
        dataset_index = int(batch_indices[i]) if i < len(batch_indices) else i
        word_groups = base.build_word_groups(input_ids[i], tokenizer)
        if not word_groups:
            continue
        token_strs = [str(tokenizer.convert_ids_to_tokens(int(tid))) for tid in input_ids[i].tolist()]
        base_prior = torch.tensor([base.priority_for_group(g, token_strs, j, word_groups) for j, g in enumerate(word_groups)], dtype=torch.float32)
        base_prior = torch.clamp(base_prior, min=0.05)

        # Compute a uniform masked-token budget once per sequence, independent of
        # mode. This keeps loss-bearing token count matched across arms.
        n_mask_uniform = max(1, int(len(word_groups) * mask_prob))
        n_mask_uniform = min(n_mask_uniform, len(word_groups))
        g_budget = make_gen(seed, "budget", global_step, dataset_index, i)
        uniform_target_words = torch.randperm(len(word_groups), generator=g_budget)[:n_mask_uniform].tolist()
        target_tokens = sum(len(word_groups[wi]) for wi in uniform_target_words)
        stats["target_masked_tokens"] += int(target_tokens)

        chosen, _mode_prior, _role = choose_words_decoupled(
            word_groups, base_prior, mode, budget_mode, target_tokens, seed, global_step, dataset_index, i
        )
        if mode == "length_matched_random":
            # Exact by construction; record mismatch just in case future adaptations
            # change the template path.
            inv = inverse_priorities(base_prior)
            g_template = make_gen(seed, "select", "inverse_priority", global_step, dataset_index, i)
            template = base.choose_priority_words(word_groups, inv, target_tokens, g_template, mode="inverse_priority", budget_mode=budget_mode)
            if sorted(len(word_groups[j]) for j in chosen) != sorted(len(word_groups[j]) for j in template):
                stats["length_template_match_failures"] += 1

        base_high_cut = torch.quantile(base_prior, 0.75).item() if base_prior.numel() >= 4 else base_prior.max().item()
        stats["available_base_high_words"] += int((base_prior >= base_high_cut).sum().item())

        for wi in chosen:
            if wi < 0 or wi >= len(word_groups):
                continue
            if base_prior[wi].item() >= base_high_cut:
                stats["selected_base_high_words"] += 1
            stats["masked_words"] += 1
            stats["selected_wordpiece_len_sum"] += len(word_groups[wi])
            stats["selected_base_priority_sum_x1000"] += int(round(float(base_prior[wi].item()) * 1000.0))
            for pos in word_groups[wi]:
                stats["masked_tokens"] += 1
                labels[i, pos] = input_ids[i, pos]
                # Corruption is independent of how selection was sampled.
                r = hash_u01(seed, "corrupt", global_step, dataset_index, pos, int(input_ids[i, pos].item()))
                if r < 0.8:
                    masked_inputs[i, pos] = mask_id
                elif r < 0.9:
                    masked_inputs[i, pos] = hash_int(len(tokenizer), seed, "replace", global_step, dataset_index, pos)

    stats["masked_token_budget_ratio"] = stats["masked_tokens"] / max(int(stats["target_masked_tokens"]), 1)
    stats["base_high_fraction_among_selected_words"] = stats["selected_base_high_words"] / max(int(stats["masked_words"]), 1)
    stats["selected_base_priority_mean"] = (stats["selected_base_priority_sum_x1000"] / 1000.0) / max(int(stats["masked_words"]), 1)
    stats["selected_wordpiece_len_mean"] = stats["selected_wordpiece_len_sum"] / max(int(stats["masked_words"]), 1)
    return masked_inputs.to(device), attention_mask.to(device), labels.to(device), stats


def add_int_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k in [
        "masked_words", "masked_tokens", "target_masked_tokens", "selected_base_high_words",
        "available_base_high_words", "selected_base_priority_sum_x1000", "selected_wordpiece_len_sum",
        "length_template_match_failures",
    ]:
        dst[k] = int(dst.get(k, 0)) + int(src.get(k, 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_checkpoint", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mask_mode", choices=MODE_CHOICES, required=True)
    ap.add_argument("--mask_budget", choices=["token_count", "word_count"], default="token_count")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_word_exposure", type=int, default=base.CONTINUATION_BUDGET)
    ap.add_argument("--start_word_exposure", type=int, default=base.CONTINUATION_START_EXPOSURE)
    ap.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL)
    ap.add_argument("--train_rng_seed", type=int, default=48044)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    init_path = Path(args.init_checkpoint)
    train_path = Path(args.train_file)
    if not init_path.exists():
        raise SystemExit(f"Init checkpoint not found: {init_path}")
    if not train_path.exists():
        raise SystemExit(f"Train file not found: {train_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({
        "event": "start", "label": args.label, "trainer_variant": "decoupled_rng_length_control",
        "mask_mode": args.mask_mode, "mask_budget": args.mask_budget,
        "init": str(init_path), "train_file": str(train_path), "device": str(device),
        "started_utc": base.now(),
    }, indent=2), flush=True)

    model = DebertaV2ForMaskedLM.from_pretrained(str(init_path))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(init_path), use_fast=True)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,} parameters; tokenizer={len(tokenizer)}")

    examples: list[dict] = []
    pool_words = 0
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                examples.append(obj)
                pool_words += obj.get("words", len(obj["text"].split()))
    print(f"Pool: {len(examples)} rows, {pool_words} words, sha256={base.sha256_file(train_path)}")

    dataset = base.ContinuationDataset(examples, tokenizer, SEQ_LENGTH)
    sched_info = base.compute_schedule(len(dataset), pool_words, args.batch_size, args.max_word_exposure)
    print(f"Dataset chunks: {len(dataset)}")
    print(f"Steps per pass: {sched_info.steps_per_pass}, passes: {sched_info.passes_needed}, total steps: {sched_info.total_steps}")
    print(f"Approx words per step: {sched_info.approx_words_per_step:.0f}; continuation LR: {sched_info.continuation_lr:.6f}")

    optim = torch.optim.AdamW(model.parameters(), lr=sched_info.continuation_lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=sched_info.total_steps)

    random.seed(args.train_rng_seed)
    torch.manual_seed(args.train_rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_rng_seed)

    model.train()
    cumulative_words = 0
    processed_chunks_total = 0
    global_step = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    saved_names: set[str] = set()
    next_ckpt = args.checkpoint_interval
    mask_stats_totals: dict[str, Any] = {
        "masked_words": 0, "masked_tokens": 0, "target_masked_tokens": 0,
        "selected_base_high_words": 0, "available_base_high_words": 0,
        "selected_base_priority_sum_x1000": 0, "selected_wordpiece_len_sum": 0,
        "length_template_match_failures": 0,
    }
    log_path = run_dir / "training_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    def ckpt_name_for_total_word_target(total_words: int) -> str:
        return f"chck_{int(round(total_words / 1_000_000))}M"

    def save_current_checkpoint(ckpt_name: str, *, target_continuation_words: int | None, loss_val: float) -> None:
        ckpt_path = run_dir / "hf_model" / ckpt_name
        if not ckpt_path.exists():
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_path, safe_serialization=True)
            tokenizer.save_pretrained(ckpt_path)
        if ckpt_name not in saved_names:
            saved_names.add(ckpt_name)
            saved_checkpoints.append({
                "name": ckpt_name,
                "path": str(ckpt_path),
                "step": global_step,
                "target_continuation_word_exposure": target_continuation_words,
                "actual_continuation_word_exposure": cumulative_words,
                "target_total_word_exposure": (args.start_word_exposure + target_continuation_words) if target_continuation_words is not None else None,
                "actual_total_word_exposure": args.start_word_exposure + cumulative_words,
                "processed_chunks": processed_chunks_total,
                "loss": round(loss_val, 4),
            })
        print(f"  >> Saved {ckpt_name} at step {global_step} actual_cont_words={cumulative_words}", flush=True)

    stop_training = False
    for pass_i in range(sched_info.passes_needed):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        pos = 0
        while pos < len(indices):
            batch_indices: list[int] = []
            while pos < len(indices) and len(batch_indices) < args.batch_size:
                projected_chunks = processed_chunks_total + len(batch_indices) + 1
                projected_words = int(round((projected_chunks / max(1, len(dataset))) * pool_words))
                if projected_words > args.max_word_exposure:
                    break
                batch_indices.append(indices[pos])
                pos += 1
            if not batch_indices:
                stop_training = True
                break

            input_ids = torch.stack([dataset[i] for i in batch_indices])
            masked_inputs, attention_mask, labels, mask_stats = decoupled_mask_batch(
                input_ids, tokenizer, MASK_PROB, device, args.mask_mode, args.mask_budget,
                args.train_rng_seed, global_step, batch_indices,
            )
            add_int_stats(mask_stats_totals, mask_stats)

            optim.zero_grad(set_to_none=True)
            out = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            if out.loss is None:
                raise RuntimeError("Model returned no loss")
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            global_step += 1
            loss_val = float(out.loss.item())
            loss_values.append(loss_val)
            processed_chunks_total += len(batch_indices)
            cumulative_words = min(args.max_word_exposure, int(round((processed_chunks_total / max(1, len(dataset))) * pool_words)))

            if global_step % args.log_every == 0 or cumulative_words >= args.max_word_exposure:
                lr_now = sched.get_last_lr()[0]
                token_budget_ratio = mask_stats_totals["masked_tokens"] / max(mask_stats_totals.get("target_masked_tokens", 0), 1)
                base_high_fraction = mask_stats_totals["selected_base_high_words"] / max(mask_stats_totals["masked_words"], 1)
                selected_len_mean = mask_stats_totals["selected_wordpiece_len_sum"] / max(mask_stats_totals["masked_words"], 1)
                entry = {
                    "step": global_step, "loss": round(loss_val, 4), "lr": round(lr_now, 8),
                    "continuation_words": cumulative_words,
                    "total_words_with_parent": args.start_word_exposure + cumulative_words,
                    "pass": pass_i + 1, "batch_chunks": len(batch_indices), "processed_chunks": processed_chunks_total,
                    "mask_mode": args.mask_mode, "mask_budget": args.mask_budget,
                    "masked_words": mask_stats_totals["masked_words"],
                    "masked_tokens": mask_stats_totals["masked_tokens"],
                    "target_masked_tokens": mask_stats_totals["target_masked_tokens"],
                    "masked_token_budget_ratio": round(token_budget_ratio, 6),
                    "base_high_fraction_among_selected_words": round(base_high_fraction, 6),
                    "selected_wordpiece_len_mean": round(selected_len_mean, 6),
                    "length_template_match_failures": mask_stats_totals["length_template_match_failures"],
                }
                log_f.write(json.dumps(entry) + "\n")
                log_f.flush()
                print(f"  step {global_step}/{sched_info.total_steps} loss={loss_val:.4f} lr={lr_now:.2e} cont_words={cumulative_words} base_high={base_high_fraction:.3f} len={selected_len_mean:.3f}", flush=True)

            while next_ckpt and cumulative_words >= next_ckpt and next_ckpt < args.max_word_exposure:
                ckpt_name = ckpt_name_for_total_word_target(args.start_word_exposure + next_ckpt)
                save_current_checkpoint(ckpt_name, target_continuation_words=next_ckpt, loss_val=loss_val)
                next_ckpt += args.checkpoint_interval

            if cumulative_words >= args.max_word_exposure:
                stop_training = True
                break
        if stop_training:
            break

    log_f.close()
    final_target_total_words = args.start_word_exposure + args.max_word_exposure
    final_ckpt_name = ckpt_name_for_total_word_target(final_target_total_words)
    save_current_checkpoint(final_ckpt_name, target_continuation_words=args.max_word_exposure,
                            loss_val=float(loss_values[-1] if loss_values else 0.0))
    token_budget_ratio = mask_stats_totals["masked_tokens"] / max(mask_stats_totals.get("target_masked_tokens", 0), 1)
    base_high_fraction = mask_stats_totals["selected_base_high_words"] / max(mask_stats_totals["masked_words"], 1)
    selected_len_mean = mask_stats_totals["selected_wordpiece_len_sum"] / max(mask_stats_totals["masked_words"], 1)
    selected_base_priority_mean = (mask_stats_totals["selected_base_priority_sum_x1000"] / 1000.0) / max(mask_stats_totals["masked_words"], 1)
    metrics = {
        "variant": f"decoupled_mask_control_{args.label}",
        "trainer_variant": "decoupled_rng_length_control",
        "mask_mode": args.mask_mode,
        "mask_budget": args.mask_budget,
        "init_checkpoint": str(init_path),
        "train_file": str(train_path),
        "train_file_sha256": base.sha256_file(train_path),
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "parent_start_word_exposure": args.start_word_exposure,
        "continuation_lr": sched_info.continuation_lr,
        "batch_size": args.batch_size,
        "seq_length": SEQ_LENGTH,
        "mask_prob": MASK_PROB,
        "word_exposure": cumulative_words,
        "continuation_word_exposure": cumulative_words,
        "actual_total_word_exposure": args.start_word_exposure + cumulative_words,
        "submit_limit_word_exposure_target": args.start_word_exposure + args.max_word_exposure,
        "actual_training_steps": global_step,
        "pool_words": pool_words,
        "pool_rows": len(examples),
        "loss_first": round(loss_values[0], 4) if loss_values else None,
        "loss_last": round(loss_values[-1], 4) if loss_values else None,
        "loss_mean": round(sum(loss_values) / len(loss_values), 4) if loss_values else None,
        "mask_stats_totals": mask_stats_totals,
        "masked_token_budget_ratio": round(token_budget_ratio, 6),
        "base_high_fraction_among_selected_words": round(base_high_fraction, 6),
        "selected_wordpiece_len_mean": round(selected_len_mean, 6),
        "selected_base_priority_mean": round(selected_base_priority_mean, 6),
        "length_template_match_failures": int(mask_stats_totals["length_template_match_failures"]),
        "saved_checkpoints": saved_checkpoints,
        "created_utc": base.now(),
        "train_rng_seed": args.train_rng_seed,
        "non_leakage_statement": "Mask priorities and length-matched controls use only training-corpus surface/tokenization cues; no downstream BabyLM labels/items and no AoA/CDI words/curves/scores were used.",
    }
    metrics_path = run_dir / "scientific_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "finished", "label": args.label, "mask_mode": args.mask_mode,
        "trainer_variant": "decoupled_rng_length_control", "mask_budget": args.mask_budget,
        "steps": global_step, "continuation_words": cumulative_words,
        "total_words_with_parent": args.start_word_exposure + cumulative_words,
        "loss_last": metrics["loss_last"],
        "base_high_fraction_among_selected_words": metrics["base_high_fraction_among_selected_words"],
        "selected_wordpiece_len_mean": metrics["selected_wordpiece_len_mean"],
        "masked_token_budget_ratio": metrics["masked_token_budget_ratio"],
        "checkpoints": len(saved_checkpoints), "metrics": str(metrics_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
