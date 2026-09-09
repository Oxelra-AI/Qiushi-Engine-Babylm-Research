#!/usr/bin/env python3
"""Evidence-visible whole-word masking continuation trainer.

This is a runnable follow-up to the research cluster continuation framework. It
keeps the same data pool, 80M clean-Qwen initialization, DeBERTa-v2 checkpoint,
optimizer family, batch size, sequence length, and total number of masked whole
words per sequence as the uniform WWM continuation. The experimental change is
only which whole words are selected for masking.

Modes:
  uniform          standard uniform WWM, same as the research continuation trainer
  evidence_visible priority-weighted WWM: named entities, numbers, and likely
                   attribute/object words are more often prediction targets
  random_priority same priority-value multiset as evidence_visible but assigned
                   to random words inside each sequence
  inverse_priority lower-priority words are selected more often than evidence
                   words, testing directionality

All token priority rules use only local surface/syntactic cues from the training
corpus. The script does not use downstream BabyLM labels/items, AoA/CDI words,
child curves, or leaderboard scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

SEQ_LENGTH = 256
MASK_PROB = 0.15
ORIGINAL_TOTAL_STEPS = 2515
WARMUP_FRACTION = 0.06
BASE_LR = 0.001
WEIGHT_DECAY = 0.01
CONTINUATION_START_EXPOSURE = 80_000_000
CONTINUATION_BUDGET = 20_000_000
CHECKPOINT_INTERVAL = 5_000_000

COPULA_VERBS = {"is", "are", "was", "were", "be", "been", "being", "became", "becomes", "become", "remains", "remained"}
RELATIONAL_VERBS = {
    "built", "discovered", "invented", "founded", "won", "lost", "defeated", "created", "wrote", "composed",
    "designed", "led", "ruled", "conquered", "established", "produced", "made", "has", "had", "contains",
    "includes", "reaches", "measures", "weighs", "located", "born", "died", "married", "joined", "served",
    "played", "directed", "published", "released", "formed", "developed", "named", "called", "known",
}
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "from", "with", "by", "of", "as",
    "into", "over", "under", "between", "through", "during", "before", "after", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "shall", "that", "this", "these", "those", "it", "its", "he", "she", "they", "we", "you", "i", "my", "your",
    "his", "her", "their", "our", "not", "no", "so", "if", "then", "than", "there", "here", "which", "who", "whom",
    "whose", "what", "when", "where", "why", "how",
}
FIRST_WORD_NOT_ENTITY = {
    "The", "A", "An", "This", "That", "These", "Those", "There", "Here", "It", "He", "She", "They", "We", "I", "You",
    "His", "Her", "Their", "Our", "But", "And", "Or", "If", "When", "Where", "While", "In", "On", "At", "For", "From",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ContinuationDataset(Dataset):
    """Tokenized chunked dataset matching the research continuation data path."""

    def __init__(self, examples: list[dict], tokenizer, seq_length: int):
        self.chunks: list[torch.Tensor] = []
        buffer: list[int] = []
        for ex in examples:
            ids = tokenizer.encode(ex["text"], add_special_tokens=False)
            buffer.extend(ids)
            while len(buffer) >= seq_length:
                self.chunks.append(torch.tensor(buffer[:seq_length], dtype=torch.long))
                buffer = buffer[seq_length:]
        if len(buffer) > seq_length // 4:
            padded = buffer + [tokenizer.pad_token_id] * (seq_length - len(buffer))
            self.chunks.append(torch.tensor(padded[:seq_length], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def clean_token(token_str: str) -> str:
    return token_str.replace("Ġ", "").replace("▁", "").strip()


def build_word_groups(input_ids_1d: torch.Tensor, tokenizer) -> list[list[int]]:
    special_ids = set(tokenizer.all_special_ids)
    word_groups: list[list[int]] = []
    current: list[int] = []
    for j in range(input_ids_1d.numel()):
        tid = int(input_ids_1d[j].item())
        if tid in special_ids or tid == tokenizer.pad_token_id:
            if current:
                word_groups.append(current)
                current = []
            continue
        tok = str(tokenizer.convert_ids_to_tokens(tid))
        if tok and is_word_start(tok) and current:
            word_groups.append(current)
            current = []
        current.append(j)
    if current:
        word_groups.append(current)
    return word_groups


def group_word(group: list[int], token_strs: list[str]) -> str:
    return "".join(clean_token(token_strs[p]) for p in group).strip()


def priority_for_group(group: list[int], token_strs: list[str], group_index: int,
                       word_groups: list[list[int]]) -> float:
    """Assign a positive priority to one whole-word group using only surface cues."""
    word = group_word(group, token_strs)
    if not word:
        return 0.0
    lower = word.lower()
    prev_word = group_word(word_groups[group_index - 1], token_strs).lower() if group_index > 0 else ""
    prev2_word = group_word(word_groups[group_index - 2], token_strs).lower() if group_index > 1 else ""
    prev_surface = clean_token(token_strs[group[0] - 1]) if group[0] > 0 else ""
    sent_start = group_index == 0 or bool(re.search(r"[.!?][\"')\]”’]*$", prev_surface))

    if re.match(r"^\d", word):
        return 2.2
    if len(word) > 1 and word[0].isupper():
        if not (sent_start and word in FIRST_WORD_NOT_ENTITY):
            # Named entities and proper names should be frequent prediction targets.
            return 2.4
    if lower in FUNCTION_WORDS:
        return 0.45
    if prev_word in COPULA_VERBS or prev_word in RELATIONAL_VERBS or prev2_word in COPULA_VERBS or prev2_word in RELATIONAL_VERBS:
        if len(lower) > 2 and lower not in FUNCTION_WORDS:
            # Attribute/object positions after relation-bearing words.
            return 1.85
    if lower in COPULA_VERBS or lower in RELATIONAL_VERBS:
        # Relation verbs are informative but should not dominate over arguments/attributes.
        return 1.15
    if len(lower) >= 7:
        return 1.15
    return 1.0


def make_priorities(word_groups: list[list[int]], token_strs: list[str], mode: str, gen: torch.Generator) -> torch.Tensor:
    base = torch.tensor([priority_for_group(g, token_strs, i, word_groups) for i, g in enumerate(word_groups)], dtype=torch.float32)
    base = torch.clamp(base, min=0.05)
    if mode == "uniform":
        return torch.ones_like(base)
    if mode == "evidence_visible":
        return base
    if mode == "inverse_priority":
        inv = 1.0 / torch.clamp(base, min=0.1)
        return inv
    if mode == "random_priority":
        perm = torch.randperm(base.numel(), generator=gen)
        return base[perm]
    raise ValueError(f"unknown mask mode: {mode}")


def choose_priority_words(word_groups: list[list[int]], priorities: torch.Tensor, target_tokens: int,
                          gen: torch.Generator, *, mode: str, budget_mode: str) -> list[int]:
    """Choose whole words while controlling either word count or subword-token count."""
    if not word_groups:
        return []
    if budget_mode == "word_count":
        n_mask = max(1, int(len(word_groups) * MASK_PROB))
        n_mask = min(n_mask, len(word_groups))
        if mode == "uniform":
            return torch.randperm(len(word_groups), generator=gen)[:n_mask].tolist()
        return torch.multinomial(priorities, num_samples=n_mask, replacement=False, generator=gen).tolist()

    # token_count mode: keep the number of loss-bearing subword positions close to
    # the uniform WWM expectation, so the intervention is masking distribution, not
    # a larger effective loss budget on fragmented words.
    remaining = set(range(len(word_groups)))
    chosen: list[int] = []
    used_tokens = 0
    target_tokens = max(1, target_tokens)
    while remaining and used_tokens < target_tokens:
        idxs = torch.tensor(sorted(remaining), dtype=torch.long)
        weights = priorities[idxs]
        if mode == "uniform":
            pick_rel = int(torch.randint(len(idxs), (1,), generator=gen).item())
        else:
            pick_rel = int(torch.multinomial(weights, num_samples=1, replacement=False, generator=gen).item())
        wi = int(idxs[pick_rel].item())
        group_tokens = len(word_groups[wi])
        if chosen and used_tokens + group_tokens > target_tokens:
            # If this word would overshoot, try to find a smaller still-priority-weighted
            # alternative that fits; otherwise stop. This preserves whole-word masking.
            fit = [j for j in remaining if used_tokens + len(word_groups[j]) <= target_tokens]
            if fit:
                fit_idxs = torch.tensor(sorted(fit), dtype=torch.long)
                fit_weights = priorities[fit_idxs]
                if mode == "uniform":
                    fit_rel = int(torch.randint(len(fit_idxs), (1,), generator=gen).item())
                else:
                    fit_rel = int(torch.multinomial(fit_weights, num_samples=1, replacement=False, generator=gen).item())
                wi = int(fit_idxs[fit_rel].item())
                group_tokens = len(word_groups[wi])
            else:
                break
        chosen.append(wi)
        remaining.remove(wi)
        used_tokens += group_tokens
    if not chosen:
        return [int(torch.argmax(priorities).item())]
    return chosen


def priority_wwm_mask_batch(input_ids: torch.Tensor, tokenizer, mask_prob: float, gen: torch.Generator,
                            device: torch.device, mode: str, budget_mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Whole-word masking with priority-weighted word choice and controlled loss budget."""
    batch_size, _seq_len = input_ids.shape
    mask_id = tokenizer.mask_token_id
    masked_inputs = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    selected_word_count = 0
    selected_token_count = 0
    high_priority_selected = 0
    high_priority_available = 0
    target_token_total = 0

    for i in range(batch_size):
        word_groups = build_word_groups(input_ids[i], tokenizer)
        if not word_groups:
            continue
        token_strs = [str(tokenizer.convert_ids_to_tokens(int(tid))) for tid in input_ids[i].tolist()]
        priorities = make_priorities(word_groups, token_strs, mode, gen)
        n_mask_uniform = max(1, int(len(word_groups) * mask_prob))
        n_mask_uniform = min(n_mask_uniform, len(word_groups))
        uniform_perm = torch.randperm(len(word_groups), generator=gen)
        uniform_target_words = uniform_perm[:n_mask_uniform].tolist()
        target_tokens = sum(len(word_groups[wi]) for wi in uniform_target_words)
        target_token_total += target_tokens
        if mode == "uniform" and budget_mode == "token_count":
            chosen = uniform_target_words
        else:
            chosen = choose_priority_words(word_groups, priorities, target_tokens, gen, mode=mode, budget_mode=budget_mode)
        high_cut = torch.quantile(priorities, 0.75).item() if priorities.numel() >= 4 else priorities.max().item()
        high_priority_available += int((priorities >= high_cut).sum().item())
        for wi in chosen:
            if priorities[wi].item() >= high_cut:
                high_priority_selected += 1
            selected_word_count += 1
            for pos in word_groups[wi]:
                selected_token_count += 1
                labels[i, pos] = input_ids[i, pos]
                r = torch.rand(1, generator=gen).item()
                if r < 0.8:
                    masked_inputs[i, pos] = mask_id
                elif r < 0.9:
                    masked_inputs[i, pos] = torch.randint(len(tokenizer), (1,), generator=gen).item()

    stats = {
        "masked_words": selected_word_count,
        "masked_tokens": selected_token_count,
        "target_masked_tokens": target_token_total,
        "masked_token_budget_ratio": selected_token_count / max(target_token_total, 1),
        "high_priority_selected": high_priority_selected,
        "high_priority_available": high_priority_available,
    }
    return masked_inputs.to(device), attention_mask.to(device), labels.to(device), stats


@dataclass
class RunSummary:
    steps_per_pass: int
    passes_needed: int
    total_steps: int
    approx_words_per_step: float
    continuation_lr: float


def compute_schedule(dataset_len: int, pool_words: int, batch_size: int, exposure_budget: int) -> RunSummary:
    steps_per_pass = math.ceil(dataset_len / batch_size)
    passes_needed = math.ceil(exposure_budget / pool_words)
    total_steps = steps_per_pass * passes_needed
    approx_words_per_step = pool_words / steps_per_pass
    warmup_steps_orig = int(ORIGINAL_TOTAL_STEPS * WARMUP_FRACTION)
    post_warmup_progress = (ORIGINAL_TOTAL_STEPS * 0.8 - warmup_steps_orig) / (ORIGINAL_TOTAL_STEPS - warmup_steps_orig)
    tail_lr_factor = 0.5 * (1 + math.cos(math.pi * post_warmup_progress))
    continuation_lr = BASE_LR * tail_lr_factor
    return RunSummary(steps_per_pass, passes_needed, total_steps, approx_words_per_step, continuation_lr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_checkpoint", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mask_mode", choices=["uniform", "evidence_visible", "random_priority", "inverse_priority"], required=True)
    ap.add_argument("--mask_budget", choices=["token_count", "word_count"], default="token_count",
                    help="token_count controls loss-bearing subword positions; word_count reproduces the earlier whole-word-count design.")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_word_exposure", type=int, default=CONTINUATION_BUDGET)
    ap.add_argument("--start_word_exposure", type=int, default=CONTINUATION_START_EXPOSURE,
                    help="Actual/ledger word exposure of the parent checkpoint; used for checkpoint naming and submit-limit accounting.")
    ap.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL)
    ap.add_argument("--train_rng_seed", type=int, default=44044)
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
    print(json.dumps({"event": "start", "label": args.label, "mask_mode": args.mask_mode,
                      "init": str(init_path), "train_file": str(train_path), "device": str(device),
                      "mask_budget": args.mask_budget,
                      "started_utc": now()}, indent=2), flush=True)

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
    print(f"Pool: {len(examples)} rows, {pool_words} words, sha256={sha256_file(train_path)}")

    dataset = ContinuationDataset(examples, tokenizer, SEQ_LENGTH)
    sched_info = compute_schedule(len(dataset), pool_words, args.batch_size, args.max_word_exposure)
    print(f"Dataset chunks: {len(dataset)}")
    print(f"Steps per pass: {sched_info.steps_per_pass}, passes: {sched_info.passes_needed}, total steps: {sched_info.total_steps}")
    print(f"Approx words per step: {sched_info.approx_words_per_step:.0f}; continuation LR: {sched_info.continuation_lr:.6f}")

    optim = torch.optim.AdamW(model.parameters(), lr=sched_info.continuation_lr, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=sched_info.total_steps)

    random.seed(args.train_rng_seed)
    torch.manual_seed(args.train_rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_rng_seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.train_rng_seed)

    model.train()
    cumulative_words = 0
    processed_chunks_total = 0
    global_step = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict] = []
    saved_names: set[str] = set()
    next_ckpt = args.checkpoint_interval
    mask_stats_totals = {"masked_words": 0, "masked_tokens": 0, "target_masked_tokens": 0, "high_priority_selected": 0, "high_priority_available": 0}
    log_path = run_dir / "training_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    def ckpt_name_for_total_word_target(total_words: int) -> str:
        # Checkpoint labels are the official million-word target labels. Rounding,
        # rather than flooring, lets a non-overexposed near-100M final state carry the
        # required `chck_100M` name while the manifest records its actual exposure.
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
                # Linear pool-pass word accounting keeps the run under the requested
                # continuation budget even when the token chunk count is not divisible
                # by batch size. The parent exposure is recorded separately.
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
            masked_inputs, attention_mask, labels, mask_stats = priority_wwm_mask_batch(
                input_ids, tokenizer, MASK_PROB, gen, device, args.mask_mode, args.mask_budget
            )
            for k in mask_stats_totals:
                mask_stats_totals[k] += int(mask_stats.get(k, 0))

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
                high_fraction = mask_stats_totals["high_priority_selected"] / max(mask_stats_totals["masked_words"], 1)
                token_budget_ratio = mask_stats_totals["masked_tokens"] / max(mask_stats_totals.get("target_masked_tokens", 0), 1)
                entry = {
                    "step": global_step,
                    "loss": round(loss_val, 4),
                    "lr": round(lr_now, 8),
                    "continuation_words": cumulative_words,
                    "total_words_with_parent": args.start_word_exposure + cumulative_words,
                    "pass": pass_i + 1,
                    "batch_chunks": len(batch_indices),
                    "processed_chunks": processed_chunks_total,
                    "mask_mode": args.mask_mode,
                    "mask_budget": args.mask_budget,
                    "masked_words": mask_stats_totals["masked_words"],
                    "masked_tokens": mask_stats_totals["masked_tokens"],
                    "target_masked_tokens": mask_stats_totals.get("target_masked_tokens", 0),
                    "masked_token_budget_ratio": round(token_budget_ratio, 6),
                    "high_priority_fraction_among_selected_words": round(high_fraction, 6),
                }
                log_f.write(json.dumps(entry) + "\n")
                log_f.flush()
                print(f"  step {global_step}/{sched_info.total_steps} loss={loss_val:.4f} lr={lr_now:.2e} cont_words={cumulative_words} highsel={high_fraction:.3f}", flush=True)

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
    high_fraction = mask_stats_totals["high_priority_selected"] / max(mask_stats_totals["masked_words"], 1)
    metrics = {
        "variant": f"evidence_visible_continuation_{args.label}",
        "mask_mode": args.mask_mode,
        "mask_budget": args.mask_budget,
        "init_checkpoint": str(init_path),
        "train_file": str(train_path),
        "train_file_sha256": sha256_file(train_path),
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
        "masked_token_budget_ratio": round(mask_stats_totals["masked_tokens"] / max(mask_stats_totals.get("target_masked_tokens", 0), 1), 6),
        "high_priority_fraction_among_selected_words": round(high_fraction, 6),
        "saved_checkpoints": saved_checkpoints,
        "created_utc": now(),
        "train_rng_seed": args.train_rng_seed,
        "non_leakage_statement": "Mask priorities use only training-corpus surface/syntactic cues; no downstream BabyLM labels/items and no AoA/CDI words/curves/scores were used.",
    }
    metrics_path = run_dir / "scientific_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "finished", "label": args.label, "mask_mode": args.mask_mode,
                      "mask_budget": args.mask_budget,
                      "steps": global_step, "continuation_words": cumulative_words,
                       "total_words_with_parent": args.start_word_exposure + cumulative_words,
                       "loss_last": metrics["loss_last"],
                      "high_priority_fraction_among_selected_words": metrics["high_priority_fraction_among_selected_words"],
                      "checkpoints": len(saved_checkpoints), "metrics": str(metrics_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
