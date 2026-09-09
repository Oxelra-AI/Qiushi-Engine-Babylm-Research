#!/usr/bin/env python3
"""research: Experience-utilization chunk-stream trainer.

Faithful word-boundary chunking for the experience-utilization experiment
(research design).  Instead of prefix-slicing seq_len_schedule that hides suffix
words while charging them, this trainer partitions each row into word-boundary
chunks so every tokenizer token from the 10M corpus is visible once per epoch.

Arms:
  U256:           10 epochs at L256 (pure visibility repair)
  U64_128_256:    3 epochs L64, 4 L128, 3 L256 (visibility + context ordering)
  U64x7_256x3:   7 epochs L64, 3 L256 (extended short context)

Invariants:
  - 10M charged words per epoch, 100M total
  - Every raw token visible once per epoch
  - 253 optimizer updates per epoch, 2530 total
  - Full effective-batch WWM before microbatch forward/backward
  - Masked-token weighted gradient accumulation
  - Continuous optimizer/LR/RNG state across stage boundaries
  - 1M-word checkpoint ladder

Use --dry_run for full accounting verification without GPU.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import DebertaV2Config, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

USER_ROOT = Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
import masking_curriculum_trainer as base  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────────
WORD_RE = re.compile(r"\S+")
ARMS: dict[str, list[tuple[int, int]]] = {
    "U256": [(256, 10)],
    "U64_128_256": [(64, 3), (128, 4), (256, 3)],
    "U64x7_256x3": [(64, 7), (256, 3)],
}
STEPS_PER_EPOCH = 253
POOL_WORDS = 10_000_000
TOTAL_EXPOSURE = 100_000_000


# ─── Chunk dataclass and construction ─────────────────────────────────────────

@dataclass
class Chunk:
    """One word-boundary chunk: a contiguous slice of tokenized words."""
    input_ids: list[int]
    word_group: list[int]
    charged_words: int
    source_row: int
    overlong_continuation: bool = False


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def find_word_index(starts: list[int], spans: list[tuple[int, int]],
                    s: int, e: int) -> int | None:
    if e <= s or not spans:
        return None
    idx = bisect.bisect_right(starts, s) - 1
    for j in (idx, idx + 1):
        if 0 <= j < len(spans):
            a, b = spans[j]
            if s < b and e > a:
                return j
    mid = (s + e - 1) // 2
    idx = bisect.bisect_right(starts, mid) - 1
    if 0 <= idx < len(spans):
        a, b = spans[idx]
        if s < b and e > a:
            return idx
    return None


def token_ids_by_whitespace_word(text: str, tokenizer
                                 ) -> tuple[list[list[int]], int, int]:
    """Assign every tokenizer token to its source whitespace word.

    Returns (per_word_ids, total_raw_tokens, unassigned_count).
    Standalone space-marker tokens (byte-level BPE Ġ) are attached to the
    following whitespace word, matching the research repair.
    """
    spans = word_spans(text)
    starts = [s for s, _ in spans]
    enc = tokenizer(text, add_special_tokens=False, truncation=False,
                    return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = list(enc.get("offset_mapping") or [])
    if len(ids) != len(offsets):
        raise RuntimeError("tokenizer offset length mismatch")
    by_word: list[list[int]] = [[] for _ in spans]
    unassigned = 0
    for tok_id, (s0, e0) in zip(ids, offsets):
        s, e = int(s0), int(e0)
        idx = find_word_index(starts, spans, s, e)
        if idx is None and 0 <= s <= e <= len(text) and text[s:e].strip() == "":
            nxt = bisect.bisect_left(starts, e)
            if 0 <= nxt < len(spans):
                idx = nxt
        if idx is None:
            unassigned += 1
        else:
            by_word[idx].append(int(tok_id))
    return by_word, len(ids), unassigned


def chunks_from_word_tokens(word_tokens: list[list[int]], L: int,
                            row_index: int) -> tuple[list[Chunk], dict[str, int]]:
    """Partition tokenized words into L-bounded chunks with word-group IDs."""
    chunks: list[Chunk] = []
    cur_ids: list[int] = []
    cur_groups: list[int] = []
    cur_words = 0
    overlong_words = 0
    continuation_chunks = 0

    def flush() -> None:
        nonlocal cur_ids, cur_groups, cur_words
        if cur_words or cur_ids:
            chunks.append(Chunk(input_ids=cur_ids, word_group=cur_groups,
                                charged_words=cur_words, source_row=row_index))
        cur_ids, cur_groups, cur_words = [], [], 0

    for ids in word_tokens:
        if len(ids) > L:
            overlong_words += 1
            flush()
            for part_start in range(0, len(ids), L):
                part = ids[part_start:part_start + L]
                is_cont = part_start > 0
                if is_cont:
                    continuation_chunks += 1
                chunks.append(Chunk(
                    input_ids=part, word_group=[0] * len(part),
                    charged_words=(1 if part_start == 0 else 0),
                    source_row=row_index, overlong_continuation=is_cont,
                ))
            continue
        if cur_ids and len(cur_ids) + len(ids) > L:
            flush()
        gid = cur_words
        cur_ids.extend(ids)
        cur_groups.extend([gid] * len(ids))
        cur_words += 1
    flush()
    return chunks, {"overlong_words": overlong_words,
                    "continuation_chunks": continuation_chunks}


# ─── Pool tokenization and chunk building ─────────────────────────────────────

def tokenize_pool(examples: list[base.Example], tokenizer
                  ) -> tuple[list[tuple[list[list[int]], int]], int]:
    """Tokenize all examples once; return per-word token lists and total tokens."""
    pool: list[tuple[list[list[int]], int]] = []
    total_tokens = 0
    for row_idx, ex in enumerate(examples):
        by_word, raw_tokens, unassigned = token_ids_by_whitespace_word(
            ex.text, tokenizer)
        if unassigned > 0:
            raise RuntimeError(f"unassigned tokenizer offsets at row {row_idx}")
        if len(by_word) != ex.words:
            raise RuntimeError(f"word count mismatch row {row_idx}: "
                               f"{len(by_word)} vs {ex.words}")
        pool.append((by_word, raw_tokens))
        total_tokens += raw_tokens
        if (row_idx + 1) % 10000 == 0:
            print(json.dumps({"event": "tokenize_progress",
                              "rows": row_idx + 1}), flush=True)
    return pool, total_tokens


def build_chunks_for_length(pool: list[tuple[list[list[int]], int]],
                            examples: list[base.Example], L: int
                            ) -> tuple[list[Chunk], dict[str, Any]]:
    """Build word-boundary chunks at length L from pre-tokenized pool."""
    all_chunks: list[Chunk] = []
    stats: dict[str, int] = {"rows": 0, "words": 0, "tokens": 0,
                              "chunks": 0, "overlong_words": 0,
                              "continuation_chunks": 0}
    for row_idx, ((by_word, raw_tokens), ex) in enumerate(zip(pool, examples)):
        row_chunks, meta = chunks_from_word_tokens(by_word, L, row_idx)
        all_chunks.extend(row_chunks)
        stats["rows"] += 1
        stats["words"] += ex.words
        stats["tokens"] += raw_tokens
        stats["chunks"] += len(row_chunks)
        stats["overlong_words"] += meta["overlong_words"]
        stats["continuation_chunks"] += meta["continuation_chunks"]
    return all_chunks, stats


def collate_chunks(chunks: list[Chunk], L: int) -> dict[str, torch.Tensor]:
    """Pack chunks into padded tensors for one optimizer update."""
    n = len(chunks)
    input_ids = torch.zeros((n, L), dtype=torch.long)
    attn_mask = torch.zeros((n, L), dtype=torch.long)
    word_grp = torch.full((n, L), -1, dtype=torch.long)
    words = torch.zeros(n, dtype=torch.long)
    for i, ch in enumerate(chunks):
        m = len(ch.input_ids)
        if m > L:
            raise RuntimeError(f"chunk length {m} > stage length {L}")
        if m != len(ch.word_group):
            raise RuntimeError("chunk id/group length mismatch")
        if m:
            input_ids[i, :m] = torch.tensor(ch.input_ids, dtype=torch.long)
            attn_mask[i, :m] = 1
            word_grp[i, :m] = torch.tensor(ch.word_group, dtype=torch.long)
        words[i] = ch.charged_words
    return {"input_ids": input_ids, "attention_mask": attn_mask,
            "word_group": word_grp, "words": words}


def distribute_into_steps(n_chunks: int,
                          n_steps: int = STEPS_PER_EPOCH
                          ) -> list[tuple[int, int]]:
    """Evenly distribute n_chunks into n_steps; return (start, end) per step."""
    base_n = n_chunks // n_steps
    rem = n_chunks % n_steps
    result, idx = [], 0
    for s in range(n_steps):
        count = base_n + (1 if s < rem else 0)
        result.append((idx, idx + count))
        idx += count
    assert idx == n_chunks, f"distribution error: {idx} vs {n_chunks}"
    return result


# ─── Args ─────────────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experience-utilization chunk-stream trainer")
    p.add_argument("--arm", required=True, choices=list(ARMS.keys()))
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--tokenizer_label", default="legal40k")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--micro_batch_size", type=int, default=64)
    # Model
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--intermediate_size", type=int, default=0,
                   help="Explicit FFN intermediate size (0=use hidden_size*ffn_mult)")
    # Optimization
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--mask_prob", type=float, default=0.15)
    # Reproducibility
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    # Checkpoints and logging
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--log_every", type=int, default=50)
    # Dry-run mode
    p.add_argument("--dry_run", action="store_true",
                   help="Full accounting verification on CPU without model")
    p.add_argument("--dry_run_epochs", type=int, default=-1,
                   help="Limit dry-run to this many total epochs (-1=all)")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _build_model(args: argparse.Namespace, tokenizer):
    """Build DeBERTa-v2 supporting explicit --intermediate_size for 12x384/1280."""
    isize = int(getattr(args, "intermediate_size", 0) or 0)
    if isize <= 0:
        return base.build_model(args, tokenizer)
    max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
    pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",")
                    if x.strip()]
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=isize,
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
    return DebertaV2ForMaskedLM(cfg)


def reset_all_rng(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer and corpus ──
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    vocab_size = len(tokenizer)

    jsonl_path = Path(args.example_jsonl)
    corpus_sha = sha256_file(jsonl_path)
    examples, pool_words, total_file_words, sample_rows = base.load_examples_jsonl(
        jsonl_path, POOL_WORDS)
    if pool_words != POOL_WORDS:
        raise RuntimeError(f"corpus words {pool_words} != {POOL_WORDS}")

    stages = ARMS[args.arm]
    total_epochs = sum(e for _, e in stages)
    total_steps = STEPS_PER_EPOCH * total_epochs  # 2530

    # Need max_seq_length for build_model compatibility
    args.max_seq_length = max(L for L, _ in stages)
    args.seq_length = args.max_seq_length
    # Resolve intermediate_size for the manifest
    effective_intermediate = (args.intermediate_size if args.intermediate_size > 0
                              else args.hidden_size * args.ffn_mult)

    print(json.dumps({
        "event": "init", "created_utc": now_utc(),
        "arm": args.arm, "stages": stages,
        "total_epochs": total_epochs, "total_steps": total_steps,
        "corpus_sha": corpus_sha, "vocab_size": vocab_size,
        "pool_rows": len(examples), "pool_words": pool_words,
        "dry_run": args.dry_run,
    }), flush=True)

    # ── Pre-tokenize the entire pool once ──
    t0 = time.time()
    token_pool, raw_tokens_per_epoch = tokenize_pool(examples, tokenizer)
    tokenize_sec = time.time() - t0
    print(json.dumps({
        "event": "pool_tokenized",
        "rows": len(token_pool), "raw_tokens": raw_tokens_per_epoch,
        "tokens_per_word": raw_tokens_per_epoch / POOL_WORDS,
        "seconds": round(tokenize_sec, 1),
    }), flush=True)

    # ── Model and optimizer (skip in dry-run) ──
    if not args.dry_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reset_all_rng(args.seed)
        if args.extra_init_seed >= 0:
            reset_all_rng(args.extra_init_seed)
        model = _build_model(args, tokenizer)
        param_count = sum(p.numel() for p in model.parameters())
        if args.train_rng_seed >= 0:
            reset_all_rng(args.train_rng_seed)
        model.to(device)
        model.train()

        optim = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay, betas=(0.9, 0.98))
        warmup_steps = max(1, int(total_steps * args.warmup_fraction))
        sched = get_cosine_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)

        gen = torch.Generator(device=device)
        gen.manual_seed(
            args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

        print(json.dumps({
            "event": "model_ready", "param_count": param_count,
            "device": str(device), "warmup_steps": warmup_steps,
        }), flush=True)
    else:
        device = torch.device("cpu")
        param_count = 0
        gen = torch.Generator(device="cpu")
        gen.manual_seed(
            args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    curriculum_state = base.MaskingCurriculumState(
        curriculum="wwm_fixed",
        mask_prob_start=args.mask_prob,
        mask_prob_end=args.mask_prob,
        switch_frac=0.7, amlm_window=10, amlm_lambda=0.2,
    )
    curriculum_state.initialize(vocab_size=vocab_size,
                                total_steps=total_steps)

    # ── Training state ──
    global_step = 0
    cumulative_words = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    next_ckpt = args.checkpoint_words
    stage_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []  # dry-run only
    dry_epochs_remaining = (args.dry_run_epochs
                            if args.dry_run and args.dry_run_epochs > 0
                            else total_epochs)

    log_path = out / "training_log.jsonl"
    logf = (log_path.open("w", encoding="utf-8")
            if not args.dry_run else None)

    # ── Source-word tracking ──
    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words

    # ── Stage loop ──
    abs_epoch = 0
    for stage_idx, (stage_length, num_epochs) in enumerate(stages):
        print(json.dumps({
            "event": "stage_start", "stage_idx": stage_idx,
            "length": stage_length, "num_epochs": num_epochs,
            "global_step": global_step,
            "cumulative_words": cumulative_words,
        }), flush=True)

        t0 = time.time()
        epoch_chunks, chunk_stats = build_chunks_for_length(
            token_pool, examples, stage_length)
        build_sec = time.time() - t0

        n_chunks = len(epoch_chunks)
        total_charged = sum(c.charged_words for c in epoch_chunks)
        total_active = sum(len(c.input_ids) for c in epoch_chunks)
        if total_charged != POOL_WORDS:
            raise RuntimeError(
                f"L{stage_length}: charged {total_charged} != {POOL_WORDS}")

        step_dist = distribute_into_steps(n_chunks)

        print(json.dumps({
            "event": "chunks_built", "length": stage_length,
            "n_chunks": n_chunks, "charged_words": total_charged,
            "active_tokens": total_active,
            "build_sec": round(build_sec, 1), **chunk_stats,
        }), flush=True)

        stage_records.append({
            "stage_idx": stage_idx, "length": stage_length,
            "num_epochs": num_epochs, "chunks_per_epoch": n_chunks,
            "charged_words_per_epoch": total_charged,
            "active_tokens_per_epoch": total_active,
            **chunk_stats,
        })

        for ep_in_stage in range(num_epochs):
            if dry_epochs_remaining <= 0:
                break

            ep_words = ep_tokens = ep_masked = ep_steps = 0

            for step_in_ep, (cs, ce) in enumerate(step_dist):
                sc = epoch_chunks[cs:ce]
                n_sc = len(sc)
                batch = collate_chunks(sc, stage_length)
                s_words = int(batch["words"].sum().item())
                s_active = int(batch["attention_mask"].sum().item())

                curriculum_state.current_step = global_step

                if not args.dry_run:
                    inp = batch["input_ids"].to(device, non_blocking=True)
                    att = batch["attention_mask"].to(device, non_blocking=True)
                    wg = batch["word_group"].to(device, non_blocking=True)

                    masked_inp, labels = base.apply_masking_curriculum(
                        inp, att, wg, tokenizer, curriculum_state, gen)
                    n_pred = int((labels != -100).sum().item())
                    if n_pred <= 0:
                        raise RuntimeError(
                            f"zero masked tokens at step {global_step}")

                    optim.zero_grad(set_to_none=True)
                    wt_loss = 0.0
                    n_rows = inp.shape[0]
                    for mb_s in range(0, n_rows, args.micro_batch_size):
                        mb_e = min(mb_s + args.micro_batch_size, n_rows)
                        mb_lab = labels[mb_s:mb_e]
                        n_mb = int((mb_lab != -100).sum().item())
                        if n_mb == 0:
                            continue
                        out_m = model(
                            input_ids=masked_inp[mb_s:mb_e],
                            attention_mask=att[mb_s:mb_e],
                            labels=mb_lab)
                        scale = n_mb / n_pred
                        (out_m.loss * scale).backward()
                        wt_loss += float(out_m.loss.detach().cpu()) * scale
                        del out_m

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    sched.step()
                    loss_float = wt_loss
                    del inp, att, wg, masked_inp, labels
                else:
                    masked_inp, labels = base.apply_masking_curriculum(
                        batch["input_ids"], batch["attention_mask"],
                        batch["word_group"], tokenizer,
                        curriculum_state, gen)
                    n_pred = int((labels != -100).sum().item())
                    loss_float = 0.0

                n_mb = math.ceil(n_sc / args.micro_batch_size)
                global_step += 1
                cumulative_words += s_words
                ep_words += s_words
                ep_tokens += s_active
                ep_masked += n_pred
                ep_steps += 1

                if args.dry_run:
                    step_records.append({
                        "global_step": global_step,
                        "epoch": abs_epoch,
                        "stage_length": stage_length,
                        "n_chunks": n_sc,
                        "charged_words": s_words,
                        "active_tokens": s_active,
                        "masked_tokens": n_pred,
                        "microbatches": n_mb,
                        "cumulative_words": cumulative_words,
                    })
                else:
                    loss_values.append(loss_float)
                    eff_mr = n_pred / max(1, s_active)
                    rec = {
                        "step": global_step, "loss": loss_float,
                        "lr": float(sched.get_last_lr()[0]),
                        "batch_words": s_words,
                        "cumulative_word_exposure": cumulative_words,
                        "seq_len": stage_length,
                        "masked_tokens": n_pred,
                        "effective_mask_rate": round(eff_mr, 4),
                        "n_chunks": n_sc, "microbatches": n_mb,
                        "elapsed_sec": round(time.time() - start_time, 1),
                    }
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                    if (global_step == 1
                            or global_step % args.log_every == 0
                            or global_step == total_steps):
                        print(json.dumps({"event": "train", **rec}),
                              flush=True)

                # Checkpoint at 1M-word milestones
                while (next_ckpt is not None
                       and cumulative_words >= next_ckpt
                       and next_ckpt <= TOTAL_EXPOSURE):
                    if next_ckpt >= 1_000_000 and next_ckpt % 1_000_000 == 0:
                        name = f"chck_{next_ckpt // 1_000_000}M"
                    else:
                        name = f"chck_{next_ckpt}w"
                    if not args.dry_run:
                        cp = out / "hf_model" / name
                        base.save_hf_checkpoint(model, tokenizer, cp)
                        print(json.dumps({
                            "event": "checkpoint", "name": name,
                            "cum_words": cumulative_words,
                        }), flush=True)
                    saved_checkpoints.append({
                        "name": name,
                        "target": next_ckpt,
                        "actual_cumulative_words": cumulative_words,
                        "global_step": global_step,
                        "stage_length": stage_length,
                        "epoch": abs_epoch,
                    })
                    next_ckpt += args.checkpoint_words

            epoch_records.append({
                "epoch": abs_epoch,
                "stage_length": stage_length,
                "epoch_in_stage": ep_in_stage,
                "steps": ep_steps,
                "charged_words": ep_words,
                "active_tokens": ep_tokens,
                "masked_tokens": ep_masked,
                "cumulative_words_after": cumulative_words,
            })
            print(json.dumps({
                "event": "epoch_done", "epoch": abs_epoch,
                "stage_length": stage_length,
                "steps": ep_steps,
                "charged_words": ep_words,
                "cumulative_words": cumulative_words,
            }), flush=True)
            abs_epoch += 1
            if args.dry_run:
                dry_epochs_remaining -= 1

        del epoch_chunks

    if logf is not None:
        logf.close()

    # Save final model
    if not args.dry_run:
        base.save_hf_checkpoint(model, tokenizer, out / "hf_model")

    # ── Verification ──
    completed_epochs = len(epoch_records)
    is_full_run = completed_epochs == total_epochs
    verification = {
        "completed_epochs": completed_epochs,
        "total_steps_executed": global_step,
        "total_charged_words": cumulative_words,
        "steps_match_expected": global_step == total_steps if is_full_run else None,
        "words_match_100M": cumulative_words == TOTAL_EXPOSURE if is_full_run else None,
        "all_epoch_words_10M": all(
            r["charged_words"] == POOL_WORDS for r in epoch_records),
        "all_epoch_steps_253": all(
            r["steps"] == STEPS_PER_EPOCH for r in epoch_records),
        "active_tokens_per_epoch": [
            r["active_tokens"] for r in epoch_records],
        "checkpoint_count": len(saved_checkpoints),
    }

    result = {
        "status": ("EXPERIENCE_UTILIZATION_DRYRUN"
                    if args.dry_run
                    else "EXPERIENCE_UTILIZATION_DONE"),
        "created_utc": now_utc(),
        "arm": args.arm,
        "stages": stages,
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": args.tokenizer_path,
        "corpus_sha": corpus_sha,
        "vocab_size": vocab_size,
        "param_count": param_count,
        "intermediate_size": effective_intermediate,
        "mask_prob": args.mask_prob,
        "raw_tokens_per_epoch": raw_tokens_per_epoch,
        "total_steps_executed": global_step,
        "total_charged_words": cumulative_words,
        "stage_records": stage_records,
        "epoch_records": epoch_records,
        "checkpoints": saved_checkpoints,
        "verification": verification,
        "elapsed_sec": round(time.time() - start_time, 1),
        "source_words": source_words,
    }

    if args.dry_run:
        out_json = out / "dryrun_metrics.json"
    else:
        out_json = out / "scientific_metrics.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    if args.dry_run and step_records:
        csv_path = out / "dryrun_step_accounting.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(step_records[0].keys()))
            w.writeheader()
            w.writerows(step_records)

    print(json.dumps({
        "event": "done",
        "arm": args.arm,
        "total_steps": global_step,
        "total_words": cumulative_words,
        "verification": verification,
        "out_json": str(out_json),
        "elapsed_sec": round(time.time() - start_time, 1),
    }), flush=True)


if __name__ == "__main__":
    main()
