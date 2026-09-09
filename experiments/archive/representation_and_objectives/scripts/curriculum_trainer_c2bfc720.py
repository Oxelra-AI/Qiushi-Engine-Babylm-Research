#!/usr/bin/env python3
"""research: AdamW + word-boundary sequence-length curriculum trainer.

Built on the trusted legal40k training machinery (compact_experience masking_curriculum_trainer).
When curriculum is '256:100000000', reproduces the fixed-256 baseline path.

Design:
1. Pre-tokenize all rows to max_seq_len=256 once (same logic as MaskedChunkDataset)
2. For each curriculum phase, chunk pre-tokenized rows at word boundaries
3. More chunks at shorter seq_len → more gradient updates per word
4. Words charged per original row (first chunk gets row.words, rest get 0)
5. Phase transitions at cumulative word-budget boundaries
6. Model created via base.build_model → exact config match to legal40k baseline
7. Masking via base.apply_masking_curriculum on full effective batch → exact WWM match

Key difference from research trainer:
  - Uses base.build_model (correct pos_att_type, max_relative_positions, pad_token_id)
  - Word-boundary chunking (no mid-wordpiece splits)
  - AdamW optimizer (not LAMB)
  - Gradient accumulation matching research (4×64 microbatches)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

USER_ROOT = Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

import masking_curriculum_trainer as base  # noqa: E402


# ─── Curriculum parsing ───────────────────────────────────────────────────────

def parse_curriculum(s: str) -> list[tuple[int, int]]:
    """Parse 'seq_len:cumulative_words,...' e.g. '64:20000000,128:50000000,256:100000000'."""
    phases = []
    for part in s.split(","):
        sl, budget = part.strip().split(":")
        phases.append((int(sl), int(budget)))
    phases.sort(key=lambda x: x[1])
    return phases


# ─── Pre-tokenization (same logic as base.MaskedChunkDataset) ─────────────────

class PreTokenizedRow:
    """A row tokenized to max_seq_len, stored as CPU tensors."""
    __slots__ = ("input_ids", "word_group", "n_tokens", "words")

    def __init__(self, input_ids: torch.Tensor, word_group: torch.Tensor,
                 n_tokens: int, words: int):
        self.input_ids = input_ids      # [max_seq_len] padded
        self.word_group = word_group    # [max_seq_len] with -1 for pad/special
        self.n_tokens = n_tokens        # real token count (before padding)
        self.words = words              # original row word count


def pre_tokenize_all(examples: list, tokenizer, max_seq_len: int, progress_every: int = 0) -> list[PreTokenizedRow]:
    """Tokenize all rows once. Uses identical word-group logic to MaskedChunkDataset."""
    special_ids = set(tokenizer.all_special_ids)
    ws_cache: dict[int, bool] = {}

    def is_ws(tid: int) -> bool:
        v = ws_cache.get(tid)
        if v is None:
            s = tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and base.is_word_start(str(s)))
            ws_cache[tid] = v
        return v

    rows: list[PreTokenizedRow] = []
    for ex in examples:
        enc = tokenizer(
            ex.text, add_special_tokens=False, truncation=True,
            max_length=max_seq_len, padding="max_length", return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        amask = enc["attention_mask"].squeeze(0)
        n_tokens = int(amask.sum().item())

        wg = torch.full_like(ids, -1)
        gid = -1
        for i in range(n_tokens):
            tid = int(ids[i])
            if tid in special_ids:
                continue
            if gid < 0 or is_ws(tid) or i == 0:
                gid += 1
            wg[i] = gid

        rows.append(PreTokenizedRow(ids, wg, n_tokens, ex.words))
        if progress_every and len(rows) % progress_every == 0:
            print(json.dumps({"event": "pre_tokenize_progress", "rows": len(rows)}), flush=True)
    return rows


# ─── Word-boundary chunking ──────────────────────────────────────────────────

def chunk_row(row: PreTokenizedRow, seq_len: int, pad_id: int,
              min_tokens: int = 8) -> list[dict[str, torch.Tensor]]:
    """Split pre-tokenized row into chunks at word boundaries.

    Returns list of dicts with 'input_ids', 'attention_mask', 'word_group', 'words'.
    First chunk of each row gets row.words; subsequent chunks get 0.
    """
    n = row.n_tokens
    if n == 0:
        return []

    ids = row.input_ids[:n]
    wg = row.word_group[:n]

    chunks: list[dict[str, torch.Tensor]] = []
    start = 0
    first_chunk = True

    while start < n:
        end = min(start + seq_len, n)

        # If not at end of row, backtrack to word boundary
        if end < n and end > start + 1:
            split_pos = end
            while split_pos > start + 1:
                # A word boundary: position where a new word group starts
                if wg[split_pos] >= 0 and wg[split_pos] != wg[split_pos - 1]:
                    break
                split_pos -= 1
            if split_pos > start + 1:
                end = split_pos
            # else: single very long word exceeding seq_len, keep full chunk

        actual_len = end - start
        if actual_len < min_tokens:
            break  # Skip very short tails

        # Build padded chunk tensors
        chunk_ids = torch.full((seq_len,), pad_id, dtype=ids.dtype)
        chunk_ids[:actual_len] = ids[start:end]

        chunk_amask = torch.zeros(seq_len, dtype=torch.long)
        chunk_amask[:actual_len] = 1

        # Renumber word groups sequentially within chunk
        old_wg = wg[start:end]
        new_wg = torch.full((seq_len,), -1, dtype=torch.long)
        seen: dict[int, int] = {}
        next_gid = 0
        for i in range(actual_len):
            g = int(old_wg[i].item())
            if g >= 0:
                if g not in seen:
                    seen[g] = next_gid
                    next_gid += 1
                new_wg[i] = seen[g]

        chunks.append({
            "input_ids": chunk_ids,
            "attention_mask": chunk_amask,
            "word_group": new_wg,
            "words": row.words if first_chunk else 0,
        })
        first_chunk = False
        start = end

    return chunks


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PhaseChunkDataset(Dataset):
    """Dataset of pre-built chunks for one curriculum phase."""
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        c = self.chunks[idx]
        return {
            "input_ids": c["input_ids"],
            "attention_mask": c["attention_mask"],
            "word_group": c["word_group"],
            "words": c["words"],
        }


def collate_chunks(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
    }


def combine_microbatches(micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.cat([b["input_ids"] for b in micro_batches], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in micro_batches], dim=0),
        "word_group": torch.cat([b["word_group"] for b in micro_batches], dim=0),
        "words": torch.cat([b["words"] for b in micro_batches], dim=0),
    }


def effective_batch_iterator(loader: DataLoader, accum_steps: int):
    buf: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        buf.append(batch)
        if len(buf) == accum_steps:
            yield combine_microbatches(buf)
            buf = []
    if buf:
        yield combine_microbatches(buf)


# ─── Args ─────────────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdamW + word-boundary curriculum trainer")
    # Data
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--example_jsonl_label", default="")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--tokenizer_label", default="legal_byte_bpe_40k")
    p.add_argument("--max_word_exposure", type=int, default=100_000_000)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint_words", type=int, default=10_000_000)

    # Curriculum
    p.add_argument("--curriculum", default="256:100000000",
                   help="seq_len:cumulative_words,... e.g. '64:20000000,128:50000000,256:100000000'")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--min_chunk_tokens", type=int, default=1,
                   help="Minimum real tokens per chunk. Use 1 for no short-tail token drops; older smoke tests used 8.")

    # Model (matching research / legal40k baseline defaults)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=0,
                   help="Override ffn_mult: if > 0, use this as intermediate_size directly")
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")

    # Optimization (matching research defaults)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Masking (fixed WWM, matching baseline)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--masking_curriculum", choices=base.MASKING_CURRICULA, default="wwm_fixed")

    # Reproducibility
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=-1,
                   help="Mask/dropout RNG seed after initialization. Matched legal40k baseline seed43022 uses 43023; pass it explicitly for endpoint work.")

    # Misc
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--amp", type=int, default=0, help="bf16 autocast (0=fp32, 1=bf16)")
    p.add_argument("--pretokenize_log_every", type=int, default=50000)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_args()
    assert args.batch_size % args.micro_batch_size == 0
    accum_steps = args.batch_size // args.micro_batch_size

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer and data (same as research) ────────────────────────────
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    pad_id = tokenizer.pad_token_id

    examples, total_file_words, total_rows, sample_rows = base.load_examples_jsonl(
        Path(args.example_jsonl), args.max_word_exposure
    )
    actual_words = sum(ex.words for ex in examples)
    print(json.dumps({
        "event": "data_loaded", "rows": len(examples), "words": actual_words,
        "sec": round(time.time() - start_time, 1),
    }), flush=True)

    # ── Parse curriculum and pre-tokenize ────────────────────────────────────
    curriculum = parse_curriculum(args.curriculum)

    t0 = time.time()
    pre_tok_rows = pre_tokenize_all(examples, tokenizer, args.max_seq_length, args.pretokenize_log_every)
    print(json.dumps({
        "event": "pre_tokenize_done", "rows": len(pre_tok_rows),
        "sec": round(time.time() - t0, 1),
    }), flush=True)

    # ── Build phase datasets ─────────────────────────────────────────────────
    phase_infos: list[dict[str, Any]] = []
    row_idx = 0
    cumulative_words_prepared = 0

    for phase_i, (seq_len, word_budget) in enumerate(curriculum):
        chunks: list[dict[str, Any]] = []
        phase_words = 0
        phase_rows_start = row_idx
        n_word_boundary_chunks = 0
        n_tail_drops = 0

        while row_idx < len(pre_tok_rows) and cumulative_words_prepared < word_budget:
            row = pre_tok_rows[row_idx]
            row_chunks = chunk_row(row, seq_len, pad_id, args.min_chunk_tokens)
            n_word_boundary_chunks += len(row_chunks)
            if row.n_tokens > seq_len:
                # Check for tail drops
                total_chunked = sum(int(c["attention_mask"].sum().item()) for c in row_chunks)
                if total_chunked < row.n_tokens:
                    n_tail_drops += 1
            chunks.extend(row_chunks)
            cumulative_words_prepared += row.words
            phase_words += row.words
            row_idx += 1

        n_steps = math.ceil(len(chunks) / args.batch_size)
        info = {
            "phase": phase_i,
            "seq_len": seq_len,
            "word_budget": word_budget,
            "chunks": chunks,
            "n_chunks": len(chunks),
            "n_steps": n_steps,
            "phase_words": phase_words,
            "rows_start": phase_rows_start,
            "rows_end": row_idx,
            "n_tail_drops": n_tail_drops,
        }
        phase_infos.append(info)
        print(json.dumps({
            "event": "phase_prepared", "phase": phase_i, "seq_len": seq_len,
            "word_budget": word_budget, "chunks": len(chunks), "steps": n_steps,
            "phase_words": phase_words, "rows": f"{phase_rows_start}-{row_idx}",
            "tail_drops": n_tail_drops,
        }), flush=True)

    total_steps = sum(pi["n_steps"] for pi in phase_infos)
    total_chunks = sum(pi["n_chunks"] for pi in phase_infos)
    geometry_manifest = {
        "event": "geometry_prepared",
        "curriculum": [(sl, wb) for sl, wb in curriculum],
        "max_seq_length": args.max_seq_length,
        "min_chunk_tokens": args.min_chunk_tokens,
        "total_rows": len(examples),
        "actual_words": actual_words,
        "total_chunks": total_chunks,
        "total_steps": total_steps,
        "phase_summary": [
            {
                "phase": pi["phase"],
                "seq_len": pi["seq_len"],
                "word_budget": pi["word_budget"],
                "rows_start": pi["rows_start"],
                "rows_end": pi["rows_end"],
                "chunks": pi["n_chunks"],
                "steps": pi["n_steps"],
                "phase_words": pi["phase_words"],
                "tail_drops": pi["n_tail_drops"],
            }
            for pi in phase_infos
        ],
    }
    (out / "curriculum_geometry_manifest.json").write_text(
        json.dumps(geometry_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(geometry_manifest), flush=True)

    # ── Create model (using trusted base.build_model) ────────────────────────
    reset_rng = lambda s: (random.seed(s), torch.manual_seed(s),
                           torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None)
    reset_rng(args.seed)
    reset_rng(args.init_seed)

    if args.intermediate_size > 0:
        # Direct config for non-integer ffn_mult (e.g., 12×384 with FFN=1280)
        from transformers import DebertaV2Config, DebertaV2ForMaskedLM
        pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]
        max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
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
        model_args = argparse.Namespace(
            hidden_size=args.hidden_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            ffn_mult=args.ffn_mult,
            max_position_embeddings=args.max_position_embeddings,
            max_seq_length=args.max_seq_length,
            position_buckets=args.position_buckets,
            max_relative_positions=args.max_relative_positions,
            deberta_pos_att_type=args.deberta_pos_att_type,
        )
        model = base.build_model(model_args, tokenizer)

    if args.train_rng_seed >= 0:
        reset_rng(args.train_rng_seed)

    param_count = sum(p.numel() for p in model.parameters())
    model.to(device)

    # Verify config matches baseline
    cfg = model.config
    print(json.dumps({
        "event": "model_created", "params": param_count,
        "pos_att_type": cfg.pos_att_type,
        "max_relative_positions": cfg.max_relative_positions,
        "pad_token_id": cfg.pad_token_id,
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
        "relative_attention": cfg.relative_attention,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
    }), flush=True)

    # ── Optimizer and scheduler ──────────────────────────────────────────────
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )
    warmup = max(1, int(total_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=warmup, num_training_steps=total_steps,
    )

    # Masking state (fixed WWM)
    curriculum_state = base.MaskingCurriculumState(
        curriculum=args.masking_curriculum,
        mask_prob_start=args.mask_prob,
        mask_prob_end=args.mask_prob,
        switch_frac=1.0,
    )
    curriculum_state.initialize(vocab_size=len(tokenizer), total_steps=total_steps)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    use_amp = bool(args.amp) and device.type == "cuda"

    print(json.dumps({
        "event": "training_start", "device": str(device), "params": param_count,
        "vocab": len(tokenizer), "total_steps": total_steps, "total_chunks": total_chunks,
        "warmup": warmup, "lr": args.learning_rate, "amp": use_amp,
        "curriculum": [(sl, wb) for sl, wb in curriculum],
        "batch_size": args.batch_size, "micro_batch_size": args.micro_batch_size,
        "accum_steps": accum_steps,
    }), flush=True)

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = 0
    cumulative_words_trained = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict[str, Any]] = []
    next_ckpt = args.checkpoint_words
    log_path = out / "training_log.jsonl"

    model.train()

    with log_path.open("w", encoding="utf-8") as logf:
        for phase_i, pi in enumerate(phase_infos):
            seq_len = pi["seq_len"]
            chunks = pi["chunks"]

            dataset = PhaseChunkDataset(chunks)
            loader = DataLoader(
                dataset, batch_size=args.micro_batch_size, shuffle=False,
                collate_fn=collate_chunks, num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

            print(json.dumps({
                "event": "phase_start", "phase": phase_i, "seq_len": seq_len,
                "chunks": len(chunks), "steps": pi["n_steps"],
            }), flush=True)

            for batch in effective_batch_iterator(loader, accum_steps):
                global_step += 1
                words = int(batch.pop("words").sum().item())

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                word_group = batch["word_group"].to(device, non_blocking=True)

                # Apply masking on full effective batch (same as research)
                curriculum_state.current_step = global_step - 1
                masked_inputs, labels = base.apply_masking_curriculum(
                    input_ids, attention_mask, word_group,
                    tokenizer, curriculum_state, gen,
                )
                n_pred_total = int((labels != -100).sum().item())
                n_candidate = int(attention_mask.bool().sum().item())
                effective_mask_rate = n_pred_total / max(1, n_candidate)
                if n_pred_total <= 0:
                    continue  # Skip empty batches (rare)

                # Microbatch forward/backward with gradient accumulation
                optim.zero_grad(set_to_none=True)
                weighted_loss_sum = 0.0
                batch_rows = input_ids.shape[0]

                for mb_start in range(0, batch_rows, args.micro_batch_size):
                    mb_end = min(mb_start + args.micro_batch_size, batch_rows)
                    sl_labels = labels[mb_start:mb_end]
                    n_pred_i = int((sl_labels != -100).sum().item())
                    if n_pred_i <= 0:
                        continue

                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            out_model = model(
                                input_ids=masked_inputs[mb_start:mb_end],
                                attention_mask=attention_mask[mb_start:mb_end],
                                labels=sl_labels,
                            )
                    else:
                        out_model = model(
                            input_ids=masked_inputs[mb_start:mb_end],
                            attention_mask=attention_mask[mb_start:mb_end],
                            labels=sl_labels,
                        )

                    loss_i = out_model.loss
                    if loss_i is None:
                        raise RuntimeError("model returned no loss")
                    scale = n_pred_i / n_pred_total
                    (loss_i * scale).backward()
                    weighted_loss_sum += float(loss_i.detach().cpu()) * scale
                    del out_model, loss_i

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                sched.step()

                cumulative_words_trained += words
                loss_float = float(weighted_loss_sum)
                loss_values.append(loss_float)

                rec = {
                    "step": global_step,
                    "phase": phase_i,
                    "seq_len": seq_len,
                    "loss": round(loss_float, 4),
                    "lr": float(sched.get_last_lr()[0]),
                    "batch_words": words,
                    "cum_words": cumulative_words_trained,
                    "masked_tokens": n_pred_total,
                    "mask_rate": round(effective_mask_rate, 4),
                    "elapsed_sec": round(time.time() - start_time, 1),
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

                if global_step == 1 or global_step % args.log_every == 0:
                    wps = cumulative_words_trained / max(1, time.time() - start_time)
                    rec["wps"] = round(wps, 0)
                    print(json.dumps({"event": "train_step", **rec}), flush=True)

                # Checkpointing
                while next_ckpt is not None and cumulative_words_trained >= next_ckpt and next_ckpt <= args.max_word_exposure:
                    if next_ckpt % 1_000_000 == 0:
                        name = f"chck_{next_ckpt // 1_000_000}M"
                    else:
                        name = f"chck_{next_ckpt}w"
                    cp = out / "hf_model" / name
                    base.save_hf_checkpoint(model, tokenizer, cp)
                    saved_checkpoints.append({
                        "name": name,
                        "target_word_exposure": next_ckpt,
                        "actual_cumulative_word_exposure": cumulative_words_trained,
                        "global_step": global_step,
                        "phase": phase_i,
                        "seq_len": seq_len,
                        "path": str(cp),
                    })
                    print(json.dumps({
                        "event": "checkpoint", "name": name,
                        "cum_words": cumulative_words_trained,
                        "step": global_step, "phase": phase_i,
                    }), flush=True)
                    next_ckpt += args.checkpoint_words

                del input_ids, attention_mask, word_group, masked_inputs, labels

            print(json.dumps({
                "event": "phase_end", "phase": phase_i, "seq_len": seq_len,
                "cum_words": cumulative_words_trained, "global_step": global_step,
            }), flush=True)

    # ── Save final model and metrics ─────────────────────────────────────────
    base.save_hf_checkpoint(model, tokenizer, out / "hf_model" / "chck_final")

    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words

    metrics = {
        "variant": f"curriculum_adamw_{args.masking_curriculum}",
        "backend": "mlm",
        "model_family": "DebertaV2ForMaskedLM",
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "word_exposure": cumulative_words_trained,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": global_step,
        "effective_batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accum_steps,
        "curriculum": [(sl, wb) for sl, wb in curriculum],
        "phase_summary": [
            {
                "phase": pi["phase"],
                "seq_len": pi["seq_len"],
                "word_budget": pi["word_budget"],
                "chunks": pi["n_chunks"],
                "steps": pi["n_steps"],
                "phase_words": pi["phase_words"],
                "tail_drops": pi["n_tail_drops"],
            }
            for pi in phase_infos
        ],
        "masking_curriculum": args.masking_curriculum,
        "mask_prob": args.mask_prob,
        "learning_rate": args.learning_rate,
        "warmup_fraction": args.warmup_fraction,
        "weight_decay": args.weight_decay,
        "amp": use_amp,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "ffn_mult": args.ffn_mult,
        "seed": args.seed,
        "init_seed": args.init_seed,
        "train_rng_seed": args.train_rng_seed,
        "max_seq_length": args.max_seq_length,
        "saved_checkpoints": saved_checkpoints,
        "source_words_consumed": source_words,
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(json.dumps({
        "event": "done",
        "params": param_count,
        "loss_first": metrics["loss_first"],
        "loss_last": metrics["loss_last"],
        "word_exposure": cumulative_words_trained,
        "total_steps": global_step,
        "total_chunks": total_chunks,
        "checkpoints": [c["name"] for c in saved_checkpoints],
        "curriculum": [(sl, wb) for sl, wb in curriculum],
        "model": str(out / "hf_model" / "chck_final"),
        "metrics": str(out / "scientific_metrics.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
