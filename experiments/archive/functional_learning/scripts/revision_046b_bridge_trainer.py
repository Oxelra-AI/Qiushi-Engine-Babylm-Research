#!/usr/bin/env python3
"""Step046b: controlled legal BabyLM bridge trainer.

Starts from the coherent86 alpha0.75 parent and trains through a legal
word-paced tail overlay that mixes relation-first answer packets with
ordinary corpus material.

Two arms (run as separate invocations):
  ordinary_wwm     : All rows get standard MLM masking (15%, 80/10/10).
  answer_allocation: Relation rows get answer-span masking; ordinary rows get MLM.

Both arms use the same overlay data, same 354 word-paced macro-updates,
same cosine LR schedule (offset from update 101 of 455), and private-adapter-only
optimization. Checkpoints are saved for separate held/Cheap7 evaluation.

The scientific question: can the replicated contextual state-selection operation
be learned as part of cumulative limited-data training without the broad fast-screen
loss seen in the standalone answer-only specialist phase?
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import gc
import json
import math
import os
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(A02_SCRIPTS))
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))

from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM  # noqa: E402
import coherent86_continuation_trainer as base_loader  # noqa: E402

PARENT_PATH = _public_path('models/frontier')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def lr_at_update(update_index0: int, total: int, warmup: int, peak: float) -> float:
    if update_index0 < warmup:
        return peak * update_index0 / max(1, warmup)
    p = (update_index0 - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def mlm_mask_row(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 special_ids: set, mask_id: int, vocab_size: int,
                 rng: random.Random, prob: float = 0.15
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard token-level MLM masking: prob% of non-special tokens, 80/10/10."""
    labels = torch.full_like(input_ids, -100)
    masked = input_ids.clone()
    for j in range(len(input_ids)):
        if attention_mask[j] == 0:
            continue
        if int(input_ids[j]) in special_ids:
            continue
        if rng.random() < prob:
            labels[j] = input_ids[j]
            p = rng.random()
            if p < 0.8:
                masked[j] = mask_id
            elif p < 0.9:
                masked[j] = rng.randint(0, vocab_size - 1)
            # else: keep original (10%)
    return masked, labels


def answer_span_mask_row(input_ids: torch.Tensor,
                         offsets: torch.Tensor,
                         answer_char_span: Sequence[int],
                         mask_id: int
                         ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Mask all tokens covering the answer character span."""
    labels = torch.full_like(input_ids, -100)
    masked = input_ids.clone()
    cs0, cs1 = int(answer_char_span[0]), int(answer_char_span[1])
    n_masked = 0
    for j in range(len(input_ids)):
        s, e = int(offsets[j][0]), int(offsets[j][1])
        if e <= s:
            continue
        if s < cs1 and e > cs0:
            labels[j] = input_ids[j]
            masked[j] = mask_id
            n_masked += 1
    return masked, labels, n_masked


def mask_row(row: Dict[str, Any], input_ids: torch.Tensor,
             attention_mask: torch.Tensor, offsets: torch.Tensor,
             arm: str, special_ids: set, mask_id: int,
             vocab_size: int, rng: random.Random
             ) -> Tuple[torch.Tensor, torch.Tensor, bool, int]:
    """Apply arm-appropriate masking to a single row.
    Returns: (masked_ids, labels, is_relation, n_masked_tokens)
    """
    is_rel = row.get("bridge_kind") == "relation_answer_packet"
    if is_rel and arm == "answer_allocation" and "answer_char_span" in row:
        masked, labels, n = answer_span_mask_row(
            input_ids, offsets, row["answer_char_span"], mask_id)
        if n == 0:
            # answer span was truncated; fall back to MLM
            masked, labels = mlm_mask_row(
                input_ids, attention_mask, special_ids, mask_id, vocab_size, rng)
            n = int((labels != -100).sum().item())
    else:
        masked, labels = mlm_mask_row(
            input_ids, attention_mask, special_ids, mask_id, vocab_size, rng)
        n = int((labels != -100).sum().item())
    return masked, labels, is_rel, n


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_model(device: torch.device, private_scale: float = 0.75):
    """Load coherent86 parent with trusted loader."""
    model, missing, unexpected = base_loader.load_model(
        PARENT_PATH, device, 128, private_scale)
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected[:10]}")
    bad_missing = [k for k in missing if "private_adapter" in k]
    if bad_missing:
        raise RuntimeError(f"Missing private keys: {bad_missing[:10]}")
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    return model


def freeze_to_private_optimizer(model, lr: float, wd: float):
    """Set up private-adapter-only optimizer."""
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    if not private_names:
        raise RuntimeError("No private_adapter parameters found")
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)
    # up projections: zero weight decay; down projections: normal
    private_up_ids = {id(p) for n, p in model.named_parameters()
                      if ".private_adapter.up." in n}
    normal_params = [p for n, p in model.named_parameters()
                     if n in private_names and id(p) not in private_up_ids]
    zero_wd_params = [p for n, p in model.named_parameters()
                      if n in private_names and id(p) in private_up_ids]
    opt = torch.optim.AdamW([
        {"params": normal_params, "weight_decay": float(wd)},
        {"params": zero_wd_params, "weight_decay": 0.0},
    ], lr=float(lr), betas=(0.9, 0.98), eps=1e-6)
    trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    return opt, {"trainable_tensors": trainable, "trainable_params": trainable_params}


def save_checkpoint(model, tokenizer, ckpt_dir: pathlib.Path, metadata: Dict[str, Any]):
    """Save full HF-compatible model checkpoint and metadata."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(ckpt_dir))
    (ckpt_dir / "bridge_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    model.train()


def train(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ---- Seed ----
    random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
    mask_rng = random.Random(args.train_seed + 7777)

    # ---- Load data ----
    t0_data = time.time()
    overlay = load_jsonl(pathlib.Path(args.overlay_jsonl))
    n_rel = sum(1 for r in overlay if r.get("bridge_kind") == "relation_answer_packet")
    n_ord = len(overlay) - n_rel
    total_words = sum(int(r.get("words", len(str(r.get("text", "")).split())))
                      for r in overlay)
    print(json.dumps({
        "event": "data_loaded",
        "rows": len(overlay),
        "relation_rows": n_rel,
        "ordinary_rows": n_ord,
        "total_words": total_words,
        "elapsed": round(time.time() - t0_data, 1),
    }), flush=True)

    # ---- Load model ----
    model = load_model(device, args.private_scale)
    private_params = sum(p.numel() for n, p in model.named_parameters()
                         if ".private_adapter." in n)
    print(json.dumps({
        "event": "model_loaded",
        "class": type(model).__name__,
        "private_params": private_params,
    }), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(PARENT_PATH), local_files_only=True, use_fast=True)
    special_ids = set(tokenizer.all_special_ids)
    mask_id = int(tokenizer.mask_token_id)
    vocab_size = int(tokenizer.vocab_size)

    opt, opt_info = freeze_to_private_optimizer(model, args.lr, args.weight_decay)
    print(json.dumps({"event": "optimizer", **opt_info}), flush=True)

    # Enable gradient checkpointing for memory
    if torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    # ---- Word-paced macro-update loop ----
    cursor = 0
    max_updates = args.max_updates
    update_logs: List[Dict[str, Any]] = []
    checkpoint_paths: List[Dict[str, Any]] = []
    cum_words = 0
    cum_rel_words = 0
    cum_ord_words = 0
    cum_rel_answer_tokens = 0
    cum_ord_mlm_tokens = 0

    model.train()
    t0_train = time.time()

    for update_i in range(max_updates):
        schedule_idx = args.schedule_offset + update_i

        # ---- Collect rows for this macro-update ----
        rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(overlay) and words < args.words_per_update:
            row = overlay[cursor]
            rows.append(row)
            words += int(row.get("words", len(str(row.get("text", "")).split())))
            cursor += 1
        if not rows:
            print(f"data exhausted at update {update_i}", flush=True)
            break

        rel_rows_in_update = sum(1 for r in rows
                                 if r.get("bridge_kind") == "relation_answer_packet")
        ord_rows_in_update = len(rows) - rel_rows_in_update
        rel_words_in_update = sum(
            int(r.get("words", 0)) for r in rows
            if r.get("bridge_kind") == "relation_answer_packet")

        # ---- Set LR ----
        lr = lr_at_update(schedule_idx, args.schedule_total, args.warmup, args.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # ---- Process micro-batches ----
        opt.zero_grad()
        n_micro = max(1, math.ceil(len(rows) / args.micro_batch))
        update_total_loss = 0.0
        update_total_tokens = 0
        update_rel_tokens = 0
        update_ord_tokens = 0
        update_rel_answer_tokens = 0

        for mb_start in range(0, len(rows), args.micro_batch):
            mb_rows = rows[mb_start:mb_start + args.micro_batch]

            # Tokenize and mask each row
            batch_masked: list = []
            batch_attn: list = []
            batch_labels: list = []
            batch_is_rel: list = []

            for row in mb_rows:
                text = str(row.get("text", ""))
                enc = tokenizer(text, max_length=args.seq_length,
                                truncation=True, padding="max_length",
                                return_tensors="pt",
                                return_offsets_mapping=True)
                ids = enc["input_ids"].squeeze(0)
                attn = enc["attention_mask"].squeeze(0)
                offs = enc["offset_mapping"].squeeze(0)

                m, lab, is_r, n_tok = mask_row(
                    row, ids, attn, offs, args.arm,
                    special_ids, mask_id, vocab_size, mask_rng)

                batch_masked.append(m)
                batch_attn.append(attn)
                batch_labels.append(lab)
                batch_is_rel.append(is_r)

                if is_r:
                    update_rel_tokens += n_tok
                    if args.arm == "answer_allocation":
                        update_rel_answer_tokens += n_tok
                else:
                    update_ord_tokens += n_tok

            batch_m = torch.stack(batch_masked).to(device)
            batch_a = torch.stack(batch_attn).to(device)
            batch_l = torch.stack(batch_labels).to(device)

            outputs = model(input_ids=batch_m, attention_mask=batch_a,
                            labels=batch_l)
            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: bad loss at update {update_i} mb {mb_start}", flush=True)
                continue

            scaled_loss = loss / n_micro
            scaled_loss.backward()

            n_batch_tokens = int((batch_l != -100).sum().item())
            update_total_loss += loss.item() * n_batch_tokens
            update_total_tokens += n_batch_tokens

        # ---- Clip and step ----
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.max_grad_norm).item()
        opt.step()

        cum_words += words
        cum_rel_words += rel_words_in_update
        cum_ord_words += words - rel_words_in_update
        cum_rel_answer_tokens += update_rel_answer_tokens
        cum_ord_mlm_tokens += update_ord_tokens

        avg_loss = update_total_loss / max(1, update_total_tokens)
        log = {
            "update": update_i,
            "schedule_idx": schedule_idx,
            "lr": lr,
            "loss": round(avg_loss, 5),
            "grad_norm": round(grad_norm, 5),
            "rows": len(rows),
            "words": words,
            "cum_words": cum_words,
            "total_tokens": update_total_tokens,
            "relation_rows": rel_rows_in_update,
            "relation_words": rel_words_in_update,
            "relation_tokens": update_rel_tokens,
            "relation_answer_tokens": update_rel_answer_tokens,
            "ordinary_rows": ord_rows_in_update,
            "ordinary_tokens": update_ord_tokens,
        }
        update_logs.append(log)

        if (update_i + 1) % 10 == 0 or update_i == 0:
            print(json.dumps({"event": "update", **log}), flush=True)

        # ---- Checkpoint ----
        if ((update_i + 1) % args.checkpoint_every == 0 or
                update_i == max_updates - 1 or
                cursor >= len(overlay)):
            ckpt_name = f"update_{update_i + 1:04d}"
            ckpt_dir = out_dir / "checkpoints" / ckpt_name
            ckpt_meta = {
                "arm": args.arm,
                "update": update_i,
                "schedule_idx": schedule_idx,
                "lr": lr,
                "cum_words": cum_words,
                "cum_rel_answer_tokens": cum_rel_answer_tokens,
                "cum_ord_mlm_tokens": cum_ord_mlm_tokens,
                "loss": round(avg_loss, 5),
                "train_seed": args.train_seed,
            }
            save_checkpoint(model, tokenizer, ckpt_dir, ckpt_meta)
            checkpoint_paths.append({
                "update": update_i + 1,
                "path": rel(ckpt_dir),
                "loss": round(avg_loss, 5),
            })
            print(json.dumps({
                "event": "checkpoint",
                "update": update_i + 1,
                "path": rel(ckpt_dir),
            }), flush=True)

    elapsed = time.time() - t0_train

    # ---- Save summary ----
    summary = {
        "status": "BRIDGE_TRAIN_DONE",
        "arm": args.arm,
        "overlay_jsonl": rel(pathlib.Path(args.overlay_jsonl)),
        "parent_path": rel(PARENT_PATH),
        "train_seed": args.train_seed,
        "private_scale": args.private_scale,
        "schedule_total": args.schedule_total,
        "schedule_offset": args.schedule_offset,
        "lr_peak": args.lr,
        "warmup": args.warmup,
        "micro_batch": args.micro_batch,
        "max_updates": max_updates,
        "completed_updates": len(update_logs),
        "total_words_consumed": cum_words,
        "total_relation_words": cum_rel_words,
        "total_ordinary_words": cum_ord_words,
        "total_relation_answer_tokens": cum_rel_answer_tokens,
        "total_ordinary_mlm_tokens": cum_ord_mlm_tokens,
        "final_loss": round(update_logs[-1]["loss"], 5) if update_logs else None,
        "elapsed_sec": round(elapsed, 1),
        "checkpoints": checkpoint_paths,
        "private_params": private_params,
    }
    (out_dir / "bridge_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # Save update log
    with open(out_dir / "update_log.jsonl", "w") as f:
        for log in update_logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60, flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlay-jsonl", required=True,
                    help="Path to interspersed or frontloaded overlay JSONL")
    ap.add_argument("--arm", required=True,
                    choices=["ordinary_wwm", "answer_allocation"],
                    help="Masking strategy for relation rows")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--max-updates", type=int, default=354)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--schedule-total", type=int, default=455)
    ap.add_argument("--schedule-offset", type=int, default=101)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="Save checkpoint every N updates")
    ap.add_argument("--construction-seed", type=int, default=40040)
    ap.add_argument("--train-seed", type=int, default=46046)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
