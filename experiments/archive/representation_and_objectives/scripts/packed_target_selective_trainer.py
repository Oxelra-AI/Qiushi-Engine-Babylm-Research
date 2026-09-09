#!/usr/bin/env python3
"""research: Target-selective trainer in the historical packed geometry.

Replicates the exact compact-view-reinvest DeBERTa training (same model,
tokenizer, optimizer, scheduler, seeds, masking, checkpointing) but with
selective loss zeroing for designated target categories.

Modes:
  full                      - standard training (reference parity check)
  drop_abs_content          - zero loss on source-absent content rewrite words
  drop_copied_content_wholeword - zero loss on matched copied content words

Uses research annotation to classify each word group in the packed stream.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, PreTrainedTokenizerFast,
    DebertaV2Config, DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

# ─── Constants ────────────────────────────────────────────────────────────────
CAT_FILLER = "filler"
CAT_SOURCE = "source"
CAT_RW_COPIED = "rw_copied"
CAT_RW_ABS_CONTENT = "rw_abs_content"
CAT_RW_ABS_OTHER = "rw_abs_other"

CAT_TO_IDX = {
    CAT_FILLER: 0, CAT_SOURCE: 1, CAT_RW_COPIED: 2,
    CAT_RW_ABS_CONTENT: 3, CAT_RW_ABS_OTHER: 4,
}
IDX_TO_CAT = {v: k for k, v in CAT_TO_IDX.items()}

MODES = ["full", "drop_abs_content", "drop_copied_content_wholeword"]


# ─── Data utilities ───────────────────────────────────────────────────────────
def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class AnnotatedChunkDataset(Dataset):
    """Like the base MaskedChunkDataset but includes per-word-group category."""

    def __init__(self, examples, tokenizer, seq_length, annotations,
                 drop_positions=None):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.annotations = annotations  # dict: example_id -> list[int] (cat indices)
        self.drop_positions = drop_positions or {}  # (example_id, word_idx) -> True
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start = {}

    def _word_start_flag(self, token_id: int) -> bool:
        v = self._word_start.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start[token_id] = v
        return v

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["text"], add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Build word groups
        group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start_flag(tid) or i == 0:
                gid += 1
            group[i] = gid

        # Build per-token category from annotation
        word_cats = self.annotations.get(ex["example_id"])
        token_cat = torch.zeros_like(input_ids)  # default: filler (0)
        if word_cats is not None:
            for i in range(input_ids.shape[0]):
                g = group[i].item()
                if g >= 0 and g < len(word_cats):
                    token_cat[i] = word_cats[g]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "token_cat": token_cat,
            "words": ex["words"],
            "example_id": ex["example_id"],
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [x[k] for x in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = torch.tensor(vals, dtype=torch.long)
    return out


# ─── Masking (WWM fixed, same as base) ───────────────────────────────────────
def apply_wwm_masking(input_ids, attention_mask, word_group, tokenizer, mask_prob, gen):
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)

    for b in range(bsz):
        groups = word_group[b]
        valid_groups = torch.unique(groups[groups >= 0])
        if valid_groups.numel() == 0:
            continue
        gp = torch.rand(valid_groups.numel(), generator=gen, device=device)
        chosen = valid_groups[gp < mask_prob]
        if chosen.numel() == 0:
            continue
        sel_b = torch.isin(groups, chosen) & candidate[b]
        select[b] = sel_b

    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True

    labels[~select] = -100
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),),
                                 generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids
    return masked_inputs, labels


# ─── Target-selective label filtering ─────────────────────────────────────────
def apply_target_filter(labels, token_cat, word_group, example_ids,
                        mode, drop_positions):
    """Zero out labels for the designated target category."""
    if mode == "full":
        return labels, {}

    bsz, seq = labels.shape
    counts = collections.Counter()
    filtered = labels.clone()

    for b in range(bsz):
        eid = example_ids[b].item()
        for i in range(seq):
            if filtered[b, i] == -100:
                continue
            cat_idx = token_cat[b, i].item()
            cat = IDX_TO_CAT.get(cat_idx, CAT_FILLER)

            if mode == "drop_abs_content":
                if cat == CAT_RW_ABS_CONTENT:
                    filtered[b, i] = -100
                    counts["dropped_abs_content"] += 1
                else:
                    counts[f"kept_{cat}"] += 1
            elif mode == "drop_copied_content_wholeword":
                gid = word_group[b, i].item()
                if gid >= 0 and (eid, gid) in drop_positions:
                    filtered[b, i] = -100
                    counts["dropped_copied_wholeword"] += 1
                else:
                    counts[f"kept_{cat}"] += 1
    return filtered, counts


# ─── Model builder (same as base + activation checkpointing) ─────────────────
def build_model(args, tokenizer):
    max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
    pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=max_pos,
        max_relative_positions=args.max_relative_positions,
        position_buckets=args.position_buckets,
        relative_attention=True,
        pos_att_type=pos_att_type,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = DebertaV2ForMaskedLM(cfg)
    # Activation checkpointing (same as research)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model


def make_portable_tokenizer(tokenizer_path):
    base = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=base.bos_token or "<s>",
        eos_token=base.eos_token or "</s>",
        unk_token=base.unk_token or "<unk>",
        pad_token=base.pad_token or "<pad>",
        mask_token=getattr(base, "mask_token", None) or "<mask>",
        model_max_length=getattr(base, "model_max_length", 1024) or 1024,
    )
    tok.init_kwargs.pop("tokenizer_class", None)
    tok.init_kwargs["tokenizer_class"] = "PreTrainedTokenizerFast"
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
    return tok


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", required=True, help="100M stream JSONL")
    p.add_argument("--annotation", required=True, help="Pool annotation JSONL")
    p.add_argument("--selection", default="", help="Whole-word selection JSONL for drop_copied mode")
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_mode", required=True, choices=MODES)
    p.add_argument("--max_word_exposure", type=int, default=100_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--lr_total_steps", type=int, default=2529)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──
    tokenizer = make_portable_tokenizer(args.tokenizer_path)

    # ── Load annotations ──
    print("Loading annotations...", flush=True)
    annotations = {}
    with open(args.annotation) as f:
        for line in f:
            a = json.loads(line)
            # Convert string categories to int indices
            annotations[a["example_id"]] = [CAT_TO_IDX.get(c, 0) for c in a["word_categories"]]

    # ── Load whole-word selection for drop_copied mode ──
    drop_positions = {}
    if args.target_mode == "drop_copied_content_wholeword" and args.selection:
        # Build (example_id, word_index) set from selection
        # First need a row_index → example_id mapping
        ann_by_row = {}
        with open(args.annotation) as f:
            for line in f:
                a = json.loads(line)
                ann_by_row[a["row_index"]] = a["example_id"]
        with open(args.selection) as f:
            for line in f:
                g = json.loads(line)
                eid = ann_by_row.get(g["row_index"])
                if eid is not None:
                    drop_positions[(eid, g["word_index"])] = True
        print(f"Loaded {len(drop_positions)} drop positions for mode {args.target_mode}", flush=True)

    # ── Load stream data ──
    print("Loading stream...", flush=True)
    selected_words = args.max_word_exposure
    if args.smoke:
        selected_words = min(500_000, selected_words)
    examples = []
    total_words = 0
    with open(args.stream) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if total_words + words > selected_words:
                break
            examples.append({
                "text": text, "words": words,
                "example_id": int(obj.get("example_id", len(examples))),
            })
            total_words += words

    print(f"Loaded {len(examples)} examples, {total_words} words", flush=True)

    # ── Dataset and DataLoader ──
    dataset = AnnotatedChunkDataset(examples, tokenizer, args.max_seq_length,
                                     annotations, drop_positions)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    total_steps = len(loader)
    print(f"Total steps: {total_steps}", flush=True)

    # ── Model ──
    def reset_all_rng(s):
        random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    reset_all_rng(args.seed)
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device(args.device)
    model.to(device)

    # Record init SHA
    init_sha = hashlib.sha256()
    for p_tensor in model.parameters():
        init_sha.update(p_tensor.data.cpu().numpy().tobytes())
    init_sha_hex = init_sha.hexdigest()
    print(f"Model: {param_count} params on {device}; init_sha={init_sha_hex}", flush=True)

    # ── Optimizer ──
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay, betas=(0.9, 0.98))
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup,
                                            num_training_steps=schedule_total)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    # ── Training loop ──
    cumulative_words = 0
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    saved_checkpoints = []
    stratum_loss_accum = collections.Counter()
    stratum_count_accum = collections.Counter()
    total_dropped = collections.Counter()

    model.train()
    for step, batch in enumerate(loader, 1):
        words = int(batch.pop("words").sum().item())
        example_ids = batch.pop("example_id")
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        word_group = batch["word_group"].to(device, non_blocking=True)
        token_cat = batch["token_cat"].to(device, non_blocking=True)

        # Apply WWM masking (same as historical)
        masked_inputs, labels = apply_wwm_masking(
            input_ids, attention_mask, word_group, tokenizer,
            args.mask_prob, gen
        )

        # Apply target-selective filter
        filtered_labels, drop_counts = apply_target_filter(
            labels, token_cat, word_group, example_ids,
            args.target_mode, drop_positions
        )
        for k, v in drop_counts.items():
            total_dropped[k] += v

        # Per-stratum loss tracking (before filtering, on masked positions)
        with torch.no_grad():
            mask_pos = labels != -100
            for cat_idx, cat_name in IDX_TO_CAT.items():
                cat_mask = mask_pos & (token_cat == cat_idx)
                if cat_mask.any():
                    stratum_count_accum[cat_name] += int(cat_mask.sum().item())

        # Forward with filtered labels
        optim.zero_grad(set_to_none=True)
        out_model = model(input_ids=masked_inputs, attention_mask=attention_mask,
                          labels=filtered_labels)
        loss = out_model.loss
        if loss is None:
            raise RuntimeError("model returned no loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        cumulative_words += words

        # Logging
        if step % args.log_every == 0 or step == total_steps:
            kept = int((filtered_labels != -100).sum().item())
            lr = sched.get_last_lr()[0]
            log_entry = {
                "e": "t", "mode": args.target_mode, "s": step,
                "l": round(loss.item(), 5), "c": cumulative_words,
                "lr": lr, "kept": kept,
            }
            print(json.dumps(log_entry, ensure_ascii=False), flush=True)

        # Checkpointing
        if next_ckpt is not None and cumulative_words >= next_ckpt:
            ckpt_name = f"chck_{next_ckpt // 1_000_000}M"
            ckpt_path = out / "hf_model" / ckpt_name
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            saved_checkpoints.append({
                "name": ckpt_name, "target": next_ckpt,
                "actual": cumulative_words, "step": step,
                "path": str(ckpt_path),
            })
            print(json.dumps({"e": "ckpt", "mode": args.target_mode,
                              "n": ckpt_name, "c": cumulative_words}), flush=True)
            next_ckpt += args.checkpoint_words

        if args.smoke and step >= 3:
            break

    # Save metrics
    metrics = {
        "status": f"PACKED_TARGET_SELECTIVE_{args.target_mode.upper()}",
        "target_mode": args.target_mode,
        "parameter_count": param_count,
        "init_sha": init_sha_hex,
        "word_exposure": cumulative_words,
        "actual_training_steps": step,
        "loss_last": round(loss.item(), 6) if 'loss' in dir() else None,
        "stratum_masked_counts": dict(stratum_count_accum),
        "total_dropped_counts": dict(total_dropped),
        "saved_checkpoints": saved_checkpoints,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
