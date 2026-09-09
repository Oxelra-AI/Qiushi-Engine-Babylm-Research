#!/usr/bin/env python3
"""research Phase 2 SOTA trainer: DeBERTa-v2 12×384, LAMB, 40k tokenizer, masking schedule.

Implements the leader's proven recipe with our data advantage:
- Architecture: DeBERTa-v2 12×384, intermediate 1280, 12 heads
- Tokenizer: 40k SentencePiece BPE
- Optimizer: LAMB with LR 0.007, cosine schedule
- Sequence length curriculum: 64→128→256
- Masking schedule: WWM for epochs 1-7 (fraction 0.7), token for epochs 8-10 (fraction 0.3)
- Data: mix_25pct (75% official + 25% aligned) or official-only for control
- Initialization: explicit extra_init_seed for reproducibility

Usage (direct SOTA attempt):
  python phase2_sota_trainer.py \
    --example_jsonl <data_path> \
    --output_dir <run_dir> \
    --tokenizer_path experiments/archive/compact_experience/data/tokenizer/hf_tokenizer_40k \
    --tokenizer_label sp40k_mix25 \
    --model_type deberta_v2 \
    --n_layer 12 --hidden_size 384 --n_head 12 \
    --intermediate_size 1280 \
    --share_att_key true --position_biased_input false \
    --optimizer lamb --learning_rate 0.007 \
    --max_seq_length 256 --batch_size 256 \
    --seq_len_schedule "0.0:64,0.3:128,0.6:256" \
    --mask_mode wwm --mask_switch_mode token --mask_switch_fraction 0.7 \
    --mask_prob 0.15 \
    --max_word_exposure 100000000 \
    --warmup_fraction 0.06 --weight_decay 0.01 \
    --seed 43 --extra_init_seed 43100 --train_rng_seed 43101 \
    --num_workers 0 --checkpoint_words 10000000 --log_every 50
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, PreTrainedTokenizerFast,
    DebertaV2Config, DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

# ========== LAMB Optimizer ==========

class LAMB(Optimizer):
    """LAMB: Layer-wise Adaptive Moments for Batch training (You et al., ICLR 2020)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01, exclude_from_layer_adaptation=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._exclude = exclude_from_layer_adaptation or set()
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                update = (exp_avg / bc1) / ((exp_avg_sq / bc2).sqrt() + eps)
                if wd > 0:
                    update = update.add(p, alpha=wd)
                # Layer-wise trust ratio
                pname = getattr(p, "_lamb_name", "")
                if pname in self._exclude:
                    trust = 1.0
                else:
                    w_norm = p.norm(2).item()
                    u_norm = update.norm(2).item()
                    trust = (w_norm / u_norm) if (w_norm > 0 and u_norm > 0) else 1.0
                p.add_(update, alpha=-lr * trust)
        return loss


# ========== Data ==========

@dataclass
class Example:
    text: str
    words: int
    example_id: int = -1
    source: str = ""


def load_examples_jsonl(path: Path, max_words: int) -> list[Example]:
    """Load examples from JSONL until max_words is reached."""
    examples = []
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row["text"]
            words = int(row.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"Word count mismatch: field={words} actual={actual}")
            if total + words > max_words:
                break
            examples.append(Example(
                text=text, words=words,
                example_id=int(row.get("example_id", len(examples))),
                source=row.get("source", ""),
            ))
            total += words
    if total != max_words:
        raise RuntimeError(f"Loaded {total} words but needed {max_words}")
    return examples


def is_word_start(token_str: str) -> bool:
    """Detect word-start tokens in SentencePiece (▁) or GPT-2 (Ġ) tokenizers."""
    return token_str.startswith("▁") or token_str.startswith("Ġ")


class MaskedChunkDataset(Dataset):
    """Tokenizes examples and precomputes word-start groups for WWM."""

    def __init__(self, examples: list[Example], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start = {}

    def _word_start_flag(self, token_id: int) -> bool:
        v = self._word_start.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start[token_id] = v
        return v

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text, add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
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
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "word_group": group, "words": ex.words}


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
    }


# ========== Masking ==========

def apply_masking(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  word_group: torch.Tensor, tokenizer, mask_mode: str,
                  mask_prob: float, gen: torch.Generator):
    """Return (masked_input_ids, labels). labels=-100 where not predicted."""
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id

    if mask_mode == "wwm":
        # Whole-word masking: select word groups, then mask all their tokens
        for b in range(bsz):
            max_gid = int(word_group[b].max().item())
            if max_gid < 0:
                labels[b, :] = -100
                continue
            n_groups = max_gid + 1
            group_mask = torch.rand(n_groups, generator=gen, device=device) < mask_prob
            token_mask = torch.zeros(seq, dtype=torch.bool, device=device)
            for g in range(n_groups):
                if group_mask[g]:
                    token_mask |= (word_group[b] == g)
            labels[b, ~token_mask] = -100
    else:
        # Token-level masking
        prob_matrix = torch.full((bsz, seq), mask_prob, device=device)
        prob_matrix[attention_mask == 0] = 0.0
        # Don't mask special tokens
        for sid in tokenizer.all_special_ids:
            prob_matrix[input_ids == sid] = 0.0
        token_mask = torch.bernoulli(prob_matrix, generator=gen).bool()
        labels[~token_mask] = -100

    # Standard BERT masking: 80% mask, 10% random, 10% keep
    masked_indices = (labels != -100)
    masked_inputs = input_ids.clone()

    # 80% → [MASK]
    replace_mask = torch.bernoulli(
        torch.full(input_ids.shape, 0.8, device=device), generator=gen).bool() & masked_indices
    masked_inputs[replace_mask] = mask_token_id

    # 10% → random token
    rand_mask = torch.bernoulli(
        torch.full(input_ids.shape, 0.5, device=device), generator=gen).bool() & masked_indices & ~replace_mask
    rand_ids = torch.randint(0, len(tokenizer), (int(rand_mask.sum()),), generator=gen, device=device)
    masked_inputs[rand_mask] = rand_ids

    return masked_inputs, labels


# ========== Model ==========

def build_model(args, tokenizer):
    """Build DeBERTa-v2 with Phase 2 / leader-matching config."""
    max_pos = max(args.max_position_embeddings, args.max_seq_length + 8)
    intermediate = args.intermediate_size if args.intermediate_size > 0 else args.hidden_size * args.ffn_mult
    pos_att_type = [x.strip() for x in args.deberta_pos_att_type.split(",") if x.strip()]

    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=intermediate,
        max_position_embeddings=max_pos,
        max_relative_positions=args.max_relative_positions,
        position_buckets=args.position_buckets,
        relative_attention=(args.deberta_relative_attention.lower() == "true"),
        pos_att_type=pos_att_type,
        share_att_key=(args.share_att_key.lower() == "true"),
        position_biased_input=(args.position_biased_input.lower() == "true"),
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id or 0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


def make_lamb_optimizer(model, lr=0.007, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6):
    """Create LAMB with standard bias/LayerNorm exclusions."""
    decay_params, no_decay_params = [], []
    excluded = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param._lamb_name = name
        if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            no_decay_params.append(param)
            excluded.add(name)
        else:
            decay_params.append(param)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return LAMB(groups, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay, exclude_from_layer_adaptation=excluded)


# ========== Utilities ==========

def reset_all_rng(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seq_length_for_progress(frac: float, schedule: list[tuple[float, int]], default_len: int) -> int:
    length = default_len
    for thresh, L in schedule:
        if frac >= thresh:
            length = L
    return length


def save_hf_checkpoint(model, tokenizer, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    # Force portable config
    tok_cfg = dst / "tokenizer_config.json"
    if tok_cfg.exists():
        cfg = json.loads(tok_cfg.read_text(encoding="utf-8"))
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.setdefault("bos_token", "<s>")
        cfg.setdefault("eos_token", "</s>")
        cfg.setdefault("unk_token", "<unk>")
        cfg.setdefault("pad_token", "<pad>")
        cfg.setdefault("mask_token", "<mask>")
        tok_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_portable_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """Load tokenizer as a portable fast tokenizer.
    
    Handles both PreTrainedTokenizerFast (baseline16k) and DebertaV2Tokenizer
    (SentencePiece-based 40k) by attempting fast conversion first.
    """
    base = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    
    if backend is not None:
        # Fast tokenizer available - wrap it portably
        tok = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token=base.bos_token if base.bos_token else "<s>",
            eos_token=base.eos_token if base.eos_token else "</s>",
            unk_token=base.unk_token if base.unk_token else "<unk>",
            pad_token=base.pad_token if base.pad_token else "<pad>",
            mask_token=getattr(base, "mask_token", None) or "<mask>",
            model_max_length=1024,
        )
        tok.init_kwargs.pop("tokenizer_class", None)
        tok.init_kwargs["tokenizer_class"] = "PreTrainedTokenizerFast"
    else:
        # Slow tokenizer (SentencePiece) - use directly
        tok = base
        # Ensure essential special tokens
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": "<pad>"})
        if tok.mask_token is None:
            tok.add_special_tokens({"mask_token": "<mask>"})
        tok.model_max_length = 1024
        return tok
    
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
    return tok


# ========== Main ==========

def build_args():
    p = argparse.ArgumentParser(description="Phase 2 SOTA trainer")
    # Data
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=100_000_000)
    p.add_argument("--example_pool_words", type=int, default=10_000_000)
    # Tokenizer
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--tokenizer_label", default="sp40k")
    # Architecture
    p.add_argument("--model_type", default="deberta_v2")
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--hidden_size", type=int, default=384)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--intermediate_size", type=int, default=1280)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--max_relative_positions", type=int, default=-1)
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--deberta_relative_attention", default="true")
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--share_att_key", default="true")
    p.add_argument("--position_biased_input", default="false")
    # Training
    p.add_argument("--optimizer", choices=["adam", "lamb"], default="lamb")
    p.add_argument("--learning_rate", type=float, default=0.007)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.06)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--seq_len_schedule", default="0.0:64,0.3:128,0.6:256")
    # Masking
    p.add_argument("--mask_mode", default="wwm", choices=["wwm", "token"])
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--mask_switch_mode", default="token", choices=["wwm", "token", "none"],
                   help="Mask mode to switch TO after mask_switch_fraction")
    p.add_argument("--mask_switch_fraction", type=float, default=0.7,
                   help="Progress fraction at which to switch mask mode (0.7 = epoch 7/10)")
    # RNG
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra_init_seed", type=int, default=-1,
                   help="If >=0, reset RNG before model init for reproducible weights")
    p.add_argument("--train_rng_seed", type=int, default=-1,
                   help="If >=0, reset RNG after model init for reproducible training order")
    # Logging
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_words", type=int, default=10_000_000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--lr_total_steps", type=int, default=0,
                   help="Override schedule total steps (0=auto from data)")
    return p.parse_args()


def main():
    args = build_args()
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(json.dumps({"event": "loading_tokenizer", "path": args.tokenizer_path}), flush=True)
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    print(json.dumps({"event": "tokenizer_loaded", "vocab_size": len(tokenizer),
                      "mask_token_id": tokenizer.mask_token_id,
                      "pad_token_id": tokenizer.pad_token_id}), flush=True)

    # Load data
    print(json.dumps({"event": "loading_data", "path": args.example_jsonl,
                      "max_words": args.max_word_exposure}), flush=True)
    examples = load_examples_jsonl(Path(args.example_jsonl), args.max_word_exposure)
    actual_words = sum(ex.words for ex in examples)
    print(json.dumps({"event": "data_loaded", "examples": len(examples),
                      "words": actual_words}), flush=True)

    # Build dataset
    reset_all_rng(args.seed)
    dataset = MaskedChunkDataset(examples, tokenizer, args.max_seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())

    # Build model with controlled initialization
    if args.extra_init_seed >= 0:
        reset_all_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    param_count = sum(p.numel() for p in model.parameters())
    emb_params = model.get_input_embeddings().weight.numel()

    # Reset training RNG after model init
    if args.train_rng_seed >= 0:
        reset_all_rng(args.train_rng_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create optimizer
    if args.optimizer == "lamb":
        optim = make_lamb_optimizer(model, lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.98))

    total_steps = len(loader)
    schedule_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup,
                                            num_training_steps=schedule_total)

    # Parse sequence length schedule
    seq_schedule = []
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(","):
            t, L = part.split(":")
            seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()

    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)

    # Masking schedule config
    mask_switch_step = int(total_steps * args.mask_switch_fraction) if args.mask_switch_mode != "none" else total_steps + 1

    print(json.dumps({
        "event": "training_start",
        "param_count": param_count,
        "embedding_params": emb_params,
        "non_embedding_params": param_count - emb_params,
        "vocab_size": len(tokenizer),
        "total_steps": total_steps,
        "optimizer": args.optimizer,
        "lr": args.learning_rate,
        "warmup_steps": warmup,
        "mask_mode_initial": args.mask_mode,
        "mask_switch_mode": args.mask_switch_mode,
        "mask_switch_step": mask_switch_step,
        "seq_schedule": args.seq_len_schedule,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "batch_size": args.batch_size,
    }), flush=True)

    # Training loop
    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values = []
    saved_checkpoints = []
    source_words = {}
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None

    model.train()
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch["words"].sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)

            # Determine current sequence length
            frac = (step - 1) / max(1, schedule_total)
            cur_len = seq_length_for_progress(frac, seq_schedule, args.max_seq_length) if seq_schedule else args.max_seq_length
            cur_len = min(cur_len, args.max_seq_length)

            # Truncate to current seq length
            input_ids = input_ids[:, :cur_len].contiguous()
            attention_mask = attention_mask[:, :cur_len].contiguous()
            word_group = word_group[:, :cur_len].contiguous()

            # Determine current mask mode (masking schedule)
            current_mask_mode = args.mask_mode
            if step > mask_switch_step and args.mask_switch_mode != "none":
                current_mask_mode = args.mask_switch_mode

            # Apply masking
            masked_inputs, labels = apply_masking(
                input_ids, attention_mask, word_group, tokenizer,
                current_mask_mode, args.mask_prob, gen)

            # Forward + backward
            optim.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            loss = out_model.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            cumulative_words += words
            loss_float = float(loss.detach().cpu())
            loss_values.append(loss_float)
            n_pred = int((labels != -100).sum().item())

            # Track source words
            # (source tracking from examples is complex; track total per epoch)

            rec = {"step": step, "loss": loss_float, "lr": float(sched.get_last_lr()[0]),
                   "batch_words": words, "cumulative_word_exposure": cumulative_words,
                   "seq_len": cur_len, "masked_tokens": n_pred,
                   "mask_mode": current_mask_mode,
                   "elapsed_sec": round(time.time() - start_time, 1)}
            logf.write(json.dumps(rec) + "\n")

            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)

            # Checkpoint saving
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                name = f"chck_{next_ckpt // 1_000_000}M"
                cp = out / "hf_model" / name
                save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({
                    "name": name, "target_word_exposure": next_ckpt,
                    "actual_cumulative_word_exposure": cumulative_words,
                    "path": str(cp)
                })
                print(json.dumps({"event": "checkpoint", "name": name,
                                   "words": cumulative_words}), flush=True)
                next_ckpt += args.checkpoint_words

    # Save final model (also at top-level hf_model/)
    save_hf_checkpoint(model, tokenizer, out / "hf_model")

    # Save scientific metrics
    metrics = {
        "variant": f"masked_{args.mask_mode}_to_{args.mask_switch_mode}",
        "backend": "mlm",
        "model_family": "DebertaV2ForMaskedLM",
        "model_type": args.model_type,
        "parameter_count": param_count,
        "embedding_parameter_count": emb_params,
        "non_embedding_parameter_count": param_count - emb_params,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": args.tokenizer_path,
        "word_exposure": cumulative_words,
        "example_pool_words_actual": args.example_pool_words,
        "selected_for_training_words": actual_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "mask_mode": args.mask_mode,
        "mask_switch_mode": args.mask_switch_mode,
        "mask_switch_fraction": args.mask_switch_fraction,
        "mask_switch_step": mask_switch_step,
        "mask_prob": args.mask_prob,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "warmup_fraction": args.warmup_fraction,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "seq_len_schedule": args.seq_len_schedule,
        "n_layer": args.n_layer,
        "hidden_size": args.hidden_size,
        "n_head": args.n_head,
        "intermediate_size": args.intermediate_size if args.intermediate_size > 0 else args.hidden_size * args.ffn_mult,
        "share_att_key": args.share_att_key,
        "position_biased_input": args.position_biased_input,
        "deberta_relative_attention": args.deberta_relative_attention,
        "deberta_pos_att_type": args.deberta_pos_att_type,
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "saved_checkpoints": saved_checkpoints,
        "elapsed_sec": round(time.time() - start_time, 1),
    }
    (out / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"event": "done", "param_count": param_count,
                      "loss_first": metrics["loss_first"],
                      "loss_last": metrics["loss_last"],
                      "word_exposure": cumulative_words,
                      "elapsed_sec": metrics["elapsed_sec"],
                      "checkpoints": [c["name"] for c in saved_checkpoints]}), flush=True)


if __name__ == "__main__":
    main()
