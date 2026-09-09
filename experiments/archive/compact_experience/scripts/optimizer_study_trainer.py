#!/usr/bin/env python3
"""research: Compact optimizer-by-data study for BabyLM Strict-Small.

Tests whether LAMB optimizer specifically amplifies the clean-Qwen
same-window paired-data effect vs generic acceleration.

Design (2×3 new arms + 2 existing AdamW-1e3 references from research):
  Optimizers: AdamW, LAMB
  Learning rates: 1e-3 (both), 7e-3 (LAMB leader), 5e-3 (AdamW high)
  Data: official-only, Qwen-aligned

All arms share:
  - DeBERTa-v2 8×480, 16k tokenizer, fixed seq256, pure 15% WWM-MLM
  - Batch128, seed43022 (init), train_rng_seed 43023
  - Cosine warmup 5%, weight_decay 0.01
  - Checkpoints at every 1M words

Logs per-step: loss, lr, grad_norm, trust_ratios_by_layer (LAMB only),
  update_norms_by_layer, pair_row_loss vs non_pair_row_loss.

"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)


# ═══════════ LAMB Optimizer ═══════════

class LAMB(Optimizer):
    """LAMB: Layer-wise Adaptive Moments (You et al., ICLR 2020).
    
    Layer-wise trust ratio scales updates per parameter tensor.
    Excludes bias, LayerNorm, and embeddings from layer adaptation.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01, exclude_from_layer_adaptation=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._exclude = exclude_from_layer_adaptation or set()
        super().__init__(params, defaults)
        self._trust_ratios: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._trust_ratios.clear()
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
                if pname in self._exclude or p.dim() <= 1:
                    trust = 1.0
                else:
                    w_norm = p.norm(2).item()
                    u_norm = update.norm(2).item()
                    trust = (w_norm / u_norm) if (w_norm > 0 and u_norm > 0) else 1.0
                if pname:
                    self._trust_ratios[pname] = trust
                p.add_(update, alpha=-lr * trust)
        return loss

    def get_trust_ratios(self) -> dict[str, float]:
        return dict(self._trust_ratios)


# ═══════════ Data ═══════════

@dataclass
class Example:
    text: str
    words: int
    example_id: int = -1
    source: str = ""
    is_pair_row: bool = False


def load_examples_jsonl(path: Path, selected_words: int) -> tuple[list[Example], int, int]:
    """Load JSONL corpus with word budget and pair-row detection.
    
    Stops at the last complete row that fits within selected_words.
    For production runs (20M, 100M), the JSONL is designed to hit exact boundaries.
    """
    examples: list[Example] = []
    total_rows = 0
    selected = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total_rows += 1
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"Word mismatch row {total_rows}: field={words} actual={actual}")
            if selected + words > selected_words:
                break
            source = str(obj.get("source", ""))
            is_pair = bool(obj.get("pair_id") or obj.get("is_pair_row") or "qwen_pair" in source)
            examples.append(Example(text=text, words=words, example_id=total_rows - 1,
                                    source=source, is_pair_row=is_pair))
            selected += words
            if selected >= selected_words:
                break
    if selected != selected_words and abs(selected - selected_words) > 160:
        raise RuntimeError(f"Selected {selected} words vs target {selected_words} (gap > 160)")
    return examples, total_rows, selected




def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════ Dataset ═══════════

BASELINE_TOKENIZER_REPO = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict"


def make_portable_tokenizer(tokenizer_path: str = "") -> PreTrainedTokenizerFast:
    if tokenizer_path:
        base = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    else:
        base = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_REPO, revision="main", use_fast=True)
    backend = getattr(base, "_tokenizer", None) or getattr(base, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError("No fast tokenizer backend found")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=base.bos_token or "<s>",
        eos_token=base.eos_token or "</s>",
        unk_token=base.unk_token or "<unk>",
        pad_token=base.pad_token or "<pad>",
        mask_token=getattr(base, "mask_token", None) or "<mask>",
        model_max_length=getattr(base, "model_max_length", 1024) or 1024,
    )
    tok.init_kwargs["tokenizer_class"] = "PreTrainedTokenizerFast"
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})
    return tok


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


class MaskedChunkDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer, seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start = {}

    def _word_start_flag(self, tid: int) -> bool:
        v = self._word_start.get(tid)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start[tid] = v
        return v

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
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
                "word_group": group, "words": ex.words, "is_pair": int(ex.is_pair_row)}


def collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "is_pair": torch.tensor([x["is_pair"] for x in batch], dtype=torch.long),
    }


# ═══════════ Masking (fixed 15% WWM) ═══════════

def apply_wwm(input_ids, attention_mask, word_group, tokenizer, mask_prob, gen):
    """Standard whole-word masking with 80/10/10 replacement."""
    device = input_ids.device
    labels = input_ids.clone()
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id

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

    masked_indices = (labels != -100)
    masked_inputs = input_ids.clone()
    replace_mask = torch.bernoulli(
        torch.full(input_ids.shape, 0.8, device=device), generator=gen).bool() & masked_indices
    masked_inputs[replace_mask] = mask_token_id
    rand_mask = torch.bernoulli(
        torch.full(input_ids.shape, 0.5, device=device), generator=gen).bool() & masked_indices & ~replace_mask
    rand_ids = torch.randint(0, len(tokenizer), (int(rand_mask.sum()),), generator=gen, device=device)
    masked_inputs[rand_mask] = rand_ids
    return masked_inputs, labels


# ═══════════ Model ═══════════

def build_model(args, tokenizer):
    """DeBERTa-v2 8×480 matching clean-Qwen config exactly."""
    config = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=512,
        relative_attention=True,
        position_buckets=256,
        max_relative_positions=256,
        pos_att_type=["p2c", "c2p"],
        type_vocab_size=0,
    )
    return DebertaV2ForMaskedLM(config)


# ═══════════ Training ═══════════

def get_param_groups(model, args):
    """Set up parameter groups with names for LAMB trust tracking."""
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    params_decay = []
    params_nodecay = []
    exclude_names = set()
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param._lamb_name = name
        if any(nd in name for nd in no_decay):
            params_nodecay.append(param)
            exclude_names.add(name)
        else:
            params_decay.append(param)
    
    return [
        {"params": params_decay, "weight_decay": args.weight_decay},
        {"params": params_nodecay, "weight_decay": 0.0},
    ], exclude_names


def compute_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total ** 0.5


def layer_update_norms(optimizer) -> dict[str, float]:
    """Compute per-layer-group update norms (sampled for efficiency)."""
    norms = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            name = getattr(p, "_lamb_name", "")
            if not name:
                continue
            # Group by layer prefix
            parts = name.split(".")
            if "layer" in parts:
                idx = parts.index("layer")
                if idx + 1 < len(parts):
                    layer_key = f"layer_{parts[idx+1]}"
                else:
                    layer_key = "other"
            elif "embeddings" in name:
                layer_key = "embeddings"
            elif "cls" in name or "lm_head" in name:
                layer_key = "head"
            else:
                layer_key = "other"
            
            gnorm = p.grad.norm(2).item()
            if layer_key in norms:
                norms[layer_key] = max(norms[layer_key], gnorm)
            else:
                norms[layer_key] = gnorm
    return norms


def main():
    p = argparse.ArgumentParser(description="research Optimizer-by-Data Study")
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--selected_words", type=int, required=True)
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--tokenizer_label", default="baseline16k")
    
    # Optimizer
    p.add_argument("--optimizer", choices=["adamw", "lamb"], default="adamw")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.98)
    p.add_argument("--eps", type=float, default=1e-8)
    
    # Model
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    
    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    
    # Reproducibility
    p.add_argument("--extra_init_seed", type=int, default=43022)
    p.add_argument("--train_rng_seed", type=int, default=43023)
    p.add_argument("--seed", type=int, default=43)
    
    args = p.parse_args()
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # ── Load data ──
    jsonl_path = Path(args.example_jsonl)
    examples, total_rows, selected = load_examples_jsonl(jsonl_path, args.selected_words)
    n_pair_rows = sum(1 for e in examples if e.is_pair_row)
    pair_words = sum(e.words for e in examples if e.is_pair_row)
    
    print(json.dumps({"event": "data_loaded", "total_rows": total_rows,
                      "selected_words": selected, "n_examples": len(examples),
                      "n_pair_rows": n_pair_rows, "pair_words": pair_words,
                      "jsonl": str(jsonl_path)}), flush=True)
    
    # ── Tokenizer ──
    tokenizer = make_portable_tokenizer(args.tokenizer_path)
    
    # ── Dataset ──
    dataset = MaskedChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())
    total_steps = len(loader)
    
    # ── Model ──
    reset_rng = lambda s: (random.seed(s), torch.manual_seed(s),
                           torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None)
    reset_rng(args.seed)
    reset_rng(args.extra_init_seed)
    model = build_model(args, tokenizer)
    reset_rng(args.train_rng_seed)
    
    param_count = sum(p.numel() for p in model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # ── Optimizer ──
    param_groups, exclude_names = get_param_groups(model, args)
    
    if args.optimizer == "lamb":
        optim = LAMB(param_groups, lr=args.learning_rate,
                     betas=(args.beta1, args.beta2), eps=args.eps,
                     weight_decay=args.weight_decay,
                     exclude_from_layer_adaptation=exclude_names)
    else:
        optim = torch.optim.AdamW(param_groups, lr=args.learning_rate,
                                   betas=(args.beta1, args.beta2), eps=args.eps)
    
    warmup = max(1, int(total_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup,
                                            num_training_steps=total_steps)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)
    
    print(json.dumps({"event": "start", "optimizer": args.optimizer,
                      "learning_rate": args.learning_rate, "beta2": args.beta2,
                      "eps": args.eps, "batch_size": args.batch_size,
                      "total_steps": total_steps, "warmup_steps": warmup,
                      "param_count": param_count, "vocab_size": len(tokenizer),
                      "selected_words": args.selected_words,
                      "output_dir": args.output_dir}), flush=True)
    
    # ── Training loop ──
    cumulative_words = 0
    next_ckpt = args.checkpoint_words
    saved_checkpoints = []
    pair_loss_accum = 0.0
    pair_count = 0
    nonpair_loss_accum = 0.0
    nonpair_count = 0
    
    model.train()
    for step, batch in enumerate(loader, 1):
        words = int(batch["words"].sum().item())
        is_pair = batch["is_pair"].to(device)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        word_group = batch["word_group"].to(device, non_blocking=True)
        
        masked_inputs, labels = apply_wwm(
            input_ids, attention_mask, word_group, tokenizer, args.mask_prob, gen)
        
        outputs = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Per-row loss for pair vs non-pair tracking
        if step % args.log_every == 0 or step <= 5:
            with torch.no_grad():
                logits = outputs.logits
                bsz = labels.shape[0]
                for b in range(bsz):
                    mask = labels[b] != -100
                    if mask.sum() == 0:
                        continue
                    row_loss = F.cross_entropy(logits[b][mask], labels[b][mask]).item()
                    if is_pair[b]:
                        pair_loss_accum += row_loss
                        pair_count += 1
                    else:
                        nonpair_loss_accum += row_loss
                        nonpair_count += 1
        
        loss.backward()
        
        # Gradient norm before step
        grad_norm = compute_grad_norm(model)
        
        optim.step()
        sched.step()
        optim.zero_grad()
        
        cumulative_words += words
        
        # Logging
        if step % args.log_every == 0 or step <= 3:
            log_entry = {
                "event": "train", "step": step, "loss": loss.item(),
                "lr": sched.get_last_lr()[0],
                "grad_norm": round(grad_norm, 4),
                "batch_words": words,
                "cumulative_word_exposure": cumulative_words,
                "masked_tokens": int((labels != -100).sum().item()),
            }
            if pair_count > 0:
                log_entry["pair_loss"] = round(pair_loss_accum / pair_count, 4)
                log_entry["nonpair_loss"] = round(nonpair_loss_accum / nonpair_count, 4) if nonpair_count > 0 else None
                log_entry["pair_rows_in_batch"] = int(is_pair.sum().item())
                pair_loss_accum = 0.0
                pair_count = 0
                nonpair_loss_accum = 0.0
                nonpair_count = 0
            
            if args.optimizer == "lamb":
                ratios = optim.get_trust_ratios()
                # Summarize trust ratios by layer group
                layer_ratios = {}
                for name, ratio in ratios.items():
                    parts = name.split(".")
                    if "layer" in parts:
                        idx = parts.index("layer")
                        if idx + 1 < len(parts):
                            key = f"L{parts[idx+1]}"
                        else:
                            key = "other"
                    elif "embeddings" in name:
                        key = "emb"
                    else:
                        key = "head"
                    if key not in layer_ratios:
                        layer_ratios[key] = []
                    layer_ratios[key].append(ratio)
                trust_summary = {k: round(sum(v)/len(v), 4) for k, v in sorted(layer_ratios.items())}
                log_entry["trust_ratios"] = trust_summary
            
            log_entry["elapsed_sec"] = round(time.time() - start_time, 1)
            print(json.dumps(log_entry), flush=True)
        
        # Checkpointing
        while next_ckpt is not None and cumulative_words >= next_ckpt:
            ckpt_name = f"chck_{next_ckpt // 1_000_000}M"
            ckpt_dir = out / "hf_model" / ckpt_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            saved_checkpoints.append({
                "name": ckpt_name, "step": step,
                "cumulative_words": cumulative_words,
                "loss": loss.item(),
            })
            print(json.dumps({"event": "checkpoint", "name": ckpt_name,
                              "cum_words": cumulative_words}), flush=True)
            if next_ckpt < 10_000_000:
                next_ckpt += args.checkpoint_words
            else:
                next_ckpt += 10_000_000
            if next_ckpt > args.selected_words:
                next_ckpt = None
    
    # ── Final summary ──
    elapsed = time.time() - start_time
    summary = {
        "event": "done",
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "beta2": args.beta2,
        "total_steps": total_steps,
        "cumulative_words": cumulative_words,
        "loss_last": loss.item(),
        "param_count": param_count,
        "checkpoints": [c["name"] for c in saved_checkpoints],
        "n_checkpoints": len(saved_checkpoints),
        "elapsed_sec": round(elapsed, 1),
        "output_dir": args.output_dir,
    }
    print(json.dumps(summary), flush=True)
    
    # Save scientific metrics
    (out / "scientific_metrics.json").write_text(json.dumps({
        "variant": f"optimizer_study_{args.optimizer}_lr{args.learning_rate}",
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "eps": args.eps,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "mask_prob": args.mask_prob,
        "hidden_size": args.hidden_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "param_count": param_count,
        "vocab_size": len(tokenizer),
        "tokenizer_label": args.tokenizer_label,
        "selected_words": args.selected_words,
        "total_steps": total_steps,
        "warmup_steps": warmup,
        "cumulative_words": cumulative_words,
        "loss_last": loss.item(),
        "n_pair_rows": n_pair_rows,
        "pair_words": pair_words,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "saved_checkpoints": saved_checkpoints,
        "elapsed_sec": round(elapsed, 1),
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": sha256_file(jsonl_path),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
