#!/usr/bin/env python3
"""research: Continuation trainer for cluster mechanism test (80M→100M).

Loads a DeBERTa-v2 checkpoint (default: chck_80M from clean-Qwen seed43022),
then continues training for 20M word exposure with a specified data pool.
Uses the same masking/tokenization mechanics as the original trainer but with:
  - Model loaded from pretrained checkpoint (not random init)
  - Fresh optimizer, LR schedule simulating the tail of the original cosine
  - No warmup (model already trained through 80% of schedule)
  - Checkpoints at 5M intervals (chck_85M, chck_90M, chck_95M, chck_100M)

Usage:
  python continuation_trainer.py \
    --init_checkpoint path/to/chck_80M \
    --train_file path/to/continuation_10M_E1_true_cluster.jsonl \
    --run_dir path/to/output \
    --label E1_true_cluster \
    --gpu 0
"""
from __future__ import annotations

import argparse
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
    AutoTokenizer,
    DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

# Constants matching original training
SEQ_LENGTH = 256
MASK_PROB = 0.15
ORIGINAL_TOTAL_STEPS = 2515  # from clean-Qwen 100M training
WARMUP_FRACTION = 0.06
BASE_LR = 0.001
WEIGHT_DECAY = 0.01
CONTINUATION_START_EXPOSURE = 80_000_000
CONTINUATION_BUDGET = 20_000_000
CHECKPOINT_INTERVAL = 5_000_000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Data utilities ───────────────────────────────────────────────────────────
class ContinuationDataset(Dataset):
    """Tokenized chunked dataset matching original MaskedChunkDataset logic."""
    
    def __init__(self, examples: list[dict], tokenizer, seq_length: int):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        
        # Tokenize all examples and chunk into seq_length sequences
        self.chunks: list[torch.Tensor] = []
        buffer: list[int] = []
        
        for ex in examples:
            ids = tokenizer.encode(ex["text"], add_special_tokens=False)
            buffer.extend(ids)
            while len(buffer) >= seq_length:
                self.chunks.append(torch.tensor(buffer[:seq_length], dtype=torch.long))
                buffer = buffer[seq_length:]
        
        # Don't discard the last partial chunk if substantial
        if len(buffer) > seq_length // 4:
            padded = buffer + [tokenizer.pad_token_id] * (seq_length - len(buffer))
            self.chunks.append(torch.tensor(padded[:seq_length], dtype=torch.long))
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def wwm_mask_batch(input_ids: torch.Tensor, tokenizer, mask_prob: float,
                   gen: torch.Generator, device: torch.device):
    """Whole-word masking matching original trainer."""
    batch_size, seq_len = input_ids.shape
    special_ids = set(tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id
    
    masked_inputs = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    for i in range(batch_size):
        # Build word groups
        word_groups: list[list[int]] = []
        current_group: list[int] = []
        
        for j in range(seq_len):
            tid = input_ids[i, j].item()
            if tid in special_ids or tid == tokenizer.pad_token_id:
                if current_group:
                    word_groups.append(current_group)
                    current_group = []
                continue
            
            token_str = tokenizer.convert_ids_to_tokens(tid)
            if token_str and is_word_start(str(token_str)) and current_group:
                word_groups.append(current_group)
                current_group = []
            current_group.append(j)
        
        if current_group:
            word_groups.append(current_group)
        
        # Select words to mask
        n_mask = max(1, int(len(word_groups) * mask_prob))
        if not word_groups:
            continue
            
        perm = torch.randperm(len(word_groups), generator=gen)
        selected_words = perm[:n_mask].tolist()
        
        for wi in selected_words:
            for pos in word_groups[wi]:
                labels[i, pos] = input_ids[i, pos]
                # 80% mask, 10% random, 10% keep
                r = torch.rand(1, generator=gen).item()
                if r < 0.8:
                    masked_inputs[i, pos] = mask_id
                elif r < 0.9:
                    masked_inputs[i, pos] = torch.randint(
                        len(tokenizer), (1,), generator=gen
                    ).item()
    
    return masked_inputs.to(device), attention_mask.to(device), labels.to(device)


def save_checkpoint(model, tokenizer, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_checkpoint", required=True,
                    help="Path to HF model dir (e.g., chck_80M)")
    ap.add_argument("--train_file", required=True,
                    help="JSONL pool for continuation (10M words)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_word_exposure", type=int, default=CONTINUATION_BUDGET)
    ap.add_argument("--checkpoint_interval", type=int, default=CHECKPOINT_INTERVAL)
    ap.add_argument("--train_rng_seed", type=int, default=43044)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    import os
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
    print(json.dumps({"event": "start", "label": args.label, "init": str(init_path),
                      "train_file": str(train_path), "device": str(device),
                      "started_utc": now()}, indent=2), flush=True)

    # === Load model from checkpoint ===
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_path))
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,} parameters from {init_path.name}")

    # === Load tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(str(init_path), use_fast=True)
    print(f"Tokenizer: vocab_size={len(tokenizer)}")

    # === Load training data ===
    examples: list[dict] = []
    pool_words = 0
    with train_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(obj)
            pool_words += obj.get("words", len(obj["text"].split()))
    
    print(f"Pool: {len(examples)} rows, {pool_words} words")

    # === Build dataset ===
    dataset = ContinuationDataset(examples, tokenizer, SEQ_LENGTH)
    print(f"Dataset chunks: {len(dataset)}")

    # Compute steps
    total_chunks_per_pass = len(dataset)
    words_per_step = (args.batch_size * SEQ_LENGTH) / (total_chunks_per_pass * SEQ_LENGTH / pool_words)
    # More accurate: words_per_step ≈ pool_words / ceil(total_chunks / batch_size)
    steps_per_pass = math.ceil(total_chunks_per_pass / args.batch_size)
    passes_needed = math.ceil(args.max_word_exposure / pool_words)
    total_steps = steps_per_pass * passes_needed
    approx_words_per_step = pool_words / steps_per_pass

    print(f"Steps per pass: {steps_per_pass}, passes: {passes_needed}, total steps: {total_steps}")
    print(f"Approx words per step: {approx_words_per_step:.0f}")

    # === Optimizer with simulated tail schedule ===
    # The original training used cosine with 6% warmup over 2515 steps.
    # At 80M (step ~2012), we're deep in the cosine tail.
    # Strategy: fresh optimizer, cosine schedule over total_steps, but start from
    # a lower base LR that matches where the original cosine was at research.
    # Original LR at research: 0.001 * 0.5 * (1 + cos(π * (2012-151)/(2515-151)))
    warmup_steps_orig = int(ORIGINAL_TOTAL_STEPS * WARMUP_FRACTION)
    post_warmup_progress = (ORIGINAL_TOTAL_STEPS * 0.8 - warmup_steps_orig) / (ORIGINAL_TOTAL_STEPS - warmup_steps_orig)
    tail_lr_factor = 0.5 * (1 + math.cos(math.pi * post_warmup_progress))
    continuation_lr = BASE_LR * tail_lr_factor
    print(f"Continuation LR: {continuation_lr:.6f} (tail factor: {tail_lr_factor:.4f})")

    optim = torch.optim.AdamW(
        model.parameters(), lr=continuation_lr,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98)
    )
    # Short cosine decay from continuation_lr to ~0 over the continuation
    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=0, num_training_steps=total_steps
    )

    # === RNG ===
    random.seed(args.train_rng_seed)
    torch.manual_seed(args.train_rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_rng_seed)
    gen = torch.Generator(device='cpu')
    gen.manual_seed(args.train_rng_seed)

    # === Training loop ===
    model.train()
    cumulative_words = 0
    global_step = 0
    loss_values: list[float] = []
    saved_checkpoints: list[dict] = []
    next_ckpt = args.checkpoint_interval

    log_path = run_dir / "training_log.jsonl"
    log_f = log_path.open("w")

    for pass_i in range(passes_needed):
        # Shuffle dataset order each pass
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        batch_buffer: list[torch.Tensor] = []
        
        for chunk_idx in indices:
            batch_buffer.append(dataset[chunk_idx])
            
            if len(batch_buffer) == args.batch_size:
                input_ids = torch.stack(batch_buffer)
                batch_buffer = []
                
                masked_inputs, attention_mask, labels = wwm_mask_batch(
                    input_ids, tokenizer, MASK_PROB, gen, device
                )
                
                optim.zero_grad(set_to_none=True)
                out = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                
                if loss is None:
                    raise RuntimeError("Model returned no loss")
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                
                global_step += 1
                loss_val = loss.item()
                loss_values.append(loss_val)
                cumulative_words += int(approx_words_per_step)
                
                if global_step % args.log_every == 0:
                    lr_now = sched.get_last_lr()[0]
                    entry = {
                        "step": global_step, "loss": round(loss_val, 4),
                        "lr": round(lr_now, 8), "words": cumulative_words,
                        "pass": pass_i + 1,
                    }
                    log_f.write(json.dumps(entry) + "\n")
                    log_f.flush()
                    print(f"  step {global_step}/{total_steps} loss={loss_val:.4f} lr={lr_now:.2e} words={cumulative_words}", flush=True)
                
                # Checkpoint
                if next_ckpt and cumulative_words >= next_ckpt:
                    ckpt_name = f"chck_{(CONTINUATION_START_EXPOSURE + cumulative_words) // 1_000_000}M"
                    ckpt_path = run_dir / "hf_model" / ckpt_name
                    save_checkpoint(model, tokenizer, ckpt_path)
                    saved_checkpoints.append({
                        "name": ckpt_name, "path": str(ckpt_path),
                        "step": global_step, "words": cumulative_words,
                        "loss": round(loss_val, 4),
                    })
                    print(f"  >> Saved {ckpt_name} at step {global_step}", flush=True)
                    next_ckpt += args.checkpoint_interval
                
                if cumulative_words >= args.max_word_exposure:
                    break
        
        if cumulative_words >= args.max_word_exposure:
            break

    log_f.close()

    # Final checkpoint
    final_ckpt_name = f"chck_{(CONTINUATION_START_EXPOSURE + cumulative_words) // 1_000_000}M"
    final_path = run_dir / "hf_model" / final_ckpt_name
    if not final_path.exists():
        save_checkpoint(model, tokenizer, final_path)
        saved_checkpoints.append({
            "name": final_ckpt_name, "path": str(final_path),
            "step": global_step, "words": cumulative_words,
            "loss": round(loss_values[-1] if loss_values else 0, 4),
        })

    # === Save metrics ===
    metrics = {
        "variant": f"continuation_{args.label}",
        "init_checkpoint": str(init_path),
        "train_file": str(train_path),
        "parameter_count": param_count,
        "vocab_size": len(tokenizer),
        "continuation_lr": continuation_lr,
        "batch_size": args.batch_size,
        "seq_length": SEQ_LENGTH,
        "mask_prob": MASK_PROB,
        "word_exposure": cumulative_words,
        "actual_training_steps": global_step,
        "pool_words": pool_words,
        "pool_rows": len(examples),
        "passes_completed": passes_needed,
        "loss_first": round(loss_values[0], 4) if loss_values else None,
        "loss_last": round(loss_values[-1], 4) if loss_values else None,
        "loss_mean": round(sum(loss_values) / len(loss_values), 4) if loss_values else None,
        "saved_checkpoints": saved_checkpoints,
        "started_utc": now(),
        "train_rng_seed": args.train_rng_seed,
    }
    
    metrics_path = run_dir / "scientific_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    
    print(json.dumps({
        "event": "finished", "label": args.label,
        "steps": global_step, "words": cumulative_words,
        "loss_last": metrics["loss_last"],
        "checkpoints": len(saved_checkpoints),
        "metrics": str(metrics_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
