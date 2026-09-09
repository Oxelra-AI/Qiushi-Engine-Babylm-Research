#!/usr/bin/env python3
"""research: bounded tail-rephase restart trainer for the legal tokenizer.

Loads the research legal chck_80M weights (discarding optimizer state), creates a
fresh AdamW optimizer with configurable peak LR, and continues for 20M words on
the frozen 10M reinvest pool with fresh shuffling.

Two variants are supported:
  --lr_mode matched   : COMPACT_EXPERIENCE-style, continuation_lr ~ 0.000109 (cosine-matched
                         to original schedule's 80% point), no warmup
  --lr_mode base      : peak_lr = BASE_LR (0.001) with warmup_fraction 0.06

Both use cosine decay to zero and save checkpoints at 85M, 90M, 95M, 100M.
Scientific purpose: determine whether fresh optimizer state (and optionally higher
LR) in the 80-100M tail can recover legal-tokenizer competence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

# === Constants ===
SEQ_LENGTH = 256
MASK_PROB = 0.15
BASE_LR = 0.001
WEIGHT_DECAY = 0.01
ORIGINAL_TOTAL_STEPS = 2529  # Legal reference run's actual steps
WARMUP_FRACTION = 0.06
PARENT_EXPOSURE = 80_000_000
CONTINUATION_BUDGET = 20_000_000
CHECKPOINT_INTERVAL = 5_000_000  # Save at 85M, 90M, 95M, 100M

LEGAL_INIT = None  # Set from args
POOL_FILE = None   # Set from args
POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ContinuationDataset(Dataset):
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


def wwm_mask_batch(input_ids: torch.Tensor, tokenizer, mask_prob: float,
                   gen: torch.Generator, device: torch.device):
    batch_size, seq_len = input_ids.shape
    special_ids = set(tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id
    masked_inputs = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    masked_words = 0
    masked_tokens = 0
    for i in range(batch_size):
        word_groups: list[list[int]] = []
        current: list[int] = []
        for j in range(seq_len):
            tid = int(input_ids[i, j].item())
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
        if not word_groups:
            continue
        n_mask = min(len(word_groups), max(1, int(len(word_groups) * mask_prob)))
        for wi in torch.randperm(len(word_groups), generator=gen)[:n_mask].tolist():
            masked_words += 1
            for pos in word_groups[wi]:
                masked_tokens += 1
                labels[i, pos] = input_ids[i, pos]
                r = torch.rand(1, generator=gen).item()
                if r < 0.8:
                    masked_inputs[i, pos] = mask_id
                elif r < 0.9:
                    masked_inputs[i, pos] = torch.randint(len(tokenizer), (1,), generator=gen).item()
    return masked_inputs.to(device), attention_mask.to(device), labels.to(device), {
        "masked_words": masked_words, "masked_tokens": masked_tokens
    }


def compute_matched_continuation_lr() -> float:
    """COMPACT_EXPERIENCE-style: approximate the original schedule's LR at the 80% point."""
    warmup = int(ORIGINAL_TOTAL_STEPS * WARMUP_FRACTION)
    progress_80 = (ORIGINAL_TOTAL_STEPS * 0.8 - warmup) / (ORIGINAL_TOTAL_STEPS - warmup)
    factor = 0.5 * (1 + math.cos(math.pi * progress_80))
    return BASE_LR * factor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_checkpoint", required=True)
    ap.add_argument("--pool_file", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--lr_mode", required=True, choices=["matched", "base"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--train_rng_seed", type=int, default=43044)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--verify_pool_hash", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_root = run_dir / "hf_model"
    model_root.mkdir(parents=True, exist_ok=True)
    init_path = Path(args.init_checkpoint)
    pool_path = Path(args.pool_file)

    if not init_path.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {init_path}")
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    if args.verify_pool_hash:
        actual = sha256_file(pool_path)
        if actual != POOL_SHA:
            raise RuntimeError(f"Pool SHA mismatch: {actual} != {POOL_SHA}")
        print(json.dumps({"event": "pool_hash_ok", "sha256": actual}), flush=True)

    # Compute LR
    if args.lr_mode == "matched":
        peak_lr = compute_matched_continuation_lr()
        warmup_steps = 0
        label = "rephase_matched_lr"
    else:
        peak_lr = BASE_LR
        warmup_steps = -1  # Computed after schedule
        label = "rephase_base_lr"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({
        "event": "start", "label": label, "lr_mode": args.lr_mode,
        "peak_lr": peak_lr, "init": str(init_path), "pool": str(pool_path),
        "device": str(device), "started_utc": now(),
    }, indent=2), flush=True)

    # Load model (no optimizer state)
    model = DebertaV2ForMaskedLM.from_pretrained(str(init_path))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(init_path), use_fast=True)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,} params; vocab={len(tokenizer)}", flush=True)

    # Load and tokenize pool
    examples: list[dict] = []
    pool_words = 0
    with pool_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            actual_w = len(str(obj["text"]).split())
            words = int(obj.get("words", actual_w))
            examples.append(obj)
            pool_words += words
    print(f"Pool: {len(examples)} rows, {pool_words} words", flush=True)

    dataset = ContinuationDataset(examples, tokenizer, SEQ_LENGTH)
    steps_per_pass = math.ceil(len(dataset) / args.batch_size)
    passes_needed = math.ceil(CONTINUATION_BUDGET / pool_words)
    total_steps = steps_per_pass * passes_needed

    if warmup_steps == -1:
        warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))

    print(json.dumps({
        "event": "schedule", "pool_rows": len(examples), "pool_words": pool_words,
        "dataset_chunks": len(dataset), "steps_per_pass": steps_per_pass,
        "passes_needed": passes_needed, "total_steps": total_steps,
        "peak_lr": peak_lr, "warmup_steps": warmup_steps,
    }, indent=2), flush=True)

    # Fresh optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                              weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)
    
    random.seed(args.train_rng_seed)
    torch.manual_seed(args.train_rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_rng_seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.train_rng_seed)

    model.train()
    cumulative_words = 0
    processed_chunks = 0
    global_step = 0
    next_ckpt = CHECKPOINT_INTERVAL
    loss_values: list[float] = []
    saved_checkpoints: list[dict] = []
    log_path = run_dir / "training_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    def save_checkpoint(target_continuation_words: int, loss_val: float) -> None:
        total_words = PARENT_EXPOSURE + cumulative_words
        name = f"chck_{int(round(total_words / 1_000_000))}M"
        ckpt_path = model_root / name
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_path, safe_serialization=True)
        tokenizer.save_pretrained(ckpt_path)
        rec = {
            "name": name, "path": str(ckpt_path), "step": global_step,
            "continuation_words": cumulative_words,
            "total_word_exposure": total_words,
            "loss": round(loss_val, 6),
        }
        saved_checkpoints.append(rec)
        print(f"  >> Saved {name} step={global_step} total_words={total_words}", flush=True)

    stop = False
    for pass_i in range(passes_needed):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        pos = 0
        while pos < len(indices):
            batch_idx: list[int] = []
            while pos < len(indices) and len(batch_idx) < args.batch_size:
                projected = int(round(((processed_chunks + len(batch_idx) + 1) / max(1, len(dataset))) * pool_words))
                if projected > CONTINUATION_BUDGET:
                    break
                batch_idx.append(indices[pos])
                pos += 1
            if not batch_idx:
                stop = True
                break

            input_ids = torch.stack([dataset[i] for i in batch_idx])
            masked_inputs, attn_mask, labels, mask_stats = wwm_mask_batch(
                input_ids, tokenizer, MASK_PROB, gen, device)
            
            outputs = model(input_ids=masked_inputs, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            sched.step()
            optim.zero_grad()

            global_step += 1
            batch_words = int(round((len(batch_idx) / max(1, len(dataset))) * pool_words))
            cumulative_words += batch_words
            processed_chunks += len(batch_idx)
            loss_val = loss.item()
            loss_values.append(loss_val)

            if global_step % args.log_every == 0 or global_step <= 5:
                lr_now = sched.get_last_lr()[0]
                rec = {
                    "step": global_step, "loss": round(loss_val, 6),
                    "lr": round(lr_now, 10), "batch_words": batch_words,
                    "continuation_words": cumulative_words,
                    "total_word_exposure": PARENT_EXPOSURE + cumulative_words,
                    "masked_tokens": mask_stats["masked_tokens"],
                    "masked_words": mask_stats["masked_words"],
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

            # Checkpoint at 5M intervals
            if cumulative_words >= next_ckpt:
                recent_loss = sum(loss_values[-50:]) / len(loss_values[-50:])
                save_checkpoint(next_ckpt, recent_loss)
                next_ckpt += CHECKPOINT_INTERVAL

        if stop:
            break

    # Final checkpoint at 100M
    if cumulative_words > 0:
        recent_loss = sum(loss_values[-50:]) / len(loss_values[-50:])
        total_words = PARENT_EXPOSURE + cumulative_words
        if total_words >= 99_000_000:
            save_checkpoint(CONTINUATION_BUDGET, recent_loss)

    log_f.close()

    # Write scientific metrics
    metrics: dict[str, Any] = {
        "status": "REPHASE_RESTART_DONE",
        "label": label,
        "lr_mode": args.lr_mode,
        "peak_lr": peak_lr,
        "warmup_steps": warmup_steps,
        "total_steps": global_step,
        "parent_exposure": PARENT_EXPOSURE,
        "continuation_words": cumulative_words,
        "total_word_exposure": PARENT_EXPOSURE + cumulative_words,
        "parameter_count": param_count,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "loss_mean_last50": sum(loss_values[-50:]) / len(loss_values[-50:]) if loss_values else None,
        "saved_checkpoints": saved_checkpoints,
        "train_rng_seed": args.train_rng_seed,
        "finished_utc": now(),
    }
    metrics_path = run_dir / "scientific_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "done", **{k: v for k, v in metrics.items() if k != "saved_checkpoints"}}, indent=2), flush=True)
    print(f"\nCheckpoints saved: {[c['name'] for c in saved_checkpoints]}", flush=True)


if __name__ == "__main__":
    main()
