#!/usr/bin/env python3
"""research: Answer-only adapter training on recombination entity-binding rows.

Implements the joint principle: uniquely sufficient variable (entity identity
determines the answer in minimal pairs) + concentrated credit (only answer
tokens receive loss; source+update+query visible).

Training:
  - Load base43022 with trusted adapter-aware loading
  - Freeze all base parameters; train only adapter parameters
  - For each row: tokenize full context, mask answer span tokens, compute
    cross-entropy loss ONLY on answer positions (labels=-100 elsewhere)
  - Evaluate every N steps on held-out: per-type accuracy + joint binding

Evaluation:
  - For each held-out row: mask answer tokens, compute log-probability of
    correct answer vs wrong state (foil)
  - Joint accuracy: fraction of DISTRACTOR pairs where both halves are correct
  - Per-type accuracy: DISTRACTOR unchanged_entity, DISTRACTOR updated_entity,
    UPDATED updated_entity

Usage:
    python answer_only_binding_train.py --epochs 10 [--smoke 4] [--plan-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

# ---------- HF cache setup BEFORE transformers import ----------
ROOT = pathlib.Path.cwd()
OUT_DEFAULT = ROOT / "experiments/archive/relation_learning/data/binding_train"
CACHE_BASE = OUT_DEFAULT / "hf_cache"
for _sub in ["hf_home", "hf_home/hub", "transformers", "modules", "datasets", "tmp"]:
    (CACHE_BASE / _sub).mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_BASE / "hf_home")
os.environ["HF_HUB_CACHE"] = str(CACHE_BASE / "hf_home/hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_BASE / "hf_home/hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_BASE / "transformers")
os.environ["HF_MODULES_CACHE"] = str(CACHE_BASE / "modules")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_BASE / "datasets")
os.environ["TMPDIR"] = str(CACHE_BASE / "tmp")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ---------- Paths ----------
TRAIN_ROWS = ROOT / "experiments/archive/relation_learning/data/recombination_rows/recombination_train.jsonl"
HELDOUT_ROWS = ROOT / "experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl"
BINDING_PAIRS = ROOT / "experiments/archive/relation_learning/data/recombination_rows/binding_pairs_heldout.jsonl"
BASE_43022_CHCK = ROOT / "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try: return str(p.relative_to(ROOT))
    except: return str(p)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def trusted_load_model(model_path: pathlib.Path, device: torch.device) -> tuple[Any, Any, dict]:
    """Load adapter-scaled model with trusted loading, return model, tokenizer, identity."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    total = sum(p.numel() for p in model.parameters())
    adapter_p = sum(p.numel() for n, p in model.named_parameters() if "adapter" in n.lower())
    identity = {
        "checkpoint_path": rel(model_path),
        "loaded_class": type(model).__module__ + "." + type(model).__qualname__,
        "total_params_loaded": total,
        "adapter_params_loaded": adapter_p,
        "adapter_scale_config": getattr(model.config, "adapter_scale", None),
        "trust_remote_code": True,
    }
    model = model.to(device)
    return model, tokenizer, identity


def freeze_base_train_adapter(model) -> dict:
    """Freeze all non-adapter parameters; set adapter parameters to requires_grad=True."""
    total = 0; adapter = 0; frozen = 0
    for name, param in model.named_parameters():
        total += 1
        if "adapter" in name.lower():
            param.requires_grad = True
            adapter += 1
        else:
            param.requires_grad = False
            frozen += 1
    return {"total_params": total, "adapter_trainable": adapter, "frozen": frozen}


def tokenize_row(tokenizer, row: dict, max_length: int = 256) -> dict | None:
    """Tokenize a recombination row and identify answer token positions."""
    context = row["context_text"]
    ans_start = row["answer_char_start"]
    ans_end = row["answer_char_end"]
    
    enc = tokenizer(context, add_special_tokens=True, truncation=True,
                     max_length=max_length, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    
    # Find token positions that overlap the answer span
    answer_positions = []
    for i, (a, b) in enumerate(offsets):
        if a == b: continue  # special tokens
        if b > ans_start and a < ans_end:
            answer_positions.append(i)
    
    if not answer_positions:
        return None
    
    # Create labels: -100 everywhere except answer positions
    labels = [-100] * len(ids)
    for pos in answer_positions:
        labels[pos] = ids[pos]
    
    return {
        "input_ids": ids,
        "labels": labels,
        "answer_positions": answer_positions,
        "row_id": row["row_id"],
        "pair_id": row["pair_id"],
        "pair_half": row.get("pair_half", ""),
        "packet_type": row.get("packet_type", ""),
        "role": row.get("role", ""),
        "answer_kind": row.get("answer_kind", ""),
    }


def collate_batch(batch: list[dict], pad_id: int) -> dict:
    """Collate a batch of tokenized rows into padded tensors."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, b in enumerate(batch):
        seq = b["input_ids"]
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1
        lab = b["labels"]
        labels[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def evaluate_heldout(model, tokenizer, heldout_toks: list[dict], binding_pairs: list[dict],
                     mask_token_id: int, pad_id: int,
                     device: torch.device, batch_size: int = 32) -> dict:
    """Evaluate held-out: answer loss per row and joint binding accuracy.
    
    Uses simultaneous masking (all answer tokens masked at once) for speed.
    For each row: compute cross-entropy loss on answer tokens.
    For binding pairs: check if model predicts top-1 correctly at each position.
    """
    model.eval()
    
    row_results = []
    for tok in heldout_toks:
        ids = tok["input_ids"]
        positions = tok["answer_positions"]
        if not positions:
            continue
        
        # Mask all answer positions simultaneously
        masked_ids = list(ids)
        for pos in positions:
            masked_ids[pos] = mask_token_id
        
        input_t = torch.tensor([masked_ids], dtype=torch.long, device=device)
        att_t = torch.ones(1, len(masked_ids), dtype=torch.long, device=device)
        logits = model(input_ids=input_t, attention_mask=att_t).logits.float()
        
        # Per-position accuracy and loss
        total_loss = 0.0
        n_correct = 0
        for pos in positions:
            target = ids[pos]
            logp = torch.nn.functional.log_softmax(logits[0, pos], dim=-1)
            total_loss -= logp[target].item()
            if logits[0, pos].argmax().item() == target:
                n_correct += 1
        
        mean_loss = total_loss / len(positions)
        token_acc = n_correct / len(positions)
        
        row_results.append({
            "row_id": tok["row_id"],
            "pair_id": tok["pair_id"],
            "pair_half": tok["pair_half"],
            "role": tok["role"],
            "packet_type": tok["packet_type"],
            "answer_kind": tok["answer_kind"],
            "mean_nll": mean_loss,
            "token_accuracy": token_acc,
            "n_answer_tokens": len(positions),
            "all_tokens_correct": int(n_correct == len(positions)),
        })
    
    # Index by row_id
    by_id = {r["row_id"]: r for r in row_results}
    
    # Binding pair joint accuracy
    joint_correct = 0
    joint_total = 0
    a_correct_count = 0
    b_correct_count = 0
    for bp in binding_pairs:
        a_id = bp["row_a_id"]
        b_id = bp["row_b_id"]
        if a_id not in by_id or b_id not in by_id:
            continue
        joint_total += 1
        a_ok = by_id[a_id]["all_tokens_correct"]
        b_ok = by_id[b_id]["all_tokens_correct"]
        if a_ok: a_correct_count += 1
        if b_ok: b_correct_count += 1
        if a_ok and b_ok: joint_correct += 1
    
    # Per-role statistics
    by_role = defaultdict(list)
    for r in row_results:
        by_role[r["role"]].append(r)
    
    result = {
        "n_scored": len(row_results),
        "mean_nll": sum(r["mean_nll"] for r in row_results) / max(1, len(row_results)),
        "mean_token_accuracy": sum(r["token_accuracy"] for r in row_results) / max(1, len(row_results)),
        "binding_pairs_total": joint_total,
        "binding_joint_correct": joint_correct,
        "binding_joint_accuracy": joint_correct / max(1, joint_total),
        "binding_a_correct": a_correct_count,
        "binding_b_correct": b_correct_count,
        "by_role": {},
    }
    for role, rows in by_role.items():
        nlls = [r["mean_nll"] for r in rows]
        accs = [r["token_accuracy"] for r in rows]
        all_tok = [r["all_tokens_correct"] for r in rows]
        result["by_role"][role] = {
            "n": len(rows),
            "mean_nll": sum(nlls) / len(nlls),
            "mean_token_accuracy": sum(accs) / len(accs),
            "all_tokens_correct_rate": sum(all_tok) / len(all_tok),
        }
    
    return result


def train_epoch(model, optimizer, train_toks: list[dict], pad_id: int,
                mask_token_id: int, device: torch.device, batch_size: int = 16) -> dict:
    """One training epoch with answer-only loss."""
    model.train()
    random.shuffle(train_toks)
    
    total_loss = 0.0
    n_batches = 0
    n_answer_tokens = 0
    
    for start in range(0, len(train_toks), batch_size):
        batch = train_toks[start:start + batch_size]
        collated = collate_batch(batch, pad_id)
        
        input_ids = collated["input_ids"].to(device)
        attention_mask = collated["attention_mask"].to(device)
        labels = collated["labels"].to(device)
        
        # Mask answer positions in input: replace with [MASK] so the model must predict
        masked_input = input_ids.clone()
        answer_mask = labels != -100
        masked_input[answer_mask] = mask_token_id
        
        outputs = model(input_ids=masked_input, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        if loss is not None and torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            n_answer_tokens += answer_mask.sum().item()
    
    return {
        "mean_loss": total_loss / max(1, n_batches),
        "n_batches": n_batches,
        "n_answer_tokens": n_answer_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs")
    ap.add_argument("--smoke", type=int, default=0, help="Use first N rows for smoke test")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    out_dir = OUT_DEFAULT
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plan = {
        "status": "BINDING_TRAIN_PLAN",
        "created_utc": now_utc(),
        "train_rows_path": rel(TRAIN_ROWS),
        "heldout_rows_path": rel(HELDOUT_ROWS),
        "binding_pairs_path": rel(BINDING_PAIRS),
        "model_path": rel(BASE_43022_CHCK),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "max_length": args.max_length,
        "smoke": args.smoke,
        "seed": args.seed,
        "design": "Answer-only adapter training: freeze base, train only adapter params. "
                  "Mask answer tokens in input, loss only on answer positions. "
                  "Joint binding criterion on DISTRACTOR pairs.",
    }
    with open(out_dir / "plan.json", "w") as f:
        json.dump(plan, f, indent=2)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    
    # Load data
    train_data = read_jsonl(TRAIN_ROWS)
    heldout_data = read_jsonl(HELDOUT_ROWS)
    binding_pairs = read_jsonl(BINDING_PAIRS)
    
    if args.smoke > 0:
        train_data = train_data[:args.smoke]
        heldout_data = heldout_data[:min(args.smoke, len(heldout_data))]
        binding_pairs = binding_pairs[:min(args.smoke // 2, len(binding_pairs))]
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...", flush=True)
    model, tokenizer, identity = trusted_load_model(BASE_43022_CHCK, device)
    
    with open(out_dir / "model_identity.json", "w") as f:
        json.dump(identity, f, indent=2)
    print(json.dumps(identity, indent=2), flush=True)
    
    if identity["adapter_params_loaded"] == 0:
        raise RuntimeError("Expected adapter parameters > 0")
    
    mask_token_id = int(tokenizer.mask_token_id)
    pad_id = int(tokenizer.pad_token_id or 0)
    
    # Tokenize
    print(f"Tokenizing {len(train_data)} train + {len(heldout_data)} heldout rows...", flush=True)
    train_toks = [t for t in (tokenize_row(tokenizer, r, args.max_length) for r in train_data) if t is not None]
    heldout_toks = [t for t in (tokenize_row(tokenizer, r, args.max_length) for r in heldout_data) if t is not None]
    
    print(f"Tokenized: {len(train_toks)} train, {len(heldout_toks)} heldout", flush=True)
    
    # Freeze base, train adapter
    freeze_stats = freeze_base_train_adapter(model)
    print(json.dumps({"freeze_stats": freeze_stats}, indent=2), flush=True)
    
    # Optimizer: only adapter parameters
    adapter_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(adapter_params, lr=args.lr, weight_decay=0.01)
    
    # Baseline evaluation
    print("Evaluating baseline (before training)...", flush=True)
    baseline_eval = evaluate_heldout(model, tokenizer, heldout_toks, binding_pairs,
                                      mask_token_id, pad_id, device)
    trajectory = [{"epoch": 0, "eval": baseline_eval}]
    print(json.dumps({"epoch": 0, "eval": baseline_eval}, indent=2), flush=True)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_result = train_epoch(model, optimizer, train_toks, pad_id,
                                    mask_token_id, device, args.batch_size)
        print(json.dumps({"epoch": epoch, "train": epoch_result}, indent=2), flush=True)
        
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_result = evaluate_heldout(model, tokenizer, heldout_toks, binding_pairs,
                                            mask_token_id, pad_id, device)
            trajectory.append({"epoch": epoch, "eval": eval_result, "train": epoch_result})
            print(json.dumps({"epoch": epoch, "eval": eval_result}, indent=2), flush=True)
    
    # Save checkpoint
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(ckpt_dir))
    
    # Save trajectory
    summary = {
        "status": "BINDING_TRAIN_DONE",
        "created_utc": now_utc(),
        "model_identity": identity,
        "freeze_stats": freeze_stats,
        "n_train_toks": len(train_toks),
        "n_heldout_toks": len(heldout_toks),
        "n_binding_pairs": len(binding_pairs),
        "trajectory": trajectory,
        "checkpoint_dir": rel(ckpt_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
