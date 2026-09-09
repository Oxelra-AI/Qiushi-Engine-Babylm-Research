#!/usr/bin/env python3
"""research: Practical chck_82M binding candidate using frozen-slow + fresh-private.

Architecture matches coherent86 exactly:
  - Frozen chck_82M slow path (35,463,008 parameters)
  - Fresh zero-initialized private adapter branch (995,584 parameters)
  - Only private_adapter.* is trained

Uses research's proven data loading, masking, and margin evaluation.
Uses research's balanced pair-batch training design.

The only change from the research balanced factorial is the model:
  research: loads base43022 chck_100M, trains existing adapter
  research: loads chck_82M into research frozen architecture, trains fresh private adapter
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import torch
import torch.nn.functional as F

_SCRIPT = _public_path('experiments/archive/relation_learning/scripts/chck82_binding_candidate.py')
_SCRIPTS = _public_path('experiments/archive/relation_learning/scripts')
_ROOT = _public_path('.')

# Set up cache BEFORE any imports that touch transformers
CACHE_BASE = os.environ.get(
    "CACHE_BASE",
    os.environ.get("TRANSFORMERS_CACHE",
                    str(_public_path('experiments/archive/relation_learning/data/chck82_binding_candidate/hf_cache')))
)
os.environ["TRANSFORMERS_CACHE"] = CACHE_BASE
os.environ["HF_HOME"] = CACHE_BASE

# Import research frozen modeling
sys.path.insert(0, str(_public_path('experiments/archive/frontier_consolidation/scripts')))
from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM

# Import research primitives
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))
os.environ.setdefault("CACHE_BASE", CACHE_BASE)
import binding_factorial_train as S73

DEFAULT_CHCK82M = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
DEFAULT_TRAIN = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_train.jsonl')
DEFAULT_HELD = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl')

def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(_ROOT))
    except: return str(p)

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def load_frozen_model(checkpoint_path: pathlib.Path, private_adapter_scale: float, device: str):
    """Load chck_82M into frozen-slow + fresh-private architecture."""
    from transformers import DebertaV2Config
    from safetensors.torch import load_file
    
    config = DebertaV2Config.from_pretrained(str(checkpoint_path))
    config.private_adapter_bottleneck = getattr(config, "adapter_bottleneck", 128)
    config.private_adapter_scale = private_adapter_scale
    config.private_adapter_enabled = True
    config.private_adapter_activation = getattr(config, "adapter_activation", "gelu")
    
    model = FrozenSlowPrivateDebertaV2ForMaskedLM(config)
    
    sf = checkpoint_path / "model.safetensors"
    state_dict = load_file(str(sf), device="cpu") if sf.exists() else torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    private_missing = [k for k in missing if "private_adapter" in k]
    # cls.predictions.decoder.weight/bias are tied to embeddings and often missing from safetensors
    tied_ok = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
    non_private_missing = [k for k in missing if "private_adapter" not in k and k not in tied_ok]
    
    if non_private_missing:
        raise ValueError(f"Non-private keys missing: {non_private_missing[:5]}")
    
    # Tie decoder weights to embeddings if missing
    if hasattr(model, "cls") and hasattr(model.cls, "predictions"):
        model.tie_weights()
    
    # Freeze everything except private_adapter
    trainable = frozen = 0
    for name, param in model.named_parameters():
        if "private_adapter" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    
    model = model.to(device)
    return model, {"model_class": type(model).__name__, "total": frozen + trainable,
                    "frozen": frozen, "trainable_private": trainable,
                    "private_adapter_scale": private_adapter_scale,
                    "private_missing_keys": len(private_missing),
                    "non_private_missing_keys": len(non_private_missing)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chck82m", type=pathlib.Path, default=DEFAULT_CHCK82M)
    p.add_argument("--train-data", type=pathlib.Path, default=DEFAULT_TRAIN)
    p.add_argument("--held-data", type=pathlib.Path, default=DEFAULT_HELD)
    p.add_argument("--out-root", type=pathlib.Path,
                   default=_public_path('experiments/archive/relation_learning/data/chck82_binding_candidate'))
    p.add_argument("--private-adapter-scale", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-fraction", type=float, default=0.06)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pair-batch-size", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=43022)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.smoke else "cpu"
    arm_name = f"scale_{args.private_adapter_scale:.2f}"
    out_dir = args.out_root / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load frozen model
    print(f"[research] Loading chck_82M from {rel(args.chck82m)}", flush=True)
    model, identity = load_frozen_model(args.chck82m, args.private_adapter_scale, device)
    print(f"[identity] {json.dumps(identity, indent=2)}", flush=True)
    
    # 2. Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.chck82m), use_fast=True)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    
    # 3. Load and prepare data using research functions
    train_rows = S73.read_jsonl(args.train_data)
    held_rows = S73.read_jsonl(args.held_data)
    
    # Filter to balanced pairs only (drop UPDATED_USE singles)
    train_rows = [r for r in train_rows if r.get("pair_half") in ("A", "B")]
    held_rows = [r for r in held_rows if r.get("pair_half") in ("A", "B")]
    
    train_toks = [x for x in (S73.tokenize_row(tokenizer, r, args.max_length) for r in train_rows) if x is not None]
    held_toks = [x for x in (S73.tokenize_row(tokenizer, r, args.max_length) for r in held_rows) if x is not None]
    
    # Group into pairs for training
    pair_groups = defaultdict(list)
    for t in train_toks:
        pair_groups[t["pair_id"]].append(t)
    complete_pairs = [(pid, rows) for pid, rows in pair_groups.items() if len(rows) == 2]
    
    # Build binding pairs for eval from the canonical file (same as research)
    BINDING_PAIRS_FILE = _public_path('experiments/archive/relation_learning/data/recombination_rows/binding_pairs_heldout.jsonl')
    binding_pairs = S73.read_jsonl(BINDING_PAIRS_FILE)
    
    # Keep original held rows for margin eval (needs context_text etc.)
    held_rows_for_eval = held_rows
    
    print(f"[data] Train: {len(train_toks)} rows, {len(complete_pairs)} pairs. "
          f"Held: {len(held_toks)} rows, {len(binding_pairs)} pairs.", flush=True)
    
    if args.validate_only:
        print(json.dumps({"status": "VALIDATE_ONLY", "identity": identity,
                          "train_toks": len(train_toks), "train_pairs": len(complete_pairs),
                          "held_toks": len(held_toks), "held_pairs": len(binding_pairs)}, indent=2), flush=True)
        return
    
    if args.smoke:
        args.epochs = 2
        args.eval_every = 1
        complete_pairs = complete_pairs[:4]
        held_rows_for_eval = held_rows_for_eval[:8]
        binding_pairs = binding_pairs[:4]
        # Skip margin eval in smoke mode (too slow on CPU); just verify training loop
        skip_eval = not torch.cuda.is_available()
    
    # 4. Training loop (reuses research collate/apply/score functions)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    
    total_updates = args.epochs * (len(complete_pairs) // args.pair_batch_size + 1)
    warmup = max(1, int(total_updates * args.warmup_fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(s / warmup, 1.0) if s < warmup else 0.5 * (1 + __import__('math').cos(__import__('math').pi * (s - warmup) / max(1, total_updates - warmup))))
    
    import random
    rng = random.Random(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    
    trajectory = []
    updates_seen = 0
    
    for epoch in range(args.epochs + 1):
        # Eval at boundary
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if args.smoke and skip_eval:
                eval_result = {"epoch": epoch, "updates_seen": updates_seen, "skip": "smoke_cpu"}
                trajectory.append(eval_result)
                print(json.dumps({"epoch": epoch, "skip": "smoke_cpu", "updates": updates_seen}), flush=True)
            else:
                eval_result = S73.score_margin_eval(model, tokenizer, held_rows_for_eval, binding_pairs,
                                                     torch.device(device), args.max_length, batch_size=32)
                eval_result["epoch"] = epoch
                eval_result["updates_seen"] = updates_seen
                trajectory.append(eval_result)
                j = eval_result.get("binding_joint_correct", 0)
                n = eval_result.get("binding_pairs_scored", 0)
                a = eval_result.get("a_correct", 0)
                b = eval_result.get("b_correct", 0)
                gf = j / max(n, 1)
                bw = n - j - (a - j) - (b - j)
                print(json.dumps({"epoch": epoch, "joint": j, "gated_frac": round(gf, 4),
                                  "a_correct": a, "b_correct": b, "both_wrong": bw,
                                  "updates": updates_seen}, indent=2), flush=True)
        
        if epoch == args.epochs:
            break
        
        # Train one epoch
        model.train()
        pair_order = list(range(len(complete_pairs)))
        rng.shuffle(pair_order)
        
        epoch_loss = epoch_labeled = 0
        for batch_start in range(0, len(pair_order), args.pair_batch_size):
            batch_idx = pair_order[batch_start:batch_start + args.pair_batch_size]
            batch_rows = []
            for idx in batch_idx:
                _, rows = complete_pairs[idx]
                batch_rows.extend(rows)
            if not batch_rows:
                continue
            
            batch_t = S73.collate(batch_rows, pad_id)
            batch_t = {k: v.to(device) for k, v in batch_t.items()}
            
            stats = S73.batch_train_step(model, optimizer, batch_t, tokenizer,
                                          "answer_clean", gen, wwm_prob=0.15)
            scheduler.step()
            updates_seen += 1
            
            if not __import__('math').isnan(stats["loss"]):
                epoch_loss += stats["loss"] * stats["labeled_positions"]
                epoch_labeled += stats["labeled_positions"]
        
        if epoch_labeled > 0 and (epoch + 1) % 5 == 0:
            print(f"[train] epoch {epoch+1}, loss={epoch_loss/epoch_labeled:.4f}, "
                  f"labeled={epoch_labeled}, updates={updates_seen}", flush=True)
    
    # 5. Save checkpoint and summary
    ckpt = out_dir / "checkpoint"
    if not args.smoke:
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
    
    summary = {
        "status": "CHCK82_BINDING_CANDIDATE",
        "created_utc": now(),
        "identity": identity,
        "private_adapter_scale": args.private_adapter_scale,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_pairs": len(complete_pairs),
        "held_pairs": len(binding_pairs),
        "updates_seen": updates_seen,
        "trajectory": trajectory,
        "checkpoint_dir": rel(ckpt) if not args.smoke else "smoke",
    }
    S73.write_json(out_dir / "summary.json", summary)
    
    print(json.dumps({"status": "CHCK82_BINDING_CANDIDATE",
                      "summary": rel(out_dir / "summary.json"),
                      "checkpoint": rel(ckpt) if not args.smoke else "smoke"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
