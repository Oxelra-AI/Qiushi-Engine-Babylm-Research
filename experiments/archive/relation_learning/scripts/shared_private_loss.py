#!/usr/bin/env python3
"""research: Per-row MLM loss for shared-versus-private fitting index.

GPU forward passes on existing frontier_consolidation checkpoints to compute deterministic
per-row MLM loss.  The SAME mask is applied to the same row across all arms
so per-row loss changes are directly comparable.

Usage:
  python shared_private_loss.py --arm D_V_43022 --gpu 0
  python shared_private_loss.py --arm D_V_43122 --gpu 1

Arms:
  D_V_43022  DeBERTa seed43022 VIEW  (converting, register-negative)
  D_V_43122  DeBERTa seed43122 VIEW  (converting, register-positive)
  D_C_43022  DeBERTa seed43022 CLEAN (non-converting baseline)
  R_V_43022  RoBERTa seed43022 VIEW  (non-converting)
  R_C_43022  RoBERTa seed43022 CLEAN (non-converting baseline)

Checkpoints evaluated: 60M, 70M, 80M, 90M, 100M

Eval sets:
  heldout: 6992 rows never trained on by any arm
  filler:  5000 sampled rows trained identically by all arms

No training, no official evaluation, no leaderboard, no upload.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────

SCRIPT_DIR = _public_path('experiments/archive/relation_learning/scripts')
# Resolve project root: go up from scripts/ ->  -> relation_learning/ -> Sessions/ -> root
# Use a marker: the directory containing "Sessions/" at its top level
_p = SCRIPT_DIR
for _ in range(8):
    if (_public_path('experiments/archive/relation_learning/scripts/Sessions')).is_dir() and (_public_path('experiments/archive/relation_learning/scripts/Knowledge')).is_dir():
        break
    _p = _public_path('experiments/archive/relation_learning')
PROJECT_ROOT = _p

frontier_consolidation_RUNS = _public_path('experiments/archive/frontier_consolidation/training/runs')
frontier_consolidation_DATA = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools')

ARM_CONFIGS = {
    "D_V_43022": {
        "model_family": "deberta",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    },
    "D_V_43122": {
        "model_family": "deberta",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    },
    "D_C_43022": {
        "model_family": "deberta",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    },
    "R_V_43022": {
        "model_family": "roberta",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/roberta_view_dose2p64x_matched_rowholdout_100M_seed43022'),
    },
    "R_C_43022": {
        "model_family": "roberta",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/roberta_clean_dose2p64x_matched_rowholdout_100M_seed43022'),
    },
}

CHECKPOINTS = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
MASK_PROB = 0.15
BATCH_SIZE = 128
MAX_SEQ_LEN = 256
FILLER_SAMPLE_N = 5000
FILLER_SAMPLE_SEED = 42

# ── data loading ─────────────────────────────────────────────────────────

def load_heldout_rows():
    """Load held-out rows (never trained on by any arm)."""
    path = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl')
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "example_id": obj["example_id"],
                "text": obj["text"],
                "source": obj["source"],
                "words": obj["words"],
            })
    return rows

def load_filler_sample():
    """Load a deterministic sample of common filler rows."""
    path = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/common_filler_rows.jsonl')
    all_rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            all_rows.append({
                "example_id": obj["example_id"],
                "text": obj["text"],
                "source": obj["source"],
                "words": obj["words"],
            })
    rng = np.random.RandomState(FILLER_SAMPLE_SEED)
    indices = rng.choice(len(all_rows), size=min(FILLER_SAMPLE_N, len(all_rows)), replace=False)
    indices.sort()
    return [all_rows[i] for i in indices]

# ── deterministic masking ────────────────────────────────────────────────

def apply_fixed_mask(input_ids, example_id, mask_token_id, special_token_ids, pad_token_id):
    """Apply a deterministic token-level mask based on example_id.
    
    Returns (masked_input_ids, labels, n_masked).
    labels has -100 for non-masked positions.
    """
    seq_len = input_ids.shape[0]
    rng = np.random.RandomState(seed=int(example_id) % (2**31))
    
    # Identify maskable positions (not special, not padding)
    maskable = torch.ones(seq_len, dtype=torch.bool)
    for sid in special_token_ids:
        maskable &= (input_ids != sid)
    maskable &= (input_ids != pad_token_id)
    
    # Generate mask probabilities for all positions
    probs = rng.random(seq_len)
    mask = torch.tensor(probs < MASK_PROB, dtype=torch.bool) & maskable
    
    # Must have at least 1 masked token for meaningful loss
    if mask.sum() == 0 and maskable.sum() > 0:
        # Force mask on first maskable position
        first_maskable = maskable.nonzero(as_tuple=True)[0][0].item()
        mask[first_maskable] = True
    
    labels = input_ids.clone()
    labels[~mask] = -100
    
    masked_input = input_ids.clone()
    # 80% [MASK], 10% random, 10% keep (standard BERT masking)
    mask_decisions = rng.random(seq_len)
    replace_mask = mask & torch.tensor(mask_decisions < 0.8, dtype=torch.bool)
    random_mask = mask & torch.tensor((mask_decisions >= 0.8) & (mask_decisions < 0.9), dtype=torch.bool)
    # keep_mask = rest
    
    masked_input[replace_mask] = mask_token_id
    if random_mask.sum() > 0:
        random_tokens = torch.tensor(rng.randint(0, 16384, size=int(random_mask.sum())), dtype=torch.long)
        masked_input[random_mask] = random_tokens
    
    return masked_input, labels, int(mask.sum().item())

# ── model loading ────────────────────────────────────────────────────────

def load_model(arm_config, checkpoint_name, device):
    """Load a model checkpoint onto the specified device."""
    from transformers import AutoModelForMaskedLM
    
    chck_path = arm_config["run_dir"] / "hf_model" / checkpoint_name
    if not chck_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {chck_path}")
    
    model = AutoModelForMaskedLM.from_pretrained(str(chck_path), torch_dtype=torch.float32)
    model.eval()
    model.to(device)
    return model

def load_tokenizer(arm_config):
    """Load the tokenizer from the run's hf_model root."""
    from transformers import AutoTokenizer
    tok_path = arm_config["run_dir"] / "hf_model"
    return AutoTokenizer.from_pretrained(str(tok_path))

# ── per-row loss computation ─────────────────────────────────────────────

@torch.no_grad()
def compute_per_row_losses(model, tokenizer, rows, device):
    """Compute per-row MLM loss with deterministic masks.
    
    Returns dict with:
      example_ids: int array [N]
      losses: float array [N]  (mean NLL per masked token)
      n_masked: int array [N]
    """
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    # Special tokens: BOS=1, EOS=2, PAD=3, MASK
    special_ids = set()
    for attr in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'cls_token_id', 'sep_token_id']:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)
    special_ids_list = list(special_ids)
    
    example_ids = []
    all_losses = []
    all_n_masked = []
    
    # Process in batches
    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[batch_start:batch_start + BATCH_SIZE]
        
        # Tokenize all rows in batch
        texts = [r["text"] for r in batch_rows]
        encodings = tokenizer(
            texts,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        batch_input_ids = encodings["input_ids"]  # [B, L]
        batch_attention_mask = encodings["attention_mask"]  # [B, L]
        
        # Apply fixed mask per row
        batch_masked_inputs = []
        batch_labels = []
        batch_n_masked = []
        
        for i, row in enumerate(batch_rows):
            eid = row["example_id"]
            masked_input, labels, n_m = apply_fixed_mask(
                batch_input_ids[i].clone(), eid,
                mask_token_id, special_ids_list, pad_token_id
            )
            batch_masked_inputs.append(masked_input)
            batch_labels.append(labels)
            batch_n_masked.append(n_m)
            example_ids.append(eid)
        
        batch_masked_inputs = torch.stack(batch_masked_inputs).to(device)
        batch_labels = torch.stack(batch_labels).to(device)
        batch_attn = batch_attention_mask.to(device)
        
        # Forward pass
        outputs = model(
            input_ids=batch_masked_inputs,
            attention_mask=batch_attn,
            labels=None,  # We compute loss manually per row
        )
        logits = outputs.logits  # [B, L, V]
        
        # Compute per-row cross-entropy on masked positions
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        # Reshape for loss computation
        B, L, V = logits.shape
        per_token_loss = loss_fn(logits.view(B * L, V), batch_labels.view(B * L))
        per_token_loss = per_token_loss.view(B, L)  # [B, L]
        
        for i in range(len(batch_rows)):
            mask_positions = (batch_labels[i] != -100)
            n_m = mask_positions.sum().item()
            if n_m > 0:
                row_loss = per_token_loss[i][mask_positions].mean().item()
            else:
                row_loss = float('nan')
            all_losses.append(row_loss)
            all_n_masked.append(n_m)
    
    return {
        "example_ids": np.array(example_ids, dtype=np.int64),
        "losses": np.array(all_losses, dtype=np.float64),
        "n_masked": np.array(all_n_masked, dtype=np.int64),
    }

# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Shared-private per-row MLM loss")
    parser.add_argument("--arm", required=True, choices=list(ARM_CONFIGS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval-set", default="both", choices=["heldout", "filler", "both"])
    parser.add_argument("--output-dir", default=str(_public_path('experiments/archive/relation_learning/data/shared_private_loss')))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    
    arm_config = ARM_CONFIGS[args.arm]
    print(f"[{args.arm}] Loading tokenizer from {arm_config['run_dir']}/hf_model", flush=True)
    tokenizer = load_tokenizer(arm_config)
    
    # Load eval sets
    eval_sets = {}
    if args.eval_set in ("heldout", "both"):
        print(f"[{args.arm}] Loading heldout rows...", flush=True)
        eval_sets["heldout"] = load_heldout_rows()
        print(f"[{args.arm}] Loaded {len(eval_sets['heldout'])} heldout rows", flush=True)
    if args.eval_set in ("filler", "both"):
        print(f"[{args.arm}] Loading filler sample...", flush=True)
        eval_sets["filler"] = load_filler_sample()
        print(f"[{args.arm}] Loaded {len(eval_sets['filler'])} filler rows", flush=True)
    
    results = {}
    
    for chck in CHECKPOINTS:
        t0 = time.time()
        print(f"\n[{args.arm}] Loading checkpoint {chck}...", flush=True)
        model = load_model(arm_config, chck, device)
        
        for eset_name, eset_rows in eval_sets.items():
            print(f"[{args.arm}] Evaluating {eset_name} ({len(eset_rows)} rows) on {chck}...", flush=True)
            row_result = compute_per_row_losses(model, tokenizer, eset_rows, device)
            
            key = f"{args.arm}__{eset_name}__{chck}"
            results[key] = row_result
            
            mean_loss = np.nanmean(row_result["losses"])
            print(f"[{args.arm}] {eset_name}/{chck}: mean_loss={mean_loss:.4f}, "
                  f"n_rows={len(row_result['losses'])}, "
                  f"mean_n_masked={row_result['n_masked'].mean():.1f}", flush=True)
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(f"[{args.arm}] {chck} done in {elapsed:.1f}s", flush=True)
    
    # Save all results as one NPZ per arm
    save_dict = {}
    for key, res in results.items():
        save_dict[f"{key}__example_ids"] = res["example_ids"]
        save_dict[f"{key}__losses"] = res["losses"]
        save_dict[f"{key}__n_masked"] = res["n_masked"]
    
    npz_path = os.path.join(args.output_dir, f"{args.arm}_per_row_losses.npz")
    np.savez_compressed(npz_path, **save_dict)
    print(f"\n[{args.arm}] Saved {npz_path} ({os.path.getsize(npz_path)} bytes)", flush=True)
    
    # Also save a summary JSON
    summary = {
        "arm": args.arm,
        "model_family": arm_config["model_family"],
        "run_dir": str(arm_config["run_dir"]),
        "checkpoints": CHECKPOINTS,
        "eval_sets": {k: len(v) for k, v in eval_sets.items()},
        "mask_prob": MASK_PROB,
        "batch_size": BATCH_SIZE,
        "mean_losses": {},
    }
    for key, res in results.items():
        summary["mean_losses"][key] = float(np.nanmean(res["losses"]))
    
    json_path = os.path.join(args.output_dir, f"{args.arm}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.arm}] Saved {json_path}", flush=True)
    
    # Final summary
    print(f"\n{'='*60}", flush=True)
    print(f"[{args.arm}] COMPLETE", flush=True)
    for key in sorted(results.keys()):
        print(f"  {key}: mean_loss={np.nanmean(results[key]['losses']):.4f}", flush=True)
    
    print(json.dumps({
        "status": f"STEP003_{args.arm}_DONE",
        "npz": npz_path,
        "summary": json_path,
        "arm": args.arm,
    }, indent=2), flush=True)

if __name__ == "__main__":
    main()
