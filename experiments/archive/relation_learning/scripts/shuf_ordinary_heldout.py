#!/usr/bin/env python3
"""research: SHUF seed43122 ordinary-heldout MLM loss comparison.

Measures deterministic-mask whole-word MLM loss on the 6,992 ordinary held-out rows
for OFF (lengthmatched) and SHUF (shuffled correspondence) at seed43122.
This is the missing piece from research/42 that completes the pre-stated SHUF
replication criterion.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

ROOT = pathlib.Path.cwd()
HELDOUT_PATH = ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/shuf_ordinary_heldout"
NOTE_PATH = ROOT / "research/notes/relation_learning/shuf_ordinary_heldout.md"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARMS = {
    "OFF": ROOT / "experiments/archive/compact_experience/training/runs/official_lengthmatched_16k_seed43122/hf_model",
    "SHUF": ROOT / "experiments/archive/relation_learning/training/runs/qwen_shuffled_control_16k_seed43122/hf_model",
}
CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
BATCH_SIZE = 64
SEQ_LEN = 256
MASK_PROB = 0.15


def read_jsonl(p: pathlib.Path) -> list[dict]:
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def deterministic_wwm_mask(input_ids: torch.Tensor, tokenizer, mask_prob: float = 0.15):
    """Generate deterministic masks using whole-word masking."""
    labels = input_ids.clone()
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
        if sid is not None:
            special_mask |= (input_ids == sid)
    
    # Whole-word groups
    tokens_str = [tokenizer.convert_ids_to_tokens(int(tid)) for tid in input_ids]
    word_starts = []
    word_groups = []
    current_group = -1
    for i, tok in enumerate(tokens_str):
        if special_mask[i]:
            word_groups.append(-1)
            continue
        if tok.startswith("▁") or tok.startswith("Ġ") or i == 0 or special_mask[i-1]:
            current_group += 1
        word_groups.append(current_group)
    
    n_words = current_group + 1
    if n_words == 0:
        labels[:] = -100
        return input_ids, labels
    
    # Deterministic selection based on token content
    rng = np.random.RandomState(int(input_ids.sum().item()) % (2**31))
    n_mask_words = max(1, int(n_words * mask_prob))
    mask_word_ids = set(rng.choice(n_words, size=n_mask_words, replace=False).tolist())
    
    masked_input = input_ids.clone()
    for i, g in enumerate(word_groups):
        if g in mask_word_ids:
            masked_input[i] = tokenizer.mask_token_id
        else:
            labels[i] = -100
    
    # Also ignore specials in labels
    labels[special_mask] = -100
    return masked_input, labels


def score_rows(model, tokenizer, rows, device, batch_size):
    """Score held-out rows and return per-row average MLM loss."""
    model.eval()
    results = []
    
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i+batch_size]
        texts = [r["text"] for r in batch_rows]
        
        enc = tokenizer(texts, padding=True, truncation=True, max_length=SEQ_LEN,
                        return_tensors="pt").to(device)
        
        all_masked_inputs = []
        all_labels = []
        for j in range(len(texts)):
            ids_j = enc.input_ids[j]
            masked_j, labels_j = deterministic_wwm_mask(ids_j, tokenizer, MASK_PROB)
            all_masked_inputs.append(masked_j)
            all_labels.append(labels_j)
        
        masked_batch = torch.stack(all_masked_inputs)
        labels_batch = torch.stack(all_labels)
        attn_mask = enc.attention_mask
        
        with torch.no_grad():
            out = model(input_ids=masked_batch, attention_mask=attn_mask, labels=labels_batch)
            logits = out.logits  # [B, T, V]
        
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fn(logits.view(-1, logits.size(-1)), labels_batch.view(-1))
        per_token_loss = per_token_loss.view(labels_batch.shape)
        
        for j in range(len(texts)):
            mask = labels_batch[j] != -100
            n_masked = mask.sum().item()
            if n_masked > 0:
                row_loss = per_token_loss[j][mask].mean().item()
            else:
                row_loss = float("nan")
            results.append({
                "row_idx": batch_rows[j].get("row_index", i + j),
                "n_masked": n_masked,
                "mean_loss": row_loss,
            })
    
    return results


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    # Load held-out rows
    rows = read_jsonl(HELDOUT_PATH)
    print(f"Held-out rows: {len(rows)}", flush=True)
    if len(rows) != 6992:
        print(f"WARNING: expected 6992, found {len(rows)}")
    
    results = {}
    meta_records = []
    
    for arm_name, arm_root in ARMS.items():
        for ck in CKPTS:
            ck_path = arm_root / ck
            if not ck_path.exists():
                print(f"SKIP {arm_name}/{ck}: not found")
                continue
            
            print(f"\n[LOAD] {arm_name}/{ck}: {ck_path}", flush=True)
            t0 = time.time()
            tokenizer = AutoTokenizer.from_pretrained(str(ck_path))
            model = AutoModelForMaskedLM.from_pretrained(str(ck_path), trust_remote_code=True).to(device)
            
            scored = score_rows(model, tokenizer, rows, device, BATCH_SIZE)
            elapsed = time.time() - t0
            
            valid = [r for r in scored if not math.isnan(r["mean_loss"])]
            mean_loss = np.mean([r["mean_loss"] for r in valid])
            
            key = f"{arm_name}_{ck}"
            results[key] = {
                "arm": arm_name, "checkpoint": ck,
                "mean_loss": float(mean_loss),
                "n_valid": len(valid), "n_total": len(scored),
            }
            meta_records.append({"arm": arm_name, "ck": ck, "elapsed_sec": round(elapsed, 1),
                                  "mean_loss": round(float(mean_loss), 6)})
            print(f"[DONE] {arm_name}/{ck}: mean_loss={mean_loss:.6f} ({len(valid)} valid rows) in {elapsed:.1f}s", flush=True)
            
            del model
            torch.cuda.empty_cache()
    
    # Compute SHUF - OFF contrasts at late checkpoints
    contrasts = []
    for ck in CKPTS:
        k_off = f"OFF_{ck}"
        k_shuf = f"SHUF_{ck}"
        if k_off in results and k_shuf in results:
            delta = results[k_shuf]["mean_loss"] - results[k_off]["mean_loss"]
            contrasts.append({"checkpoint": ck, 
                              "OFF_loss": results[k_off]["mean_loss"],
                              "SHUF_loss": results[k_shuf]["mean_loss"],
                              "delta_SHUF_minus_OFF": delta})
    
    # Save results
    out = {
        "status": "SHUF_ORDINARY_HELDOUT_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "contrasts": contrasts,
        "meta": meta_records,
    }
    (OUT_DIR / "shuf_ordinary_heldout_result.json").write_text(json.dumps(out, indent=2))
    
    # Write note
    lines = [
        "# research: SHUF seed43122 ordinary-heldout MLM loss",
        "",
        "Deterministic whole-word masking at p=0.15 on 6,992 held-out rows.",
        "",
        "## Results",
        "",
        "| Checkpoint | OFF loss | SHUF loss | Δ (SHUF−OFF) |",
        "|---|---|---|---|",
    ]
    for c in contrasts:
        lines.append(f"| {c['checkpoint']} | {c['OFF_loss']:.6f} | {c['SHUF_loss']:.6f} | {c['delta_SHUF_minus_OFF']:+.6f} |")
    lines.append("")
    lines.append(f"Data: `experiments/archive/relation_learning/data/shuf_ordinary_heldout`")
    
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.write_text("\n".join(lines) + "\n")
    
    print("\n" + "\n".join(lines), flush=True)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
