#!/usr/bin/env python3
"""research: Identity-under-variation representation probe.

Test whether VIEW-trained checkpoints produce more surface-invariant representations
than CLEAN/REPEAT checkpoints, using natural (source, rewrite) paraphrase pairs.

The pairs come from `selected_matched_max_pairs.jsonl`, which contains the original
FineWeb sentences and their compressed/paraphrased versions.

For each pair:
  - Forward-pass source_text and rewrite_text through each checkpoint
  - Extract mean-pooled hidden states from the final transformer layer
  - Compute cosine similarity between source and rewrite representations
  - Compare across VIEW, CLEAN, and REPEAT arms

Pre-stated null: If VIEW similarity = CLEAN similarity, VIEW did not learn surface
invariance. If VIEW > CLEAN, it supports identity-under-variation mechanism.

No training, upload, or benchmark scoring.
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
import time
import numpy as np
import torch
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
OUT_DIR = WS / "data" / "invariance_probe"
frontier_consolidation_RUNS = ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs"

PAIRS_FILE = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "dose_distribution_select" / "selected_matched_max_pairs.jsonl"

# Arm configs with checkpoint paths
ARM_CONFIGS = {
    "D_V_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_V_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_C_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_C_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_R_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
}


def load_pairs(path: pathlib.Path, n: int = 500) -> list[dict]:
    """Load paraphrase pairs from the matched pairs file."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("source_text") and d.get("rewrite_text"):
                pairs.append({
                    "pair_id": d.get("pair_id", ""),
                    "source_text": d["source_text"],
                    "rewrite_text": d["rewrite_text"],
                    "source_words": d.get("source_words", 0),
                    "rewrite_words": d.get("rewrite_words", 0),
                    "content_overlap": d.get("content_overlap", 0),
                    "length_ratio": d.get("length_ratio", 0),
                })
            if len(pairs) >= n:
                break
    return pairs


def extract_representations(
    model, tokenizer, texts: list[str], device: str, batch_size: int = 64
) -> np.ndarray:
    """Extract mean-pooled last-layer hidden states for a list of texts."""
    model.eval()
    all_reps = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # Tokenize with padding
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get last hidden state
        last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
        
        # Mean pool over non-padding tokens
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq, 1)
        masked_hidden = last_hidden * attention_mask
        mean_pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        
        all_reps.append(mean_pooled.cpu().numpy())
    
    return np.concatenate(all_reps, axis=0)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity between two representation matrices."""
    # Normalize
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
    return (a_norm * b_norm).sum(axis=1)


def run_probe(arm_name: str, run_dir: pathlib.Path, checkpoint: str, pairs: list[dict],
              device: str, gpu: int) -> dict[str, Any]:
    """Run the invariance probe for one arm."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    model_path = run_dir / "hf_model" / checkpoint
    if not model_path.exists():
        return {"error": f"Missing checkpoint: {model_path}"}
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    print(f"  [{arm_name}] Loading model from {checkpoint}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(run_dir / "hf_model"))
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), output_hidden_states=True)
    model.to(device)
    model.eval()
    
    source_texts = [p["source_text"] for p in pairs]
    rewrite_texts = [p["rewrite_text"] for p in pairs]
    
    print(f"  [{arm_name}] Extracting source representations ({len(source_texts)} texts)...", flush=True)
    source_reps = extract_representations(model, tokenizer, source_texts, device)
    
    print(f"  [{arm_name}] Extracting rewrite representations ({len(rewrite_texts)} texts)...", flush=True)
    rewrite_reps = extract_representations(model, tokenizer, rewrite_texts, device)
    
    # Matched-pair similarity: source_i with its own rewrite_i
    matched_sim = cosine_similarity_rows(source_reps, rewrite_reps)
    
    # Random-pair baseline: source_i with rewrite_(i+shift) mod N
    n = len(pairs)
    shifts = [n // 3, n // 2, 2 * n // 3]  # Three different random shifts
    random_sims = []
    for shift in shifts:
        shifted_rewrite = np.roll(rewrite_reps, shift, axis=0)
        random_sims.append(cosine_similarity_rows(source_reps, shifted_rewrite))
    random_sim = np.mean(random_sims, axis=0)
    
    # Same-text control: source_i with source_i (should be 1.0)
    same_sim = cosine_similarity_rows(source_reps, source_reps)
    
    # Invariance signal: matched - random
    invariance_signal = matched_sim - random_sim
    
    del model
    torch.cuda.empty_cache()
    
    result = {
        "arm": arm_name,
        "checkpoint": checkpoint,
        "n_pairs": n,
        "matched_sim_mean": float(np.mean(matched_sim)),
        "matched_sim_std": float(np.std(matched_sim)),
        "random_sim_mean": float(np.mean(random_sim)),
        "random_sim_std": float(np.std(random_sim)),
        "same_sim_mean": float(np.mean(same_sim)),
        "invariance_signal_mean": float(np.mean(invariance_signal)),
        "invariance_signal_std": float(np.std(invariance_signal)),
        "invariance_signal_positive_frac": float(np.mean(invariance_signal > 0)),
    }
    
    print(f"  [{arm_name}] matched={result['matched_sim_mean']:.4f} "
          f"random={result['random_sim_mean']:.4f} "
          f"invariance={result['invariance_signal_mean']:.4f}", flush=True)
    
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--checkpoint", default="chck_100M")
    ap.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS.keys()))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load pairs
    pairs = load_pairs(PAIRS_FILE, args.n_pairs)
    print(f"Loaded {len(pairs)} paraphrase pairs", flush=True)
    print(f"  Mean source_words: {np.mean([p['source_words'] for p in pairs]):.1f}", flush=True)
    print(f"  Mean rewrite_words: {np.mean([p['rewrite_words'] for p in pairs]):.1f}", flush=True)
    print(f"  Mean content_overlap: {np.mean([p['content_overlap'] for p in pairs]):.3f}", flush=True)
    
    if args.plan_only:
        print(json.dumps({
            "status": "INVARIANCE_PROBE_PLAN",
            "n_pairs": len(pairs),
            "arms": args.arms,
            "checkpoint": args.checkpoint,
        }, indent=2), flush=True)
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    
    for arm_name in args.arms:
        if arm_name not in ARM_CONFIGS:
            print(f"  [SKIP] Unknown arm: {arm_name}", flush=True)
            continue
        run_dir = ARM_CONFIGS[arm_name]
        t0 = time.time()
        r = run_probe(arm_name, run_dir, args.checkpoint, pairs, device, args.gpu)
        r["elapsed_sec"] = round(time.time() - t0, 1)
        results.append(r)
    
    # Summary comparison
    print(f"\n{'='*70}", flush=True)
    print(f"IDENTITY-UNDER-VARIATION PROBE RESULTS ({args.checkpoint})", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Arm':<15s} {'Matched':>10s} {'Random':>10s} {'Invariance':>12s} {'Positive%':>10s}", flush=True)
    print(f"{'-'*57}", flush=True)
    for r in results:
        print(f"{r['arm']:<15s} {r['matched_sim_mean']:>10.4f} {r['random_sim_mean']:>10.4f} "
              f"{r['invariance_signal_mean']:>12.4f} {r['invariance_signal_positive_frac']:>10.1%}", flush=True)
    
    # VIEW vs CLEAN comparison
    view_arms = [r for r in results if "V_" in r["arm"]]
    clean_arms = [r for r in results if "C_" in r["arm"]]
    repeat_arms = [r for r in results if "R_" in r["arm"]]
    
    if view_arms and clean_arms:
        view_mean = np.mean([r["invariance_signal_mean"] for r in view_arms])
        clean_mean = np.mean([r["invariance_signal_mean"] for r in clean_arms])
        print(f"\nVIEW mean invariance: {view_mean:.4f}")
        print(f"CLEAN mean invariance: {clean_mean:.4f}")
        print(f"VIEW - CLEAN gap: {view_mean - clean_mean:.4f}")
        
        if repeat_arms:
            repeat_mean = np.mean([r["invariance_signal_mean"] for r in repeat_arms])
            print(f"REPEAT mean invariance: {repeat_mean:.4f}")
    
    # Save
    summary = {
        "status": "INVARIANCE_PROBE_DONE",
        "n_pairs": len(pairs),
        "checkpoint": args.checkpoint,
        "results": results,
    }
    summary_path = OUT_DIR / "invariance_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved: {summary_path.relative_to(ROOT)}", flush=True)
    print(json.dumps({"status": summary["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
