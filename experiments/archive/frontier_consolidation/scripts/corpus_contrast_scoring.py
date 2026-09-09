#!/usr/bin/env python3
"""research Phase 2+3: Score frozen corpus contrasts and analyze within-coordinate.

Loads the frozen corpus contrast set (built independently of model information),
scores all four context-target combinations on existing legal16-80M checkpoints,
and tests whether within-coordinate interaction tracks broad competence.

within-coordinate comparisons get greatest weight;
if interaction does not consistently track broad transfer on independently
built contrasts, return to broader representation search.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast

SCRIPT_DIR = _public_path('experiments/archive/frontier_consolidation/scripts')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
STUDY = _public_path('experiments/archive/frontier_consolidation')
ROOT = _public_path('.')  # project root
CONTRAST_FILE = _public_path('experiments/archive/frontier_consolidation/data/frozen_corpus_contrasts/frozen_corpus_contrasts.jsonl')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/corpus_contrast_scoring')

# Within-coordinate legal16-80M models only (same tokenizer, architecture, seed)
MODELS = {
    "a02_legal16_clean_80M": {
        "ckpt": "experiments/archive/frontier_consolidation/training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M",
        "broad_cheap7": 41.6022,
        "note": "clean-Qwen control",
    },
    "a02_legal16_reinvest_80M": {
        "ckpt": "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
        "broad_cheap7": 42.9486,
        "note": "compact-view reinvest reference",
    },
    "a02_step075_strict_innov_80M": {
        "ckpt": "experiments/archive/frontier_consolidation/training/runs/strict_content_innovation_wwm_reinvest_seed43022_80M/hf_model/chck_80M",
        "broad_cheap7": 41.8957,
        "note": "closed innovation-probability route",
    },
}


def load_contrasts(path: pathlib.Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_tokenizer(model_path: pathlib.Path):
    try:
        tok = AutoTokenizer.from_pretrained(str(model_path), padding_side="right", trust_remote_code=True)
    except Exception:
        tok = PreTrainedTokenizerFast.from_pretrained(str(model_path), padding_side="right")
    if tok.pad_token_id is None:
        if tok.cls_token_id is not None:
            tok.pad_token_id = tok.cls_token_id
        elif tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        else:
            tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.mask_token_id is None:
        raise RuntimeError(f"Tokenizer at {model_path} has no mask_token_id")
    return tok


def combo_encoding(tokenizer, context: str, target: str, max_length: int = 256) -> dict:
    completion = " " + target
    sentence = " ".join([context, target])
    enc = tokenizer(sentence, return_offsets_mapping=True, truncation=True, max_length=max_length)
    tokens = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = enc["offset_mapping"]
    start_char_idx = len(sentence) - len(completion)
    phrase_indices = []
    target_tokens = []
    for i, (start, end) in enumerate(offsets):
        if end > start_char_idx:
            phrase_indices.append(i)
            target_tokens.append(tokens[i])
    if not phrase_indices:
        return {"valid": False, "tokens": [], "attn": [], "indices": [], "targets": []}
    return {"valid": True, "tokens": tokens, "attn": attn, "indices": phrase_indices, "targets": target_tokens}


def score_encoded_batch(model, tokenizer, encs: list[dict], device, batch_size: int = 128) -> list[float]:
    records = []
    invalid = [False for _ in encs]
    for ex_i, enc in enumerate(encs):
        if not enc["valid"]:
            invalid[ex_i] = True
            continue
        toks = enc["tokens"]
        attn = enc["attn"]
        for idx, tgt in zip(enc["indices"], enc["targets"]):
            masked = list(toks)
            masked[idx] = tokenizer.mask_token_id
            records.append((ex_i, masked, list(attn), idx, int(tgt)))
    sums = [0.0 for _ in encs]
    if not records:
        return [float("nan") for _ in encs]
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        max_len = max(len(r[1]) for r in batch)
        input_ids = []
        attn_mask = []
        idxs = []
        tgts = []
        exs = []
        for ex_i, toks, attn, idx, tgt in batch:
            pad_n = max_len - len(toks)
            input_ids.append(toks + [pad_id] * pad_n)
            attn_mask.append(attn + [0] * pad_n)
            idxs.append(idx)
            tgts.append(tgt)
            exs.append(ex_i)
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
                attention_mask=torch.tensor(attn_mask, dtype=torch.long, device=device),
            )
            logits = out["logits"] if isinstance(out, dict) or hasattr(out, "__getitem__") else out.logits
            if logits.size(1) != max_len:
                logits = logits[:, -max_len:]
            batch_idx = torch.arange(logits.shape[0], device=device)
            masked_logits = logits[batch_idx, torch.tensor(idxs, dtype=torch.long, device=device)]
            lp = F.log_softmax(masked_logits, dim=-1)
            vals = torch.gather(lp, -1, torch.tensor(tgts, dtype=torch.long, device=device).unsqueeze(-1)).squeeze(-1).detach().cpu().tolist()
        for ex_i, val in zip(exs, vals):
            sums[ex_i] += float(val)
    for i, bad in enumerate(invalid):
        if bad:
            sums[i] = float("nan")
    return sums


def safe_mean(xs):
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Phase 2: Score ──
    contrasts = load_contrasts(CONTRAST_FILE)
    n_contrasts = len(contrasts)
    sha = hashlib.sha256(open(CONTRAST_FILE, "rb").read()).hexdigest()
    print(json.dumps({"event": "loaded_contrasts", "n": n_contrasts, "sha256": sha}))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    all_model_results = {}
    
    for model_name, cfg in MODELS.items():
        ckpt = ROOT / cfg["ckpt"] if not pathlib.Path(cfg["ckpt"]).is_absolute() else pathlib.Path(cfg["ckpt"])
        print(json.dumps({"event": "model_start", "model": model_name, "rows": n_contrasts}))
        t0 = time.time()
        
        tokenizer = load_tokenizer(ckpt)
        model = AutoModelForMaskedLM.from_pretrained(str(ckpt), torch_dtype=dtype).to(device).eval()
        
        # Encode all four combos for all contrasts
        ROW_BATCH = 64
        records = []
        for batch_start in range(0, n_contrasts, ROW_BATCH):
            batch_rows = contrasts[batch_start:batch_start + ROW_BATCH]
            encs_11, encs_21, encs_12, encs_22 = [], [], [], []
            for c in batch_rows:
                encs_11.append(combo_encoding(tokenizer, c["Context1"], c["Target1"]))
                encs_21.append(combo_encoding(tokenizer, c["Context2"], c["Target1"]))
                encs_12.append(combo_encoding(tokenizer, c["Context1"], c["Target2"]))
                encs_22.append(combo_encoding(tokenizer, c["Context2"], c["Target2"]))
            
            s11 = score_encoded_batch(model, tokenizer, encs_11, device)
            s21 = score_encoded_batch(model, tokenizer, encs_21, device)
            s12 = score_encoded_batch(model, tokenizer, encs_12, device)
            s22 = score_encoded_batch(model, tokenizer, encs_22, device)
            
            for i, c in enumerate(batch_rows):
                official_margin = s11[i] - s21[i]
                interaction = (s11[i] + s22[i]) - (s12[i] + s21[i])
                correct = 1 if s11[i] >= s21[i] else 0
                records.append({
                    "uid": c["uid"],
                    "family": c["family"],
                    "source_a": c.get("source_a", ""),
                    "source_b": c.get("source_b", ""),
                    "s11": s11[i], "s21": s21[i], "s12": s12[i], "s22": s22[i],
                    "official_margin": official_margin,
                    "interaction": interaction,
                    "correct": correct,
                })
        
        elapsed = time.time() - t0
        
        # Per-family summaries
        by_family = {}
        for fam in ["within_source", "cross_source"]:
            fam_recs = [r for r in records if r["family"] == fam]
            if fam_recs:
                by_family[fam] = {
                    "n": len(fam_recs),
                    "accuracy": 100.0 * sum(r["correct"] for r in fam_recs) / len(fam_recs),
                    "mean_interaction": safe_mean([r["interaction"] for r in fam_recs]),
                    "mean_margin": safe_mean([r["official_margin"] for r in fam_recs]),
                }
        
        overall = {
            "n": len(records),
            "accuracy": 100.0 * sum(r["correct"] for r in records) / len(records),
            "mean_interaction": safe_mean([r["interaction"] for r in records]),
            "mean_margin": safe_mean([r["official_margin"] for r in records]),
        }
        
        result = {
            "model_name": model_name,
            "broad_cheap7": cfg["broad_cheap7"],
            "note": cfg["note"],
            "elapsed_sec": round(elapsed, 2),
            "overall": overall,
            "by_family": by_family,
            "records": records,
        }
        all_model_results[model_name] = result
        
        # Save per-model
        with open(OUT_DIR / f"{model_name}.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print(json.dumps({"event": "model_done", "model": model_name,
                          "accuracy": overall["accuracy"],
                          "mean_interaction": overall["mean_interaction"],
                          "elapsed_sec": round(elapsed, 2)}))
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    
    # ── Phase 3: Analyze within-coordinate ──
    print("\n" + "="*60)
    print("WITHIN-COORDINATE ANALYSIS (legal16 80M)")
    print("="*60)
    
    model_names = sorted(all_model_results.keys())
    
    # Overall interaction vs broad_cheap7
    interactions = [all_model_results[m]["overall"]["mean_interaction"] for m in model_names]
    broad_vals = [all_model_results[m]["broad_cheap7"] for m in model_names]
    
    for m in model_names:
        r = all_model_results[m]
        print(f"\n{m}: cheap7={r['broad_cheap7']}, corpus_interaction={r['overall']['mean_interaction']:.6f}, "
              f"corpus_accuracy={r['overall']['accuracy']:.2f}")
        for fam, fv in r["by_family"].items():
            print(f"  {fam}: interaction={fv['mean_interaction']:.6f}, accuracy={fv['accuracy']:.2f}")
    
    # Check consistency: does reinvest > research > clean in interaction, matching broad order?
    print("\n--- Within-coordinate ordering ---")
    rank_broad = sorted(model_names, key=lambda m: all_model_results[m]["broad_cheap7"], reverse=True)
    rank_interaction = sorted(model_names, key=lambda m: all_model_results[m]["overall"]["mean_interaction"], reverse=True)
    rank_within_src = sorted(model_names, key=lambda m: all_model_results[m]["by_family"].get("within_source", {}).get("mean_interaction", 0), reverse=True)
    rank_cross_src = sorted(model_names, key=lambda m: all_model_results[m]["by_family"].get("cross_source", {}).get("mean_interaction", 0), reverse=True)
    
    print(f"  Rank by broad_cheap7:       {rank_broad}")
    print(f"  Rank by overall interaction: {rank_interaction}")
    print(f"  Rank by within_src interaction: {rank_within_src}")
    print(f"  Rank by cross_src interaction:  {rank_cross_src}")
    
    # Pairwise treatment deltas
    print("\n--- Treatment deltas (reinvest - clean at 80M) ---")
    for fam in ["overall", "within_source", "cross_source"]:
        r_val = all_model_results["a02_legal16_reinvest_80M"]
        c_val = all_model_results["a02_legal16_clean_80M"]
        if fam == "overall":
            delta_int = r_val["overall"]["mean_interaction"] - c_val["overall"]["mean_interaction"]
            delta_acc = r_val["overall"]["accuracy"] - c_val["overall"]["accuracy"]
        else:
            r_fam = r_val["by_family"].get(fam, {})
            c_fam = c_val["by_family"].get(fam, {})
            delta_int = (r_fam.get("mean_interaction", 0) or 0) - (c_fam.get("mean_interaction", 0) or 0)
            delta_acc = (r_fam.get("accuracy", 0) or 0) - (c_fam.get("accuracy", 0) or 0)
        print(f"  {fam}: Δinteraction={delta_int:+.6f}, Δaccuracy={delta_acc:+.2f}")
    
    # research vs reinvest delta
    print("\n--- research vs reinvest delta ---")
    for fam in ["overall", "within_source", "cross_source"]:
        s_val = all_model_results["a02_step075_strict_innov_80M"]
        r_val = all_model_results["a02_legal16_reinvest_80M"]
        if fam == "overall":
            delta_int = s_val["overall"]["mean_interaction"] - r_val["overall"]["mean_interaction"]
            delta_acc = s_val["overall"]["accuracy"] - r_val["overall"]["accuracy"]
        else:
            s_fam = s_val["by_family"].get(fam, {})
            r_fam = r_val["by_family"].get(fam, {})
            delta_int = (s_fam.get("mean_interaction", 0) or 0) - (r_fam.get("mean_interaction", 0) or 0)
            delta_acc = (s_fam.get("accuracy", 0) or 0) - (r_fam.get("accuracy", 0) or 0)
        print(f"  {fam}: Δinteraction={delta_int:+.6f}, Δaccuracy={delta_acc:+.2f}")
    
    # Save comparison summary
    comparison = {
        "status": "CORPUS_CONTRAST_SCORED",
        "frozen_contrasts": {
            "path": str(CONTRAST_FILE),
            "sha256": sha,
            "n_contrasts": n_contrasts,
        },
        "models": {},
        "within_coordinate_consistency": {},
    }
    for m in model_names:
        r = all_model_results[m]
        comparison["models"][m] = {
            "broad_cheap7": r["broad_cheap7"],
            "note": r["note"],
            "overall_interaction": r["overall"]["mean_interaction"],
            "overall_accuracy": r["overall"]["accuracy"],
            "by_family": {fam: {"interaction": fv["mean_interaction"], "accuracy": fv["accuracy"]}
                          for fam, fv in r["by_family"].items()},
        }
    
    # Check: do all three families (overall, within_source, cross_source) rank models consistently with broad_cheap7?
    concordance = {
        "broad_rank": rank_broad,
        "overall_interaction_rank": rank_interaction,
        "within_src_interaction_rank": rank_within_src,
        "cross_src_interaction_rank": rank_cross_src,
        "all_families_match_broad": rank_interaction == rank_broad and rank_within_src == rank_broad and rank_cross_src == rank_broad,
        "overall_matches_broad": rank_interaction == rank_broad,
        "within_src_matches_broad": rank_within_src == rank_broad,
        "cross_src_matches_broad": rank_cross_src == rank_broad,
    }
    comparison["within_coordinate_consistency"] = concordance
    
    with open(_public_path('experiments/archive/frontier_consolidation/data/corpus_contrast_scoring/corpus_contrast_comparison.json'), "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Final decision
    print("\n" + "="*60)
    if concordance["all_families_match_broad"]:
        print("RESULT: All contrast families rank models consistently with broad cheap7.")
        print("Interaction on independently built corpus contrasts tracks within-coordinate broad transfer.")
    else:
        print("RESULT: Interaction ordering on corpus contrasts does NOT consistently match broad cheap7 ordering.")
        print("return to broader representation search.")
    print("="*60)


if __name__ == "__main__":
    main()
