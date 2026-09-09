#!/usr/bin/env python3
"""research: diagnose the temporal-change bottleneck from research.

Three convergent diagnostics:
1. Representational similarity: do pretrained CLS embeddings distinguish
   "Before" vs "After" hypothesis variants?  If not, the model literally
   cannot separate temporal queries and changed-focal rows create
   irreconcilable label conflicts.
2. Context temporal cue diagnostic: does the pretrained model represent
   "earlier" vs "later" context sentences differently?  This tests whether
   temporal selection is representationally possible.
3. Synthetic fitting test: can the model fit changed-only rows without any
   stable base?  This separates "base stable prior dominance" from
   "changed rows are representationally unlearnable".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
DEFAULT_MODEL = "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M"

ALIAS_PAIRS = [
    ("Arlo", "Soren"), ("Rowan", "Kael"), ("Lysander", "Percival"),
    ("Emeric", "Thaddeus"), ("Caius", "Leander"), ("Dorian", "Ezekiel"),
]

HYP_BEFORE_TEMPLATES = [
    "Before the later ranking update, {X} held the higher ranking than {Y}.",
    "Initially, {X} outranked {Y}.",
]
HYP_AFTER_TEMPLATES = [
    "After the later ranking update, {X} held the higher ranking than {Y}.",
    "Later, {X} outranked {Y}.",
]

INIT_CTX_TEMPLATES = [
    "In the earlier ATP ranking, {H} was above {Lo}.",
    "At the first ranking date, {H} stood higher than {Lo}.",
    "In the prior ranking snapshot, {H} had the better position than {Lo}.",
]
LATER_CTX_TEMPLATES = [
    "In the later ATP ranking, {H} was above {Lo}.",
    "At the later ranking date, {H} stood higher than {Lo}.",
    "In the subsequent ranking snapshot, {H} had the better position than {Lo}.",
]


def cls_embed(model, tokenizer, texts, device, batch_size=16):
    """Get CLS embeddings for a list of texts."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        emb = out.last_hidden_state[:, 0, :]  # CLS
        all_embs.append(emb.cpu().float())
    return torch.cat(all_embs, dim=0)


def diagnostic_1_hypothesis_similarity(model, tokenizer, device):
    """Test whether Before vs After hypothesis variants are representationally distinct."""
    results = []
    for x_name, y_name in ALIAS_PAIRS:
        for tpl_idx in range(min(len(HYP_BEFORE_TEMPLATES), len(HYP_AFTER_TEMPLATES))):
            h_before = HYP_BEFORE_TEMPLATES[tpl_idx].format(X=x_name, Y=y_name)
            h_after = HYP_AFTER_TEMPLATES[tpl_idx].format(X=x_name, Y=y_name)
            # Also compute same-time different-direction
            h_before_rev = HYP_BEFORE_TEMPLATES[tpl_idx].format(X=y_name, Y=x_name)
            h_after_rev = HYP_AFTER_TEMPLATES[tpl_idx].format(X=y_name, Y=x_name)
            
            embs = cls_embed(model, tokenizer, [h_before, h_after, h_before_rev, h_after_rev], device)
            
            cos_before_after = F.cosine_similarity(embs[0:1], embs[1:2]).item()
            cos_before_rev = F.cosine_similarity(embs[0:1], embs[2:3]).item()
            cos_after_rev = F.cosine_similarity(embs[1:2], embs[3:4]).item()
            cos_same_time_diff_dir = F.cosine_similarity(embs[0:1], embs[2:3]).item()
            
            results.append({
                "x": x_name, "y": y_name, "tpl_idx": tpl_idx,
                "cos_before_vs_after_same_dir": cos_before_after,
                "cos_before_vs_before_rev_dir": cos_before_rev,
                "cos_after_vs_after_rev_dir": cos_after_rev,
                "cos_same_time_diff_dir": cos_same_time_diff_dir,
                "h_before": h_before, "h_after": h_after,
            })
    
    ba_cos = [r["cos_before_vs_after_same_dir"] for r in results]
    dir_cos = [r["cos_before_vs_before_rev_dir"] for r in results]
    
    return {
        "n_pairs": len(results),
        "before_vs_after_cos_mean": float(np.mean(ba_cos)),
        "before_vs_after_cos_std": float(np.std(ba_cos)),
        "before_vs_before_rev_cos_mean": float(np.mean(dir_cos)),
        "before_vs_before_rev_cos_std": float(np.std(dir_cos)),
        "diagnostic": "before_after_nearly_identical" if np.mean(ba_cos) > 0.95 else
                      "before_after_moderately_similar" if np.mean(ba_cos) > 0.85 else
                      "before_after_distinguishable",
        "pairs": results
    }


def diagnostic_2_context_temporal_cues(model, tokenizer, device):
    """Test whether earlier vs later context sentences are representationally distinct."""
    results = []
    for h_name, lo_name in ALIAS_PAIRS:
        for tpl_idx in range(min(len(INIT_CTX_TEMPLATES), len(LATER_CTX_TEMPLATES))):
            ctx_init = INIT_CTX_TEMPLATES[tpl_idx].format(H=h_name, Lo=lo_name)
            ctx_later = LATER_CTX_TEMPLATES[tpl_idx].format(H=h_name, Lo=lo_name)
            # Same context but reversed entities
            ctx_init_rev = INIT_CTX_TEMPLATES[tpl_idx].format(H=lo_name, Lo=h_name)
            ctx_later_rev = LATER_CTX_TEMPLATES[tpl_idx].format(H=lo_name, Lo=h_name)
            
            embs = cls_embed(model, tokenizer, [ctx_init, ctx_later, ctx_init_rev, ctx_later_rev], device)
            
            cos_init_later = F.cosine_similarity(embs[0:1], embs[1:2]).item()
            cos_init_init_rev = F.cosine_similarity(embs[0:1], embs[2:3]).item()
            cos_later_later_rev = F.cosine_similarity(embs[1:2], embs[3:4]).item()
            
            results.append({
                "h": h_name, "lo": lo_name, "tpl_idx": tpl_idx,
                "cos_init_vs_later_same_dir": cos_init_later,
                "cos_init_vs_init_rev_dir": cos_init_init_rev,
                "cos_later_vs_later_rev_dir": cos_later_later_rev,
            })
    
    il_cos = [r["cos_init_vs_later_same_dir"] for r in results]
    dir_cos = [r["cos_init_vs_init_rev_dir"] for r in results]
    
    return {
        "n_pairs": len(results),
        "init_vs_later_cos_mean": float(np.mean(il_cos)),
        "init_vs_later_cos_std": float(np.std(il_cos)),
        "init_vs_init_revdir_cos_mean": float(np.mean(dir_cos)),
        "init_vs_init_revdir_cos_std": float(np.std(dir_cos)),
        "temporal_separation": float(np.mean(dir_cos)) - float(np.mean(il_cos)),
        "diagnostic": "temporal_cues_collapsed" if np.mean(il_cos) > 0.95 else
                      "temporal_cues_weakly_distinct" if np.mean(il_cos) > 0.85 else
                      "temporal_cues_distinct",
        "pairs": results
    }


def diagnostic_3_full_context_temporal_selection(model, tokenizer, device):
    """Test whether full temporal context (init + later) with before/after hyp
    can be distinguished.  This is the closest to the actual training setting."""
    results = []
    for h_name, lo_name in ALIAS_PAIRS[:3]:
        for tpl_idx in range(min(len(INIT_CTX_TEMPLATES), len(LATER_CTX_TEMPLATES))):
            # Stable world: init = later (same direction)
            ctx_stable = (
                f"Focal initial ranking: {INIT_CTX_TEMPLATES[tpl_idx].format(H=h_name, Lo=lo_name)} "
                f"Focal later ranking: {LATER_CTX_TEMPLATES[tpl_idx].format(H=h_name, Lo=lo_name)}"
            )
            # Changed world: init ≠ later (reversed direction)
            ctx_changed = (
                f"Focal initial ranking: {INIT_CTX_TEMPLATES[tpl_idx].format(H=h_name, Lo=lo_name)} "
                f"Focal later ranking: {LATER_CTX_TEMPLATES[tpl_idx].format(H=lo_name, Lo=h_name)}"
            )
            
            for hyp_tpl_idx in range(min(len(HYP_BEFORE_TEMPLATES), len(HYP_AFTER_TEMPLATES))):
                hyp_before = HYP_BEFORE_TEMPLATES[hyp_tpl_idx].format(X=h_name, Y=lo_name)
                hyp_after = HYP_AFTER_TEMPLATES[hyp_tpl_idx].format(X=h_name, Y=lo_name)
                
                texts = [
                    f"{ctx_stable} [SEP] {hyp_before}",   # stable + before
                    f"{ctx_stable} [SEP] {hyp_after}",    # stable + after (same answer)
                    f"{ctx_changed} [SEP] {hyp_before}",  # changed + before (init answer)
                    f"{ctx_changed} [SEP] {hyp_after}",   # changed + after (DIFF answer)
                ]
                
                embs = cls_embed(model, tokenizer, texts, device)
                
                # Within stable: before vs after should be similar (same answer)
                cos_stable_ba = F.cosine_similarity(embs[0:1], embs[1:2]).item()
                # Within changed: before vs after should differ (different answers!)
                cos_changed_ba = F.cosine_similarity(embs[2:3], embs[3:4]).item()
                # Cross: stable-before vs changed-before (same answer, diff context)
                cos_cross_before = F.cosine_similarity(embs[0:1], embs[2:3]).item()
                # Cross: stable-after vs changed-after (DIFF answers)
                cos_cross_after = F.cosine_similarity(embs[1:2], embs[3:4]).item()
                
                results.append({
                    "ctx_tpl": tpl_idx, "hyp_tpl": hyp_tpl_idx,
                    "cos_stable_before_vs_after": cos_stable_ba,
                    "cos_changed_before_vs_after": cos_changed_ba,
                    "cos_cross_before": cos_cross_before,
                    "cos_cross_after": cos_cross_after,
                    "ba_gap_stable_vs_changed": cos_stable_ba - cos_changed_ba,
                })
    
    stable_ba = [r["cos_stable_before_vs_after"] for r in results]
    changed_ba = [r["cos_changed_before_vs_after"] for r in results]
    gap = [r["ba_gap_stable_vs_changed"] for r in results]
    
    return {
        "stable_before_after_cos_mean": float(np.mean(stable_ba)),
        "changed_before_after_cos_mean": float(np.mean(changed_ba)),
        "ba_gap_stable_vs_changed_mean": float(np.mean(gap)),
        "diagnostic": "before_after_representationally_identical_in_context" if np.mean(gap) < 0.01 else
                      "weak_contextual_temporal_separation" if np.mean(gap) < 0.05 else
                      "context_enables_temporal_selection",
        "pairs": results
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = WS / "data/temporal_bottleneck_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(json.dumps({"status": "DIAGNOSTIC_START", "device": device}), flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModel.from_pretrained(DEFAULT_MODEL).to(device).eval()
    
    # Diagnostic 1: hypothesis similarity
    d1 = diagnostic_1_hypothesis_similarity(model, tokenizer, device)
    print(json.dumps({"diagnostic": "hypothesis_similarity",
                       "before_after_cos": d1["before_vs_after_cos_mean"],
                       "direction_rev_cos": d1["before_vs_before_rev_cos_mean"],
                       "result": d1["diagnostic"]}), flush=True)
    
    # Diagnostic 2: context temporal cues
    d2 = diagnostic_2_context_temporal_cues(model, tokenizer, device)
    print(json.dumps({"diagnostic": "context_temporal_cues",
                       "init_later_cos": d2["init_vs_later_cos_mean"],
                       "dir_rev_cos": d2["init_vs_init_revdir_cos_mean"],
                       "temporal_sep": d2["temporal_separation"],
                       "result": d2["diagnostic"]}), flush=True)
    
    # Diagnostic 3: full context temporal selection
    d3 = diagnostic_3_full_context_temporal_selection(model, tokenizer, device)
    print(json.dumps({"diagnostic": "full_context_temporal",
                       "stable_ba_cos": d3["stable_before_after_cos_mean"],
                       "changed_ba_cos": d3["changed_before_after_cos_mean"],
                       "gap": d3["ba_gap_stable_vs_changed_mean"],
                       "result": d3["diagnostic"]}), flush=True)
    
    del model
    torch.cuda.empty_cache()
    
    summary = {
        "status": "DIAGNOSTIC_DONE",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis_similarity": {k: v for k, v in d1.items() if k != "pairs"},
        "context_temporal_cues": {k: v for k, v in d2.items() if k != "pairs"},
        "full_context_temporal": {k: v for k, v in d3.items() if k != "pairs"},
    }
    
    (out_dir / "temporal_bottleneck_diagnostic.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    # Human-readable
    md = ["# research temporal bottleneck diagnostic\n"]
    md.append(f"## Hypothesis similarity (before vs after)")
    md.append(f"- Before vs After same-direction cosine: **{d1['before_vs_after_cos_mean']:.4f}** (±{d1['before_vs_after_cos_std']:.4f})")
    md.append(f"- Before vs Before reversed-direction cosine: **{d1['before_vs_before_rev_cos_mean']:.4f}** (±{d1['before_vs_before_rev_cos_std']:.4f})")
    md.append(f"- Verdict: **{d1['diagnostic']}**")
    md.append(f"\nThe before/after temporal cue in the hypothesis {'is barely distinguishable' if d1['before_vs_after_cos_mean'] > 0.95 else 'may be distinguishable'} — switching entity direction changes the representation {'more' if d1['before_vs_before_rev_cos_mean'] < d1['before_vs_after_cos_mean'] else 'less'} than switching temporal reference.\n")
    
    md.append(f"## Context temporal cues (earlier vs later)")
    md.append(f"- Init vs Later same-direction cosine: **{d2['init_vs_later_cos_mean']:.4f}** (±{d2['init_vs_later_cos_std']:.4f})")
    md.append(f"- Init vs Init reversed-direction cosine: **{d2['init_vs_init_revdir_cos_mean']:.4f}** (±{d2['init_vs_init_revdir_cos_std']:.4f})")
    md.append(f"- Temporal separation (dir_rev - init_later): **{d2['temporal_separation']:.4f}**")
    md.append(f"- Verdict: **{d2['diagnostic']}**\n")
    
    md.append(f"## Full context + hypothesis temporal selection")
    md.append(f"- Stable world before/after cos: **{d3['stable_before_after_cos_mean']:.4f}**")
    md.append(f"- Changed world before/after cos: **{d3['changed_before_after_cos_mean']:.4f}**")
    md.append(f"- Gap (stable - changed): **{d3['ba_gap_stable_vs_changed_mean']:.4f}**")
    md.append(f"- Verdict: **{d3['diagnostic']}**")
    md.append(f"\nIf the gap is near zero, the pretrained model does not distinguish before/after queries even when the context contains contradictory temporal states.\n")
    
    (out_dir / "temporal_bottleneck_diagnostic.md").write_text("\n".join(md), encoding="utf-8")
    
    print(json.dumps({"status": "DIAGNOSTIC_DONE",
                       "summary_json": str(out_dir / "temporal_bottleneck_diagnostic.json")}), flush=True)


if __name__ == "__main__":
    main()
