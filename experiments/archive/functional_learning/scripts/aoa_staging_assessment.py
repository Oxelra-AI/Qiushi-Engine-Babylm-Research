#!/usr/bin/env python3
"""research: AoA staging assessment and early-stop checkpoint builder.

Assesses the AoA checkpoint situation for coherent86 and dense candidates under
the BabyLM Strict-Small early-stopping convention (README lines 209, 218):
"we require [...] checkpoints [...] or up until the one you trained"

Training ancestry:
  research (frontier_consolidation): DebertaV2ForMaskedLM + slow adapter, 100M words, all chck_1M..chck_100M
  chck_82M: slow-adapter-only 82M endpoint
  coherent86: private-adapter replay from chck_82M, ~86.005M total words
  dense_seed62064: dense focus from coherent86, ~89.168M total words

Under early stopping:
  - coherent86 (86.005M): needs chck_1M..chck_9M, chck_10M..chck_80M = 17 checkpoints
  - dense (89.168M): same 17 checkpoints (80M is last 10M milestone ≤ 89M)
  Both share identical ancestral checkpoints 1M-80M from research.
  The final endpoint differs: coherent86 final vs dense final.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone

A02 = Path("experiments/archive/frontier_consolidation")
A01 = Path("experiments/archive/functional_learning")
OUT_ROOT = A01 / "data/aoa_staging_assessment"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# research ancestral ladder
LADDER = A02 / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model"

# Final endpoints
COHERENT86 = A01 / "data/automodel_repair/repaired_coherent86_alpha075"
DENSE_64 = A01 / "data/automodel_repair/repaired_dense_seed62064_u0080"
DENSE_65 = A01 / "data/automodel_repair/repaired_dense_seed62065_u0080"

# Generate early-stop checkpoint list
# strict-small: chck_1M..chck_9M, chck_10M..chck_100M
# early stop at ~86M or ~89M: need through chck_80M
EARLY_STOP_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i*10}M" for i in range(1, 9)]  # 1M-9M + 10M-80M = 17
EARLY_STOP_WORDS = [i * 1_000_000 for i in range(1, 10)] + [10 * i * 1_000_000 for i in range(1, 9)]

FULL_STRICT_SMALL_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i*10}M" for i in range(1, 11)]


def assess():
    results = {
        "status": "AOA_STAGING_ASSESSMENT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "convention": "BabyLM Strict-Small early-stop (README lines 209, 218)",
        "full_revisions": len(FULL_STRICT_SMALL_STEPS),
        "early_stop_revisions": len(EARLY_STOP_STEPS),
    }
    
    # Check ancestral checkpoint availability
    ancestral_status = {}
    for step in EARLY_STOP_STEPS:
        ckpt_dir = LADDER / step
        config = ckpt_dir / "config.json"
        weights = ckpt_dir / "model.safetensors"
        
        available = ckpt_dir.is_dir()
        has_config = config.is_file() if available else False
        has_weights = weights.is_file() if available else False
        
        ancestral_status[step] = {
            "available": available,
            "has_config": has_config,
            "has_weights": has_weights,
            "complete": available and has_config and has_weights,
        }
        
        if has_config:
            with open(config) as f:
                cfg = json.load(f)
            ancestral_status[step]["architectures"] = cfg.get("architectures", [])
            ancestral_status[step]["auto_map"] = cfg.get("auto_map", {})
    
    results["ancestral_checkpoints"] = ancestral_status
    n_available = sum(1 for v in ancestral_status.values() if v["complete"])
    results["ancestral_available"] = n_available
    results["ancestral_complete"] = n_available == len(EARLY_STOP_STEPS)
    
    # Check final endpoints
    endpoints = {}
    for label, path in [("coherent86", COHERENT86), ("dense_seed62064", DENSE_64), ("dense_seed62065", DENSE_65)]:
        available = path.is_dir()
        has_config = (path / "config.json").is_file() if available else False
        has_weights = (path / "model.safetensors").is_file() if available else False
        
        info = {
            "available": available,
            "has_config": has_config,
            "has_weights": has_weights,
            "complete": available and has_config and has_weights,
        }
        
        if has_config:
            with open(path / "config.json") as f:
                cfg = json.load(f)
            info["architectures"] = cfg.get("architectures", [])
            info["auto_map"] = cfg.get("auto_map", {})
        
        endpoints[label] = info
    
    results["endpoints"] = endpoints
    
    # Key findings
    findings = []
    
    if results["ancestral_complete"]:
        findings.append(f"All {len(EARLY_STOP_STEPS)} ancestral checkpoints available from research ladder")
        
        # Check architecture consistency
        archs = set()
        for step, status in ancestral_status.items():
            archs.update(status.get("architectures", []))
        findings.append(f"Ancestral architectures: {archs}")
        findings.append("Ancestral checkpoints are stock DebertaV2ForMaskedLM + slow adapter")
        findings.append("Final endpoints are FrozenSlowPrivateDebertaV2ForMaskedLM + both adapters")
        findings.append("Architecture change at 82M is genuine training history, not reconstruction")
    
    findings.append(f"Coherent86 total words: ~86,005,295 → last 10M milestone: 80M → 17 checkpoints needed")
    findings.append(f"Dense total words: ~89,168,037 → last 10M milestone: 80M → 17 checkpoints needed")
    findings.append("Both share identical ancestral ladder; only final endpoint differs")
    findings.append("AoA difference between coherent86 and dense will be very small (same trajectory through 80M)")
    
    results["findings"] = findings
    
    # Staging plan
    results["staging_plan"] = {
        "checkpoints_to_stage": EARLY_STOP_STEPS,
        "word_exposures": EARLY_STOP_WORDS,
        "source": str(LADDER),
        "method": "symlink ancestral + copy repaired final endpoint",
        "architecture_note": (
            "Ancestral checkpoints use DebertaV2ForMaskedLM with slow adapter (scale 1.75). "
            "Final endpoint uses FrozenSlowPrivateDebertaV2ForMaskedLM with slow + private adapters. "
            "AoA evaluation loads each with AutoModelForMaskedLM which covers both architectures."
        ),
        "early_stop_note": (
            "The official code hardcodes 19 revisions (STRICT_SMALL_FAST_REVISIONS in collate_preds.py). "
            "The README says to edit for early stopping. A custom AoA runner avoids modifying the "
            "official code by loading each checkpoint from its own directory."
        ),
    }
    
    # Save
    out_json = OUT_ROOT / "aoa_staging_assessment.json"
    out_md = (OUT_ROOT.parents[4] / 'research/documents/functional_learning/data/aoa_staging_assessment/aoa_staging_assessment.md')
    
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(out_md, "w") as f:
        f.write("# research AoA Staging Assessment\n\n")
        f.write("## Convention\n\n")
        f.write("BabyLM Strict-Small early-stopping: checkpoints up to the amount trained.\n")
        f.write(f"Full revisions: {len(FULL_STRICT_SMALL_STEPS)}, Early-stop revisions: {len(EARLY_STOP_STEPS)}\n\n")
        f.write("## Ancestral Checkpoints\n\n")
        f.write(f"Available: {n_available}/{len(EARLY_STOP_STEPS)} from research ladder\n\n")
        for step, status in ancestral_status.items():
            mark = "✓" if status["complete"] else "✗"
            f.write(f"- {mark} {step}: arch={status.get('architectures', '?')}\n")
        f.write("\n## Endpoints\n\n")
        for label, info in endpoints.items():
            mark = "✓" if info["complete"] else "✗"
            f.write(f"- {mark} {label}: arch={info.get('architectures', '?')}\n")
        f.write("\n## Findings\n\n")
        for finding in findings:
            f.write(f"- {finding}\n")
        f.write("\n## Next Steps\n\n")
        f.write("1. Write custom AoA runner that loads each checkpoint from its own directory\n")
        f.write("2. Stage symlinks to ancestral checkpoints + repaired endpoints\n")
        f.write("3. Run AoA surprisal computation on all staged checkpoints\n")
        f.write("4. Compute AoA score using official AoAEvaluator\n")
    
    print(json.dumps({
        "ancestral_complete": results["ancestral_complete"],
        "n_ancestral": n_available,
        "n_needed": len(EARLY_STOP_STEPS),
        "endpoints_complete": all(v["complete"] for v in endpoints.values()),
    }, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    assess()
