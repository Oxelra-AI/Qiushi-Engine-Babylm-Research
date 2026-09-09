#!/usr/bin/env python3
"""research: measure β from calibration arm ladders and recalibrate AoA prediction.

After the two 30M calibration arms complete, this script:
1. Extracts per-word mean surprisals from each calibration checkpoint
   using the official AoA surprisal machinery
2. Computes per-word surprisal differences (calibration - v4) at matched checkpoints
3. Computes per-word log-exposure differences from the schedule predictor
4. Regresses surprisal differences on log-exposure differences to measure β
5. Uses measured β to recalibrate the across-pass AoA predictor

Inputs:
  - V4 ladder surprisals (already computed: research AoA data)
  - Calibration arm checkpoints (30M words, 1M-interval checkpoints)
  - Per-word exposure tables from the prep script

This script runs on CPU for analysis; the surprisal extraction step uses GPU.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, importlib.util, json, math, pathlib, re, sys, time
from collections import Counter
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

ROOT = _public_path('.')
V4_SURPRISAL = _public_path('experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0/collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/surprisal.json')
V4_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder')
PREP_DIR = _public_path('experiments/archive/relation_learning/data/aoa_calibration_prep')
OUT = _public_path('experiments/archive/relation_learning/data/beta_measurement')
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Import research utilities for AoA fitting
PREDICTOR = _public_path('experiments/archive/relation_learning/scripts/aoa_pass1_order_predictor.py')


def load_mod():
    spec = importlib.util.spec_from_file_location("pred", PREDICTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PREDICTOR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pred"] = mod
    spec.loader.exec_module(mod)
    return mod


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_v4_word_surprisals():
    """Load v4 per-word mean surprisals from the official ladder."""
    data = json.loads(V4_SURPRISAL.read_text(encoding="utf-8"))
    # The surprisal data has per-word, per-checkpoint mean surprisals
    # Format: data["words"][word]["steps"][step_str]["mean_surprisal"]
    word_data = {}
    meta = data.get("metadata", {})
    for word, info in data.get("words", {}).items():
        steps = info.get("steps", {})
        word_surprisals = {}
        for step_str, sdata in steps.items():
            ms = sdata.get("mean_surprisal")
            if ms is not None and math.isfinite(ms):
                word_surprisals[int(step_str)] = float(ms)
        if word_surprisals:
            word_data[word] = word_surprisals
    return word_data, meta


def extract_arm_surprisals(arm_dir: pathlib.Path, checkpoints: list[int]):
    """Extract per-word surprisals from calibration arm checkpoints.
    
    This is a placeholder for the actual extraction which requires GPU.
    After running the official extractor on each checkpoint, load the results.
    """
    surprisals = {}
    for ck in checkpoints:
        name = f"chck_{ck // 1_000_000}M"
        surp_path = arm_dir / "aoa_surprisals" / name / "surprisal.json"
        if surp_path.exists():
            data = json.loads(surp_path.read_text(encoding="utf-8"))
            for word, info in data.get("words", {}).items():
                if word not in surprisals:
                    surprisals[word] = {}
                steps = info.get("steps", {})
                for step_str, sdata in steps.items():
                    ms = sdata.get("mean_surprisal")
                    if ms is not None and math.isfinite(ms):
                        surprisals[word][ck] = float(ms)
    return surprisals


def measure_beta(v4_surp, arm_surp, exposure_deltas, checkpoints):
    """Measure β: regress Δsurprisal on Δlog_exposure across words and checkpoints.
    
    β is the slope in: Δsurprisal_w,c ≈ β · Δlog_exposure_w,c + noise
    """
    X = []  # log-exposure differences
    Y = []  # surprisal differences
    word_ck = []
    
    for word in v4_surp:
        if word not in arm_surp or word not in exposure_deltas:
            continue
        for ck in checkpoints:
            v4_s = v4_surp[word].get(ck)
            arm_s = arm_surp[word].get(ck)
            dlog = exposure_deltas[word].get(ck)
            if v4_s is not None and arm_s is not None and dlog is not None and abs(dlog) > 1e-6:
                X.append(dlog)
                Y.append(arm_s - v4_s)
                word_ck.append((word, ck))
    
    if len(X) < 10:
        return {"n_obs": len(X), "beta": None, "r2": None, "status": "insufficient_data"}
    
    X = np.array(X)
    Y = np.array(Y)
    
    # OLS regression: Y = beta * X + intercept
    A = np.column_stack([X, np.ones(len(X))])
    result = np.linalg.lstsq(A, Y, rcond=None)
    beta = float(result[0][0])
    intercept = float(result[0][1])
    
    Y_pred = beta * X + intercept
    ss_res = float(np.sum((Y - Y_pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Pearson correlation
    r_stat, p_val = pearsonr(X, Y)
    
    return {
        "n_obs": len(X),
        "beta": beta,
        "intercept": intercept,
        "r2": r2,
        "pearson_r": float(r_stat),
        "pearson_p": float(p_val),
        "mean_abs_dlog": float(np.abs(X).mean()),
        "mean_abs_dsurp": float(np.abs(Y).mean()),
        "status": "measured"
    }


def recalibrate_prediction(measured_beta, schedule_scores_path):
    """Recalibrate the research across-pass AoA predictions with measured β."""
    if measured_beta is None:
        return {"status": "no_beta"}
    
    # Load research schedule prediction scores
    if not pathlib.Path(schedule_scores_path).exists():
        return {"status": "no_schedule_scores"}
    
    import csv
    rows = []
    with open(schedule_scores_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # For each legal schedule, recompute the predicted AoA score using measured β
    # The research scores used a fitted β; we now replace with measured β
    recalibrated = []
    for row in rows:
        if row.get("oracle_only", "False") == "True":
            continue
        mode = row.get("mode", "")
        old_beta = float(row.get("beta", 0))
        old_r = float(row.get("unclipped_r", 0)) if row.get("unclipped_r") else None
        
        # The relationship is approximately: new_r ≈ old_r * measured_beta / old_beta
        # This is a rough rescaling; the actual recalibration should rerun the predictor
        if old_beta != 0 and old_r is not None:
            scale = measured_beta / old_beta
            estimated_r = old_r * scale
            recalibrated.append({
                "mode": mode,
                "old_beta": old_beta,
                "measured_beta": measured_beta,
                "old_unclipped_r": old_r,
                "estimated_r_at_measured_beta": estimated_r,
                "passes_significance": abs(estimated_r) > 0.112  # rough n≈220 gate
            })
    
    return {"status": "recalibrated", "schedules": recalibrated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-arm-dir", default=str(_public_path('experiments/archive/relation_learning/data/schedule_arm')))
    parser.add_argument("--enrichment-arm-dir", default=str(_public_path('experiments/archive/relation_learning/data/enrichment_arm')))
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Matched checkpoints: 1M through 30M
    checkpoints = list(range(1_000_000, 31_000_000, 1_000_000))
    
    # Load v4 word surprisals
    v4_surp, v4_meta = load_v4_word_surprisals()
    print(json.dumps({"event": "v4_loaded", "words": len(v4_surp)}), flush=True)
    
    # Check if calibration arm surprisals have been extracted
    schedule_dir = pathlib.Path(args.schedule_arm_dir)
    enrichment_dir = pathlib.Path(args.enrichment_arm_dir)
    
    # This script can also just verify training completion and prepare extraction commands
    schedule_metrics = schedule_dir / "scientific_metrics.json"
    enrichment_metrics = enrichment_dir / "scientific_metrics.json"
    
    status = {"v4_words": len(v4_surp), "checkpoints": len(checkpoints)}
    
    if schedule_metrics.exists():
        sm = json.loads(schedule_metrics.read_text(encoding="utf-8"))
        status["schedule_arm"] = {
            "word_exposure": sm.get("word_exposure"),
            "training_steps": sm.get("actual_training_steps"),
            "checkpoints_saved": len(sm.get("saved_checkpoints", []))
        }
    else:
        status["schedule_arm"] = {"status": "training_not_complete"}
    
    if enrichment_metrics.exists():
        em = json.loads(enrichment_metrics.read_text(encoding="utf-8"))
        status["enrichment_arm"] = {
            "word_exposure": em.get("word_exposure"),
            "training_steps": em.get("actual_training_steps"),
            "checkpoints_saved": len(em.get("saved_checkpoints", []))
        }
    else:
        status["enrichment_arm"] = {"status": "training_not_complete"}
    
    summary = {"status": "BETA_MEASUREMENT_STATUS", "created_utc": now(), **status}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
