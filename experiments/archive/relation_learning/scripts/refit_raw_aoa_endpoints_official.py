#!/usr/bin/env python3
"""research official-semantics raw AoA refits for endpoints.

The BabyLM AoA evaluator clips non-significant correlations to zero. This script
uses the same curve semantics as `evaluation_pipeline/utils.py` but records the
raw Pearson r,p and fitted words before clipping. It is intentionally separate
from the earlier research draft, which used a looser child-curve fit.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from transformers import AutoTokenizer

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/data/raw_aoa_refits_official')
CDI_PATH = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
TOKENIZER_PATH = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model')

ENDPOINTS = {
    "coherent86": _public_path('experiments/archive/functional_learning/data/batched_aoa_measured/full/coherent86/full/AoA_word/surprisal.json'),
    "dense_seed62064": _public_path('experiments/archive/functional_learning/data/batched_aoa_measured/full/dense_seed62064/full/AoA_word/surprisal.json'),
    "dense_seed62065": _public_path('experiments/archive/functional_learning/data/batched_aoa_measured/full/dense_seed62065/full/AoA_word/surprisal.json'),
    "clean_pres_seed62064": _public_path('experiments/archive/functional_learning/data/batched_aoa_clean_eval_measured/clean_pres_lambda1_eval_seed62064/full/AoA_word/surprisal.json'),
    "clean_pres_seed62065": _public_path('experiments/archive/functional_learning/data/batched_aoa_measured/clean_pres_lambda1_eval_seed62065/full/AoA_word/surprisal.json'),
    "densemask_sparselabel_seed62064": _public_path('experiments/archive/functional_learning/data/batched_aoa_measured/densemask_sparselabel_seed62064/full/AoA_word/surprisal.json'),
}


def sigmoid_function(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a / (1 + np.exp(-b * (x - c))) + d


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_cdi_child_aoas(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    age_columns = [str(age) for age in range(16, 31) if str(age) in df.columns]
    if not age_columns:
        raise RuntimeError(f"No official 16..30 age columns in {path}")
    ages = np.array([int(c) for c in age_columns], dtype=float)
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        word = str(row["word"]).strip()
        proportions = row[age_columns].to_numpy(dtype=float)
        valid_mask = ~np.isnan(proportions)
        if valid_mask.sum() < 3:
            continue
        valid_ages = ages[valid_mask]
        valid_props = proportions[valid_mask]
        try:
            initial_guess = [1.0, 0.1, float(np.mean(valid_ages)), 0.0]
            bounds = ([0, 0, valid_ages[0], -0.5], [2.0, 1.0, valid_ages[-1], 0.5])
            popt, _ = curve_fit(sigmoid_function, valid_ages, valid_props, p0=initial_guess, bounds=bounds, maxfev=1000)
            a, b, c, d = [float(x) for x in popt]
            if b <= 0 or a <= 0:
                continue
            y_target = 0.5
            if y_target <= d or y_target >= a + d:
                continue
            aoa = c - np.log((a / (y_target - d)) - 1) / b
            if aoa < valid_ages[0] or aoa > valid_ages[-1]:
                continue
            out[word] = float(aoa)
        except Exception:
            continue
    return out


def compute_model_aoa(mean_surprisals: List[float], steps: List[float], vocab_size: int, n_subword_tokens: int) -> Optional[float]:
    if len(mean_surprisals) != len(steps):
        raise ValueError("surprisal data and training steps must have same length")
    if len(mean_surprisals) < 3:
        return None
    steps_arr = np.array(steps, dtype=float)
    surp_arr = np.array(mean_surprisals, dtype=float)
    valid_mask = ~np.isnan(surp_arr)
    if not np.any(valid_mask) or valid_mask.sum() < 3:
        return None
    valid_steps = steps_arr[valid_mask]
    valid_surps = surp_arr[valid_mask]
    random_chance_surprisal = n_subword_tokens * np.log(vocab_size)
    min_surprisal = np.min(valid_surps)
    threshold_surprisal = random_chance_surprisal - 0.5 * (random_chance_surprisal - min_surprisal)
    try:
        neg_surps = -valid_surps
        log_steps = np.log10(valid_steps + 1)
        rng = np.max(neg_surps) - np.min(neg_surps)
        if rng <= 1e-12:
            return None
        initial_guess = [rng, 1.0, float(np.mean(log_steps)), float(np.min(neg_surps))]
        lower = [0.0, 0.0, float(np.min(log_steps) - 1), float(np.min(neg_surps) - 2 * rng - 1)]
        upper = [10 * rng + 1, 100.0, float(np.max(log_steps) + 1), float(np.max(neg_surps) + 1)]
        popt, _ = curve_fit(sigmoid_function, log_steps, neg_surps, p0=initial_guess, bounds=(lower, upper), maxfev=20000)
        a, b, c, d = [float(x) for x in popt]
        neg_threshold = -threshold_surprisal
        if b <= 1e-6 or a <= 1e-6:
            return None
        if neg_threshold <= d or neg_threshold >= a + d:
            return None
        log_aoa_step = c - np.log((a / (neg_threshold - d)) - 1) / b
        aoa_step = 10 ** log_aoa_step - 1
        if aoa_step < valid_steps[0] or aoa_step > valid_steps[-1]:
            return None
        return float(log_aoa_step)
    except Exception:
        return None


def extract_step_number(step_name: Any) -> Optional[float]:
    import re
    if isinstance(step_name, (int, float)):
        return float(step_name)
    match = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", str(step_name), re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).upper() if match.group(2) else ""
    return number * {"K": 1000, "M": 1000000, "B": 1000000000}.get(unit, 1)


def load_mean_curves(path: Path) -> Tuple[dict, Dict[str, Tuple[List[float], List[float]]], int]:
    t0 = time.time()
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    metadata = obj.get("metadata", {})
    records = obj.get("results", [])
    acc: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    n_records = 0
    for r in records:
        if not isinstance(r, dict):
            continue
        step_val = extract_step_number(r.get("step"))
        if step_val is None:
            continue
        try:
            val = float(r["surprisal"])
        except Exception:
            continue
        if not math.isfinite(val):
            continue
        word = str(r.get("target_word", "")).strip()
        if not word:
            continue
        acc[word][float(step_val)].append(val)
        n_records += 1
    curves: Dict[str, Tuple[List[float], List[float]]] = {}
    for word, by_step in acc.items():
        steps = sorted(by_step)
        curves[word] = (steps, [float(np.mean(by_step[s])) for s in steps])
    print(json.dumps({"event": "loaded", "path": rel(path), "records": n_records, "words": len(curves), "sec": round(time.time()-t0, 1)}), flush=True)
    return metadata, curves, n_records


def subword_lengths(tokenizer, words: List[str]) -> Dict[str, int]:
    prefix_ids = tokenizer("The", add_special_tokens=False)["input_ids"]
    out = {}
    for w in words:
        ids = tokenizer("The " + w, add_special_tokens=False)["input_ids"]
        out[w] = max(1, len(ids) - len(prefix_ids))
    return out


def fit_endpoint(name: str, path: Path, child_aoas: Dict[str, float], tokenizer) -> Tuple[dict, List[dict]]:
    metadata, curves, n_records = load_mean_curves(path)
    lens = subword_lengths(tokenizer, list(curves.keys()))
    rows = []
    for word, (steps, surps) in curves.items():
        if word not in child_aoas:
            continue
        model_aoa = compute_model_aoa(surps, steps, tokenizer.vocab_size, lens[word])
        if model_aoa is None:
            continue
        rows.append({
            "endpoint": name,
            "word": word,
            "child_aoa": child_aoas[word],
            "model_aoa_log10_step": model_aoa,
            "n_subword_tokens": lens[word],
            "n_points": len(steps),
            "first_step": min(steps),
            "last_step": max(steps),
        })
    if len(rows) >= 3:
        # Same order as official compute_curve_fitness: pearsonr(model_aoas, child_aoas)
        r, p = pearsonr([r["model_aoa_log10_step"] for r in rows], [r["child_aoa"] for r in rows])
        raw_r = float(r); raw_p = float(p)
    else:
        raw_r = float("nan"); raw_p = float("nan")
    summary = {
        "endpoint": name,
        "surprisal_path": rel(path),
        "metadata_model_name": metadata.get("model_name"),
        "metadata_total_steps": metadata.get("total_steps"),
        "metadata_completed_steps": metadata.get("completed_steps"),
        "n_surprisal_records": n_records,
        "raw_pearson_r": raw_r,
        "raw_pearson_p": raw_p,
        "fitted_word_count": len(rows),
        "official_clipped_score": 0.0 if (math.isfinite(raw_p) and raw_p > 0.1) else raw_r,
        "near_significant_harmful_negative_p_le_0p2": bool(math.isfinite(raw_r) and raw_r < 0 and raw_p <= 0.2),
        "significant_negative_p_le_0p1": bool(math.isfinite(raw_r) and raw_r < 0 and raw_p <= 0.1),
        "significant_positive_p_le_0p1": bool(math.isfinite(raw_r) and raw_r > 0 and raw_p <= 0.1),
    }
    return summary, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    child_aoas = load_cdi_child_aoas(CDI_PATH)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), use_fast=True, local_files_only=True)
    summaries = []
    all_rows = []
    missing = []
    for name, path in ENDPOINTS.items():
        if not path.exists():
            missing.append({"endpoint": name, "path": rel(path)})
            continue
        summary, rows = fit_endpoint(name, path, child_aoas, tokenizer)
        summaries.append(summary)
        all_rows.extend(rows)
        pd.DataFrame(rows).to_csv(OUT / f"{name}_word_fits.csv", index=False)
        print(json.dumps({"endpoint": name, "r": summary["raw_pearson_r"], "p": summary["raw_pearson_p"], "n": summary["fitted_word_count"]}), flush=True)
    word_sets = {s["endpoint"]: set(r["word"] for r in all_rows if r["endpoint"] == s["endpoint"]) for s in summaries}
    overlaps = []
    for a, A in word_sets.items():
        for b, B in word_sets.items():
            overlaps.append({"endpoint_a": a, "endpoint_b": b, "overlap": len(A & B), "n_a": len(A), "n_b": len(B)})
    pd.DataFrame(summaries).to_csv(_public_path('experiments/archive/relation_learning/data/raw_aoa_refits_official/raw_aoa_endpoint_summary.csv'), index=False)
    pd.DataFrame(all_rows).to_csv(_public_path('experiments/archive/relation_learning/data/raw_aoa_refits_official/raw_aoa_all_word_fits.csv'), index=False)
    pd.DataFrame(overlaps).to_csv(_public_path('experiments/archive/relation_learning/data/raw_aoa_refits_official/raw_aoa_fit_word_overlap.csv'), index=False)
    payload = {
        "status": "RAW_AOA_REFITS_OFFICIAL_SEMANTICS",
        "cdi_path": rel(CDI_PATH),
        "tokenizer_path": rel(TOKENIZER_PATH),
        "semantics": "Official AoAEvaluator: bounded child sigmoid to threshold 0.5; mean surprisal per word/checkpoint; subword length by prefix subtraction; model AoA is log10(step+1) at halfway from random surprisal to min surprisal; raw Pearson before p>0.1 clipping.",
        "summaries": summaries,
        "missing": missing,
        "outputs": {
            "summary_csv": rel(_public_path('experiments/archive/relation_learning/data/raw_aoa_refits_official/raw_aoa_endpoint_summary.csv')),
            "word_fits_csv": rel(_public_path('experiments/archive/relation_learning/data/raw_aoa_refits_official/raw_aoa_all_word_fits.csv')),
            "overlap_csv": rel(_public_path('experiments/archive/relation_learning/data/raw_aoa_refits_official/raw_aoa_fit_word_overlap.csv')),
        },
    }
    (_public_path('experiments/archive/relation_learning/data/raw_aoa_refits_official/summary.json')).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# research raw AoA refits with official semantics", "", payload["semantics"], "", "| endpoint | raw r | p | fitted words | official clipped | harmful negative p<=0.2 |", "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['endpoint']} | {s['raw_pearson_r']:.6f} | {s['raw_pearson_p']:.6g} | {s['fitted_word_count']} | {s['official_clipped_score']:.6f} | {s['near_significant_harmful_negative_p_le_0p2']} |")
    if missing:
        lines += ["", "Missing files:", json.dumps(missing, indent=2)]
    lines += ["", "Outputs:", json.dumps(payload["outputs"], indent=2)]
    (_public_path('research/documents/relation_learning/data/raw_aoa_refits_official/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "done", "out": rel(OUT), "n_endpoints": len(summaries)}), flush=True)


if __name__ == "__main__":
    main()
