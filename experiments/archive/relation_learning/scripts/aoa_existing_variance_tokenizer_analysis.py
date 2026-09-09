#!/usr/bin/env python3
"""research: AoA prior evidence, tokenizer geometry, and score dispersion analysis.

This script reads existing official AoA outputs and the v4/coherent86 word-curve
records to answer two route-critical questions before 100M AoA spending:

1. How much did earlier full-scale masking-priority manipulations move the official
   AoA statistic, compared with the movement needed to make v4 positive?
2. Is the official AoA statistic in this lineage dominated by tokenizer/subword
   geometry or fit-set effects rather than exposure timing?

Outputs are research-facing evidence under data/aoa_existing_variance_tokenizer/.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, t as student_t

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/data/aoa_existing_variance_tokenizer')

WORD_RECORDS = _public_path('experiments/archive/relation_learning/data/aoa_curve_fit_analysis/word_curve_fit_records.jsonl')
EXPOSURE_CSV = _public_path('experiments/archive/relation_learning/data/aoa_stream_exposure_timing/word_exposure_timing.csv')
LOCAL_SCORES_CSV = _public_path('experiments/archive/relation_learning/data/aoa_curve_fit_analysis/local_aoa_score_files.csv')
V4_SURPRISAL = _public_path('experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0/collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/surprisal.json')

ROOT = _public_path('experiments/archive/compact_experience/data/mask_endpoint_full_eval/aoa_outputs')
LABELS = [
    "mask_uniform_control_90M",
    "mask_uniform_control_100M",
    "mask_evidence_visible_90M",
    "mask_evidence_visible_100M",
    "mask_inverse_priority_95M",
    "mask_inverse_priority_100M",
]
DENSITY_SCORE_PATHS = {
    "representation_frontier_studies_step017_compact_view_core": _public_path('experiments/archive/frontier_consolidation/data/density_full_eval/aoa_outputs/compact_view_core/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json'),
    "representation_frontier_studies_step018_density_compact_repeat_core": _public_path('experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_compact_repeat_core/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json'),
    "representation_frontier_studies_step018_density_near_view": _public_path('experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_near_view/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json'),
    "representation_frontier_studies_step018_density_near_repeat": _public_path('experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_near_repeat/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json'),
}

VOCAB_SIZE = 16384
LN_VOCAB = math.log(VOCAB_SIZE)
STEP_RE = re.compile(r"(\d+(?:\.\d+)?)([KMB]?)", re.I)


def sigmoid(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a / (1.0 + np.exp(-b * (x - c))) + d


def extract_step(step: str | int | float) -> float | None:
    if isinstance(step, (int, float)):
        return float(step)
    m = STEP_RE.search(str(step))
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).upper()
    return val * {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(unit, 1)


def pearson_summary(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) < 3:
        return {"n": len(x), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def ols_r2(y: np.ndarray, X: np.ndarray) -> float | None:
    if len(y) < 3:
        return None
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return None
    return 1.0 - ss_res / ss_tot


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def score_record(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    rec = obj.get("curve_fitness_record", {})
    return {
        "path": str(path.relative_to(ROOT)),
        "aoa": obj.get("aoa"),
        "curve_fitness": rec.get("curve_fitness"),
        "p_value": rec.get("p_value"),
        "n_words": rec.get("n_words"),
        "valid_words": rec.get("valid_words", []),
        "model_aoas": rec.get("model_aoas", []),
        "child_aoas": rec.get("child_aoas", []),
    }


def load_word_records() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with WORD_RECORDS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["word"]] = rec
    return out


def load_exposures() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with EXPOSURE_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["word"]] = row
    return out


def try_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def model_aoa_official_like(surprisals: list[float], steps: list[float], n_subword: int) -> float | None:
    if len(surprisals) < 3:
        return None
    steps_arr = np.asarray(steps, dtype=float)
    surp = np.asarray(surprisals, dtype=float)
    mask = np.isfinite(surp)
    if mask.sum() < 3:
        return None
    steps_arr = steps_arr[mask]
    surp = surp[mask]
    random_chance = n_subword * LN_VOCAB
    min_s = float(np.min(surp))
    threshold = random_chance - 0.5 * (random_chance - min_s)
    neg = -surp
    log_steps = np.log10(steps_arr + 1.0)
    rng = float(np.max(neg) - np.min(neg))
    if rng <= 1e-12:
        return None
    p0 = [rng, 1.0, float(np.mean(log_steps)), float(np.min(neg))]
    lower = [0.0, 0.0, float(np.min(log_steps) - 1), float(np.min(neg) - 2 * rng - 1)]
    upper = [10 * rng + 1, 100.0, float(np.max(log_steps) + 1), float(np.max(neg) + 1)]
    try:
        popt, _ = curve_fit(sigmoid, log_steps, neg, p0=p0, bounds=(lower, upper), maxfev=20000)
        a, b, c, d = popt
        neg_threshold = -threshold
        if b <= 1e-6 or a <= 1e-6 or neg_threshold <= d or neg_threshold >= a + d:
            return None
        log_aoa = c - np.log((a / (neg_threshold - d)) - 1.0) / b
        step = 10 ** log_aoa - 1
        if step < steps_arr.min() or step > steps_arr.max():
            return None
        return float(log_aoa)
    except Exception:
        return None


def model_aoa_progress_norm(surprisals: list[float], steps: list[float], n_subword: int) -> float | None:
    """Fit a 0..1 progress curve: progress=(random-surprisal)/(random-best)."""
    if len(surprisals) < 3:
        return None
    steps_arr = np.asarray(steps, dtype=float)
    surp = np.asarray(surprisals, dtype=float)
    mask = np.isfinite(surp)
    if mask.sum() < 3:
        return None
    steps_arr = steps_arr[mask]
    surp = surp[mask]
    random_chance = n_subword * LN_VOCAB
    min_s = float(np.min(surp))
    denom = random_chance - min_s
    if denom <= 1e-8:
        return None
    progress = (random_chance - surp) / denom
    log_steps = np.log10(steps_arr + 1.0)
    rng = float(np.max(progress) - np.min(progress))
    if rng <= 1e-8:
        return None
    p0 = [max(rng, 1e-3), 1.0, float(np.mean(log_steps)), float(np.min(progress))]
    lower = [0.0, 0.0, float(np.min(log_steps) - 1), float(np.min(progress) - 2 * rng - 1)]
    upper = [10 * rng + 1, 100.0, float(np.max(log_steps) + 1), float(np.max(progress) + 1)]
    try:
        popt, _ = curve_fit(sigmoid, log_steps, progress, p0=p0, bounds=(lower, upper), maxfev=20000)
        a, b, c, d = popt
        target = 0.5
        if b <= 1e-6 or a <= 1e-6 or target <= d or target >= a + d:
            return None
        log_aoa = c - np.log((a / (target - d)) - 1.0) / b
        step = 10 ** log_aoa - 1
        if step < steps_arr.min() or step > steps_arr.max():
            return None
        return float(log_aoa)
    except Exception:
        return None


def aggregate_surprisal_curves(path: Path) -> dict[str, tuple[list[float], list[float]]]:
    t0 = time.time()
    obj = load_json(path)
    acc: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in obj["results"]:
        s = extract_step(r.get("step"))
        if s is None:
            continue
        try:
            val = float(r["surprisal"])
        except Exception:
            continue
        if math.isfinite(val):
            acc[str(r["target_word"]).lower()][s].append(val)
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for w, by_step in acc.items():
        steps = sorted(by_step)
        curves[w] = (steps, [float(np.mean(by_step[s])) for s in steps])
    print(json.dumps({"event": "surprisal_curves_loaded", "words": len(curves), "sec": round(time.time()-t0, 1)}), flush=True)
    return curves


def analyze_existing_scores() -> dict[str, Any]:
    # Important family: full-scale masking-priority triple.
    measured_aoa_deltas = {}
    for label in LABELS:
        p = ROOT / label / "hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json"
        measured_aoa_deltas[label] = score_record(p)

    density = {}
    for label, path in DENSITY_SCORE_PATHS.items():
        if path.exists():
            density[label] = score_record(path)

    # De-duplicated local score landscape from research.
    unique = {}
    with LOCAL_SCORES_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row["path"]
            aoa = try_float(row.get("aoa"))
            if aoa is None:
                continue
            unique[path] = aoa
    vals = list(unique.values())
    nonzero = [v for v in vals if abs(v) > 1e-12]

    def stats(vals_: list[float]) -> dict[str, Any]:
        if not vals_:
            return {"n": 0}
        return {
            "n": len(vals_),
            "mean": float(statistics.mean(vals_)),
            "sd_sample": float(statistics.stdev(vals_)) if len(vals_) > 1 else 0.0,
            "min": float(min(vals_)),
            "max": float(max(vals_)),
            "median": float(statistics.median(vals_)),
        }

    # research within-family effects.
    s46 = {k: float(v["aoa"]) for k, v in measured_aoa_deltas.items()}
    deltas = {
        "uniform_100M_minus_90M": s46["mask_uniform_control_100M"] - s46["mask_uniform_control_90M"],
        "evidence_visible_100M_minus_uniform_100M": s46["mask_evidence_visible_100M"] - s46["mask_uniform_control_100M"],
        "evidence_visible_90M_minus_uniform_90M": s46["mask_evidence_visible_90M"] - s46["mask_uniform_control_90M"],
        "inverse_priority_100M_minus_uniform_100M": s46["mask_inverse_priority_100M"] - s46["mask_uniform_control_100M"],
        "inverse_priority_100M_minus_evidence_visible_100M": s46["mask_inverse_priority_100M"] - s46["mask_evidence_visible_100M"],
        "max_minus_min": max(s46.values()) - min(s46.values()),
    }

    # Fit-set overlap: important because AoA score changes can enter by changing fitted words.
    fit_overlaps = {}
    pairs = [
        ("evidence100_vs_uniform100", "mask_evidence_visible_100M", "mask_uniform_control_100M"),
        ("inverse100_vs_uniform100", "mask_inverse_priority_100M", "mask_uniform_control_100M"),
        ("uniform100_vs_uniform90", "mask_uniform_control_100M", "mask_uniform_control_90M"),
    ]
    for name, a, b in pairs:
        A = set(measured_aoa_deltas[a]["valid_words"])
        B = set(measured_aoa_deltas[b]["valid_words"])
        fit_overlaps[name] = {
            "n_a": len(A), "n_b": len(B), "n_overlap": len(A & B),
            "only_a": sorted(A - B)[:30], "only_b": sorted(B - A)[:30],
            "n_only_a": len(A - B), "n_only_b": len(B - A),
        }

    # Positive p=0.1 two-sided boundary for v4's 225 words.
    n_v4 = 225
    tcrit = float(student_t.ppf(1 - 0.1 / 2, n_v4 - 2))
    rcrit = tcrit / math.sqrt(tcrit * tcrit + (n_v4 - 2))
    v4_unclipped = -0.03485418140387646
    required_delta = rcrit - v4_unclipped
    all_sd = stats(vals).get("sd_sample") or 0.0
    nz_sd = stats(nonzero).get("sd_sample") or 0.0
    spread = deltas["max_minus_min"]

    return {
        "records": {k: {kk: vv for kk, vv in v.items() if kk not in {"valid_words", "model_aoas", "child_aoas"}} for k, v in measured_aoa_deltas.items()},
        "deltas": deltas,
        "fit_set_overlap": fit_overlaps,
        "density_records": {k: {kk: vv for kk, vv in v.items() if kk not in {"valid_words", "model_aoas", "child_aoas"}} for k, v in density.items()},
        "local_score_landscape": {
            "all_unique_stored_scores": stats(vals),
            "nonzero_stored_scores": stats(nonzero),
            "n_zero_or_clipped": sum(1 for v in vals if abs(v) <= 1e-12),
            "n_unique_paths": len(vals),
        },
        "positive_boundary_units": {
            "n_words_reference": n_v4,
            "p_value_two_sided": 0.1,
            "positive_r_boundary": rcrit,
            "v4_unclipped_r": v4_unclipped,
            "required_delta_from_v4_to_positive_boundary": required_delta,
            "required_delta_div_all_unique_sd": required_delta / all_sd if all_sd else None,
            "required_delta_div_nonzero_sd": required_delta / nz_sd if nz_sd else None,
            "required_delta_div_step046_mask_family_spread": required_delta / spread if spread else None,
            "required_delta_div_step046_evidence100_minus_uniform100": required_delta / abs(deltas["evidence_visible_100M_minus_uniform_100M"]),
        },
    }


def analyze_tokenizer_geometry() -> dict[str, Any]:
    records = load_word_records()
    exposures = load_exposures()

    # Official stored fits from research records.
    official_ok = []
    child_ok = []
    by_sub = defaultdict(list)
    status_by_sub = defaultdict(Counter)
    for w, r in records.items():
        sub = int(r.get("subword_len") or 0) if r.get("subword_len") is not None else None
        if r.get("child_status") == "ok":
            child_ok.append(w)
            if sub is not None:
                status_by_sub[sub][r.get("model_status", "missing")] += 1
                by_sub[sub].append(w)
        if r.get("child_status") == "ok" and r.get("model_status") == "ok":
            lf = try_float(exposures.get(w, {}).get("stream_log_freq_per_million"))
            official_ok.append({
                "word": w,
                "child_aoa": float(r["child_aoa"]),
                "model_aoa": float(r["model_aoa"]),
                "subword_len": int(r["subword_len"]),
                "char_len": len(w),
                "logfreq": lf,
            })

    def corr_for(rows: list[dict[str, Any]], xkey: str, ykey: str) -> dict[str, Any]:
        xs, ys = [], []
        for row in rows:
            x = row.get(xkey); y = row.get(ykey)
            if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y)):
                xs.append(float(x)); ys.append(float(y))
        return pearson_summary(xs, ys)

    official_corr_all = corr_for(official_ok, "model_aoa", "child_aoa")
    official_corr_single = corr_for([r for r in official_ok if r["subword_len"] == 1], "model_aoa", "child_aoa")
    official_corr_multi = corr_for([r for r in official_ok if r["subword_len"] > 1], "model_aoa", "child_aoa")
    token_predict_model = corr_for(official_ok, "subword_len", "model_aoa")
    char_predict_model = corr_for(official_ok, "char_len", "model_aoa")
    freq_predict_model = corr_for(official_ok, "logfreq", "model_aoa")
    token_predict_child_all = []
    for w in child_ok:
        r = records[w]
        if r.get("subword_len") is not None and r.get("child_aoa") is not None:
            token_predict_child_all.append({"subword_len": int(r["subword_len"]), "child_aoa": float(r["child_aoa"]), "char_len": len(w)})

    # R2 comparisons on official ok set.
    y = np.asarray([r["model_aoa"] for r in official_ok], dtype=float)
    logf = np.asarray([r["logfreq"] if r["logfreq"] is not None else np.nan for r in official_ok], dtype=float)
    sub = np.asarray([r["subword_len"] for r in official_ok], dtype=float)
    ch = np.asarray([r["char_len"] for r in official_ok], dtype=float)
    mask = np.isfinite(logf)
    r2 = {
        "model_aoa_from_logfreq": ols_r2(y[mask], logf[mask, None]),
        "model_aoa_from_subword_len": ols_r2(y, sub[:, None]),
        "model_aoa_from_char_len": ols_r2(y, ch[:, None]),
        "model_aoa_from_logfreq_plus_subword": ols_r2(y[mask], np.column_stack([logf[mask], sub[mask]])),
        "model_aoa_from_logfreq_plus_char": ols_r2(y[mask], np.column_stack([logf[mask], ch[mask]])),
        "model_aoa_from_logfreq_plus_subword_plus_char": ols_r2(y[mask], np.column_stack([logf[mask], sub[mask], ch[mask]])),
    }

    subword_table = []
    for sublen in sorted(by_sub):
        words = by_sub[sublen]
        rows_child = [records[w] for w in words]
        rows_fit = [r for r in official_ok if r["subword_len"] == sublen]
        child_aoas = [float(r["child_aoa"]) for r in rows_child if r.get("child_aoa") is not None]
        model_aoas = [float(r["model_aoa"]) for r in rows_fit]
        logfs = [r["logfreq"] for r in rows_fit if r.get("logfreq") is not None]
        subword_table.append({
            "subword_len": sublen,
            "n_child_ok_words": len(words),
            "n_model_fit_ok": len(rows_fit),
            "fit_rate_given_child_ok": len(rows_fit) / max(1, len(words)),
            "mean_child_aoa": float(np.mean(child_aoas)) if child_aoas else None,
            "mean_model_aoa": float(np.mean(model_aoas)) if model_aoas else None,
            "mean_logfreq_fit_words": float(np.mean(logfs)) if logfs else None,
            "model_status_counts": dict(status_by_sub[sublen]),
        })

    # Refit variants from actual mean surprisal curves.
    curves = aggregate_surprisal_curves(V4_SURPRISAL)
    refit_rows = []
    for w, r in records.items():
        if r.get("child_status") != "ok" or w not in curves or r.get("subword_len") is None:
            continue
        steps, surps = curves[w]
        sublen = int(r["subword_len"])
        off_refit = model_aoa_official_like(surps, steps, sublen)
        per_piece = model_aoa_official_like([s / sublen for s in surps], steps, 1)
        progress = model_aoa_progress_norm(surps, steps, sublen)
        refit_rows.append({
            "word": w,
            "child_aoa": float(r["child_aoa"]),
            "subword_len": sublen,
            "char_len": len(w),
            "logfreq": try_float(exposures.get(w, {}).get("stream_log_freq_per_million")),
            "official_record_model_aoa": try_float(r.get("model_aoa")),
            "official_refit_model_aoa": off_refit,
            "per_piece_model_aoa": per_piece,
            "progress_norm_model_aoa": progress,
        })

    def corr_variant(key: str, rows: list[dict[str, Any]], single: bool | None = None) -> dict[str, Any]:
        xs, ys = [], []
        for row in rows:
            if single is True and row["subword_len"] != 1:
                continue
            if single is False and row["subword_len"] == 1:
                continue
            if row.get(key) is None:
                continue
            xs.append(float(row[key])); ys.append(float(row["child_aoa"]))
        return pearson_summary(xs, ys)

    variant_corrs = {
        "official_record_all": official_corr_all,
        "official_record_single_token": official_corr_single,
        "official_record_multi_token": official_corr_multi,
        "official_refit_all": corr_variant("official_refit_model_aoa", refit_rows),
        "official_refit_single_token": corr_variant("official_refit_model_aoa", refit_rows, True),
        "official_refit_multi_token": corr_variant("official_refit_model_aoa", refit_rows, False),
        "per_piece_all": corr_variant("per_piece_model_aoa", refit_rows),
        "per_piece_single_token": corr_variant("per_piece_model_aoa", refit_rows, True),
        "per_piece_multi_token": corr_variant("per_piece_model_aoa", refit_rows, False),
        "progress_norm_all": corr_variant("progress_norm_model_aoa", refit_rows),
        "progress_norm_single_token": corr_variant("progress_norm_model_aoa", refit_rows, True),
        "progress_norm_multi_token": corr_variant("progress_norm_model_aoa", refit_rows, False),
    }

    # Save refit rows for inspection.
    refit_csv = _public_path('experiments/archive/relation_learning/data/aoa_existing_variance_tokenizer/v4_refit_variant_word_records.csv')
    with refit_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(refit_rows[0].keys()) if refit_rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(refit_rows)

    # Does subword length predict inclusion? Simple table already gives rates; add logistic-free contrasts.
    child_single = [records[w] for w in child_ok if int(records[w].get("subword_len") or 0) == 1]
    child_multi = [records[w] for w in child_ok if int(records[w].get("subword_len") or 0) > 1]
    multi_words = [w for w in child_ok if int(records[w].get("subword_len") or 0) > 1]
    early_child_quantile = np.quantile([float(records[w]["child_aoa"]) for w in child_ok if records[w].get("child_aoa") is not None], 0.25)
    early_multi = [w for w in multi_words if float(records[w]["child_aoa"]) <= early_child_quantile]

    return {
        "official_correlations_and_predictors": {
            "model_child_all_official_ok": official_corr_all,
            "model_child_single_token_official_ok": official_corr_single,
            "model_child_multi_token_official_ok": official_corr_multi,
            "subword_len_vs_model_aoa_official_ok": token_predict_model,
            "char_len_vs_model_aoa_official_ok": char_predict_model,
            "whole_stream_logfreq_vs_model_aoa_official_ok": freq_predict_model,
            "subword_len_vs_child_aoa_all_child_ok": corr_for(token_predict_child_all, "subword_len", "child_aoa"),
            "char_len_vs_child_aoa_all_child_ok": corr_for(token_predict_child_all, "char_len", "child_aoa"),
            "ols_r2": r2,
        },
        "subword_status_table": subword_table,
        "fit_set_and_child_early_multi": {
            "n_child_ok_single_token": len(child_single),
            "n_child_ok_multi_token": len(child_multi),
            "child_aoa_25pct_month": float(early_child_quantile),
            "n_multi_token_child_early_25pct": len(early_multi),
            "multi_token_child_early_words": sorted(early_multi),
        },
        "refit_variant_correlations": variant_corrs,
        "refit_variant_csv": str(refit_csv.relative_to(ROOT)),
    }


def write_markdown(summary: dict[str, Any]) -> None:
    existing = summary["existing_aoa_score_evidence"]
    tok = summary["tokenizer_geometry_analysis"]
    lines = []
    lines.append("# research AoA existing evidence and tokenizer geometry analysis")
    lines.append("")
    lines.append(f"Created UTC: {summary['created_utc']}")
    lines.append("")
    lines.append("## 1. Existing full-scale AoA movements")
    lines.append("")
    lines.append("### research masking-priority family")
    lines.append("")
    lines.append("| arm | official AoA/correlation | p | fitted words |")
    lines.append("|---|---:|---:|---:|")
    for k, rec in existing["records"].items():
        lines.append(f"| {k} | {rec.get('aoa')} | {rec.get('p_value')} | {rec.get('n_words')} |")
    lines.append("")
    lines.append("Key differences:")
    for k, v in existing["deltas"].items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("The evidence-visible masking arm improves the 100M uniform control by only about "
                 f"{existing['deltas']['evidence_visible_100M_minus_uniform_100M']:.6f}; inverse-priority worsens it by "
                 f"{existing['deltas']['inverse_priority_100M_minus_uniform_100M']:.6f}. The whole research family spans "
                 f"{existing['deltas']['max_minus_min']:.6f}.")
    lines.append("")
    lines.append("### REPRESENTATION_FRONTIER_STUDIES density AoA anchors")
    lines.append("")
    lines.append("| record | AoA/correlation | p | fitted words |")
    lines.append("|---|---:|---:|---:|")
    for k, rec in existing["density_records"].items():
        lines.append(f"| {k} | {rec.get('aoa')} | {rec.get('p_value')} | {rec.get('n_words')} |")
    lines.append("")
    b = existing["positive_boundary_units"]
    lines.append("### Positive-threshold scale")
    lines.append("")
    lines.append(f"For n={b['n_words_reference']} fitted words and two-sided p=0.1, the positive correlation boundary is r={b['positive_r_boundary']:.6f}. Coherent86/v4 has r={b['v4_unclipped_r']:.6f}, so the required movement is {b['required_delta_from_v4_to_positive_boundary']:.6f}.")
    lines.append(f"This movement is {b['required_delta_div_all_unique_sd']:.2f} times the sample sd of de-duplicated stored local AoA scores, {b['required_delta_div_nonzero_sd']:.2f} times the sd of nonzero stored scores, {b['required_delta_div_step046_mask_family_spread']:.2f} times the entire research masking-priority spread, and {b['required_delta_div_step046_evidence100_minus_uniform100']:.1f} times the 100M evidence-visible versus uniform difference.")
    lines.append("")
    lines.append("## 2. Tokenizer and fit-set geometry")
    lines.append("")
    c = tok["official_correlations_and_predictors"]
    lines.append("### Official v4 fitted set")
    lines.append("")
    for name, rec in c.items():
        if name == "ols_r2":
            continue
        lines.append(f"- {name}: n={rec.get('n')}, Pearson r={rec.get('pearson_r')}, p={rec.get('pearson_p')}, Spearman r={rec.get('spearman_r')}")
    lines.append("")
    lines.append("OLS R2 for official fitted model AoA:")
    for k, v in c["ols_r2"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Subword length table")
    lines.append("")
    lines.append("| subword length | child-ok words | model-fit words | fit rate | mean child AoA | mean model AoA | status counts |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for row in tok["subword_status_table"]:
        lines.append(f"| {row['subword_len']} | {row['n_child_ok_words']} | {row['n_model_fit_ok']} | {row['fit_rate_given_child_ok']:.3f} | {row['mean_child_aoa']} | {row['mean_model_aoa']} | {row['model_status_counts']} |")
    lines.append("")
    lines.append("### Refit variants")
    lines.append("")
    lines.append("| variant | n | Pearson r | p | Spearman r |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, rec in tok["refit_variant_correlations"].items():
        lines.append(f"| {k} | {rec.get('n')} | {rec.get('pearson_r')} | {rec.get('pearson_p')} | {rec.get('spearman_r')} |")
    lines.append("")
    lines.append("## Interpretation for the ongoing AoA calibration")
    lines.append("")
    lines.append("The existing 100M masking-priority family shows that a legal masking-priority channel can move the official AoA correlation, but prior movements are much smaller than the shift needed to make v4 positive and they can also move in the wrong direction. The current 30M beta measurement remains valuable because it isolates the child-enrichment credit hypothesis; however, a positive beta should be read together with this full-scale prior rather than converted mechanically into 100M spending.")
    lines.append("")
    lines.append("The tokenizer analysis separates two possible explanations. If token count or length explains more fitted model-AoA variance than corpus frequency, then AoA is dominated by evaluation geometry; if not, the dominant measured variable remains frequency/credit. The tables above make that comparison directly.")
    (_public_path('research/documents/relation_learning/data/aoa_existing_variance_tokenizer/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    existing = analyze_existing_scores()
    tokenizer = analyze_tokenizer_geometry()
    summary = {
        "status": "AOA_EXISTING_VARIANCE_TOKENIZER_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "word_records": str(WORD_RECORDS.relative_to(ROOT)),
            "exposure_csv": str(EXPOSURE_CSV.relative_to(ROOT)),
            "local_scores_csv": str(LOCAL_SCORES_CSV.relative_to(ROOT)),
            "v4_surprisal": str(V4_SURPRISAL.relative_to(ROOT)),
            "root": str(ROOT.relative_to(ROOT)),
        },
        "existing_aoa_score_evidence": existing,
        "tokenizer_geometry_analysis": tokenizer,
        "elapsed_sec": round(time.time() - t0, 1),
        "outputs": {
            "summary_json": "experiments/archive/relation_learning/data/aoa_existing_variance_tokenizer/summary.json",
            "summary_md": "research/documents/relation_learning/data/aoa_existing_variance_tokenizer/summary.md",
            "refit_variant_csv": tokenizer.get("refit_variant_csv"),
        },
    }
    (_public_path('experiments/archive/relation_learning/data/aoa_existing_variance_tokenizer/summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary)
    print(json.dumps({
        "status": summary["status"],
        "elapsed_sec": summary["elapsed_sec"],
        "summary_json": summary["outputs"]["summary_json"],
        "summary_md": summary["outputs"]["summary_md"],
        "span": existing["deltas"]["max_minus_min"],
        "required_delta": existing["positive_boundary_units"]["required_delta_from_v4_to_positive_boundary"],
        "official_single_token_r": tokenizer["refit_variant_correlations"]["official_record_single_token"]["pearson_r"],
        "progress_norm_all_r": tokenizer["refit_variant_correlations"]["progress_norm_all"]["pearson_r"],
    }), flush=True)


if __name__ == "__main__":
    main()
