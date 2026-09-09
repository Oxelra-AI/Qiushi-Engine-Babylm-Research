#!/usr/bin/env python3
"""research: raw AoA refits for endpoint surprisal files.

The official leaderboard stores AoA as zero when the Pearson relation between fitted
model AoA and child AoA is not significant. This script preserves the raw relation
for candidate safety checks: r, p, fitted-word count, overlap, and word records.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from transformers import AutoTokenizer

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/data/raw_aoa_refits')
EVAL_ROOT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa')
CDI_PATH = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
TOKENIZER_PATH = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model')
VOCAB_SIZE = 16384

ENDPOINTS = {
    "coherent86": _public_path('experiments/archive/functional_learning/data/batched_aoa_clean_eval_measured/coherent86/full/AoA_word/surprisal.json'),
    "dense_seed62064": _public_path('experiments/archive/functional_learning/data/batched_aoa_measured/full/dense_seed62064/full/AoA_word/surprisal.json'),
    "dense_seed62065": _public_path('experiments/archive/functional_learning/data/batched_aoa_measured/full/dense_seed62065/full/AoA_word/surprisal.json'),
    "clean_pres_seed62064": _public_path('experiments/archive/functional_learning/data/batched_aoa_clean_eval_measured/clean_pres_lambda1_eval_seed62064/full/AoA_word/surprisal.json'),
    "clean_pres_seed62065": _public_path('experiments/archive/functional_learning/data/batched_aoa_measured/clean_pres_lambda1_eval_seed62065/full/AoA_word/surprisal.json'),
    "densemask_sparselabel_seed62064": _public_path('experiments/archive/functional_learning/data/batched_aoa_measured/densemask_sparselabel_seed62064/full/AoA_word/surprisal.json'),
}


def sigmoid(x, a, b, c, d):
    return a / (1.0 + np.exp(-b * (x - c))) + d


def read_cdi(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """Return child curves by word: proportions over age bins and raw non-null counts."""
    df = pd.read_csv(path)
    # The BabyLM AoA file has one row per word and month/age columns. Keep the
    # implementation permissive because older challenge snapshots used slightly
    # different column names.
    word_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in {"word", "target_word", "item_definition", "definition"} or "word" == lc:
            word_col = c
            break
    if word_col is None:
        word_col = df.columns[0]
    age_cols = []
    ages = []
    for c in df.columns:
        if c == word_col:
            continue
        s = str(c)
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits:
            try:
                ages.append(float(digits))
                age_cols.append(c)
            except Exception:
                pass
    if not age_cols:
        raise RuntimeError(f"No age/proportion columns found in {path}; columns={list(df.columns)}")
    order = np.argsort(np.asarray(ages, dtype=float))
    age_cols = [age_cols[i] for i in order]
    ages_arr = np.asarray([ages[i] for i in order], dtype=float)
    out = {}
    for _, row in df.iterrows():
        word = str(row[word_col]).strip().lower()
        vals = pd.to_numeric(row[age_cols], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if mask.sum() >= 3:
            out[word] = (ages_arr[mask], vals[mask], int(mask.sum()))
    return out


def fit_child_aoa(ages: np.ndarray, props: np.ndarray) -> Optional[float]:
    # Official CDI proportions can be percentages or [0,1]; normalize when needed.
    y = props.astype(float)
    if np.nanmax(y) > 1.5:
        y = y / 100.0
    try:
        popt, _ = curve_fit(sigmoid, ages, y, p0=[1.0, 0.3, float(np.nanmedian(ages)), 0.0], maxfev=20000)
        a, b, c, d = [float(x) for x in popt]
        if not np.isfinite([a, b, c, d]).all() or abs(a) < 1e-9 or abs(b) < 1e-9:
            return None
        # 50% of the fitted asymptotic rise.
        target = 0.5 * a + d
        ratio = a / (target - d) - 1.0
        if ratio <= 0:
            return None
        x = c - math.log(ratio) / b
        if not np.isfinite(x):
            return None
        return float(x)
    except Exception:
        return None


def tokenizer_word_pieces(tokenizer, word: str) -> int:
    ids = tokenizer.encode(word, add_special_tokens=False)
    return max(1, len(ids))


def fit_model_aoa(word: str, rows: List[Tuple[float, float]], n_pieces: int) -> Optional[Dict[str, float]]:
    # rows: (word_count, mean_surprisal). Official logic fits performance = -surprisal
    # over log10(word_count) and uses a random-ceiling threshold scaled by subwords.
    clean = [(float(s), float(v)) for s, v in rows if np.isfinite(s) and np.isfinite(v) and s > 0]
    if len(clean) < 3:
        return None
    clean.sort()
    steps = np.asarray([x[0] for x in clean], dtype=float)
    y = -np.asarray([x[1] for x in clean], dtype=float)
    x = np.log10(steps)
    ceiling = -float(n_pieces) * math.log(VOCAB_SIZE)
    # Acquisition threshold: halfway from random ceiling to the best observed score.
    best = float(np.nanmax(y))
    if not np.isfinite(best) or best <= ceiling:
        return None
    threshold = (best + ceiling) / 2.0
    try:
        amp0 = max(1e-3, best - ceiling)
        p0 = [amp0, 1.0, float(np.nanmedian(x)), ceiling]
        bounds = ([-1e3, -50.0, x.min() - 5.0, -1e4], [1e3, 50.0, x.max() + 5.0, 1e4])
        popt, _ = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=20000)
        a, b, c, d = [float(z) for z in popt]
        if not np.isfinite([a, b, c, d]).all() or abs(a) < 1e-9 or abs(b) < 1e-9:
            return None
        ratio = a / (threshold - d) - 1.0
        if ratio <= 0:
            return None
        log_aoa = c - math.log(ratio) / b
        if not np.isfinite(log_aoa):
            return None
        aoa_step = 10.0 ** log_aoa
        if aoa_step < steps.min() or aoa_step > steps.max():
            return None
        return {
            "model_aoa_word_count": float(aoa_step),
            "threshold": float(threshold),
            "best_perf": float(best),
            "ceiling": float(ceiling),
            "n_points": int(len(clean)),
            "first_word_count": float(steps.min()),
            "last_word_count": float(steps.max()),
        }
    except Exception:
        return None


def load_surprisal(path: Path) -> Tuple[dict, Dict[str, Dict[float, List[float]]]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    metadata = obj.get("metadata", {}) if isinstance(obj, dict) else {}
    records = obj.get("results") or obj.get("rows") or obj.get("data") or []
    # Some files may store a dict of rows. Flatten defensively.
    if isinstance(records, dict):
        if "rows" in records and isinstance(records["rows"], list):
            records = records["rows"]
        else:
            tmp = []
            for v in records.values():
                if isinstance(v, list):
                    tmp.extend(v)
                elif isinstance(v, dict):
                    tmp.append(v)
            records = tmp
    acc: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if not isinstance(r, dict):
            continue
        w = str(r.get("target_word") or r.get("word") or r.get("target") or "").strip().lower()
        if not w:
            continue
        wc = r.get("word_count", r.get("step", r.get("training_step")))
        sv = r.get("surprisal", r.get("mean_surprisal", r.get("score")))
        try:
            wc_f = float(wc)
            sv_f = float(sv)
        except Exception:
            continue
        if np.isfinite(wc_f) and np.isfinite(sv_f):
            acc[w][wc_f].append(sv_f)
    return metadata, acc


def summarize_endpoint(name: str, path: Path, child_curves, tokenizer) -> Tuple[dict, List[dict]]:
    metadata, acc = load_surprisal(path)
    rows = []
    for word, by_step in acc.items():
        if word not in child_curves:
            continue
        ages, props, child_points = child_curves[word]
        child_aoa = fit_child_aoa(ages, props)
        if child_aoa is None:
            continue
        mean_rows = [(wc, float(np.mean(vals))) for wc, vals in by_step.items() if vals]
        n_pieces = tokenizer_word_pieces(tokenizer, word)
        fit = fit_model_aoa(word, mean_rows, n_pieces)
        if fit is None:
            continue
        row = {
            "endpoint": name,
            "word": word,
            "child_aoa": float(child_aoa),
            "model_aoa_word_count": fit["model_aoa_word_count"],
            "model_aoa_log10_word_count": float(math.log10(fit["model_aoa_word_count"])),
            "n_subword_tokens": int(n_pieces),
            "n_model_points": fit["n_points"],
            "n_child_points": int(child_points),
            "first_word_count": fit["first_word_count"],
            "last_word_count": fit["last_word_count"],
            "threshold": fit["threshold"],
            "best_perf": fit["best_perf"],
            "ceiling": fit["ceiling"],
        }
        rows.append(row)
    if len(rows) >= 3:
        r, p = pearsonr([x["child_aoa"] for x in rows], [x["model_aoa_log10_word_count"] for x in rows])
        r = float(r); p = float(p)
    else:
        r = float("nan"); p = float("nan")
    summary = {
        "endpoint": name,
        "surprisal_path": str(path.relative_to(ROOT)),
        "metadata_model_name": metadata.get("model_name"),
        "metadata_total_steps": metadata.get("total_steps"),
        "metadata_completed_steps": metadata.get("completed_steps"),
        "raw_pearson_r_child_vs_model_log10_aoa": r,
        "raw_pearson_p": p,
        "fitted_word_count": len(rows),
        "leaderboard_stored_score": 0.0 if (np.isfinite(r) and p > 0.1) else r,
        "near_significant_harmful_negative": bool(np.isfinite(r) and r < 0 and p <= 0.2),
        "significant_negative_at_p_0p1": bool(np.isfinite(r) and r < 0 and p <= 0.1),
        "significant_positive_at_p_0p1": bool(np.isfinite(r) and r > 0 and p <= 0.1),
    }
    return summary, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    child_curves = read_cdi(CDI_PATH)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), use_fast=True)
    summaries = []
    all_rows = []
    missing = []
    for name, path in ENDPOINTS.items():
        if not path.exists():
            missing.append({"endpoint": name, "path": str(path.relative_to(ROOT))})
            continue
        summary, rows = summarize_endpoint(name, path, child_curves, tokenizer)
        summaries.append(summary)
        all_rows.extend(rows)
        pd.DataFrame(rows).to_csv(OUT / f"{name}_word_fits.csv", index=False)
        print(json.dumps({"endpoint": name, "r": summary["raw_pearson_r_child_vs_model_log10_aoa"], "p": summary["raw_pearson_p"], "n": summary["fitted_word_count"]}), flush=True)
    # Pairwise overlap table among fitted words.
    word_sets = {s["endpoint"]: set(r["word"] for r in all_rows if r["endpoint"] == s["endpoint"]) for s in summaries}
    overlap_rows = []
    for a in word_sets:
        for b in word_sets:
            overlap_rows.append({"endpoint_a": a, "endpoint_b": b, "overlap": len(word_sets[a] & word_sets[b]), "n_a": len(word_sets[a]), "n_b": len(word_sets[b])})
    pd.DataFrame(summaries).to_csv(_public_path('experiments/archive/relation_learning/data/raw_aoa_refits/raw_aoa_endpoint_summary.csv'), index=False)
    pd.DataFrame(all_rows).to_csv(_public_path('experiments/archive/relation_learning/data/raw_aoa_refits/raw_aoa_all_word_fits.csv'), index=False)
    pd.DataFrame(overlap_rows).to_csv(_public_path('experiments/archive/relation_learning/data/raw_aoa_refits/raw_aoa_fit_word_overlap.csv'), index=False)
    out_json = {"status": "RAW_AOA_REFITS", "cdi_path": str(CDI_PATH.relative_to(ROOT)), "tokenizer_path": str(TOKENIZER_PATH.relative_to(ROOT)), "summaries": summaries, "missing": missing, "outputs": {"summary_csv": str((_public_path('experiments/archive/relation_learning/data/raw_aoa_refits/raw_aoa_endpoint_summary.csv')).relative_to(ROOT)), "word_fits_csv": str((_public_path('experiments/archive/relation_learning/data/raw_aoa_refits/raw_aoa_all_word_fits.csv')).relative_to(ROOT)), "overlap_csv": str((_public_path('experiments/archive/relation_learning/data/raw_aoa_refits/raw_aoa_fit_word_overlap.csv')).relative_to(ROOT))}}
    with (_public_path('experiments/archive/relation_learning/data/raw_aoa_refits/summary.json')).open("w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, sort_keys=True)
    lines = ["# research raw AoA refits", "", f"CDI: `{out_json['cdi_path']}`", f"Tokenizer: `{out_json['tokenizer_path']}`", "", "| endpoint | raw r | p | fitted words | stored if official | harmful negative? |", "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['endpoint']} | {s['raw_pearson_r_child_vs_model_log10_aoa']:.6f} | {s['raw_pearson_p']:.6g} | {s['fitted_word_count']} | {s['leaderboard_stored_score']:.6f} | {s['near_significant_harmful_negative']} |")
    if missing:
        lines += ["", "Missing files:", json.dumps(missing, indent=2)]
    lines += ["", "Word-fit and overlap records are saved beside this summary."]
    (_public_path('research/documents/relation_learning/data/raw_aoa_refits/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "done", "out": str(OUT.relative_to(ROOT)), "n_endpoints": len(summaries)}), flush=True)


if __name__ == "__main__":
    main()
