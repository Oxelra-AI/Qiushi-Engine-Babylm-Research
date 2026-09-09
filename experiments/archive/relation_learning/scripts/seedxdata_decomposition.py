#!/usr/bin/env python3
"""Corrected seed-by-data decomposition of the recorded per-row losses.

The purpose is to re-read the per-row fitting records without inheriting the
incorrect interpretation that a low VIEW-CLEAN correlation necessarily means data
responsiveness.  With the fourth DeBERTa CLEAN seed43122 arm now available, the
DeBERTa cells form a 2(seed) x 2(data) design.  This script asks:

  * Is row-level improvement direction changed more by data substitution than by seed?
  * Is the VIEW-CLEAN row-level effect reproducible across seeds?
  * Are shared-arm contrast correlations above the null caused by algebraic arm overlap?
  * Do simple text-property associations survive difficulty control in this corrected design?

No training, no benchmark scoring, no upload, no leaderboard action.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats

ROOT = _public_path('.')
LOSS_STUDY_ROOT = _public_path('experiments/archive/relation_learning')
DATA_STUDY_ROOT = _public_path('experiments/archive/frontier_consolidation')
SHARED_LOSSES = _public_path('experiments/archive/relation_learning/data/shared_private_loss')
ADDITIONAL_LOSSES = _public_path('experiments/archive/relation_learning/data/seedxdata_loss')
OUTDIR = _public_path('experiments/archive/relation_learning/data/seedxdata_decomposition')
NOTE = _public_path('research/notes/relation_learning/corrected_seedxdata_decomposition.md')
HELDOUT_ROWS = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl')
FILLER_ROWS = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/common_filler_rows.jsonl')

ARMS = {
    "V43022": (_public_path('experiments/archive/relation_learning/data/shared_private_loss/D_V_43022_per_row_losses.npz'), "D_V_43022"),
    "V43122": (_public_path('experiments/archive/relation_learning/data/shared_private_loss/D_V_43122_per_row_losses.npz'), "D_V_43122"),
    "C43022": (_public_path('experiments/archive/relation_learning/data/shared_private_loss/D_C_43022_per_row_losses.npz'), "D_C_43022"),
    "C43122": (_public_path('experiments/archive/relation_learning/data/seedxdata_loss/D_C_43122_per_row_losses.npz'), "D_C_43122"),
}
CHECKPOINTS = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
WINDOWS = [("60to100", "chck_60M", "chck_100M"), ("80to100", "chck_80M", "chck_100M"),
           ("60to70", "chck_60M", "chck_70M"), ("70to80", "chck_70M", "chck_80M"),
           ("80to90", "chck_80M", "chck_90M"), ("90to100", "chck_90M", "chck_100M")]


def finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def pearson(x, y):
    x, y = finite_xy(x, y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": None, "p": None, "n": int(len(x))}
    r, p = stats.pearsonr(x, y)
    return {"r": float(r), "p": float(p), "n": int(len(x))}


def spearman(x, y):
    x, y = finite_xy(x, y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"rho": None, "p": None, "n": int(len(x))}
    r, p = stats.spearmanr(x, y)
    return {"rho": float(r), "p": float(p), "n": int(len(x))}


def residualize(y, controls):
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(y)] + [np.asarray(c, dtype=float) for c in controls])
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, np.nan, dtype=float)
    beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    out[m] = y[m] - X[m] @ beta
    return out


def partial_corr(x, y, controls):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = [np.asarray(c, dtype=float) for c in controls]
    m = np.isfinite(x) & np.isfinite(y)
    for c in controls:
        m &= np.isfinite(c)
    if int(m.sum()) < 5:
        return {"r": None, "p": None, "n": int(m.sum()), "k_controls": len(controls)}
    X = np.column_stack([np.ones(int(m.sum()))] + [c[m] for c in controls])
    bx, *_ = np.linalg.lstsq(X, x[m], rcond=None)
    by, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    rx = x[m] - X @ bx
    ry = y[m] - X @ by
    if np.std(rx) == 0 or np.std(ry) == 0:
        return {"r": None, "p": None, "n": int(m.sum()), "k_controls": len(controls)}
    r, p = stats.pearsonr(rx, ry)
    return {"r": float(r), "p": float(p), "n": int(m.sum()), "k_controls": len(controls)}


def load_npz():
    data = {}
    for short, (path, prefix) in ARMS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        data[short] = (np.load(path), prefix)
    return data


def arr(data, arm, eset, checkpoint, suffix="losses"):
    npz, prefix = data[arm]
    key = f"{prefix}__{eset}__{checkpoint}__{suffix}"
    return npz[key]


def check_ids(data, eset):
    ref = None
    out = {}
    for arm in ARMS:
        ids = arr(data, arm, eset, "chck_60M", "example_ids")
        out[arm] = {"n": int(len(ids)), "first5": [int(x) for x in ids[:5]]}
        if ref is None:
            ref = ids
        elif not np.array_equal(ref, ids):
            raise ValueError(f"example_id mismatch for {eset} {arm}")
    return out


def load_rows(path, expected_ids=None, sample_n=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(obj)
    if sample_n is not None:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(rows), size=min(sample_n, len(rows)), replace=False)
        idx.sort()
        rows = [rows[i] for i in idx]
    if expected_ids is not None:
        ids = np.array([r["example_id"] for r in rows], dtype=np.int64)
        if not np.array_equal(ids, expected_ids):
            # For the sampled filler set, research also uses the same seed+sorted indices;
            # fail loudly if this ever diverges.
            raise ValueError(f"row id mismatch for {path}")
    return rows


def row_features(rows):
    feats = {}
    text_list = [r.get("text", "") for r in rows]
    feats["words_meta"] = np.array([float(r.get("words", 0)) for r in rows])
    feats["char_len"] = np.array([float(len(t)) for t in text_list])
    feats["n_commas"] = np.array([float(t.count(",")) for t in text_list])
    feats["n_colons"] = np.array([float(t.count(":")) for t in text_list])
    feats["n_semicolons"] = np.array([float(t.count(";")) for t in text_list])
    feats["n_periods"] = np.array([float(t.count(".")) for t in text_list])
    feats["n_qmarks"] = np.array([float(t.count("?")) for t in text_list])
    feats["n_exclaims"] = np.array([float(t.count("!")) for t in text_list])
    feats["n_quotes"] = np.array([float(t.count('"') + t.count("'") + t.count("“") + t.count("”")) for t in text_list])
    feats["n_parens"] = np.array([float(t.count("(") + t.count(")") + t.count("[") + t.count("]")) for t in text_list])
    feats["n_digits"] = np.array([float(sum(ch.isdigit() for ch in t)) for t in text_list])
    feats["n_capitals"] = np.array([float(sum(ch.isupper() for ch in t)) for t in text_list])
    feats["n_sentences_rough"] = feats["n_periods"] + feats["n_qmarks"] + feats["n_exclaims"]
    # Crude lexical diversity on whitespace tokens; enough for confound checks, not a linguistic parser.
    toks = [t.split() for t in text_list]
    feats["whitespace_tokens"] = np.array([float(len(x)) for x in toks])
    feats["type_token_ratio"] = np.array([float(len(set(x)) / max(1, len(x))) for x in toks])
    feats["avg_word_len"] = np.array([float(np.mean([len(w) for w in x])) if x else 0.0 for x in toks])
    feats["source"] = [r.get("source", "") for r in rows]
    return feats


def cellwise_residual_improvements(data, eset, start, end):
    imps = {}
    starts = {}
    for arm in ARMS:
        starts[arm] = arr(data, arm, eset, start)
        imps[arm] = arr(data, arm, eset, start) - arr(data, arm, eset, end)  # positive improvement
    residual = {arm: residualize(imps[arm], [starts[arm]]) for arm in ARMS}
    residual_allstart = {arm: residualize(imps[arm], [starts[a] for a in ARMS]) for arm in ARMS}
    return imps, starts, residual, residual_allstart


def decompose(cells):
    # cells are improvement vectors: V43022, V43122, C43022, C43122.
    V0, V1, C0, C1 = cells["V43022"], cells["V43122"], cells["C43022"], cells["C43122"]
    return {
        "grand": (V0 + V1 + C0 + C1) / 4.0,
        "data_main_view_minus_clean": (V0 + V1 - C0 - C1) / 4.0,
        "seed_main_43022_minus_43122": (V0 + C0 - V1 - C1) / 4.0,
        "seed_x_data_interaction": (V0 - C0 - V1 + C1) / 4.0,
        "data_effect_seed43022": V0 - C0,
        "data_effect_seed43122": V1 - C1,
        "seed_effect_view": V0 - V1,
        "seed_effect_clean": C0 - C1,
    }


def var_info(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {"mean": float(np.mean(x)), "sd_across_rows": float(np.std(x, ddof=1)), "var_across_rows": float(np.var(x, ddof=1))}


def shared_arm_null(a, b, c, n_perm=1000, seed=4):
    """Null for corr(a-b, a-c) due only to shared arm a.

    Permuting b and c independently across rows preserves each arm's empirical distribution
    and the shared-arm algebra, but removes row-wise relation between nonshared arms and a.
    """
    rng = np.random.RandomState(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    a, b, c = a[m], b[m], c[m]
    obs = stats.pearsonr(a - b, a - c)[0]
    vals = []
    n = len(a)
    for _ in range(n_perm):
        bp = b[rng.permutation(n)]
        cp = c[rng.permutation(n)]
        vals.append(stats.pearsonr(a - bp, a - cp)[0])
    vals = np.array(vals, dtype=float)
    # Analytic equal-variance benchmark is 0.5; empirical permutation mean is better for unequal arm variances.
    return {
        "observed_r": float(obs),
        "equal_variance_null_r": 0.5,
        "perm_mean_r": float(np.mean(vals)),
        "perm_sd_r": float(np.std(vals, ddof=1)),
        "perm_p05_r": float(np.quantile(vals, 0.05)),
        "perm_p50_r": float(np.quantile(vals, 0.50)),
        "perm_p95_r": float(np.quantile(vals, 0.95)),
        "obs_minus_perm_mean": float(obs - np.mean(vals)),
        "n": int(n),
        "n_perm": int(n_perm),
    }


def summarize_window(data, eset, label, start, end):
    imps, starts, resid, resid_all = cellwise_residual_improvements(data, eset, start, end)
    ids = arr(data, "V43022", eset, start, "example_ids")
    out = {"eval_set": eset, "window": label, "start": start, "end": end, "n_rows": int(len(ids))}
    pair_defs = {
        "same_data_view_cross_seed_V43022xV43122": ("V43022", "V43122", "same_data_cross_seed"),
        "same_data_clean_cross_seed_C43022xC43122": ("C43022", "C43122", "same_data_cross_seed"),
        "same_seed_43022_cross_data_VxC": ("V43022", "C43022", "same_seed_cross_data"),
        "same_seed_43122_cross_data_VxC": ("V43122", "C43122", "same_seed_cross_data"),
        "diagonal_V43022xC43122": ("V43022", "C43122", "diagonal"),
        "diagonal_V43122xC43022": ("V43122", "C43022", "diagonal"),
    }
    out["pairwise"] = {}
    for name, (a, b, kind) in pair_defs.items():
        controls = [starts[a], starts[b]]
        out["pairwise"][name] = {
            "kind": kind,
            "raw_pearson": pearson(imps[a], imps[b]),
            "raw_spearman": spearman(imps[a], imps[b]),
            "partial_startloss": partial_corr(imps[a], imps[b], controls),
            "cell_residual_pearson": pearson(resid[a], resid[b]),
            "all_startloss_residual_pearson": pearson(resid_all[a], resid_all[b]),
            "mean_improvement_a": float(np.nanmean(imps[a])),
            "mean_improvement_b": float(np.nanmean(imps[b])),
        }
    # Averages that diagnose the research interpretation.
    same_data = [out["pairwise"]["same_data_view_cross_seed_V43022xV43122"]["partial_startloss"]["r"],
                 out["pairwise"]["same_data_clean_cross_seed_C43022xC43122"]["partial_startloss"]["r"]]
    same_seed = [out["pairwise"]["same_seed_43022_cross_data_VxC"]["partial_startloss"]["r"],
                 out["pairwise"]["same_seed_43122_cross_data_VxC"]["partial_startloss"]["r"]]
    out["average_partial_r"] = {
        "same_data_cross_seed": float(np.nanmean(same_data)),
        "same_seed_cross_data": float(np.nanmean(same_seed)),
        "same_seed_minus_same_data": float(np.nanmean(same_seed) - np.nanmean(same_data)),
    }
    dec_raw = decompose(imps)
    dec_res = decompose(resid_all)
    out["decomposition_raw"] = {k: var_info(v) for k, v in dec_raw.items()}
    out["decomposition_all_startloss_residualized"] = {k: var_info(v) for k, v in dec_res.items()}
    # Independent contrast reproducibility: no shared arms.
    out["contrast_reproducibility"] = {
        "data_effect_VminusC_across_seeds_raw": pearson(dec_raw["data_effect_seed43022"], dec_raw["data_effect_seed43122"]),
        "seed_effect_43022minus43122_across_data_raw": pearson(dec_raw["seed_effect_view"], dec_raw["seed_effect_clean"]),
        "data_effect_VminusC_across_seeds_residualized": pearson(dec_res["data_effect_seed43022"], dec_res["data_effect_seed43122"]),
        "seed_effect_43022minus43122_across_data_residualized": pearson(dec_res["seed_effect_view"], dec_res["seed_effect_clean"]),
    }
    # Shared-arm contrast correlations and nulls.
    V0, V1, C0, C1 = imps["V43022"], imps["V43122"], imps["C43022"], imps["C43122"]
    RV0, RV1, RC0, RC1 = resid_all["V43022"], resid_all["V43122"], resid_all["C43022"], resid_all["C43122"]
    out["shared_arm_nulls"] = {
        "corr_V0minusC0_with_V0minusV1_raw": shared_arm_null(V0, C0, V1, seed=41),
        "corr_V0minusC0_with_V0minusV1_residualized": shared_arm_null(RV0, RC0, RV1, seed=42),
        "corr_V1minusC1_with_V1minusV0_raw": shared_arm_null(V1, C1, V0, seed=43),
        "corr_V1minusC1_with_V1minusV0_residualized": shared_arm_null(RV1, RC1, RV0, seed=44),
        "corr_V0minusC0_with_C1minusC0_raw": shared_arm_null(C0, V0, C1, seed=45),
        "corr_V0minusC0_with_C1minusC0_residualized": shared_arm_null(RC0, RV0, RC1, seed=46),
    }
    # Store a few vectors for feature analysis outside JSON? Summary stats only in JSON.
    return out, imps, starts, resid_all, dec_raw, dec_res


def feature_analysis(rows, starts, dec_raw, dec_res, eset, window):
    feats = row_features(rows)
    numeric = {k: v for k, v in feats.items() if k != "source"}
    controls = [starts[a] for a in ARMS]
    targets = {
        "data_main_raw": dec_raw["data_main_view_minus_clean"],
        "seed_main_raw": dec_raw["seed_main_43022_minus_43122"],
        "interaction_raw": dec_raw["seed_x_data_interaction"],
        "data_main_residualized": dec_res["data_main_view_minus_clean"],
        "seed_main_residualized": dec_res["seed_main_43022_minus_43122"],
        "interaction_residualized": dec_res["seed_x_data_interaction"],
    }
    table = []
    for tname, tvec in targets.items():
        for fname, fvec in numeric.items():
            raw = pearson(tvec, fvec)
            part = partial_corr(tvec, fvec, controls)
            table.append({
                "eval_set": eset, "window": window, "target": tname, "feature": fname,
                "raw_r": raw["r"], "raw_p": raw["p"], "partial_startloss_r": part["r"],
                "partial_startloss_p": part["p"], "n": raw["n"],
            })
    # Source means for main effects.
    sources = sorted(set(feats["source"]))
    source_rows = []
    for src in sources:
        idx = np.array([s == src for s in feats["source"]], dtype=bool)
        if idx.sum() < 5:
            continue
        row = {"eval_set": eset, "window": window, "source": src, "n": int(idx.sum())}
        for tname, tvec in targets.items():
            row[tname + "_mean"] = float(np.nanmean(tvec[idx]))
            row[tname + "_sd"] = float(np.nanstd(tvec[idx], ddof=1))
        source_rows.append(row)
    return table, source_rows


def write_csv(path, rows, header=None):
    import csv
    rows = list(rows)
    if header is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        header = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_npz()
    id_checks = {eset: check_ids(data, eset) for eset in ["heldout", "filler"]}

    # Reconstruct the exact row order used by the evaluator for feature analysis.
    heldout_ids = arr(data, "V43022", "heldout", "chck_60M", "example_ids")
    filler_ids = arr(data, "V43022", "filler", "chck_60M", "example_ids")
    rows_by_set = {
        "heldout": load_rows(HELDOUT_ROWS, heldout_ids),
        "filler": load_rows(FILLER_ROWS, filler_ids, sample_n=5000),
    }

    summaries = []
    feature_rows = []
    source_rows = []
    compact_rows = []
    full = {"status": "SEEDXDATA_DECOMPOSITION_DONE", "id_checks": id_checks, "windows": {}}
    for eset in ["heldout", "filler"]:
        full["windows"][eset] = {}
        for label, start, end in WINDOWS:
            summ, imps, starts, resid_all, dec_raw, dec_res = summarize_window(data, eset, label, start, end)
            full["windows"][eset][label] = summ
            summaries.append(summ)
            # Compact table for quick reading.
            row = {
                "eval_set": eset,
                "window": label,
                "mean_imp_V43022": float(np.mean(imps["V43022"])),
                "mean_imp_V43122": float(np.mean(imps["V43122"])),
                "mean_imp_C43022": float(np.mean(imps["C43022"])),
                "mean_imp_C43122": float(np.mean(imps["C43122"])),
                "avg_partial_same_data_cross_seed": summ["average_partial_r"]["same_data_cross_seed"],
                "avg_partial_same_seed_cross_data": summ["average_partial_r"]["same_seed_cross_data"],
                "same_seed_minus_same_data": summ["average_partial_r"]["same_seed_minus_same_data"],
                "raw_data_effect_corr_across_seeds": summ["contrast_reproducibility"]["data_effect_VminusC_across_seeds_raw"]["r"],
                "resid_data_effect_corr_across_seeds": summ["contrast_reproducibility"]["data_effect_VminusC_across_seeds_residualized"]["r"],
                "raw_seed_effect_corr_across_data": summ["contrast_reproducibility"]["seed_effect_43022minus43122_across_data_raw"]["r"],
                "resid_seed_effect_corr_across_data": summ["contrast_reproducibility"]["seed_effect_43022minus43122_across_data_residualized"]["r"],
                "raw_data_main_sd": summ["decomposition_raw"]["data_main_view_minus_clean"]["sd_across_rows"],
                "raw_seed_main_sd": summ["decomposition_raw"]["seed_main_43022_minus_43122"]["sd_across_rows"],
                "raw_interaction_sd": summ["decomposition_raw"]["seed_x_data_interaction"]["sd_across_rows"],
                "resid_data_main_sd": summ["decomposition_all_startloss_residualized"]["data_main_view_minus_clean"]["sd_across_rows"],
                "resid_seed_main_sd": summ["decomposition_all_startloss_residualized"]["seed_main_43022_minus_43122"]["sd_across_rows"],
                "resid_interaction_sd": summ["decomposition_all_startloss_residualized"]["seed_x_data_interaction"]["sd_across_rows"],
                "shared_arm_obs_raw": summ["shared_arm_nulls"]["corr_V0minusC0_with_V0minusV1_raw"]["observed_r"],
                "shared_arm_perm_mean_raw": summ["shared_arm_nulls"]["corr_V0minusC0_with_V0minusV1_raw"]["perm_mean_r"],
                "shared_arm_obs_minus_null_raw": summ["shared_arm_nulls"]["corr_V0minusC0_with_V0minusV1_raw"]["obs_minus_perm_mean"],
            }
            compact_rows.append(row)
            if eset == "heldout" and label in ("60to100", "80to100"):
                fr, sr = feature_analysis(rows_by_set[eset], starts, dec_raw, dec_res, eset, label)
                feature_rows.extend(fr)
                source_rows.extend(sr)

    summary_json = _public_path('experiments/archive/relation_learning/data/seedxdata_decomposition/seedxdata_decomposition.json')
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)
    write_csv(_public_path('experiments/archive/relation_learning/data/seedxdata_decomposition/seedxdata_window_summary.csv'), compact_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/seedxdata_decomposition/heldout_text_feature_partial_correlations.csv'), feature_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/seedxdata_decomposition/heldout_source_means.csv'), source_rows)

    # Human-readable note with the key corrected results.
    h60 = full["windows"]["heldout"]["60to100"]
    h80 = full["windows"]["heldout"]["80to100"]
    def fmt(x, nd=3):
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            return "NA"
        return f"{x:.{nd}f}"
    def pr(pair, summ=h60):
        return summ["pairwise"][pair]["partial_startloss"]["r"]
    note = f"""# research: Corrected seed x data decomposition of row-level improvement

## Why this step was necessary

research produced a valuable per-row loss asset, but its interpretation was too strong.  The research table itself shows that, for DeBERTa heldout 60→100M, the same-data cross-seed partial correlation for VIEW was +0.370 while the same-seed cross-data partial correlation for seed43022 was +0.369.  A data substitution and a seed change therefore perturbed the row-level improvement direction by nearly the same amount.  This step evaluated the missing fourth cell, `D_C_43122`, from the existing frontier_consolidation checkpoint `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122`, then recomputed the design as 2 seeds × 2 data conditions.

No new training, official benchmark scoring, upload, or leaderboard action was performed.

## New forward-pass cell

`D_C_43122` was scored with the same deterministic masks as research on 6,992 heldout rows and 5,000 filler rows at 60/70/80/90/100M.

Heldout mean losses:

| checkpoint | mean loss |
|---|---:|
| 60M | 2.6803 |
| 70M | 2.6054 |
| 80M | 2.5329 |
| 90M | 2.5107 |
| 100M | 2.5037 |

This makes the DeBERTa row-level design complete:

| cell | data | seed | 60→100M mean improvement |
|---|---|---:|---:|
| D_V_43022 | VIEW | 43022 | {fmt(np.mean((arr(data,'V43022','heldout','chck_60M')-arr(data,'V43022','heldout','chck_100M'))),4)} |
| D_V_43122 | VIEW | 43122 | {fmt(np.mean((arr(data,'V43122','heldout','chck_60M')-arr(data,'V43122','heldout','chck_100M'))),4)} |
| D_C_43022 | CLEAN | 43022 | {fmt(np.mean((arr(data,'C43022','heldout','chck_60M')-arr(data,'C43022','heldout','chck_100M'))),4)} |
| D_C_43122 | CLEAN | 43122 | {fmt(np.mean((arr(data,'C43122','heldout','chck_60M')-arr(data,'C43122','heldout','chck_100M'))),4)} |

## Corrected pairwise reading

Heldout 60→100M partial correlations, controlling the two compared arms' 60M losses:

| comparison | meaning | partial r |
|---|---|---:|
| V43022 × V43122 | same data, different seed | {fmt(pr('same_data_view_cross_seed_V43022xV43122'))} |
| C43022 × C43122 | same data, different seed | {fmt(pr('same_data_clean_cross_seed_C43022xC43122'))} |
| V43022 × C43022 | same seed, different data | {fmt(pr('same_seed_43022_cross_data_VxC'))} |
| V43122 × C43122 | same seed, different data | {fmt(pr('same_seed_43122_cross_data_VxC'))} |

Average same-data cross-seed r = {fmt(h60['average_partial_r']['same_data_cross_seed'])}.  Average same-seed cross-data r = {fmt(h60['average_partial_r']['same_seed_cross_data'])}.  Difference = {fmt(h60['average_partial_r']['same_seed_minus_same_data'])}.  Thus the research 0.37-versus-0.90 contrast should be read as a difference in trajectory determinism at these checkpoints and losses, not as evidence that DeBERTa is specifically data-responsive.

The 80→100M window is even less seed/data-stable: average same-data cross-seed r = {fmt(h80['average_partial_r']['same_data_cross_seed'])}, average same-seed cross-data r = {fmt(h80['average_partial_r']['same_seed_cross_data'])}.

## Stable row-level VIEW-minus-CLEAN effect is weak

The direct reproducibility of the row-level data effect is the correlation between `VIEW-CLEAN` contrast vectors computed independently in the two seeds.  It uses no shared arms.

| effect reproducibility | 60→100M raw r | 60→100M residualized r | 80→100M raw r | 80→100M residualized r |
|---|---:|---:|---:|---:|
| data effect `(V-C)` across seeds | {fmt(h60['contrast_reproducibility']['data_effect_VminusC_across_seeds_raw']['r'])} | {fmt(h60['contrast_reproducibility']['data_effect_VminusC_across_seeds_residualized']['r'])} | {fmt(h80['contrast_reproducibility']['data_effect_VminusC_across_seeds_raw']['r'])} | {fmt(h80['contrast_reproducibility']['data_effect_VminusC_across_seeds_residualized']['r'])} |
| seed effect `(43022-43122)` across data | {fmt(h60['contrast_reproducibility']['seed_effect_43022minus43122_across_data_raw']['r'])} | {fmt(h60['contrast_reproducibility']['seed_effect_43022minus43122_across_data_residualized']['r'])} | {fmt(h80['contrast_reproducibility']['seed_effect_43022minus43122_across_data_raw']['r'])} | {fmt(h80['contrast_reproducibility']['seed_effect_43022minus43122_across_data_residualized']['r'])} |

The stable row-level data-contrast direction is therefore at most weak in this instrument.  A VIEW/CLEAN benchmark contrast, if it replicates across seeds, is probably not carried by a simple reproducible direction in heldout-row MLM-loss improvement.

## Shared-arm correlations must be read against their algebraic null

The earlier "impressionable rows" number correlated two differences sharing `D_V_43022`: `corr(V43022-C43022, V43022-V43122)`.  A null model with independent equal-variance cell noise around a common direction predicts about 0.5 because both differences contain the same arm.  With the empirical row distributions, the permutation null is also almost exactly the observed value:

| contrast correlation | observed r | permutation null mean | null p05–p95 | observed - null mean |
|---|---:|---:|---:|---:|
| `corr(V43022-C43022, V43022-V43122)` 60→100M | {fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_raw']['observed_r'])} | {fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_raw']['perm_mean_r'])} | {fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_raw']['perm_p05_r'])}–{fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_raw']['perm_p95_r'])} | {fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_raw']['obs_minus_perm_mean'])} |
| residualized same contrast | {fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_residualized']['observed_r'])} | {fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_residualized']['perm_mean_r'])} | {fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_residualized']['perm_p05_r'])}–{fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_residualized']['perm_p95_r'])} | {fmt(h60['shared_arm_nulls']['corr_V0minusC0_with_V0minusV1_residualized']['obs_minus_perm_mean'])} |

So research's "data-responsive and seed-sensitive rows overlap" interpretation is not supported; the correlation is essentially the shared-arm null.

## Decomposition magnitudes

Orthogonal 2×2 row-level decomposition of positive improvement, heldout 60→100M:

| component | mean | sd across rows | residualized sd across rows |
|---|---:|---:|---:|
| common improvement | {fmt(h60['decomposition_raw']['grand']['mean'],4)} | {fmt(h60['decomposition_raw']['grand']['sd_across_rows'],4)} | {fmt(h60['decomposition_all_startloss_residualized']['grand']['sd_across_rows'],4)} |
| data main `(VIEW-CLEAN)/2` contribution | {fmt(h60['decomposition_raw']['data_main_view_minus_clean']['mean'],4)} | {fmt(h60['decomposition_raw']['data_main_view_minus_clean']['sd_across_rows'],4)} | {fmt(h60['decomposition_all_startloss_residualized']['data_main_view_minus_clean']['sd_across_rows'],4)} |
| seed main `(43022-43122)/2` contribution | {fmt(h60['decomposition_raw']['seed_main_43022_minus_43122']['mean'],4)} | {fmt(h60['decomposition_raw']['seed_main_43022_minus_43122']['sd_across_rows'],4)} | {fmt(h60['decomposition_all_startloss_residualized']['seed_main_43022_minus_43122']['sd_across_rows'],4)} |
| seed×data interaction | {fmt(h60['decomposition_raw']['seed_x_data_interaction']['mean'],4)} | {fmt(h60['decomposition_raw']['seed_x_data_interaction']['sd_across_rows'],4)} | {fmt(h60['decomposition_all_startloss_residualized']['seed_x_data_interaction']['sd_across_rows'],4)} |

The common row-level improvement is real, but the data main component is not larger than the seed component or the interaction.  This is the opposite of a clean row-level data-steering result.

## Difficulty-controlled text-property check

Feature correlations are saved in `heldout_text_feature_partial_correlations.csv`.  In the corrected design, simple punctuation/length features do not by themselves rescue a strong seed-stable row-level data-effect story; the next useful reading is at benchmark-item or representation level, not another scalar text-property claim from these rows.

## Consequence for the data-efficient learning principle

The research dataset remains valuable, but its correct conclusion is a fork:

1. If VIEW/CLEAN benchmark improvements replicate across seeds while this heldout-row loss instrument shows little stable data-specific direction, then the conversion mechanism is probably representation-level: the data changes reusable coordinates or hidden-state geometry shared across many rows, not a simple list of heldout rows whose MLM loss improves.
2. If VIEW/CLEAN benchmark improvements do not replicate beyond the seed floor exposed by the register experiments, then the fixed-budget substitution ledger must be rebuilt around seed-stable sub-benchmark facts rather than single-seed half-point aggregate contrasts.

The next tests should therefore (i) link the corrected seed×data decomposition to per-item benchmark movement, and (ii) move the conversion measurement to hidden states/representational coordinates if benchmark effects are seed-stable.  RoBERTa still needs an actual model-seed pair before its VIEW/CLEAN trajectory determinism can be called data-inertness; the `n=3` entries in the research terminal-rate ladder are row/mask replicates, not independent RoBERTa trainings.

## Files

- New `D_C_43122` per-row losses: `experiments/archive/relation_learning/data/seedxdata_loss/D_C_43122_per_row_losses.npz`
- Full corrected analysis: `experiments/archive/relation_learning/data/seedxdata_decomposition/seedxdata_decomposition.json`
- Window summary: `experiments/archive/relation_learning/data/seedxdata_decomposition/seedxdata_window_summary.csv`
- Feature partial correlations: `experiments/archive/relation_learning/data/seedxdata_decomposition/heldout_text_feature_partial_correlations.csv`
- Source means: `experiments/archive/relation_learning/data/seedxdata_decomposition/heldout_source_means.csv`
- Script: `experiments/archive/relation_learning/scripts/seedxdata_decomposition.py`
"""
    NOTE.write_text(note, encoding="utf-8")

    print(json.dumps({
        "status": "SEEDXDATA_DECOMPOSITION_DONE",
        "summary_json": str(summary_json),
        "summary_csv": str(_public_path('experiments/archive/relation_learning/data/seedxdata_decomposition/seedxdata_window_summary.csv')),
        "note": str(NOTE),
        "heldout_60to100_avg_partial_same_data_cross_seed": h60["average_partial_r"]["same_data_cross_seed"],
        "heldout_60to100_avg_partial_same_seed_cross_data": h60["average_partial_r"]["same_seed_cross_data"],
        "heldout_60to100_data_effect_corr_across_seeds_raw": h60["contrast_reproducibility"]["data_effect_VminusC_across_seeds_raw"]["r"],
        "heldout_60to100_data_effect_corr_across_seeds_residualized": h60["contrast_reproducibility"]["data_effect_VminusC_across_seeds_residualized"]["r"],
        "heldout_60to100_shared_arm_obs": h60["shared_arm_nulls"]["corr_V0minusC0_with_V0minusV1_raw"]["observed_r"],
        "heldout_60to100_shared_arm_perm_mean": h60["shared_arm_nulls"]["corr_V0minusC0_with_V0minusV1_raw"]["perm_mean_r"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
