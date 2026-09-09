#!/usr/bin/env python3
"""Adjusted analysis of frozen source-use probe records.

Purpose: sharpen the next MLM-native mechanism route after the causal reciprocal
branch closed. We analyze whether compact views have any source-conditioned
ordering interaction residual after accounting for copy opportunity and target
strata, rather than comparing raw family means that are dominated by compact's
large source-absent share.

Outputs:
  - data/adjusted_source_use_probe/adjusted_source_use_probe.json
  - data/adjusted_source_use_probe/adjusted_source_use_probe.md

This is CPU-only analysis of existing records; it is not a training result and
cannot establish downstream causality.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
from typing import Any

import numpy as np
import pandas as pd

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
DEFAULT_RECORDS = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/frozen_source_use_probe_chck82_full/source_use_probe_records.csv"
)
DEFAULT_OUT = ROOT / "data/adjusted_source_use_probe"

FAMILIES = [
    "compact_vs_compact_scrambled",
    "prefix_fluent_vs_prefix_scrambled",
    "sourcewide_onegap_vs_sourcewide_onegap_scrambled",
]
COMPACT = "compact_vs_compact_scrambled"
PREFIX = "prefix_fluent_vs_prefix_scrambled"
ONEGAP = "sourcewide_onegap_vs_sourcewide_onegap_scrambled"


def finite_mean(x: pd.Series) -> float | None:
    vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    return float(vals.mean())


def describe_subset(df: pd.DataFrame) -> dict[str, Any]:
    vals = pd.to_numeric(df["I_f"], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    out: dict[str, Any] = {
        "n_targets": int(len(vals)),
        "n_pairs": int(df["pair_id"].nunique()),
    }
    if len(vals):
        out.update(
            mean_I_f=float(vals.mean()),
            median_I_f=float(np.median(vals)),
            positive_frac=float(np.mean(vals > 0)),
            q10=float(np.quantile(vals, 0.10)),
            q90=float(np.quantile(vals, 0.90)),
            mean_source_effect_ord=finite_mean(df["source_effect_ord"]),
            mean_source_effect_scr=finite_mean(df["source_effect_scr"]),
            mean_nll_ord_s=finite_mean(df["nll_ord_s"]),
            mean_nll_scr_s=finite_mean(df["nll_scr_s"]),
            mean_nll_ord=finite_mean(df["nll_ord"]),
            mean_nll_scr=finite_mean(df["nll_scr"]),
        )
    return out


def family_subset_stats(df: pd.DataFrame) -> dict[str, Any]:
    return {fam: describe_subset(df[df.family == fam]) for fam in FAMILIES}


def _bin_source_decile(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "absent"
    if not math.isfinite(v):
        return "absent"
    if v <= 1:
        return "head01"
    if v <= 4:
        return "mid24"
    if v <= 7:
        return "late57"
    return "tail89"


def _bin_gap(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "absent"
    if not math.isfinite(v):
        return "absent"
    if v <= 8:
        return "near"
    if v <= 20:
        return "mid"
    if v <= 40:
        return "far"
    return "very_far"


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source_decile_bin"] = df["source_decile_min"].map(_bin_source_decile)
    df["gap_bin"] = df["min_visible_gap"].map(_bin_gap)
    df["ord_pos_bin"] = pd.cut(
        df["ord_word_pos"],
        bins=[-1, 2, 6, 12, 10_000],
        labels=["pos0_2", "pos3_6", "pos7_12", "pos13plus"],
    ).astype(str)
    # Collapse rare lexical classes for stable common-support reweighting.
    df["lex3"] = df["lex_class"].where(
        df["lex_class"].isin(["content", "function", "capitalized_content"]),
        "other",
    )
    df["copy3"] = df["copy_zone"].where(
        df["copy_zone"].isin(["prefix_only", "tail_only", "both_prefix_tail", "absent"]),
        "unknown",
    )
    df["match3"] = df["source_match_bin"].replace({"2_3": "multi", "4plus": "multi"})
    return df


def stratum_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = (
        df.groupby(["family"] + keys, dropna=False)
        .agg(n=("I_f", "size"), mean=("I_f", "mean"), pairs=("pair_id", "nunique"))
        .reset_index()
    )
    return g


def reweighted_family_means(
    df: pd.DataFrame,
    keys: list[str],
    target: str = "pooled_common",
    min_per_family: int = 5,
) -> dict[str, Any]:
    """Compute direct-standardized family means on common strata.

    Strata are retained only if every family has at least min_per_family target
    records. Target distribution is pooled over retained strata unless target is
    a family name, in which case that family's retained-stratum distribution is
    used. This intentionally drops compact source-absent strata when controls
    lack support; source-absent compact targets are reported separately.
    """
    families = list(FAMILIES)
    tab = stratum_table(df, keys)
    # Build stratum key as tuples.
    tab["_stratum"] = list(map(tuple, tab[keys].astype(str).to_numpy()))
    counts = tab.pivot(index="_stratum", columns="family", values="n").fillna(0)
    retained = counts.index[(counts[families] >= min_per_family).all(axis=1)].tolist()
    if not retained:
        return {"status": "NO_COMMON_SUPPORT", "keys": keys, "min_per_family": min_per_family}

    dfr = df.copy()
    dfr["_stratum"] = list(map(tuple, dfr[keys].astype(str).to_numpy()))
    dfr = dfr[dfr["_stratum"].isin(retained)].copy()

    # Target weights.
    if target in families:
        target_counts = dfr[dfr.family == target]["_stratum"].value_counts().to_dict()
    else:
        target_counts = dfr["_stratum"].value_counts().to_dict()
    total = sum(target_counts.values())
    weights = {k: v / total for k, v in target_counts.items()}

    means_by = (
        dfr.groupby(["family", "_stratum"], dropna=False)["I_f"]
        .mean()
        .reset_index()
    )
    means_lookup = {(r["family"], r["_stratum"]): float(r["I_f"]) for _, r in means_by.iterrows()}
    counts_lookup = (
        dfr.groupby(["family", "_stratum"], dropna=False)["I_f"]
        .size()
        .to_dict()
    )

    means = {}
    retained_counts = {}
    for fam in families:
        s = 0.0
        for st, w in weights.items():
            s += w * means_lookup[(fam, st)]
        means[fam] = float(s)
        retained_counts[fam] = int((dfr.family == fam).sum())

    contrasts = {
        "compact_minus_prefix": float(means[COMPACT] - means[PREFIX]),
        "compact_minus_onegap": float(means[COMPACT] - means[ONEGAP]),
        "compact_minus_mean_extracts": float(means[COMPACT] - 0.5 * (means[PREFIX] + means[ONEGAP])),
        "onegap_minus_prefix": float(means[ONEGAP] - means[PREFIX]),
    }

    return {
        "status": "OK",
        "keys": keys,
        "target_distribution": target,
        "min_per_family": min_per_family,
        "n_retained_strata": int(len(retained)),
        "n_possible_strata": int(counts.shape[0]),
        "retained_fraction_by_family": {
            fam: float(retained_counts[fam] / max(1, int((df.family == fam).sum()))) for fam in families
        },
        "retained_target_counts_by_family": retained_counts,
        "weighted_means": means,
        "contrasts": contrasts,
        "largest_target_weight_strata": [
            {"stratum": list(st), "weight": float(w)}
            for st, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:20]
        ],
    }


def bootstrap_reweighted(
    df: pd.DataFrame,
    keys: list[str],
    target: str,
    min_per_family: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Pair-cluster bootstrap for reweighted contrasts.

    Resample pair_id clusters within each family. Some bootstrap replicates may
    lose rare common-support strata; we recompute common support per replicate,
    which makes intervals conservative for sparse strata.
    """
    rng = random.Random(seed)
    fam_groups: dict[str, dict[Any, pd.DataFrame]] = {}
    pair_ids: dict[str, list[Any]] = {}
    for fam in FAMILIES:
        sub = df[df.family == fam]
        groups = {pid: g.copy() for pid, g in sub.groupby("pair_id", sort=False)}
        fam_groups[fam] = groups
        pair_ids[fam] = list(groups.keys())

    vals: dict[str, list[float]] = collections.defaultdict(list)
    ok = 0
    fail = 0
    for _ in range(n_boot):
        parts = []
        for fam in FAMILIES:
            ids = pair_ids[fam]
            draw = [rng.choice(ids) for _ in ids]
            # Add bootstrap instance id to avoid accidental same-index grouping downstream.
            parts.extend(fam_groups[fam][pid] for pid in draw)
        boot = pd.concat(parts, ignore_index=True)
        res = reweighted_family_means(boot, keys, target=target, min_per_family=min_per_family)
        if res.get("status") != "OK":
            fail += 1
            continue
        ok += 1
        for k, v in res["contrasts"].items():
            vals[k].append(float(v))
        for fam, v in res["weighted_means"].items():
            vals[f"mean::{fam}"].append(float(v))

    out: dict[str, Any] = {"n_boot_requested": n_boot, "n_boot_ok": ok, "n_boot_failed": fail}
    summaries = {}
    for k, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        if len(a) == 0:
            continue
        summaries[k] = {
            "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "q025": float(np.quantile(a, 0.025)),
            "q500": float(np.quantile(a, 0.5)),
            "q975": float(np.quantile(a, 0.975)),
            "p_gt_0": float(np.mean(a > 0)),
        }
    out["summaries"] = summaries
    return out


def matched_pair_family_deltas(df: pd.DataFrame, subset: pd.Series | np.ndarray) -> dict[str, Any]:
    """Use common pair IDs across families to compare per-pair means.

    This is a different adjustment: each family was probed on the same sampled
    pair IDs where possible, but target sets differ. We aggregate within pair and
    compare family differences over common pairs for the selected subset.
    """
    d = df[subset].copy()
    fam_pair = d.groupby(["family", "pair_id"], dropna=False)["I_f"].mean().reset_index()
    piv = fam_pair.pivot(index="pair_id", columns="family", values="I_f")
    out: dict[str, Any] = {
        "n_common_pairs": 0,
        "subset_n_targets_by_family": {fam: int((d.family == fam).sum()) for fam in FAMILIES},
    }
    missing_families = [fam for fam in FAMILIES if fam not in piv.columns]
    if missing_families:
        out["missing_families"] = missing_families
        out["support_note"] = "At least one family has no targets in this selected subset; pair-matched family deltas are unsupported."
        return out
    common = piv.dropna(subset=FAMILIES)
    out["n_common_pairs"] = int(len(common))
    if len(common) == 0:
        out["support_note"] = "No common pair IDs retain targets for all families in this subset."
        return out
    for name, series in {
        "compact_minus_prefix": common[COMPACT] - common[PREFIX],
        "compact_minus_onegap": common[COMPACT] - common[ONEGAP],
        "compact_minus_mean_extracts": common[COMPACT] - 0.5 * (common[PREFIX] + common[ONEGAP]),
        "onegap_minus_prefix": common[ONEGAP] - common[PREFIX],
    }.items():
        arr = series.to_numpy(dtype=float)
        out[name] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "se": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            "q025": float(np.quantile(arr, 0.025)),
            "q975": float(np.quantile(arr, 0.975)),
            "p_gt_0": float(np.mean(arr > 0)),
        }
    return out


def regression_residualization(df: pd.DataFrame, controls: list[str]) -> dict[str, Any]:
    """OLS residual family effect after categorical/numeric controls.

    Uses a full dummy design with one family baseline (prefix) and cluster
    bootstrap by pair_id within family. This is descriptive; no causal claim.
    """
    d = df.copy()
    y = d["I_f"].to_numpy(dtype=float)
    # Prepare controls. Missing numerical values get sentinel bins before dummying.
    design = []
    names = []
    intercept = np.ones(len(d), dtype=float)
    design.append(intercept)
    names.append("intercept")
    # family dummies: compact and onegap, prefix baseline.
    for fam in [COMPACT, ONEGAP]:
        design.append((d.family == fam).to_numpy(dtype=float))
        names.append(f"family::{fam}")
    for c in controls:
        if c in ["source_decile_min", "source_decile_max", "min_visible_gap", "ord_word_pos"]:
            vals = pd.to_numeric(d[c], errors="coerce").astype(float)
            med = float(vals[np.isfinite(vals)].median()) if np.isfinite(vals).any() else 0.0
            miss = ~np.isfinite(vals)
            filled = vals.mask(miss, med).to_numpy(dtype=float)
            # Standardize for conditioning.
            sd = float(np.std(filled)) or 1.0
            design.append((filled - float(np.mean(filled))) / sd)
            names.append(f"num::{c}")
            design.append(miss.to_numpy(dtype=float))
            names.append(f"missing::{c}")
        else:
            cats = d[c].astype(str).fillna("NA")
            levels = sorted(cats.unique().tolist())
            # Drop first level to avoid exact collinearity with intercept.
            for lev in levels[1:]:
                design.append((cats == lev).to_numpy(dtype=float))
                names.append(f"{c}::{lev}")
    X = np.vstack(design).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    # Conventional row residual variance for orientation only.
    dof = max(1, len(y) - np.linalg.matrix_rank(X))
    sigma2 = float(np.sum(resid ** 2) / dof)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    coeffs = {names[i]: {"beta": float(beta[i]), "row_se": float(se[i])} for i in range(len(names))}
    return {
        "n": int(len(d)),
        "rank": int(np.linalg.matrix_rank(X)),
        "n_columns": int(X.shape[1]),
        "baseline_family": PREFIX,
        "controls": controls,
        "family_coefficients": {
            "compact_minus_prefix_adjusted": coeffs.get(f"family::{COMPACT}"),
            "onegap_minus_prefix_adjusted": coeffs.get(f"family::{ONEGAP}"),
            "compact_minus_onegap_adjusted": {
                "beta": float(beta[names.index(f"family::{COMPACT}")] - beta[names.index(f"family::{ONEGAP}")]),
                "row_se_not_valid_for_difference": None,
            },
        },
        "selected_coefficients": {k: v for k, v in coeffs.items() if k.startswith("family::")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(DEFAULT_RECORDS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=218013)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.records)
    df = add_bins(df)
    # Ensure expected order/values.
    df = df[df.family.isin(FAMILIES)].copy()
    df["is_copied"] = df["source_match_bin"].astype(str) != "0"
    df["is_absent"] = df["source_match_bin"].astype(str) == "0"

    raw = {
        "all": family_subset_stats(df),
        "copied_only": family_subset_stats(df[df.is_copied]),
        "content_copied_only": family_subset_stats(df[df.is_copied & df.contentlike.astype(bool)]),
        "function_copied_only": family_subset_stats(df[df.is_copied & (~df.contentlike.astype(bool))]),
        "compact_absent_only": describe_subset(df[(df.family == COMPACT) & df.is_absent]),
    }

    # Main adjusted comparisons on copied targets; absent compact has no real support in other families.
    copied = df[df.is_copied].copy()
    main_keys = ["copy3", "match3", "lex3", "source_decile_bin", "gap_bin", "ord_pos_bin"]
    coarser_keys = ["copy3", "match3", "lex3", "source_decile_bin"]
    no_gap_keys = ["copy3", "match3", "lex3"]

    adjusted = {
        "copied_main_pooled_common": reweighted_family_means(copied, main_keys, target="pooled_common", min_per_family=5),
        "copied_main_compact_distribution": reweighted_family_means(copied, main_keys, target=COMPACT, min_per_family=5),
        "copied_coarser_pooled_common": reweighted_family_means(copied, coarser_keys, target="pooled_common", min_per_family=10),
        "copied_no_gap_pooled_common": reweighted_family_means(copied, no_gap_keys, target="pooled_common", min_per_family=20),
    }

    boot = {
        "copied_main_pooled_common": bootstrap_reweighted(copied, main_keys, "pooled_common", 5, args.bootstrap, args.seed),
        "copied_coarser_pooled_common": bootstrap_reweighted(copied, coarser_keys, "pooled_common", 10, args.bootstrap, args.seed + 1),
    }

    # Pair-matched deltas for selected meaningful subsets.
    pair_matched = {
        "all_targets": matched_pair_family_deltas(df, np.ones(len(df), dtype=bool)),
        "copied_targets": matched_pair_family_deltas(df, df.is_copied.to_numpy()),
        "unique_copied_content": matched_pair_family_deltas(
            df, ((df.source_match_bin.astype(str) == "1_unique") & df.contentlike.astype(bool)).to_numpy()
        ),
        "tail_only_copied": matched_pair_family_deltas(df, ((df.copy_zone == "tail_only") & df.is_copied).to_numpy()),
        "prefix_only_copied": matched_pair_family_deltas(df, ((df.copy_zone == "prefix_only") & df.is_copied).to_numpy()),
    }

    regressions = {
        "copied_ols_main": regression_residualization(
            copied,
            controls=["copy3", "match3", "lex3", "source_decile_bin", "gap_bin", "ord_pos_bin"],
        ),
        "copied_content_ols": regression_residualization(
            copied[copied.contentlike.astype(bool)],
            controls=["copy3", "match3", "source_decile_bin", "gap_bin", "ord_pos_bin"],
        ),
    }

    # Explain composition differences that make raw family means misleading.
    composition = {}
    for fam in FAMILIES:
        sub = df[df.family == fam]
        composition[fam] = {
            "n_targets": int(len(sub)),
            "absent_frac": float(np.mean(sub.is_absent)),
            "unique_frac": float(np.mean(sub.source_match_bin.astype(str) == "1_unique")),
            "tail_only_frac": float(np.mean(sub.copy_zone == "tail_only")),
            "prefix_only_frac": float(np.mean(sub.copy_zone == "prefix_only")),
            "contentlike_frac": float(np.mean(sub.contentlike.astype(bool))),
            "complete_bpe_copy_frac": float(np.mean(sub.complete_bpe_copy.astype(bool))),
        }

    conclusions = []
    raw_all = {fam: raw["all"][fam].get("mean_I_f") for fam in FAMILIES}
    raw_copied = {fam: raw["copied_only"][fam].get("mean_I_f") for fam in FAMILIES}
    conclusions.append(
        "Raw compact mean I_f is lower than prefix/onegap, but compact has a large source-absent target share; raw family means mostly mix retrieval-opportunity composition with family effects."
    )
    adj_main = adjusted["copied_main_pooled_common"]
    if adj_main.get("status") == "OK":
        cm = adj_main["contrasts"]["compact_minus_mean_extracts"]
        conclusions.append(
            f"On copied targets with common-support reweighting over copy/multiplicity/lexical/source-position/gap bins, compact-minus-mean-extractive I_f is {cm:+.3f} nats; this measures whether compact has a residual ordered-source interaction after copy-opportunity adjustment."
        )
    absent_comp = raw["compact_absent_only"].get("mean_I_f")
    conclusions.append(
        f"Compact source-absent targets retain positive but much weaker ordering interaction (mean I_f {absent_comp:.3f} nats) than copied compact targets; this supports weak semantic/context association, not strong direct retrieval."
    )
    conclusions.append(
        "This analysis remains a frozen reconstruction probe; it can sharpen predictions for a boundary-blocked MLM visibility intervention but cannot settle which training signal caused downstream Supplement/EWoK transitions."
    )

    result = {
        "status": "ADJUSTED_SOURCE_USE_PROBE_ANALYSIS",
        "records_path": str(args.records),
        "n_records": int(len(df)),
        "families": FAMILIES,
        "composition": composition,
        "raw_family_stats": raw,
        "adjusted_reweighted": adjusted,
        "bootstrap": boot,
        "pair_matched_deltas": pair_matched,
        "regressions": regressions,
        "interpretive_conclusions": conclusions,
        "method_notes": [
            "Common-support reweighting is restricted to copied targets because compact has many source-absent targets while prefix/onegap have almost none; forcing absent targets into a family comparison would be unsupported.",
            "Pair-cluster bootstrap resamples pair_id clusters within family and recomputes common support per replicate.",
            "OLS row standard errors are included only for orientation; pair-clustered bootstrap/reweighting and pair-matched deltas carry the main interpretation.",
            "All analyses use existing A02 research frozen chck82 source-use records; no new model training or evaluation is performed.",
        ],
    }

    json_path = out_dir / "adjusted_source_use_probe.json"
    md_path = out_dir / "adjusted_source_use_probe.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def fmt(x: Any, nd: int = 3) -> str:
        if x is None:
            return "NA"
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)

    lines = []
    lines.append("# research adjusted source-use probe analysis")
    lines.append("")
    lines.append("CPU-only analysis of A02 research frozen chck82 source-use records. This is a mechanism-sharpening analysis, not training evidence.")
    lines.append("")
    lines.append("## Family composition")
    lines.append("| family | n | absent frac | unique frac | tail-only frac | contentlike frac | complete-BPE-copy frac | raw mean I_f | copied mean I_f |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fam in FAMILIES:
        c = composition[fam]
        lines.append(
            f"| {fam} | {c['n_targets']} | {fmt(c['absent_frac'])} | {fmt(c['unique_frac'])} | {fmt(c['tail_only_frac'])} | {fmt(c['contentlike_frac'])} | {fmt(c['complete_bpe_copy_frac'])} | {fmt(raw_all[fam])} | {fmt(raw_copied[fam])} |"
        )
    lines.append("")
    lines.append("## Reweighted copied-target contrasts")
    for name, res in adjusted.items():
        lines.append(f"### {name}")
        if res.get("status") != "OK":
            lines.append(f"Status: {res.get('status')}")
            continue
        lines.append(
            f"Retained strata {res['n_retained_strata']}/{res['n_possible_strata']}; retained fractions: "
            + ", ".join(f"{fam}={fmt(v)}" for fam, v in res["retained_fraction_by_family"].items())
        )
        lines.append("Weighted means: " + ", ".join(f"{fam}={fmt(v)}" for fam, v in res["weighted_means"].items()))
        lines.append("Contrasts: " + ", ".join(f"{k}={fmt(v)}" for k, v in res["contrasts"].items()))
        lines.append("")
    lines.append("## Bootstrap intervals for main copied reweighting")
    for name, b in boot.items():
        lines.append(f"### {name} (ok {b['n_boot_ok']}/{b['n_boot_requested']})")
        for k, s in b.get("summaries", {}).items():
            if not (k.startswith("compact_minus") or k.startswith("mean::compact")):
                continue
            lines.append(f"- {k}: median {fmt(s['q500'])}, 95% [{fmt(s['q025'])}, {fmt(s['q975'])}], p>0 {fmt(s['p_gt_0'])}")
    lines.append("")
    lines.append("## Pair-matched deltas over common pair IDs")
    for subset_name, res in pair_matched.items():
        lines.append(f"### {subset_name}: common pairs {res.get('n_common_pairs')}")
        for k in ["compact_minus_prefix", "compact_minus_onegap", "compact_minus_mean_extracts", "onegap_minus_prefix"]:
            if k in res:
                v = res[k]
                lines.append(f"- {k}: mean {fmt(v['mean'])}, 95% quantile [{fmt(v['q025'])}, {fmt(v['q975'])}], p>0 {fmt(v['p_gt_0'])}")
    lines.append("")
    lines.append("## Compact source-absent targets")
    lines.append("Compact absent-only: " + ", ".join(f"{k}={fmt(v)}" for k, v in raw["compact_absent_only"].items() if isinstance(v, (int, float))))
    lines.append("")
    lines.append("## Interpretation")
    for c in conclusions:
        lines.append(f"- {c}")
    lines.append("")
    lines.append(f"JSON: `{json_path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "json": str(json_path),
        "md": str(md_path),
        "raw_mean_I_f": raw_all,
        "raw_copied_mean_I_f": raw_copied,
        "main_adjusted": adjusted["copied_main_pooled_common"].get("contrasts"),
        "main_boot_compact_minus_mean_extracts": boot["copied_main_pooled_common"].get("summaries", {}).get("compact_minus_mean_extracts"),
        "compact_absent_mean_I_f": absent_comp,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
