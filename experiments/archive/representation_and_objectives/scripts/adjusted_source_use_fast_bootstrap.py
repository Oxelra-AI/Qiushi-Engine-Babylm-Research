#!/usr/bin/env python3
"""Fast fixed-support pair-cluster bootstrap for research adjusted source-use analysis.

This complements adjusted_source_use_probe_analysis.py. It keeps the
observed common-support strata and direct-standardization weights fixed, then
resamples pair_id clusters within each family. This gives an uncertainty scale
for copied-target adjusted source-use contrasts without the very slow per-
replicate dataframe/common-support recomputation.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RECORDS = pathlib.Path("experiments/archive/frontier_consolidation/data/frozen_source_use_probe_chck82_full/source_use_probe_records.csv")
DEFAULT_OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/adjusted_source_use_probe")
FAMILIES = [
    "compact_vs_compact_scrambled",
    "prefix_fluent_vs_prefix_scrambled",
    "sourcewide_onegap_vs_sourcewide_onegap_scrambled",
]
COMPACT = FAMILIES[0]
PREFIX = FAMILIES[1]
ONEGAP = FAMILIES[2]


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
    df["ord_pos_bin"] = pd.cut(df["ord_word_pos"], bins=[-1, 2, 6, 12, 10_000], labels=["pos0_2", "pos3_6", "pos7_12", "pos13plus"]).astype(str)
    df["lex3"] = df["lex_class"].where(df["lex_class"].isin(["content", "function", "capitalized_content"]), "other")
    df["copy3"] = df["copy_zone"].where(df["copy_zone"].isin(["prefix_only", "tail_only", "both_prefix_tail", "absent"]), "unknown")
    df["match3"] = df["source_match_bin"].replace({"2_3": "multi", "4plus": "multi"})
    return df


def quant_summary(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "q025": float(np.quantile(arr, 0.025)),
        "q500": float(np.quantile(arr, 0.5)),
        "q975": float(np.quantile(arr, 0.975)),
        "p_gt_0": float(np.mean(arr > 0)),
    }


def family_matrices(df: pd.DataFrame, retained: list[tuple[str, ...]], keys: list[str]) -> dict[str, dict[str, Any]]:
    st_to_idx = {st: i for i, st in enumerate(retained)}
    out: dict[str, dict[str, Any]] = {}
    for fam in FAMILIES:
        sub = df[df.family == fam].copy()
        sub["_stratum"] = list(map(tuple, sub[keys].astype(str).to_numpy()))
        sub = sub[sub["_stratum"].isin(st_to_idx)].copy()
        pair_ids = sorted(sub["pair_id"].unique().tolist())
        pid_to_idx = {pid: i for i, pid in enumerate(pair_ids)}
        sums = np.zeros((len(pair_ids), len(retained)), dtype=np.float64)
        counts = np.zeros((len(pair_ids), len(retained)), dtype=np.int32)
        for r in sub[["pair_id", "_stratum", "I_f"]].itertuples(index=False, name=None):
            pid, st, ival = r[0], r[1], r[2]
            i = pid_to_idx[pid]
            j = st_to_idx[st]
            v = float(ival)
            if math.isfinite(v):
                sums[i, j] += v
                counts[i, j] += 1
        out[fam] = {"pair_ids": pair_ids, "sums": sums, "counts": counts}
    return out


def observed_weighted_means(mats: dict[str, dict[str, Any]], weights: np.ndarray) -> dict[str, float]:
    means = {}
    for fam, m in mats.items():
        sums = m["sums"].sum(axis=0)
        counts = m["counts"].sum(axis=0)
        if np.any(counts <= 0):
            raise RuntimeError(f"Observed retained stratum missing for {fam}")
        means[fam] = float(np.dot(weights, sums / counts))
    return means


def run_boot(df: pd.DataFrame, keys: list[str], min_per_family: int, target_distribution: str, n_boot: int, seed: int) -> dict[str, Any]:
    df = df.copy()
    df["_stratum"] = list(map(tuple, df[keys].astype(str).to_numpy()))
    counts = df.groupby(["family", "_stratum"], dropna=False).size().unstack(0).fillna(0)
    retain_mask = (counts[FAMILIES] >= min_per_family).all(axis=1)
    retained = counts.index[retain_mask].tolist()
    if not retained:
        return {"status": "NO_COMMON_SUPPORT", "keys": keys, "min_per_family": min_per_family}
    dfr = df[df["_stratum"].isin(retained)].copy()
    if target_distribution in FAMILIES:
        w_counts = dfr[dfr.family == target_distribution]["_stratum"].value_counts().to_dict()
    else:
        w_counts = dfr["_stratum"].value_counts().to_dict()
    weights = np.asarray([w_counts.get(st, 0) for st in retained], dtype=np.float64)
    weights = weights / weights.sum()

    mats = family_matrices(dfr, retained, keys)
    observed = observed_weighted_means(mats, weights)
    observed_contrasts = {
        "compact_minus_prefix": observed[COMPACT] - observed[PREFIX],
        "compact_minus_onegap": observed[COMPACT] - observed[ONEGAP],
        "compact_minus_mean_extracts": observed[COMPACT] - 0.5 * (observed[PREFIX] + observed[ONEGAP]),
        "onegap_minus_prefix": observed[ONEGAP] - observed[PREFIX],
    }

    rng = np.random.default_rng(seed)
    boot_means = {fam: [] for fam in FAMILIES}
    boot_contrasts = {k: [] for k in observed_contrasts}
    skipped = 0
    for _ in range(n_boot):
        means = {}
        ok = True
        for fam in FAMILIES:
            sums = mats[fam]["sums"]
            cnts = mats[fam]["counts"]
            idx = rng.integers(0, sums.shape[0], size=sums.shape[0])
            bs = sums[idx].sum(axis=0)
            bc = cnts[idx].sum(axis=0)
            if np.any(bc <= 0):
                ok = False
                break
            means[fam] = float(np.dot(weights, bs / bc))
        if not ok:
            skipped += 1
            continue
        cons = {
            "compact_minus_prefix": means[COMPACT] - means[PREFIX],
            "compact_minus_onegap": means[COMPACT] - means[ONEGAP],
            "compact_minus_mean_extracts": means[COMPACT] - 0.5 * (means[PREFIX] + means[ONEGAP]),
            "onegap_minus_prefix": means[ONEGAP] - means[PREFIX],
        }
        for fam in FAMILIES:
            boot_means[fam].append(means[fam])
        for k, v in cons.items():
            boot_contrasts[k].append(v)

    return {
        "status": "OK",
        "keys": keys,
        "min_per_family": min_per_family,
        "target_distribution": target_distribution,
        "n_retained_strata": int(len(retained)),
        "n_possible_strata": int(counts.shape[0]),
        "n_boot_requested": int(n_boot),
        "n_boot_ok": int(n_boot - skipped),
        "n_boot_skipped_missing_stratum": int(skipped),
        "observed_weighted_means": observed,
        "observed_contrasts": observed_contrasts,
        "bootstrap_means": {fam: quant_summary(np.asarray(vals)) for fam, vals in boot_means.items() if vals},
        "bootstrap_contrasts": {k: quant_summary(np.asarray(vals)) for k, vals in boot_contrasts.items() if vals},
        "retained_fraction_by_family": {fam: float((dfr.family == fam).sum() / max(1, (df.family == fam).sum())) for fam in FAMILIES},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(DEFAULT_RECORDS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=218117)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.records)
    df = add_bins(df)
    df = df[df.family.isin(FAMILIES)].copy()
    copied = df[df.source_match_bin.astype(str) != "0"].copy()

    main_keys = ["copy3", "match3", "lex3", "source_decile_bin", "gap_bin", "ord_pos_bin"]
    coarser_keys = ["copy3", "match3", "lex3", "source_decile_bin"]
    results = {
        "status": "FAST_FIXED_SUPPORT_BOOTSTRAP",
        "records_path": str(args.records),
        "n_boot": args.n_boot,
        "note": "Fixed observed common-support strata and weights; pair_id clusters resampled within each family. Restricted to copied targets because source-absent compact targets lack support in extractive families.",
        "copied_main_pooled_common": run_boot(copied, main_keys, 5, "pooled_common", args.n_boot, args.seed),
        "copied_main_compact_distribution": run_boot(copied, main_keys, 5, COMPACT, args.n_boot, args.seed + 1),
        "copied_coarser_pooled_common": run_boot(copied, coarser_keys, 10, "pooled_common", args.n_boot, args.seed + 2),
    }

    json_path = out_dir / "adjusted_source_use_fast_bootstrap.json"
    md_path = out_dir / "adjusted_source_use_fast_bootstrap.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = ["# research fast fixed-support bootstrap", "", results["note"], ""]
    for name in ["copied_main_pooled_common", "copied_main_compact_distribution", "copied_coarser_pooled_common"]:
        r = results[name]
        lines.append(f"## {name}")
        if r.get("status") != "OK":
            lines.append(f"Status: {r.get('status')}")
            continue
        lines.append(f"Retained strata {r['n_retained_strata']}/{r['n_possible_strata']}; boot ok {r['n_boot_ok']}/{r['n_boot_requested']}")
        lines.append("Observed contrasts: " + ", ".join(f"{k}={v:+.3f}" for k, v in r["observed_contrasts"].items()))
        for k in ["compact_minus_prefix", "compact_minus_onegap", "compact_minus_mean_extracts"]:
            s = r["bootstrap_contrasts"][k]
            lines.append(f"- {k}: median {s['q500']:+.3f}, 95% [{s['q025']:+.3f}, {s['q975']:+.3f}], p>0 {s['p_gt_0']:.3f}")
        lines.append("")
    lines.append(f"JSON: `{json_path}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": results["status"],
        "json": str(json_path),
        "md": str(md_path),
        "main_compact_minus_mean_extracts": results["copied_main_pooled_common"].get("bootstrap_contrasts", {}).get("compact_minus_mean_extracts"),
        "compact_distribution_compact_minus_mean_extracts": results["copied_main_compact_distribution"].get("bootstrap_contrasts", {}).get("compact_minus_mean_extracts"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
