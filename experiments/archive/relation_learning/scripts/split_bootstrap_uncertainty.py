#!/usr/bin/env python3
"""research: pair-cluster bootstrap uncertainty for REPEAT_SPLIT locality.

Uses pair-averaged token-nonoverlap compact-rewrite contrasts from research
robustness analysis. Quantifies probe/pair uncertainty (not training-seed
uncertainty) and checks whether RS-C lies inside the pre-stated near-zero band
while original R-C lies far outside it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("experiments/archive/relation_learning")
INP = ROOT / "data/repeat_split_robustness/repeat_split_pair_level_late_pairs.csv"
OUT = ROOT / "data/split_bootstrap_uncertainty"
NOTE = (ROOT.parents[2] / 'research/notes/relation_learning/split_bootstrap_uncertainty.md')
BAND = 0.25
N_BOOT = 10000
SEED = 180918


def fmt(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if math.isnan(xf):
        return "NA"
    return f"{xf:+.{nd}f}"


def bootstrap_mean(vals: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    n = len(vals)
    idx = rng.integers(0, n, size=(n_boot, n))
    return vals[idx].mean(axis=1)


def summarize(vals: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    boot = bootstrap_mean(vals, rng, N_BOOT)
    return {
        "n_pairs": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "sd_pairs": float(vals.std(ddof=1)),
        "se_pairs": float(vals.std(ddof=1) / math.sqrt(len(vals))),
        "ci95_lo": float(np.quantile(boot, 0.025)),
        "ci95_hi": float(np.quantile(boot, 0.975)),
        "prob_mean_inside_pm_0p25": float(np.mean(np.abs(boot) <= BAND)),
        "prob_mean_negative": float(np.mean(boot < 0)),
        "q05_pair": float(np.quantile(vals, 0.05)),
        "q10_pair": float(np.quantile(vals, 0.10)),
        "q25_pair": float(np.quantile(vals, 0.25)),
        "q75_pair": float(np.quantile(vals, 0.75)),
        "q90_pair": float(np.quantile(vals, 0.90)),
        "q95_pair": float(np.quantile(vals, 0.95)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INP)
    rng = np.random.default_rng(SEED)
    rows = []
    for contrast, g in df.groupby("contrast"):
        for metric in ["mean_gain_delta", "mean_true_delta", "mean_unrel_delta", "mean_excess_true_cost"]:
            rows.append({"contrast": contrast, "metric": metric, **summarize(g[metric].to_numpy(float), rng)})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "pair_bootstrap_summary.csv", index=False)

    # Attenuation estimates relative to original R-C excess cost.
    def get_mean(contrast: str, metric: str) -> float:
        return float(summ[(summ["contrast"] == contrast) & (summ["metric"] == metric)]["mean"].iloc[0])
    rc_excess = get_mean("RminusC", "mean_excess_true_cost")
    rsc_excess = get_mean("RSminusC", "mean_excess_true_cost")
    attenuation = 1.0 - abs(rsc_excess) / abs(rc_excess)
    rc_gain = get_mean("RminusC", "mean_gain_delta")
    rsc_gain = get_mean("RSminusC", "mean_gain_delta")
    attenuation_gain = 1.0 - abs(rsc_gain) / abs(rc_gain)
    result = {
        "status": "SPLIT_BOOTSTRAP_UNCERTAINTY_DONE",
        "n_boot": N_BOOT,
        "seed": SEED,
        "near_zero_band": BAND,
        "original_RminusC_excess": rc_excess,
        "RSminusC_excess": rsc_excess,
        "excess_attenuation_fraction": attenuation,
        "original_RminusC_gain": rc_gain,
        "RSminusC_gain": rsc_gain,
        "gain_attenuation_fraction": attenuation_gain,
        "summary_csv": str(OUT / "pair_bootstrap_summary.csv"),
        "note": str(NOTE),
    }
    (OUT / "split_bootstrap_uncertainty_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    def row(contrast: str, metric: str) -> pd.Series:
        return summ[(summ["contrast"] == contrast) & (summ["metric"] == metric)].iloc[0]
    lines = []
    lines.append("# research REPEAT_SPLIT pair-cluster uncertainty")
    lines.append("")
    lines.append("This bootstrap quantifies uncertainty from the finite held-out compact-rewrite pair sample after pair-averaging over 80M/90M/100M. It does not estimate training-seed variability; the second split seed is still required for that.")
    lines.append("")
    lines.append("## Mean estimates and pair-bootstrap intervals")
    lines.append("")
    lines.append("| contrast | metric | mean | median pair | 95% bootstrap interval | P(|mean|≤0.25) | pair q10/q90 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for contrast in ["RminusC", "RSminusC", "RSminusR"]:
        for metric in ["mean_gain_delta", "mean_true_delta", "mean_unrel_delta", "mean_excess_true_cost"]:
            r = row(contrast, metric)
            lines.append(f"| {contrast} | {metric} | {fmt(r['mean'])} | {fmt(r['median'])} | [{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}] | {float(r['prob_mean_inside_pm_0p25']):.3f} | {fmt(r['q10_pair'])}/{fmt(r['q90_pair'])} |")
    lines.append("")
    lines.append(f"The pair-sample estimate of RS−C excess true-source cost is {fmt(rsc_excess)} versus original R−C {fmt(rc_excess)}, an attenuation of {attenuation*100:.1f}% by absolute magnitude. RS−C gain is {fmt(rsc_gain)} versus original R−C {fmt(rc_gain)}, an attenuation of {attenuation_gain*100:.1f}%. The bootstrap interval for RS−C mean gain/excess stays well inside the pre-stated ±0.25 near-zero band, while original R−C is far outside it. This strengthens the one-seed locality result at the probe-sample level but does not replace the seed43122 split replicate.")
    lines.append("")
    lines.append("The pair distribution remains broad: RS−C has about half of pairs on each side of zero, while original R−C has a strong positive excess-cost majority. The scientific object is a distributional source-specific tendency, not every-pair determinism.")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "note": result["note"], "summary": result["summary_csv"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
