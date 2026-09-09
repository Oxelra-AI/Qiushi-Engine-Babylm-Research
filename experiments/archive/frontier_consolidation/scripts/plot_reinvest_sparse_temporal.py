#!/usr/bin/env python3
"""research: plot compact_view_reinvest sparse temporal seed trajectories.

This visual evidence helps distinguish task-specific early split, late divergence, and the EWoK
fast-vs-official coordinate difference before the clean sparse DiD control arrives.
"""
from __future__ import annotations

import csv
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
ANALYSIS = STUDY / "data/reinvest_sparse_temporal_analysis/reinvest_sparse_temporal_analysis.json"
OUT_FIG = STUDY / "figures/reinvest_sparse_temporal_seed_gaps.png"
OUT_CSV = STUDY / "data/reinvest_sparse_temporal_analysis/reinvest_sparse_temporal_seed_gaps.csv"

GROUP_LABELS = {
    "blimp_worst": "BLiMP selected loss slices",
    "blimp_control": "BLiMP selected control slices",
    "supplement_all": "Supplement fast",
    "ewok_all": "EWoK fast",
    "entity_full": "Entity full",
}


def main() -> None:
    d = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    exposures = [int(x) for x in d["exposures_m"]]
    groups = list(d["group_trajectories"].keys())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_group", "exposure_m", "seed43022", "seed43122", "gap_43122_minus_43022"])
        for g in groups:
            rec = d["group_trajectories"][g]
            for exp in exposures:
                w.writerow([
                    g, exp,
                    rec["seed43022_scores_by_exposure"][str(exp)],
                    rec["seed43122_scores_by_exposure"][str(exp)],
                    rec["delta_by_exposure"][str(exp)],
                ])

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.0), sharex=True)
    ax = axes[0]
    for g in groups:
        rec = d["group_trajectories"][g]
        y0 = [rec["seed43022_scores_by_exposure"][str(exp)] for exp in exposures]
        y1 = [rec["seed43122_scores_by_exposure"][str(exp)] for exp in exposures]
        label = GROUP_LABELS.get(g, g)
        ax.plot(exposures, y0, marker="o", linestyle="-", alpha=0.85, label=f"43022 {label}")
        ax.plot(exposures, y1, marker="x", linestyle="--", alpha=0.85, label=f"43122 {label}")
    ax.set_ylabel("Score")
    ax.set_title("compact_view_reinvest sparse task-slice scores across checkpoints")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=7.5, frameon=False)

    ax2 = axes[1]
    for g in groups:
        rec = d["group_trajectories"][g]
        y = [rec["delta_by_exposure"][str(exp)] for exp in exposures]
        ax2.plot(exposures, y, marker="o", label=GROUP_LABELS.get(g, g))
    ax2.axhline(0, color="black", linewidth=0.8)
    # show full official EWoK correction at 100M
    ew = d["ewok_coordinate_shift"]["official_full_7618_100M"]["delta_43122_minus_43022"]
    ax2.scatter([100], [ew], s=85, marker="*", color="tab:red", zorder=5, label="EWoK full official 100M")
    ax2.annotate(f"full official EWoK {ew:+.2f}", xy=(100, ew), xytext=(68, ew + 2.0),
                 arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    ax2.set_xlabel("Training exposure (M words)")
    ax2.set_ylabel("Seed gap: 43122 - 43022")
    ax2.set_title("Seed gap trajectories: late BLiMP/Supplement divergence, EWoK fast-vs-official contrast")
    ax2.grid(True, alpha=0.25)
    ax2.legend(ncol=2, fontsize=8, frameon=False)
    ax2.set_xticks(exposures)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180)
    print(json.dumps({"status": "REINVEST_SPARSE_TEMPORAL_FIGURE", "figure": str(OUT_FIG), "csv": str(OUT_CSV)}, indent=2))

if __name__ == "__main__":
    main()
