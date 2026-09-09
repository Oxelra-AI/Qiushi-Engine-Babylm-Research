#!/usr/bin/env python3
"""Plot concise expression-factorial evidence from research analysis JSON."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CELL_ORDER = ["orig", "query_only", "update_only", "both_changed"]
CELL_LABELS = ["original", "query\nchanged", "update\nchanged", "both\nchanged"]
METRICS = [
    ("neutral_both_full_source", "neutral\nsource"),
    ("retain_both_full_source", "distractor\nretains source"),
    ("self_update_both_full_new", "self-update\naccepts new"),
    ("recipient_only_flip_both_queries", "complete\noperation"),
]
FAIL_METRICS = [
    ("retain_cross_only_replacement_override", "source>wronɡ\nbut new wins"),
    ("self_update_semantic_inertia", "new>wrong\nbut source wins"),
    ("neutral_selector_available_but_operation_fails", "neutral ok\nbut op fails"),
]


def get_cell(info: Dict[str, Any], model: str, cell: str, metric: str) -> float:
    return float(info["factorial_effects"][model]["cell_means_over_pairs"][cell][metric])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.loads(pathlib.Path(args.analysis).read_text(encoding="utf-8"))
    model = "specialist_seed40040" if "specialist_seed40040" in data["factorial_effects"] else data["models"][-1]

    x = np.arange(len(CELL_ORDER))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    ax = axes[0]
    for j, (metric, label) in enumerate(METRICS):
        vals = [get_cell(data, model, c, metric) for c in CELL_ORDER]
        ax.bar(x + (j - 1.5) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_LABELS)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("pair-level success rate")
    ax.set_title("Available parts vs complete operation")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for j, (metric, label) in enumerate(FAIL_METRICS):
        vals = [get_cell(data, model, c, metric) for c in CELL_ORDER]
        ax.bar(x + (j - 1.0) * 0.22, vals, 0.22, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CELL_LABELS)
    ax.set_ylim(0, 1.04)
    ax.set_title("Dissociated expression failures")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("research pilot: expression change separates source retrieval from update use")
    fig.tight_layout(rect=[0, 0.0, 1, 0.94])
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(json.dumps({"status": "EXPRESSION_FACTORIAL_PLOT_DONE", "out": str(out), "model": model}, ensure_ascii=False))


if __name__ == "__main__":
    main()
