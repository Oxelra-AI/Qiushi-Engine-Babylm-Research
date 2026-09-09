#!/usr/bin/env python3
"""Fit-filtered factorial summary for research independent-orientation runs."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

QUERIES = ["event_role", "focal_state", "untouched_state"]
EVALS = [
    "atp_trainTrain_trainHyp", "atp_trainTrain_heldHyp",
    "atp_heldHeld_trainHyp", "atp_heldHeld_heldHyp",
    "atp_dirDir_trainHyp", "atp_dirDir_heldHyp",
    "atp_trainEvent_dirState_trainHyp", "atp_dirEvent_trainState_trainHyp",
    "atp_trainEvent_nonState_trainHyp", "atp_nonEvent_trainState_trainHyp",
]
NEEDED = ["Etrue_Rtrue", "Eflip_Rtrue", "Etrue_Rflip", "Eflip_Rflip"]


def metric(seed_result, ev, q):
    return seed_result["evals"][ev]["contrastive"]["by_query"].get(q, float("nan"))


def stat(xs):
    vals = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return {"mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
            "n": len(vals), "vals": vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out_dir", default="experiments/archive/representation_and_objectives/data/fitrepair3_factorial_effects")
    ap.add_argument("--fit_threshold", type=float, default=0.99)
    args = ap.parse_args()
    d = json.loads(Path(args.summary).read_text("utf-8"))
    by_seed = {}
    for r in d["per_seed"]:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r
    out = {"source": args.summary, "fit_threshold": args.fit_threshold,
           "included_seeds_by_eval_query": {}, "effects": {}}
    lines = [f"# Fit-filtered research factorial effects for {args.summary}\n\n"]
    lines.append(f"A seed is included only when all four orientation arms have `best_train_acc >= {args.fit_threshold}`. This removes optimization non-measurements such as the underfit joint-flip seed.\n\n")
    for ev in EVALS:
        out["effects"][ev] = {}
        lines.append(f"## {ev}\n\n")
        lines.append("| query | included seeds | Delta_E | Delta_R | interaction | cells |\n")
        lines.append("|---|---:|---:|---:|---:|---|\n")
        for q in QUERIES:
            de, dr, ix, cells = [], [], [], []
            for seed, arms in sorted(by_seed.items()):
                if not all(a in arms and float(arms[a].get("best_train_acc", -1)) >= args.fit_threshold for a in NEEDED):
                    continue
                tt, ft, tf, ff = [metric(arms[a], ev, q) for a in NEEDED]
                de.append(((tt - ft) + (tf - ff)) / 2.0)
                dr.append(((tt - tf) + (ft - ff)) / 2.0)
                ix.append(tt - ft - tf + ff)
                cells.append({"seed": seed, "TT": tt, "FT": ft, "TF": tf, "FF": ff})
            out["effects"][ev][q] = {"delta_event": stat(de), "delta_rank": stat(dr),
                                       "interaction": stat(ix), "cells": cells}
            lines.append(
                f"| {q} | {len(de)} | {stat(de)['mean']:.3f}±{stat(de)['std']:.3f} | "
                f"{stat(dr)['mean']:.3f}±{stat(dr)['std']:.3f} | "
                f"{stat(ix)['mean']:.3f}±{stat(ix)['std']:.3f} | {cells} |\n")
        lines.append("\n")
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    (od / "fit_filtered_factorial_effects.json").write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    (od / "fit_filtered_factorial_effects.md").write_text("".join(lines), "utf-8")
    print(json.dumps({"status": "fit_filtered_factorial_done",
                      "md": str(od / "fit_filtered_factorial_effects.md"),
                      "json": str(od / "fit_filtered_factorial_effects.json")}))

if __name__ == "__main__":
    main()
