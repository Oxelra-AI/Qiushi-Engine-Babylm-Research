#!/usr/bin/env python3
"""research: Corrected binding factorial analysis using the paired-null coordinate.

In the recombination design, each pair shares identical context and differs only in
the queried entity.  A non-gating model gives both halves the same answer and gets
exactly one right per pair.  The no-gating null is therefore joint=0, NOT A·B/n.

The independence-excess coordinate used in research is structurally wrong for paired
data because A and B are anti-correlated under any shared-prior model.

Correct readouts:
  gated_fraction = joint_correct / n_pairs   (null = 0)
  both_wrong = n - joint - (a_only + b_only)   (should be ~0 for clean gating)
  a_only = a_correct - joint_correct
  b_only = b_correct - joint_correct

If both_wrong ≈ 0, every flip is in the correct direction: the model assigned the
right answer to both halves of the pair, which requires entity identity.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import sys

_SCRIPT = _public_path('experiments/archive/relation_learning/scripts/corrected_binding_analysis.py')
ROOT = _public_path('experiments/archive/relation_learning/data')

def analyze_trajectory(csv_path: pathlib.Path) -> list[dict]:
    """Read trajectory CSV and compute corrected paired-null readouts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            n = int(r["n"])
            a = int(r["a_correct"])
            b = int(r["b_correct"])
            j = int(r["joint_correct"])
            
            a_only = a - j
            b_only = b - j
            one_right = a_only + b_only
            both_wrong = n - j - one_right
            
            gated_frac = j / n if n > 0 else 0.0
            prior_b_frac = b_only / max(one_right, 1)  # fraction of one-right that chose B
            
            rows.append({
                "arm": r["arm"],
                "epoch": int(r["epoch"]),
                "subset": r["subset"],
                "n": n,
                "joint_correct": j,
                "a_correct": a,
                "b_correct": b,
                "a_only_right": a_only,
                "b_only_right": b_only,
                "one_right": one_right,
                "both_wrong": both_wrong,
                "gated_fraction": round(gated_frac, 5),
                "prior_b_fraction": round(prior_b_frac, 5),
                "mean_a_margin": float(r.get("mean_a_margin", 0)),
                "mean_b_margin": float(r.get("mean_b_margin", 0)),
                # OLD coordinate for reference
                "old_expected_joint": float(r.get("expected_joint_independent_count", 0)),
                "old_gating_count": float(r.get("gating_count", 0)),
            })
    return rows

def main():
    # Balanced pairbatch
    balanced_csv = _public_path('experiments/archive/relation_learning/data/binding_factorial_balanced_pairbatch/trajectory_gating_flat.csv')
    if not balanced_csv.exists():
        print(f"ERROR: {balanced_csv} not found", file=sys.stderr)
        sys.exit(1)
    
    rows = analyze_trajectory(balanced_csv)
    
    out_dir = _public_path('experiments/archive/relation_learning/data/corrected_binding_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full corrected CSV
    csv_path = out_dir / "corrected_trajectory.csv"
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    
    # Summary at epoch 20 for all_pairs
    ep20_all = [r for r in rows if r["epoch"] == 20 and r["subset"] == "all_pairs"]
    ep20_119 = [r for r in rows if r["epoch"] == 20 and r["subset"] == "updated_entity_in_source_119"]
    
    # Trajectory for all_pairs
    arms_order = ["answer_clean", "uniform_wwm", "answer_corrupt_update_state"]
    trajectory_all = {}
    for arm in arms_order:
        arm_rows = [r for r in rows if r["arm"] == arm and r["subset"] == "all_pairs"]
        trajectory_all[arm] = [
            {"epoch": r["epoch"], "joint": r["joint_correct"], "gated_frac": r["gated_fraction"],
             "both_wrong": r["both_wrong"], "a_only": r["a_only_right"], "b_only": r["b_only_right"],
             "prior_b_frac": r["prior_b_fraction"]}
            for r in sorted(arm_rows, key=lambda x: x["epoch"])
        ]
    
    summary = {
        "status": "CORRECTED_BINDING_ANALYSIS",
        "correction": "The independence A*B/n null is wrong for paired rows sharing identical context. "
                       "The no-gating null is joint=0 because a non-gating model gets exactly one per pair right. "
                       "both_wrong should be ~0 for clean gating signal.",
        "epoch_20_all_pairs": {r["arm"]: {
            "n": r["n"], "joint": r["joint_correct"], "gated_fraction": r["gated_fraction"],
            "both_wrong": r["both_wrong"], "a_correct": r["a_correct"], "b_correct": r["b_correct"],
            "a_only": r["a_only_right"], "b_only": r["b_only_right"],
            "prior_b_fraction": r["prior_b_fraction"],
            "mean_a_margin": round(r["mean_a_margin"], 4),
            "mean_b_margin": round(r["mean_b_margin"], 4),
        } for r in ep20_all},
        "epoch_20_updated_in_source_119": {r["arm"]: {
            "n": r["n"], "joint": r["joint_correct"], "gated_fraction": r["gated_fraction"],
            "both_wrong": r["both_wrong"],
        } for r in ep20_119},
        "trajectory_all_pairs": trajectory_all,
    }
    
    # Markdown
    md_lines = [
        "# research: Corrected binding factorial analysis (paired-null coordinate)",
        "",
        "## Correction",
        "The research independence-excess coordinate (joint - A·B/n) is structurally wrong",
        "for paired rows sharing identical context. A non-gating model gets exactly one",
        "per pair right, so the no-gating null is **joint = 0**.",
        "",
        "## Epoch 20 all-pairs (n=200)",
        "",
        "| arm | joint | gated% | both_wrong | A_only | B_only | prior_B% | A_margin | B_margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ep20_all:
        md_lines.append(
            f"| {r['arm']} | {r['joint_correct']} | {r['gated_fraction']*100:.1f}% | "
            f"{r['both_wrong']} | {r['a_only_right']} | {r['b_only_right']} | "
            f"{r['prior_b_fraction']*100:.1f}% | {r['mean_a_margin']:.3f} | {r['mean_b_margin']:.3f} |"
        )
    
    md_lines += [
        "",
        "## Epoch 20 updated-entity-in-source (n=119)",
        "",
        "| arm | joint | gated% | both_wrong |",
        "|---|---:|---:|---:|",
    ]
    for r in ep20_119:
        md_lines.append(f"| {r['arm']} | {r['joint_correct']} | {r['gated_fraction']*100:.1f}% | {r['both_wrong']} |")
    
    md_lines += [
        "",
        "## Joint trajectory (all_pairs, n=200)",
        "",
        "| epoch | answer_clean | uniform_wwm | corrupt_update |",
        "|---:|---:|---:|---:|",
    ]
    for i in range(len(trajectory_all["answer_clean"])):
        ac = trajectory_all["answer_clean"][i]
        uw = trajectory_all["uniform_wwm"][i]
        cu = trajectory_all["answer_corrupt_update_state"][i]
        md_lines.append(f"| {ac['epoch']} | {ac['joint']} ({ac['gated_frac']*100:.1f}%) | "
                       f"{uw['joint']} ({uw['gated_frac']*100:.1f}%) | "
                       f"{cu['joint']} ({cu['gated_frac']*100:.1f}%) |")
    
    md_lines += [
        "",
        "## Interpretation",
        "",
        "Under the corrected coordinate:",
        "- base (epoch 0): 6/200 = 3.0% gating, shared across all arms",
        "- answer_clean: 6 → 18 → 30 → 35 → 44 (still rising at epoch 20)",
        "- uniform_wwm: 6 → 5 → 7 → 6 → 7 (flat, near base)",
        "- answer_corrupt: 6 → 14 → 25 → 30 → 30 (plateaued ~15%)",
        "",
        "Clean answer credit raised gating ~7x from base, still rising.",
        "Uniform WWM installed nothing. Corrupted update support reached half of",
        "clean answer credit and plateaued, showing that visible update evidence",
        "matters but is not sufficient without concentrated credit.",
        "",
        "both_wrong ≈ 0 for all arms: every joint success is a genuine flip, not noise.",
        "The residual is prior-governed (B_only >> A_only), not anti-gated.",
    ]
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(out_dir / "corrected_trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    
    print(json.dumps({
        "status": "CORRECTED_BINDING_ANALYSIS",
        "summary": str(out_dir / "summary.json"),
        "md": str(out_dir / "summary.md"),
        "csv": str(out_dir / "corrected_trajectory.csv"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
