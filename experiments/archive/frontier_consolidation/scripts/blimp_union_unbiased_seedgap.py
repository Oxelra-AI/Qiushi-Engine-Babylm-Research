#!/usr/bin/env python3
"""research: remove the post-hoc selection bias in the reinvest sparse temporal
BLiMP seed-gap read.

The research sparse probe split BLiMP into two groups by seed43122's *100M* outcome:
- blimp_worst  = 10 files where seed43122 was much below seed43022 at 100M
- blimp_control = 8 files where seed43122 matched or exceeded seed43022 at 100M

Because that split conditions on the 100M seed43122 outcome, the two groups'
seed-gap trajectories are contaminated by selection (winner's curse /
regression to the mean): blimp_worst is biased negative and blimp_control biased
positive at 100M by construction. Their UNION (18 files) is an unselected BLiMP
panel whose seed gap is an unbiased estimate of the reinvest BLiMP seed spread
over these files, and it is the quantity that should be compared against the
clean-Qwen control.

This script recomputes:
  1. per-exposure item-weighted seed gap on the union of the 18 BLiMP files,
  2. per-exposure macro (unweighted-over-file) seed gap on the union,
  3. the early (1->10M) and late (40->100M) changes of the union gap,
  4. a comparison showing how much the reported blimp_worst / blimp_control
     numbers are inflated relative to the union.

CPU-only, reads the already-delivered reinvest sparse summary. No training.
"""
from __future__ import annotations

import json
import pathlib
import statistics

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
REINVEST = STUDY / "training/runs/sparse_temporal_probe_main_direct/sparse_temporal_probe_summary.json"
OUT_DIR = STUDY / "data/blimp_union_unbiased_seedgap"

# Each BLiMP file has 1000 items in the strict-small pipeline; the probe reports
# per-file accuracies in percent. For an item-weighted union mean we would need
# per-file item counts. All BLiMP files here are the standard 1000-item files, so
# item-weighted == macro over these 18 equally sized files. We compute macro and
# note this equivalence explicitly.
BLIMP_FILE_ITEMS = 1000


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_index(summary: dict) -> dict:
    out = {}
    for row in summary.get("rows", []):
        out[(str(row["seed"]), int(row["exposure_m"]))] = row
    return out


def subscores(row: dict, group: str) -> dict:
    rec = row.get("scores", {}).get(group)
    if rec is None:
        return {}
    return {str(k): float(v) for k, v in rec.get("subtask_scores", {}).items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load(REINVEST)
    ix = row_index(summary)
    exposures = sorted(set(int(e) for e in summary.get("exposures_m", [])))

    tg = summary.get("task_groups", {})
    worst_stems = list(tg.get("blimp_worst", {}).get("stems", []))
    control_stems = list(tg.get("blimp_control", {}).get("stems", []))
    union_stems = worst_stems + control_stems
    assert len(set(union_stems)) == len(union_stems), "overlap between worst and control stems"

    per_exposure = {}
    union_gap_by_exp = {}
    worst_gap_by_exp = {}
    control_gap_by_exp = {}
    for exp in exposures:
        a = ix.get(("43022", exp))
        b = ix.get(("43122", exp))
        if not a or not b:
            continue
        # gather subtask scores from both groups
        a_scores = {}
        b_scores = {}
        for grp in ("blimp_worst", "blimp_control"):
            a_scores.update(subscores(a, grp))
            b_scores.update(subscores(b, grp))
        common = [s for s in union_stems if s in a_scores and s in b_scores]
        union_gaps = [b_scores[s] - a_scores[s] for s in common]
        worst_gaps = [b_scores[s] - a_scores[s] for s in worst_stems if s in a_scores and s in b_scores]
        control_gaps = [b_scores[s] - a_scores[s] for s in control_stems if s in a_scores and s in b_scores]
        union_macro_gap = statistics.mean(union_gaps) if union_gaps else None
        # item-weighted == macro since all files are 1000 items
        per_exposure[str(exp)] = {
            "n_union_files": len(common),
            "seed43022_union_macro": statistics.mean([a_scores[s] for s in common]) if common else None,
            "seed43122_union_macro": statistics.mean([b_scores[s] for s in common]) if common else None,
            "union_macro_gap_43122_minus_43022": union_macro_gap,
            "worst_group_macro_gap": statistics.mean(worst_gaps) if worst_gaps else None,
            "control_group_macro_gap": statistics.mean(control_gaps) if control_gaps else None,
            "note": "item-weighted equals macro because all BLiMP files are 1000-item files",
        }
        union_gap_by_exp[exp] = union_macro_gap
        worst_gap_by_exp[exp] = statistics.mean(worst_gaps) if worst_gaps else None
        control_gap_by_exp[exp] = statistics.mean(control_gaps) if control_gaps else None

    def change(d, e0, e1):
        if e0 in d and e1 in d and d[e0] is not None and d[e1] is not None:
            return d[e1] - d[e0]
        return None

    trajectory = {
        "union_gap_by_exposure": {str(k): v for k, v in union_gap_by_exp.items()},
        "worst_gap_by_exposure": {str(k): v for k, v in worst_gap_by_exp.items()},
        "control_gap_by_exposure": {str(k): v for k, v in control_gap_by_exp.items()},
        "union_early_change_1_to_10M": change(union_gap_by_exp, 1, 10),
        "union_late_change_40_to_100M": change(union_gap_by_exp, 40, 100),
        "union_final_100M": union_gap_by_exp.get(100),
    }

    interpretation = {
        "selection_bias": (
            "blimp_worst and blimp_control were defined by seed43122's 100M outcome, "
            "so their individual 100M gaps (-14.05 and +10.06) are inflated by selection. "
            "The 18-file union gap is the unbiased reinvest BLiMP seed spread and is the "
            "correct quantity to compare against clean-Qwen."
        ),
        "union_vs_selected": (
            f"At 100M the union macro gap is {union_gap_by_exp.get(100)}, versus the selected "
            f"worst {worst_gap_by_exp.get(100)} and control {control_gap_by_exp.get(100)}."
        ),
    }

    result = {
        "status": "BLIMP_UNION_UNBIASED_SEEDGAP",
        "source_summary": str(REINVEST),
        "worst_stems": worst_stems,
        "control_stems": control_stems,
        "union_n_files": len(union_stems),
        "per_exposure": per_exposure,
        "trajectory": trajectory,
        "interpretation": interpretation,
    }
    out_json = OUT_DIR / "blimp_union_unbiased_seedgap.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = ["# research — unbiased BLiMP union seed-gap (reinvest sparse temporal)\n\n"]
    lines.append(interpretation["selection_bias"] + "\n\n")
    lines.append("## Union (18 BLiMP files) macro seed gap seed43122-minus-seed43022 by exposure\n")
    for exp in exposures:
        rec = per_exposure.get(str(exp))
        if rec:
            lines.append(
                f"- {exp}M: union_gap={rec['union_macro_gap_43122_minus_43022']:.3f} "
                f"(worst={rec['worst_group_macro_gap']:.3f}, control={rec['control_group_macro_gap']:.3f})\n"
            )
    lines.append("\n## Trajectory\n")
    lines.append(f"- union early change 1->10M: {trajectory['union_early_change_1_to_10M']}\n")
    lines.append(f"- union late change 40->100M: {trajectory['union_late_change_40_to_100M']}\n")
    lines.append(f"- union final 100M gap: {trajectory['union_final_100M']}\n\n")
    lines.append(interpretation["union_vs_selected"] + "\n\n")
    lines.append(f"Machine-readable output: `{out_json}`\n")
    (OUT_DIR / "blimp_union_unbiased_seedgap.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "trajectory": trajectory, "out_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
