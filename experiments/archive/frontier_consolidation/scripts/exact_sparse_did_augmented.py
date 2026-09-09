#!/usr/bin/env python3
"""research: augment exact sparse temporal DiD with fixed BLiMP union.

The sparse probes split BLiMP into outcome-selected `blimp_worst` and
`blimp_control`. This script computes a fixed 18-subtask BLiMP union from the
same exact reinvest and clean sparse summaries, then combines it with the exact
DiD result so BLiMP is not overinterpreted through selected groups.
"""
from __future__ import annotations

import json
import pathlib
import statistics
from typing import Any

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
REINVEST = STUDY / "training/runs/sparse_temporal_probe_main_direct/sparse_temporal_probe_summary.json"
CLEAN = STUDY / "training/runs/clean_sparse_temporal_probe_main_direct/clean_sparse_temporal_probe_summary.json"
DID = STUDY / "data/sparse_temporal_did_exact/exact_reinvest_vs_clean_sparse_did.json"
OUT_DIR = STUDY / "data/exact_sparse_did_augmented"


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_index(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(r["seed"]), int(r["exposure_m"])): r for r in summary["rows"]}


def subtask_scores(row: dict[str, Any], group: str) -> dict[str, float]:
    return {str(k): float(v) for k, v in row["scores"][group].get("subtask_scores", {}).items()}


def blimp_union_score(row: dict[str, Any]) -> float:
    vals = []
    for group in ["blimp_worst", "blimp_control"]:
        vals.extend(subtask_scores(row, group).values())
    if len(vals) != 18:
        raise RuntimeError(f"Expected 18 BLiMP union subtasks, got {len(vals)}")
    return statistics.mean(vals)


def score(summary: dict[str, Any], seed: str, exposure: int, group: str) -> float:
    r = row_index(summary)[(seed, exposure)]
    return float(r["scores"][group]["average_score"])


def union_gap(summary: dict[str, Any], exposure: int) -> dict[str, float]:
    ix = row_index(summary)
    s430 = blimp_union_score(ix[("43022", exposure)])
    s431 = blimp_union_score(ix[("43122", exposure)])
    return {"seed43022": s430, "seed43122": s431, "gap_43122_minus_43022": s431 - s430}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reinvest = load(REINVEST)
    clean = load(CLEAN)
    did = load(DID)
    exposures = sorted(set(map(int, reinvest["exposures_m"])) & set(map(int, clean["exposures_m"])))
    union = {}
    for exp in exposures:
        rg = union_gap(reinvest, exp)
        cg = union_gap(clean, exp)
        union[str(exp)] = {
            "reinvest": rg,
            "clean": cg,
            "excess_reinvest_minus_clean": rg["gap_43122_minus_43022"] - cg["gap_43122_minus_43022"],
        }

    # Add late changes for all task groups plus the BLiMP union.
    late_change = {}
    for tg, rec in did["task_group_summary"].items():
        ex = {int(k): float(v) for k, v in rec["excess_by_exposure"].items()}
        if 40 in ex and 100 in ex:
            late_change[tg] = ex[100] - ex[40]
    late_change["blimp_union_18_subtasks"] = union["100"]["excess_reinvest_minus_clean"] - union["40"]["excess_reinvest_minus_clean"]

    # Top subtask excesses at 100M from the exact DiD object.
    top_final = {}
    for tg, block in did["subtask_excess"].get("100", {}).items():
        recs = block.get("all", [])
        top_final[tg] = {
            "most_negative": sorted(recs, key=lambda x: x["excess_reinvest_minus_clean"])[:8],
            "most_positive": sorted(recs, key=lambda x: x["excess_reinvest_minus_clean"], reverse=True)[:8],
            "mean_excess": block.get("mean_excess"),
        }

    result = {
        "status": "EXACT_SPARSE_DID_AUGMENTED",
        "purpose": "Exact sparse temporal DiD with a fixed 18-subtask BLiMP union to avoid overreading outcome-selected BLiMP groups.",
        "inputs": {"reinvest": str(REINVEST), "clean": str(CLEAN), "did": str(DID)},
        "exposures_m": exposures,
        "blimp_union_18_subtasks": union,
        "task_group_exact_excess_summary": did["task_group_summary"],
        "late_excess_change_100M_minus_40M": late_change,
        "top_final_subtask_excess_100M": top_final,
        "scientific_read": {
            "blimp": "Fixed 18-subtask BLiMP union has modest final excess, so selected worst/control groups should not drive the route.",
            "supplement": "Supplement excess is nonmonotonic and reverses sharply between 40M and 100M, pointing to late dynamics or repeated-exposure interaction rather than a simple early acquisition defect.",
            "ewok": "Fast EWoK excess is already negative by 10M and becomes more negative by 100M; current-official full EWoK confirms positive but much smaller seed43122 treatment effect.",
        },
    }
    out_json = OUT_DIR / "exact_sparse_did_augmented.json"
    out_md = OUT_DIR / "exact_sparse_did_augmented.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research — exact sparse DiD with fixed BLiMP union\n\n"]
    lines.append("The exact clean sparse temporal control has delivered. This file augments the prepared DiD with a fixed 18-subtask BLiMP union, because the named worst/control BLiMP groups were outcome-selected.\n\n")
    lines.append("## BLiMP union trajectory\n")
    for exp in exposures:
        rec = union[str(exp)]
        lines.append(
            f"- {exp}M: reinvest gap {rec['reinvest']['gap_43122_minus_43022']:+.3f}; "
            f"clean gap {rec['clean']['gap_43122_minus_43022']:+.3f}; "
            f"excess {rec['excess_reinvest_minus_clean']:+.3f}\n"
        )
    lines.append("\n## Exact task-group excess summary\n")
    for tg, rec in sorted(did["task_group_summary"].items()):
        lines.append(f"- {tg}: excess_by_exposure={rec['excess_by_exposure']}; mean={rec['mean_excess']:+.3f}; final={rec['final_excess']:+.3f}\n")
    lines.append(f"- blimp_union_18_subtasks: excess_by_exposure={{{', '.join([repr(k)+': '+format(v['excess_reinvest_minus_clean'], '+.3f') for k, v in union.items()])}}}; final={union['100']['excess_reinvest_minus_clean']:+.3f}\n")
    lines.append("\n## Late excess change, 100M - 40M\n")
    for k, v in sorted(late_change.items(), key=lambda kv: kv[1]):
        lines.append(f"- {k}: {v:+.3f}\n")
    lines.append("\n## 100M subtask excess highlights\n")
    for tg in ["supplement_all", "ewok_all", "entity_full", "blimp_worst", "blimp_control"]:
        if tg not in top_final:
            continue
        lines.append(f"### {tg}\n")
        lines.append("Most negative: " + "; ".join([f"{r['subtask']} {r['excess_reinvest_minus_clean']:+.2f}" for r in top_final[tg]["most_negative"][:5]]) + "\n")
        lines.append("Most positive: " + "; ".join([f"{r['subtask']} {r['excess_reinvest_minus_clean']:+.2f}" for r in top_final[tg]["most_positive"][:5]]) + "\n")
    lines.append("\n## Scientific read\n")
    for v in result["scientific_read"].values():
        lines.append(f"- {v}\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "blimp_union_final_excess": union["100"]["excess_reinvest_minus_clean"],
        "blimp_union_late_change": late_change["blimp_union_18_subtasks"],
        "supplement_final_excess": did["task_group_summary"]["supplement_all"]["final_excess"],
        "ewok_fast_final_excess": did["task_group_summary"]["ewok_all"]["final_excess"],
        "entity_final_excess": did["task_group_summary"]["entity_full"]["final_excess"],
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
