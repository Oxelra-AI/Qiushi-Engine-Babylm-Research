#!/usr/bin/env python3
"""research: separate FW within-pair mechanism delta from absolute SOTA progress.

Reads the research cheap official-compatible evaluation summary and compares each
FW arm against the existing fully legal compact-view-reinvest trajectory. The
within-pair compact-vs-breadth delta answers how to spend the expanded FineWeb
companion budget; the absolute delta versus the legal reference answers whether
that family is moving toward the 41.8+ complete target.

This script does not train or evaluate models. Run it after
eval_fw_comparison.py has written fw_comparison_eval_summary.json.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
DEFAULT_SUMMARY = WORKSPACE / "data/fw_comparison_eval/fw_comparison_eval_summary.json"
OUT_ROOT = WORKSPACE / "data/fw_absolute_decision"

SCALAR_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

# Existing legal compact-view-reinvest reference. 70M/80M from research; 100M
# from research research same-pool-tokenizer full endpoint. This is not a paired
# control for research because tokenizer and FW source set differ, but it is the
# active performance trajectory that the new FW family must improve to matter.
LEGAL_REFERENCE = {
    "70M": {
        "BLiMP": 65.39,
        "Supplement": 59.31,
        "EWoK": 50.47,
        "Entity": 26.98,
        "COMPS": 51.82,
        "GlobalPIQA": 35.55,
        "Reading": 8.74,
        "cheap7": 42.6086,
        "source": "experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_70M.json",
    },
    "80M": {
        "BLiMP": 66.11,
        "Supplement": 60.66,
        "EWoK": 51.01,
        "Entity": 27.06,
        "COMPS": 51.93,
        "GlobalPIQA": 35.58,
        "Reading": 8.29,
        "cheap7": 42.9486,
        "source": "experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_80M.json",
    },
    "100M": {
        "BLiMP": 65.8707181799453,
        "Supplement": 61.16566092036889,
        "EWoK": 50.39323748109589,
        "Entity": 27.400833994026197,
        "COMPS": 52.00834536316919,
        "GlobalPIQA": 36.0631067961165,
        "Reading": 8.13816768550987,
        "cheap7": 43.0057,
        "Overall": 41.257770896404615,
        "SuperGLUE": 70.27986764740969,
        "AoA": 0.0,
        "source": "experiments/archive/frontier_consolidation/data/compliant_endpoint_results/compliant_endpoint_results_summary.json",
    },
}

LIVE_LEADER_OVERALL = 41.8
CURRENT_BEST_LEGAL_OVERALL = 41.257770896404615
CURRENT_OVERALL_GAP = LIVE_LEADER_OVERALL - CURRENT_BEST_LEGAL_OVERALL
CHEAP7_GAIN_EQUIVALENT_IF_ONLY_CHEAP_COLUMNS_MOVE = CURRENT_OVERALL_GAP * 9.0 / 7.0
PAIR_DELTA_FOR_MECHANISM = 0.3
ABS_DELTA_THAT_MATTERS = 0.3
ABS_DELTA_APPROACHING_FULL_GAP = 0.55


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_summary(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing research evaluation summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def result_for(summary: dict[str, Any], arm: str, ckpt: str) -> dict[str, Any] | None:
    return summary.get("results", {}).get(f"{arm}_{ckpt}")


def abs_comparison(row: dict[str, Any], ckpt: str) -> dict[str, Any]:
    ref = LEGAL_REFERENCE.get(ckpt)
    if row is None:
        return {"checkpoint": ckpt, "status": "MISSING_RESULT"}
    if row.get("status") != "OK":
        return {
            "checkpoint": ckpt,
            "arm": row.get("arm"),
            "status": "RESULT_NOT_OK",
            "result_status": row.get("status"),
        }
    if ref is None:
        return {
            "checkpoint": ckpt,
            "arm": row.get("arm"),
            "status": "NO_REFERENCE_FOR_CHECKPOINT",
            "scores": row.get("scores", {}),
        }
    scores = row["scores"]
    deltas = {col: round(float(scores[col]) - float(ref[col]), 4) for col in SCALAR_COLUMNS}
    cheap_delta = round(float(scores["cheap7"]) - float(ref["cheap7"]), 4)
    broad_positive = sum(1 for col in SCALAR_COLUMNS if deltas[col] > 0.0)
    broad_negative = sum(1 for col in SCALAR_COLUMNS if deltas[col] < 0.0)
    strong_losses = {col: d for col, d in deltas.items() if d <= -1.0}
    strong_gains = {col: d for col, d in deltas.items() if d >= 1.0}
    return {
        "checkpoint": ckpt,
        "arm": row.get("arm"),
        "status": "OK",
        "cheap7": scores["cheap7"],
        "legal_ref_cheap7": ref["cheap7"],
        "cheap7_delta_vs_legal_ref": cheap_delta,
        "column_deltas_vs_legal_ref": deltas,
        "positive_columns": broad_positive,
        "negative_columns": broad_negative,
        "strong_losses_le_minus_1": strong_losses,
        "strong_gains_ge_plus_1": strong_gains,
        "reference_source": ref.get("source"),
    }


def best_abs_at_checkpoint(abs_rows: list[dict[str, Any]], ckpt: str) -> dict[str, Any] | None:
    ok = [r for r in abs_rows if r.get("status") == "OK" and r.get("checkpoint") == ckpt]
    if not ok:
        return None
    return max(ok, key=lambda r: float(r["cheap7_delta_vs_legal_ref"]))


def within_pair(summary: dict[str, Any], ckpt: str) -> dict[str, Any] | None:
    comp = summary.get("comparisons", {}).get(ckpt)
    return comp


def synthesize(summary: dict[str, Any], abs_rows: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoints = summary.get("checkpoints", [])
    available_ckpts = set(checkpoints)
    best80 = best_abs_at_checkpoint(abs_rows, "80M")
    best100 = best_abs_at_checkpoint(abs_rows, "100M")
    best70 = best_abs_at_checkpoint(abs_rows, "70M")

    actions: list[str] = []
    interpretation: list[str] = []

    interpretation.append(
        "Within-pair compact-minus-breadth answers the FW companion-budget mechanism; "
        "absolute delta versus the existing legal trajectory answers whether the family is useful for the 41.8+ target."
    )
    interpretation.append(
        f"Current complete legal endpoint is Overall {CURRENT_BEST_LEGAL_OVERALL:.4f}; reaching {LIVE_LEADER_OVERALL:.1f} needs +{CURRENT_OVERALL_GAP:.4f} Overall. "
        f"If only the seven cheap columns moved, this corresponds to about +{CHEAP7_GAIN_EQUIVALENT_IF_ONLY_CHEAP_COLUMNS_MOVE:.4f} cheap7 at 100M."
    )

    if "70M" not in available_ckpts or "80M" not in available_ckpts:
        actions.append("Run research cheap evaluation at 70M and 80M for both arms before using any FW relative result.")
        return {"status": "NEED_70M_80M_EVAL", "interpretation": interpretation, "recommended_next_actions": actions}

    pair80 = within_pair(summary, "80M")
    if pair80 and pair80.get("status") == "OK":
        d = pair80.get("cheap7_delta_compact_minus_breadth")
        if d is not None:
            if d >= PAIR_DELTA_FOR_MECHANISM:
                interpretation.append(f"At 80M the within-pair result favors compact_view by {d:+.4f} cheap7.")
            elif d <= -PAIR_DELTA_FOR_MECHANISM:
                interpretation.append(f"At 80M the within-pair result favors source_breadth by {d:+.4f} cheap7.")
            else:
                interpretation.append(f"At 80M compact and breadth are close within the FW family ({d:+.4f} cheap7).")

    best_mature = best80 or best70
    if best_mature:
        interpretation.append(
            f"Best mature early checkpoint versus matched legal reference: {best_mature.get('arm')} {best_mature.get('checkpoint')} "
            f"has Δcheap7={best_mature['cheap7_delta_vs_legal_ref']:+.4f}, "
            f"positive columns {best_mature['positive_columns']}/7."
        )

    # If both 70M and 80M are flat or worse, source_repeat would only explain a
    # family that is not improving the active legal trajectory.
    if best80 and best70:
        if best80["cheap7_delta_vs_legal_ref"] <= 0.0 and best70["cheap7_delta_vs_legal_ref"] <= 0.0:
            interpretation.append(
                "Both 70M and 80M best FW arms are flat or below the existing legal trajectory; "
                "a source-repeat arm would explain a non-advancing family rather than move toward the target."
            )
            actions.append("Do not launch source_repeat from the relative FW result alone.")
            if "100M" not in available_ckpts:
                actions.append("Evaluate a 100M endpoint only if training has completed and the 80M per-column pattern shows broad recovery not visible in cheap7.")
            return {"status": "FW_FAMILY_NOT_ADVANCING_AT_70_80", "interpretation": interpretation, "recommended_next_actions": actions}

    # If the best 80M arm is meaningfully above reference, cheap 100M can test
    # whether the gain survives the late phase before any full official work.
    if best80 and best80["cheap7_delta_vs_legal_ref"] >= ABS_DELTA_THAT_MATTERS:
        if "100M" not in available_ckpts:
            actions.append(
                f"Evaluate 100M cheap7 for the promising arm(s) first; 80M best is {best80.get('arm')} with Δcheap7={best80['cheap7_delta_vs_legal_ref']:+.4f}."
            )
            if pair80 and pair80.get("status") == "OK" and abs(pair80.get("cheap7_delta_compact_minus_breadth", 0.0)) < PAIR_DELTA_FOR_MECHANISM:
                actions.append("Postpone source_repeat until 100M cheap7 shows the FW family can materially improve the legal trajectory.")
            return {"status": "PROMISING_80M_NEEDS_100M_CHEAP", "interpretation": interpretation, "recommended_next_actions": actions}

    # If 100M is present, use it to decide the next minimum evaluation.
    if best100:
        interpretation.append(
            f"Best 100M arm versus current legal endpoint: {best100.get('arm')} has Δcheap7={best100['cheap7_delta_vs_legal_ref']:+.4f}, "
            f"positive columns {best100['positive_columns']}/7."
        )
        if best100["cheap7_delta_vs_legal_ref"] >= CHEAP7_GAIN_EQUIVALENT_IF_ONLY_CHEAP_COLUMNS_MOVE:
            actions.append("Run full official-compatible evaluation for the 100M winner only, because cheap7 movement is large enough to plausibly close the complete gap if SuperGLUE/AoA do not fall.")
            return {"status": "WINNER_READY_FOR_FULL_EVAL", "interpretation": interpretation, "recommended_next_actions": actions}
        if best100["cheap7_delta_vs_legal_ref"] >= ABS_DELTA_APPROACHING_FULL_GAP:
            actions.append("Run the minimum missing official columns for the 100M winner (SuperGLUE and AoA, then full collation if favorable), because cheap7 is close enough that non-cheap columns could decide the complete endpoint.")
            return {"status": "WINNER_NEEDS_MINIMUM_FULL_COLUMNS", "interpretation": interpretation, "recommended_next_actions": actions}
        if best100["cheap7_delta_vs_legal_ref"] >= ABS_DELTA_THAT_MATTERS:
            actions.append("The 100M winner is an advancing substrate but likely still short by cheap7 alone; inspect per-column pattern and decide whether a focused official-column read is warranted before more training.")
            return {"status": "ADVANCING_BUT_NOT_YET_SOTA_SCALE", "interpretation": interpretation, "recommended_next_actions": actions}
        actions.append("Do not run source_repeat solely for attribution; the trained FW family has not shown enough absolute gain over the legal endpoint.")
        return {"status": "FW_FAMILY_NOT_ENOUGH_AT_100M", "interpretation": interpretation, "recommended_next_actions": actions}

    # Otherwise the 70/80 evidence is present but not promising enough; use 100M
    # only if the late trajectory or per-column pattern can decide the route.
    if best80:
        if best80["cheap7_delta_vs_legal_ref"] > 0.0:
            actions.append("If the 100M endpoints are already trained, evaluate the better 80M arm at 100M before any new GPU training; the 80M gain is positive but small.")
            return {"status": "SMALL_POSITIVE_80M_CONSIDER_100M_WINNER", "interpretation": interpretation, "recommended_next_actions": actions}
        actions.append("No new training. The 70M/80M FW evidence does not yet beat the existing legal trajectory.")
        return {"status": "NO_NEW_TRAINING_FROM_70_80", "interpretation": interpretation, "recommended_next_actions": actions}

    actions.append("Evaluation summary is incomplete; run or repair research cheap evaluation before interpreting the FW result.")
    return {"status": "INCOMPLETE_SUMMARY", "interpretation": interpretation, "recommended_next_actions": actions}


def write_markdown(decision: dict[str, Any], out_md: pathlib.Path) -> None:
    lines = [
        "# research FW absolute-progress decision",
        "",
        f"Created UTC: {decision['created_utc']}",
        "",
        "The FW comparison has two meanings: within-pair compact minus breadth decides the FineWeb companion-budget mechanism; absolute movement versus the current legal trajectory decides whether the family can move toward the complete 41.8+ target.",
        "",
        f"Current complete legal endpoint: Overall {CURRENT_BEST_LEGAL_OVERALL:.4f}; live target {LIVE_LEADER_OVERALL:.1f}; gap {CURRENT_OVERALL_GAP:.4f}. If only cheap7 moves, this is about +{CHEAP7_GAIN_EQUIVALENT_IF_ONLY_CHEAP_COLUMNS_MOVE:.4f} cheap7 at 100M.",
        "",
        "## Absolute comparison versus existing legal trajectory",
        "",
        "| Checkpoint | Arm | cheap7 | legal ref | Δcheap7 | +cols | -cols | ΔBLiMP | ΔSuppl | ΔEWoK | ΔEntity | ΔCOMPS | ΔGPIQA | ΔReading |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decision.get("absolute_comparisons", []):
        if row.get("status") != "OK":
            lines.append(f"| {row.get('checkpoint')} | {row.get('arm')} | {row.get('status')} | | | | | | | | | | | |")
            continue
        d = row["column_deltas_vs_legal_ref"]
        lines.append(
            f"| {row['checkpoint']} | {row['arm']} | {row['cheap7']:.4f} | {row['legal_ref_cheap7']:.4f} | "
            f"{row['cheap7_delta_vs_legal_ref']:+.4f} | {row['positive_columns']} | {row['negative_columns']} | "
            f"{d['BLiMP']:+.2f} | {d['Supplement']:+.2f} | {d['EWoK']:+.2f} | {d['Entity']:+.2f} | "
            f"{d['COMPS']:+.2f} | {d['GlobalPIQA']:+.2f} | {d['Reading']:+.3f} |"
        )
    lines.extend(["", "## Within-pair compact minus breadth", "", "| Checkpoint | Δcheap7 compact-breadth | Result |", "|---|---:|---|"])
    for ckpt, comp in decision.get("within_pair_comparisons", {}).items():
        if comp is None or comp.get("status") != "OK":
            lines.append(f"| {ckpt} | {None if comp is None else comp.get('status')} | |")
            continue
        lines.append(f"| {ckpt} | {comp['cheap7_delta_compact_minus_breadth']:+.4f} | {comp.get('decision_rule', {}).get('result')} |")
    lines.extend(["", "## Interpretation", ""])
    for item in decision.get("synthesis", {}).get("interpretation", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Next action", ""])
    lines.append(f"Status: `{decision.get('synthesis', {}).get('status')}`")
    for item in decision.get("synthesis", {}).get("recommended_next_actions", []):
        lines.append(f"- {item}")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = ap.parse_args()

    summary_path = pathlib.Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = USER_ROOT / summary_path
    summary = load_summary(summary_path)

    checkpoints = summary.get("checkpoints", [])
    abs_rows: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        for arm in ["compact_view", "source_breadth"]:
            abs_rows.append(abs_comparison(result_for(summary, arm, ckpt), ckpt))

    pair_map = {ckpt: within_pair(summary, ckpt) for ckpt in checkpoints}
    synthesis = synthesize(summary, abs_rows)
    decision = {
        "status": "FW_ABSOLUTE_PROGRESS_DECISION",
        "created_utc": now_utc(),
        "source_summary": str(summary_path),
        "live_leader_overall": LIVE_LEADER_OVERALL,
        "current_best_legal_overall": CURRENT_BEST_LEGAL_OVERALL,
        "current_overall_gap": CURRENT_OVERALL_GAP,
        "cheap7_gain_equivalent_if_only_cheap_columns_move": CHEAP7_GAIN_EQUIVALENT_IF_ONLY_CHEAP_COLUMNS_MOVE,
        "legal_reference": LEGAL_REFERENCE,
        "absolute_comparisons": abs_rows,
        "within_pair_comparisons": pair_map,
        "synthesis": synthesis,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = OUT_ROOT / "fw_absolute_decision.json"
    out_md = OUT_ROOT / "fw_absolute_decision.md"
    out_json.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(decision, out_md)
    print(json.dumps({
        "status": decision["status"],
        "synthesis_status": synthesis.get("status"),
        "out_json": str(out_json),
        "out_md": str(out_md),
        "recommended_next_actions": synthesis.get("recommended_next_actions", []),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
