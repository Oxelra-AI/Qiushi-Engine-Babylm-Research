#!/usr/bin/env python3
"""Summarize research/044 cluster continuation no-AoA trajectory results.

Reads the per-arm trajectory summaries produced by eval_checkpoint_trajectory_fullzeroshot.py
and writes a durable comparison focused on whether the true natural cluster arm E1
beats matched controls E2/E3/E4 and recovers the clean-Qwen deficits in EWoK,
COMPS, and GlobalPIQA without sacrificing the existing strengths.

The script reports both best-per-arm contrasts and same-checkpoint contrasts. It
reads only post-training no-AoA evaluation outputs; it does not use AoA/CDI words
or downstream item content for training/selection.
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

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/cluster_continuation_eval')
DEFAULT_NOTE = _public_path('research/notes/compact_experience/cluster_continuation_noaoa_result.md')
ARMS = ["E1_true_cluster", "E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
TARGET_RECOVERY = ["EWoK", "COMPS", "GlobalPIQA"]
PRESERVE = ["BLiMP", "Supplement", "Entity", "Reading"]

# Clean-Qwen seed43022 chck_100M no-AoA profile from the trusted current frontier.
CLEAN_QWEN = {
    "BLiMP": 66.84,
    "Supplement": 62.84,
    "EWoK": 50.19,
    "Entity": 25.76,
    "COMPS": 51.78,
    "GlobalPIQA": 36.62,
    "Reading": 7.76,
    "equal7_full_eval": 43.112857142857145,
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_summary(out_root: pathlib.Path, arm: str) -> dict[str, Any] | None:
    target = f"step043_{arm}"
    path = out_root / f"{target}_trajectory_summary.json"
    if not path.exists():
        nested = out_root / target / f"{target}_trajectory_summary.json"
        if not nested.exists():
            return None
        path = nested
    obj = json.loads(path.read_text())
    obj["summary_path"] = str(path)
    return obj


def get_best(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("best_by_equal7_full_eval") or summary.get("best") or {}


def best_row(summary: dict[str, Any]) -> dict[str, Any]:
    best = get_best(summary)
    row = dict(best.get("row") or {})
    row["checkpoint"] = best.get("checkpoint")
    row["equal7_full_eval"] = float(row.get("equal7_full_eval", 0.0))
    return row


def col_sum_delta(row_a: dict[str, Any], row_b: dict[str, Any], cols: list[str]) -> float:
    return sum(float(row_a.get(c, 0.0)) - float(row_b.get(c, 0.0)) for c in cols)


def entry_from_row(arm: str, row: dict[str, Any], summary_path: str | None) -> dict[str, Any]:
    return {
        "arm": arm,
        "best_checkpoint": row.get("checkpoint"),
        "equal7": float(row.get("equal7_full_eval", 0.0)),
        "delta_vs_clean_equal7": float(row.get("equal7_full_eval", 0.0)) - CLEAN_QWEN["equal7_full_eval"],
        "target_recovery_sum_delta_vs_clean": col_sum_delta(row, CLEAN_QWEN, TARGET_RECOVERY),
        "preserve_sum_delta_vs_clean": col_sum_delta(row, CLEAN_QWEN, PRESERVE),
        "scores": {c: float(row.get(c, 0.0)) for c in COLUMNS},
        "deltas_vs_clean": {c: float(row.get(c, 0.0)) - float(CLEAN_QWEN[c]) for c in COLUMNS},
        "summary_path": summary_path,
    }


def contrast_rows(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, Any]:
    return {
        "equal7": float(row_a.get("equal7_full_eval", 0.0)) - float(row_b.get("equal7_full_eval", 0.0)),
        "target_recovery_sum": col_sum_delta(row_a, row_b, TARGET_RECOVERY),
        "preserve_sum": col_sum_delta(row_a, row_b, PRESERVE),
        "per_column": {col: float(row_a.get(col, 0.0)) - float(row_b.get(col, 0.0)) for col in COLUMNS},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    note_path = pathlib.Path(args.note)
    out_root.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm in ARMS:
        summary = load_summary(out_root, arm)
        if summary is None:
            missing.append(arm)
            continue
        summaries[arm] = summary
        rows.append(entry_from_row(arm, best_row(summary), summary.get("summary_path")))

    rows.sort(key=lambda x: x["equal7"], reverse=True)
    by_arm = {r["arm"]: r for r in rows}

    best_checkpoint_contrasts: dict[str, Any] = {}
    if "E1_true_cluster" in by_arm:
        e1_best = get_best(summaries["E1_true_cluster"]).get("row", {})
        for ctrl in ["E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]:
            if ctrl not in summaries:
                continue
            c_best = get_best(summaries[ctrl]).get("row", {})
            best_checkpoint_contrasts[f"E1_best_minus_{ctrl}_best"] = contrast_rows(e1_best, c_best)

    same_checkpoint_contrasts: dict[str, Any] = {}
    if "E1_true_cluster" in summaries:
        e1_table = summaries["E1_true_cluster"].get("table", {})
        for ckpt, e1_row in e1_table.items():
            if e1_row.get("equal7_full_eval") is None:
                continue
            ckpt_entry: dict[str, Any] = {}
            for ctrl in ["E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]:
                ctrl_row = summaries.get(ctrl, {}).get("table", {}).get(ckpt, {})
                if ctrl_row.get("equal7_full_eval") is None:
                    continue
                ckpt_entry[f"E1_minus_{ctrl}"] = contrast_rows(e1_row, ctrl_row)
            if ckpt_entry:
                same_checkpoint_contrasts[ckpt] = ckpt_entry

    e1_beats_all_equal7 = bool(
        "E1_true_cluster" in by_arm and all(
            ctrl in by_arm and by_arm["E1_true_cluster"]["equal7"] > by_arm[ctrl]["equal7"]
            for ctrl in ["E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]
        )
    )
    e1_best_target_recovery = bool(
        "E1_true_cluster" in by_arm and all(
            ctrl in by_arm and by_arm["E1_true_cluster"]["target_recovery_sum_delta_vs_clean"] > by_arm[ctrl]["target_recovery_sum_delta_vs_clean"]
            for ctrl in ["E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]
        )
    )

    decision_signal = "missing_outputs"
    if not missing:
        if e1_beats_all_equal7 and e1_best_target_recovery:
            decision_signal = "promote_E1_for_full_nine_column_check"
        else:
            decision_signal = "do_not_promote_cluster_E1_as_frontier_route"

    payload = {
        "status": "CLUSTER_CONTINUATION_NOAOA_SUMMARY",
        "created_utc": now(),
        "out_root": str(out_root),
        "clean_qwen_reference": CLEAN_QWEN,
        "missing_arms": missing,
        "ranked_results": rows,
        "best_checkpoint_contrasts": best_checkpoint_contrasts,
        "same_checkpoint_contrasts": same_checkpoint_contrasts,
        "decision_signal": decision_signal,
        "non_leakage_statement": "No official AoA/CDI words, child curves, or downstream item content used for training or route construction; this is post-training no-AoA measurement only.",
    }
    out_json = out_root / "cluster_continuation_noaoa_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# research cluster continuation no-AoA result",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        "Clean-Qwen reference: equal7 = {:.6f}; target recovery columns are EWoK, COMPS, GlobalPIQA.".format(CLEAN_QWEN["equal7_full_eval"]),
        "",
        "| arm | best | equal7 | Δequal7 vs clean | Δ(EWoK+COMPS+GPIQA) vs clean | Δpreserve(BLiMP+Supp+Entity+Reading) vs clean | EWoK | COMPS | GPIQA | Supp | Entity | Read | BLiMP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["scores"]
        lines.append(
            "| {arm} | {ckpt} | {eq:.4f} | {deq:+.4f} | {dtarget:+.4f} | {dpreserve:+.4f} | {ewok:.2f} | {comps:.2f} | {gpiqa:.2f} | {supp:.2f} | {entity:.2f} | {read:.3f} | {blimp:.2f} |".format(
                arm=r["arm"], ckpt=r["best_checkpoint"], eq=r["equal7"], deq=r["delta_vs_clean_equal7"],
                dtarget=r["target_recovery_sum_delta_vs_clean"], dpreserve=r["preserve_sum_delta_vs_clean"],
                ewok=s["EWoK"], comps=s["COMPS"], gpiqa=s["GlobalPIQA"], supp=s["Supplement"],
                entity=s["Entity"], read=s["Reading"], blimp=s["BLiMP"],
            )
        )
    lines.extend(["", "## E1 best-checkpoint contrasts", ""])
    for name, c in best_checkpoint_contrasts.items():
        lines.append(f"- `{name}`: equal7 {c['equal7']:+.4f}; target-recovery-sum {c['target_recovery_sum']:+.4f}; preserve-sum {c['preserve_sum']:+.4f}; per-column {json.dumps(c['per_column'], ensure_ascii=False)}")
    lines.extend(["", "## Same-checkpoint E1 contrasts", ""])
    for ckpt, ckpt_contrasts in same_checkpoint_contrasts.items():
        lines.append(f"### {ckpt}")
        for name, c in ckpt_contrasts.items():
            lines.append(f"- `{name}`: equal7 {c['equal7']:+.4f}; target-recovery-sum {c['target_recovery_sum']:+.4f}; preserve-sum {c['preserve_sum']:+.4f}; per-column {json.dumps(c['per_column'], ensure_ascii=False)}")
    lines.extend(["", f"Scientific signal: `{decision_signal}`.", ""])
    if missing:
        lines.append(f"Missing arms: {missing}. Rerun or repair evaluation before scientific interpretation.")
    elif decision_signal.startswith("promote"):
        lines.append("E1 is the best matched cluster arm by equal7 and by the targeted recovery sum; next work should freeze the winning checkpoint and run full nine-column evaluation including SuperGLUE and AoA as final measurement.")
    else:
        lines.append("E1 does not dominate the matched controls on the no-AoA profile needed for a frontier route; the natural shared-anchor cluster family should not receive a 100M run unless a distinct construction changes the active ingredient substantially.")
    note_path.write_text("\n".join(lines) + "\n")

    print(json.dumps({"out": str(out_json), "note": str(note_path), "decision_signal": decision_signal, "missing_arms": missing}, indent=2), flush=True)


if __name__ == "__main__":
    main()
