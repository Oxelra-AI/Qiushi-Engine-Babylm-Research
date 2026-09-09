#!/usr/bin/env python3
"""Summarize full evaluation of compact_view_reinvest.

Safe on partial results. It reads the per-target JSON from
data/compact_reinvest_full_eval/per_target, computes official
nine-column scores when possible, and contrasts the result against the visible
41.8 leader, the inherited COMPACT_EXPERIENCE clean-Qwen coordinate, and the existing fast
compact-view/core surfaces.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))

from babylm_official_scoring import OFFICIAL_OVERALL_KEYS, compute_overall_from_tasks, target_scores_from_tasks, normalize_aoa_record  # noqa: E402

A01_WORKSPACE = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
OUT_ROOT = A01_WORKSPACE / "data" / "compact_reinvest_full_eval"
PER_TARGET = OUT_ROOT / "per_target"
SUMMARY = OUT_ROOT / "compact_reinvest_full_eval_summary.json"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/compact_reinvest_full_eval_summary.md')
FAST_REINVEST = USER_ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_noaoa_eval_reinvest" / "density_noaoa_eval_summary.json"
DENSITY_ANALYSIS = A01_WORKSPACE / "data" / "compact_view_density_review" / "compact_view_density_review.json"

VISIBLE_LEADER = {
    "BLiMP": 67.20, "Supplement": 56.01, "EWoK": 56.07, "Entity": 28.45,
    "COMPS": 53.57, "SuperGLUE": 69.79, "GlobalPIQA": 39.67,
    "Reading": 5.42, "AoA": 0.0, "Overall": 41.80,
}
COMPACT_EXPERIENCE_CLEAN = {
    "BLiMP": 66.84, "Supplement": 62.84, "EWoK": 50.19, "Entity": 25.76,
    "COMPS": 51.78, "SuperGLUE": 70.30861598316157, "GlobalPIQA": 36.62,
    "Reading": 7.76, "AoA": 0.0, "Overall": 41.34429066479573,
}


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_row(path: pathlib.Path) -> dict[str, Any]:
    payload = read_json(path)
    tasks = payload.get("tasks", {})
    if isinstance(tasks.get("AoA"), dict):
        tasks["AoA"] = normalize_aoa_record(tasks["AoA"])
    scores = target_scores_from_tasks(tasks)
    overall = payload.get("official_overall") or compute_overall_from_tasks(tasks)
    return {
        "target": payload.get("target", path.stem),
        "per_target_json": str(path),
        "run_dir": payload.get("run_dir"),
        "model_root": payload.get("model_root"),
        "model_path": payload.get("model_path"),
        "endpoint": payload.get("endpoint"),
        "scores": scores,
        "official_overall": overall,
        "completed_columns": [k for k, v in scores.items() if v is not None],
        "missing_columns": [k for k, v in scores.items() if v is None],
        "task_returncodes": {k: v.get("returncode") for k, v in tasks.items() if isinstance(v, dict)},
        "superglue_tasks": tasks.get("SuperGLUE", {}).get("tasks") if isinstance(tasks.get("SuperGLUE"), dict) else None,
        "aoa_record": tasks.get("AoA") if isinstance(tasks.get("AoA"), dict) else None,
        "run_summary": payload.get("run_summary"),
        "started_utc": payload.get("started_utc"),
        "finished_utc": payload.get("finished_utc"),
    }


def flat(row_or_ref: dict[str, Any]) -> dict[str, Any]:
    if "scores" in row_or_ref and "official_overall" in row_or_ref:
        out = dict(row_or_ref.get("scores") or {})
        oo = row_or_ref.get("official_overall") or {}
        out["Overall"] = oo.get("Overall")
        out["NLP_average"] = oo.get("NLP_average")
        out["Human_like_average"] = oo.get("Human_like_average")
        return out
    out = dict(row_or_ref)
    if "GlobalPIQA_mean" in out and "GlobalPIQA" not in out:
        out["GlobalPIQA"] = out["GlobalPIQA_mean"]
    return out


def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    af, bf = flat(a), flat(b)
    keys = OFFICIAL_OVERALL_KEYS + ["Overall", "NLP_average", "Human_like_average"]
    return {k: None if af.get(k) is None or bf.get(k) is None else round(float(af[k]) - float(bf[k]), 6) for k in keys}


def seven_sum(scores: dict[str, Any]) -> float | None:
    keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    vals = [scores.get(k) for k in keys]
    if any(v is None for v in vals):
        return None
    return sum(float(v) for v in vals)


def main() -> None:
    rows: dict[str, Any] = {}
    if PER_TARGET.exists():
        for p in sorted(PER_TARGET.glob("*.json")):
            try:
                row = extract_row(p)
                rows[row["target"]] = row
            except Exception as exc:
                rows[p.stem] = {"target": p.stem, "per_target_json": str(p), "error": repr(exc)}
    reinvest = rows.get("compact_view_reinvest")

    fast = read_json(FAST_REINVEST) if FAST_REINVEST.exists() else None
    density_analysis = read_json(DENSITY_ANALYSIS) if DENSITY_ANALYSIS.exists() else None
    contrasts: dict[str, Any] = {}
    if reinvest and "scores" in reinvest:
        contrasts["compact_view_reinvest_minus_visible_leader"] = diff(reinvest, VISIBLE_LEADER)
        contrasts["compact_view_reinvest_minus_compact_experience_clean_qwen"] = diff(reinvest, COMPACT_EXPERIENCE_CLEAN)
        if density_analysis:
            core_fast = density_analysis.get("component_rows", {}).get("compact_view_core_seed43022")
            if core_fast:
                # Fast core has no SuperGLUE/AoA; diff only the overlapping visible columns.
                core_ref = {
                    "BLiMP": core_fast.get("BLiMP"), "Supplement": core_fast.get("Supplement"),
                    "EWoK": core_fast.get("EWoK"), "Entity": core_fast.get("Entity"),
                    "COMPS": core_fast.get("COMPS"), "GlobalPIQA": core_fast.get("GlobalPIQA_mean"),
                    "Reading": core_fast.get("Reading"),
                }
                contrasts["compact_view_reinvest_full_minus_core_fast_overlap"] = diff(reinvest, core_ref)

    if reinvest and "scores" in reinvest:
        s = reinvest["scores"]
        oo = reinvest.get("official_overall") or {}
        seven = seven_sum(s)
        sg_aoa = None
        if s.get("SuperGLUE") is not None and s.get("AoA") is not None:
            sg_aoa = float(s["SuperGLUE"]) + float(s["AoA"])
        interpretation = {
            "full_overall": oo.get("Overall"),
            "submit_ready_overall": oo.get("submit_ready_overall"),
            "aoa_status": oo.get("aoa_status"),
            "seven_column_sum": seven,
            "superglue_plus_aoa": sg_aoa,
            "passes_visible_41p8": (oo.get("Overall") is not None and float(oo["Overall"]) > 41.8),
            "passes_round_42p0": (oo.get("Overall") is not None and float(oo["Overall"]) >= 42.0),
            "scientific_reading": "Full compact_view_reinvest score should be read as the SOTA-facing endpoint; core view/repeat remains the mechanism comparison.",
        }
    else:
        interpretation = {"status": "awaiting_or_failed_per_target", "scientific_reading": "No completed compact_view_reinvest row is available yet."}

    payload = {
        "status": "COMPACT_REINVEST_FULL_EVAL_SUMMARY",
        "per_target_dir": str(PER_TARGET),
        "targets": rows,
        "contrasts": contrasts,
        "reference": {
            "visible_leader": VISIBLE_LEADER,
            "compact_experience_clean_qwen": COMPACT_EXPERIENCE_CLEAN,
            "fast_reinvest_source": str(FAST_REINVEST),
            "density_analysis_review_source": str(DENSITY_ANALYSIS),
            "fast_reinvest": fast,
        },
        "interpretation": interpretation,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    lines = [
        "# research compact_view_reinvest full evaluation summary",
        "",
        f"Summary JSON: `{SUMMARY}`",
        f"Per-target dir: `{PER_TARGET}`",
        "",
        "## Completed rows",
        "",
        "| target | Overall | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | SuperGLUE | Reading | AoA | submit-ready | missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, row in rows.items():
        if "scores" not in row:
            lines.append(f"| {name} | ERROR | | | | | | | | | | | {row.get('error')} |")
            continue
        s = row["scores"]; oo = row.get("official_overall") or {}
        lines.append(
            f"| {name} | {fmt(oo.get('Overall'))} | {fmt(s.get('BLiMP'))} | {fmt(s.get('Supplement'))} | {fmt(s.get('EWoK'))} | {fmt(s.get('Entity'))} | {fmt(s.get('COMPS'))} | {fmt(s.get('GlobalPIQA'))} | {fmt(s.get('SuperGLUE'))} | {fmt(s.get('Reading'))} | {fmt(s.get('AoA'))} | {oo.get('submit_ready_overall')} | {', '.join(row.get('missing_columns', []))} |"
        )
    lines += ["", "## Key contrasts", ""]
    for name, rec in contrasts.items():
        vals = ", ".join(f"{k}={v}" for k, v in rec.items())
        lines.append(f"- **{name}**: {vals}")
    lines += ["", "## Interpretation", "", json.dumps(interpretation, indent=2, ensure_ascii=False)]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": str(SUMMARY), "note": str(NOTE), "targets": list(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
