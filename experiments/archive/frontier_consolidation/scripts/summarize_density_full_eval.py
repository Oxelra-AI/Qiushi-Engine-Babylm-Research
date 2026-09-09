#!/usr/bin/env python3
"""Summarize research density compact full official-compatible evaluation.

Reads completed per-target JSON files produced by
`full_eval_density_compact.py`, computes official-like nine-column scores,
and contrasts the compact-core candidate against the inherited clean-Qwen result,
the visible leader surface, and its matched compact-repeat control when available.
The script is safe on partial runs and records missing columns explicitly.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys
from typing import Any

USER_ROOT = _public_path('.')
COMPACT_EXPERIENCE_SCRIPTS = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
from babylm_official_scoring import OFFICIAL_OVERALL_KEYS, NLP_KEYS, compute_overall_from_tasks, target_scores_from_tasks, normalize_aoa_record  # noqa: E402

WORKSPACE = pathlib.Path("experiments/archive/frontier_consolidation")
PER_TARGET = WORKSPACE / "data" / "density_full_eval" / "per_target"
OUT = WORKSPACE / "data" / "density_full_eval" / "density_full_eval_summary.json"
NOTE = (_PUBLIC_ROOT / 'research/notes/frontier_consolidation/density_full_eval_summary.md')
FAST = WORKSPACE / "data" / "density_noaoa_eval_compact_core" / "density_noaoa_eval_summary.json"
CLEAN_SUMMARY = _public_path('experiments/archive/compact_experience/data/full_eval/full_eval_summary.json')
LEADER = {
    "source": "Visible BabyLM 2026 Strict-Small leaderboard/model card for wwm_curriculum_simplification_40k",
    "Overall": 41.80,
    "BLiMP": 67.20,
    "Supplement": 56.01,
    "EWoK": 56.07,
    "Entity": 28.45,
    "COMPS": 53.57,
    "GlobalPIQA": 39.67,
    "SuperGLUE": 69.79,
    "Reading": 5.42,
    "AoA": 0.0,
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
    row = {
        "target": payload.get("target", path.stem),
        "per_target_json": str(path),
        "run_dir": payload.get("run_dir"),
        "model_path": payload.get("model_path"),
        "scores": scores,
        "official_overall": overall,
        "completed_columns": [k for k, v in scores.items() if v is not None],
        "missing_columns": [k for k, v in scores.items() if v is None],
        "task_returncodes": {k: v.get("returncode") for k, v in tasks.items() if isinstance(v, dict)},
        "aoa_status": overall.get("aoa_status"),
        "submit_ready_overall": overall.get("submit_ready_overall"),
        "started_utc": payload.get("started_utc"),
        "finished_utc": payload.get("finished_utc"),
        "run_summary": payload.get("run_summary"),
    }
    return row


def flat_scores(row: dict[str, Any]) -> dict[str, float | None]:
    out = dict(row.get("scores", {}))
    oo = row.get("official_overall", {}) or {}
    for k in ["Overall", "NLP_average", "Human_like_average"]:
        out[k] = oo.get(k)
    return out


def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    af = flat_scores(a)
    bf = flat_scores(b)
    keys = OFFICIAL_OVERALL_KEYS + ["Overall", "NLP_average", "Human_like_average"]
    return {k: None if af.get(k) is None or bf.get(k) is None else round(float(af[k]) - float(bf[k]), 6) for k in keys}


def ref_row(name: str, scores: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": name,
        "scores": {k: scores.get(k) for k in OFFICIAL_OVERALL_KEYS},
        "official_overall": {
            "Overall": scores.get("Overall"),
            "NLP_average": scores.get("NLP_average"),
            "Human_like_average": scores.get("Human_like_average"),
        },
    }


def main() -> None:
    rows = {}
    if PER_TARGET.exists():
        for p in sorted(PER_TARGET.glob("*.json")):
            try:
                r = extract_row(p)
                rows[r["target"]] = r
            except Exception as exc:
                rows[p.stem] = {"target": p.stem, "per_target_json": str(p), "error": repr(exc)}

    clean = None
    clean_all = None
    if CLEAN_SUMMARY.exists():
        clean_all = read_json(CLEAN_SUMMARY)
        clean_scores = clean_all.get("targets", {}).get("qwen_clean_aligned")
        if clean_scores:
            clean = ref_row("compact_experience_clean_qwen", clean_scores)
    leader = ref_row("visible_leader", LEADER)

    contrasts = {}
    cvc = rows.get("compact_view_core")
    crc = rows.get("compact_repeat_core")
    cvr = rows.get("compact_view_reinvest")
    if cvc and crc and "scores" in cvc and "scores" in crc:
        contrasts["compact_view_core_minus_compact_repeat_core"] = diff(cvc, crc)
    if cvc and clean and "scores" in cvc:
        contrasts["compact_view_core_minus_compact_experience_clean_qwen"] = diff(cvc, clean)
    if cvc and "scores" in cvc:
        contrasts["compact_view_core_minus_visible_leader"] = diff(cvc, leader)
    if cvr and cvc and "scores" in cvr and "scores" in cvc:
        contrasts["compact_view_reinvest_minus_compact_view_core"] = diff(cvr, cvc)

    fast = read_json(FAST) if FAST.exists() else None
    payload = {
        "status": "DENSITY_FULL_EVAL_SUMMARY",
        "per_target_dir": str(PER_TARGET),
        "targets": rows,
        "contrasts": contrasts,
        "reference": {
            "visible_leader": LEADER,
            "compact_experience_clean_qwen": clean.get("scores") if clean else None,
            "compact_experience_clean_qwen_source": str(CLEAN_SUMMARY),
        },
        "fast_screen_source": str(FAST),
        "fast_screen": fast,
        "interpretation": (
            "Full-eval summary for the protected compact-core density candidate. "
            "Incomplete rows should not be read as scientific failures; use task_returncodes and missing_columns."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research density compact full-evaluation summary",
        "",
        f"Summary JSON: `{OUT}`",
        f"Per-target dir: `{PER_TARGET}`",
        "",
        "## Completed target rows",
        "",
        "| target | Overall | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | SuperGLUE | Reading | AoA | submit-ready | missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, row in rows.items():
        if "scores" not in row:
            lines.append(f"| {name} | ERROR | | | | | | | | | | | {row.get('error')} |")
            continue
        s = row["scores"]; oo = row.get("official_overall", {}) or {}
        fmt = lambda x: "" if x is None else f"{float(x):.4f}"
        lines.append(
            f"| {name} | {fmt(oo.get('Overall'))} | {fmt(s.get('BLiMP'))} | {fmt(s.get('Supplement'))} | {fmt(s.get('EWoK'))} | {fmt(s.get('Entity'))} | {fmt(s.get('COMPS'))} | {fmt(s.get('GlobalPIQA'))} | {fmt(s.get('SuperGLUE'))} | {fmt(s.get('Reading'))} | {fmt(s.get('AoA'))} | {oo.get('submit_ready_overall')} | {', '.join(row.get('missing_columns', []))} |"
        )
    lines.extend(["", "## Key contrasts", ""])
    for name, rec in contrasts.items():
        vals = ", ".join(f"{k}={v}" for k, v in rec.items())
        lines.append(f"- **{name}**: {vals}")
    lines.extend([
        "",
        "## Reading rule",
        "",
        "The compact-core mechanism is judged first by `compact_view_core_minus_compact_repeat_core` and then by full comparison with the inherited COMPACT_EXPERIENCE clean-Qwen result and visible leader. Reinvestment, if present, is an extension rather than a verdict on the compact-core mechanism.",
    ])
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": str(OUT), "note": str(NOTE), "targets": list(rows)}, indent=2))


if __name__ == "__main__":
    main()
