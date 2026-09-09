#!/usr/bin/env python3
"""Repair private-scale SuperGLUE summary files from completed payloads.

research/157 SuperGLUE wrappers may crash after successful official-compatible
SuperGLUE evaluation because the private-scale cheap summaries store `scores`
but not a top-level `cheap7`.  This script reconstructs the intended summary
from the completed SuperGLUE payload and cheap score vector.  It does not rerun
SuperGLUE, upload, or submit anything.
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
from statistics import mean
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
CHCK82 = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def payload_superglue(payload: dict[str, Any]) -> float:
    sg = payload.get("tasks", {}).get("SuperGLUE", {})
    if "superglue_mean" in sg:
        return float(sg["superglue_mean"])
    oo = payload.get("official_overall", {}).get("scores", {})
    if oo.get("SuperGLUE") is not None:
        return float(oo["SuperGLUE"])
    raise KeyError("SuperGLUE score not found in payload")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--cheap-summary", required=True)
    ap.add_argument("--superglue-payload", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--endpoint", default="final")
    args = ap.parse_args()

    cheap_summary_path = pathlib.Path(args.cheap_summary)
    sg_payload_path = pathlib.Path(args.superglue_payload)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cheap = read_json(cheap_summary_path)
    payload = read_json(sg_payload_path)
    protected = read_json(CHCK82)
    protected_scores = {k: float(v) for k, v in protected["score_arithmetic"]["scores"].items() if v is not None}
    protected_overall = float(protected["score_arithmetic"]["overall_reported"])
    protected_cheap7 = float(mean(protected_scores[c] for c in CHEAP_COLS))

    cheap_scores = {k: float(v) for k, v in cheap["scores"].items() if v is not None}
    cheap7 = float(mean(cheap_scores[c] for c in CHEAP_COLS))
    sg = payload_superglue(payload)
    overall = float(mean([*(cheap_scores[c] for c in CHEAP_COLS), sg, 0.0]))

    sg_task = payload.get("tasks", {}).get("SuperGLUE", {})
    summary = {
        "status": "REPAIRED_PRIVATE_SCALE_SUPERGLUE_SUMMARY",
        "created_utc": now(),
        "label": args.label,
        "target": args.target,
        "run_dir": rel(args.run_dir) if args.run_dir else payload.get("run_dir"),
        "endpoint": args.endpoint,
        "cheap_summary": rel(cheap_summary_path),
        "cheap_scores": cheap_scores,
        "cheap7": cheap7,
        "superglue": sg,
        "aoa_assumed_for_projection": 0.0,
        "projected_overall_with_aoa0": overall,
        "protected_chck82": {
            "overall": protected_overall,
            "cheap7": protected_cheap7,
            "superglue": protected_scores["SuperGLUE"],
            "source": rel(CHCK82),
        },
        "deltas_vs_chck82": {
            "cheap7": cheap7 - protected_cheap7,
            "superglue": sg - protected_scores["SuperGLUE"],
            "projected_overall_with_aoa0": overall - protected_overall,
        },
        "payload_path": rel(sg_payload_path),
        "superglue_subtask_metrics": sg_task.get("superglue_primary_metric_details", []),
        "repair_reason": "Original SuperGLUE wrapper completed the official-compatible SuperGLUE run but crashed while reading a missing top-level cheap7 key in the private-scale cheap summary. This file reconstructs only the arithmetic summary from existing payloads.",
        "scientific_reading": "Endpoint arithmetic only; research closed private-scale amplitude and anchor-confidence gating as a general relation/state mechanism.",
    }
    out_json = out_dir / f"{args.target}_superglue_summary.json"
    out_md = out_dir / f"{args.target}_superglue_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(
        f"# research repaired private-scale SuperGLUE summary — {args.target}\n\n"
        f"Status: **{summary['status']}**\n\n"
        f"SuperGLUE: `{sg}`; cheap7: `{cheap7}`.\n\n"
        f"Projected Overall(AoA0): `{overall}` (delta vs chck82 `{overall - protected_overall:+.12f}`).\n\n"
        f"Payload: `{rel(sg_payload_path)}`\n\n"
        f"This is a repaired arithmetic summary from an already completed SuperGLUE payload; no evaluation was rerun.\n\n"
        f"JSON: `{rel(out_json)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "out_json": rel(out_json), "superglue": sg, "cheap7": cheap7, "overall_with_aoa0": overall, "delta_vs_chck82": overall - protected_overall}, indent=2), flush=True)


if __name__ == "__main__":
    main()
