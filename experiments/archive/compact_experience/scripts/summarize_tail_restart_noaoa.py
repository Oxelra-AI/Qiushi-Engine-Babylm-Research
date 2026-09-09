#!/usr/bin/env python3
"""Summarize no-AoA trajectory for the clean-parent tail restart."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time

ROOT = _public_path('experiments/archive/compact_experience')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/tail_restart_noaoa_eval')
SUMMARY = _public_path('experiments/archive/compact_experience/data/tail_restart_noaoa_eval/clean_tail_restart_ladder_trajectory_summary.json')
OUT = _public_path('experiments/archive/compact_experience/data/tail_restart_noaoa_eval/tail_restart_noaoa_summary.json')
NOTE = _public_path('research/notes/compact_experience/tail_restart_noaoa_summary.md')
CLEAN = {
    "BLiMP": 66.84, "Supplement": 62.84, "EWoK": 50.19, "Entity": 25.76,
    "COMPS": 51.78, "GlobalPIQA": 36.62, "Reading": 7.76,
    "equal7_full_eval": 43.112857142857145,
}
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    if not SUMMARY.exists():
        raise FileNotFoundError(SUMMARY)
    obj = json.loads(SUMMARY.read_text(encoding="utf-8"))
    best = obj.get("best_by_equal7_full_eval") or obj.get("best", {})
    table = obj.get("table", {})
    rows = []
    for ckpt, row in table.items():
        if not isinstance(row, dict) or row.get("equal7_full_eval") is None:
            continue
        rows.append({
            "checkpoint": ckpt,
            "equal7": float(row["equal7_full_eval"]),
            "delta_vs_clean_equal7": float(row["equal7_full_eval"]) - CLEAN["equal7_full_eval"],
            "scores": {c: float(row.get(c, 0.0)) for c in COLUMNS},
            "deltas_vs_clean": {c: float(row.get(c, 0.0)) - CLEAN[c] for c in COLUMNS},
        })
    rows.sort(key=lambda r: r["equal7"], reverse=True)
    payload = {
        "status": "TAIL_RESTART_NOAOA_SUMMARY",
        "created_utc": now(),
        "trajectory_summary": str(SUMMARY),
        "clean_qwen_reference": CLEAN,
        "best_checkpoint": best.get("checkpoint"),
        "best_row": rows[0] if rows else None,
        "ranked_results": rows,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research clean-tail restart no-AoA trajectory",
        "",
        f"Created UTC: {payload['created_utc']}",
        "",
        "| checkpoint | equal7 | Δequal7 vs clean | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["scores"]
        lines.append(f"| {r['checkpoint']} | {r['equal7']:.4f} | {r['delta_vs_clean_equal7']:+.4f} | {s['BLiMP']:.2f} | {s['Supplement']:.2f} | {s['EWoK']:.2f} | {s['Entity']:.2f} | {s['COMPS']:.2f} | {s['GlobalPIQA']:.2f} | {s['Reading']:.3f} |")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "best_checkpoint": payload["best_checkpoint"], "best_equal7": rows[0]["equal7"] if rows else None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
