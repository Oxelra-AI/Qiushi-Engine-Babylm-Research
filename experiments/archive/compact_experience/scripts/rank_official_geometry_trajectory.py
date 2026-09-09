#!/usr/bin/env python3
"""Rank official-only geometry trajectory screens against clean-Qwen/cap120 control.

Uses only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading. AoA/SuperGLUE are
not used for ranking.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import json
import pathlib
from statistics import mean
from typing import Any, Dict, List

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_ROOT = _public_path('experiments/archive/compact_experience/data/official_geometry_trajectory_screen')
CLEAN_ROOT = _public_path('experiments/archive/compact_experience/data/trajectory_screen')
CAP_RANK = _public_path('experiments/archive/compact_experience/data/contextual_cap120_trajectory_ranking.json')
OUT = _public_path('experiments/archive/compact_experience/data/official_geometry_trajectory_ranking.json')
NOTE = _public_path('research/notes/compact_experience/official_geometry_trajectory_ranking.md')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RECOVERY = ["BLiMP", "EWoK", "COMPS", "Reading"]


def load(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from_summary(path: pathlib.Path, family: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    d = load(path)
    target = str(d.get("target") or path.name.replace("_trajectory_summary.json", ""))
    rows = []
    for ckpt, row in (d.get("table") or {}).items():
        rec: Dict[str, Any] = {"target": target, "family": family, "checkpoint": ckpt, "equal7_full_eval": row.get("equal7_full_eval")}
        for k in KEYS:
            rec[k] = row.get(k)
        rec["complete"] = rec["equal7_full_eval"] is not None and all(rec[k] is not None for k in KEYS)
        if rec["complete"]:
            rec["recovery_R"] = mean(float(rec[k]) for k in RECOVERY)
        rows.append(rec)
    return rows


def best(rows: List[Dict[str, Any]], target: str, key: str = "equal7_full_eval") -> Dict[str, Any] | None:
    xs = [r for r in rows if r.get("target") == target and r.get("complete") and r.get(key) is not None]
    return max(xs, key=lambda r: float(r[key]), default=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.glob("*_trajectory_summary.json")):
        rows += rows_from_summary(p, "official_geometry")
    rows += rows_from_summary(_public_path('experiments/archive/compact_experience/data/trajectory_screen/clean_qwen_seed43022_trajectory_summary.json'), "clean_qwen_reference")
    complete = [r for r in rows if r.get("complete")]
    ranked = sorted(complete, key=lambda r: float(r["equal7_full_eval"]), reverse=True)
    cap = load(CAP_RANK) if CAP_RANK.exists() else {}
    payload = {
        "status": "OFFICIAL_GEOMETRY_TRAJECTORY_RANKING",
        "non_leakage_statement": "Ranks only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading. AoA/CDI words, child curves, AoA outputs, AoA predictions, and SuperGLUE are not used.",
        "root": str(root),
        "num_rows": len(rows),
        "num_complete_rows": len(complete),
        "best_by_target": {t: best(complete, t) for t in sorted({r['target'] for r in complete})},
        "top_rows": ranked[: args.top_k],
        "reference_clean_best": best(complete, "clean_qwen_seed43022"),
        "reference_cap120_control_best": cap.get("best_control_equal7"),
        "reference_cap120_treatment_best": cap.get("best_treatment_equal7"),
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research official geometry trajectory ranking", "", payload["non_leakage_statement"], "", f"Complete rows: {len(complete)} / {len(rows)}", "", "## Top rows"]
    for i, r in enumerate(ranked[: args.top_k], 1):
        lines.append(f"{i}. `{r['target']}` `{r['checkpoint']}` equal7={float(r['equal7_full_eval']):.6f} R={float(r['recovery_R']):.6f} BLiMP={r.get('BLiMP')} Supplement={r.get('Supplement')} EWoK={r.get('EWoK')} Entity={r.get('Entity')} COMPS={r.get('COMPS')} GlobalPIQA={r.get('GlobalPIQA')} Reading={r.get('Reading')}")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "best_by_target": payload["best_by_target"], "top_rows": ranked[: min(5, len(ranked))]}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
