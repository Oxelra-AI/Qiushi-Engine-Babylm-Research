#!/usr/bin/env python3
"""Rank acquisition-curriculum trajectories using only no-AoA columns."""
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
DEFAULT_ROOT = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_trajectory_screen')
CLEAN_ROOT = _public_path('experiments/archive/compact_experience/data/trajectory_screen')
OUT = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_trajectory_ranking.json')
NOTE = _public_path('research/notes/compact_experience/acquisition_curriculum_trajectory_ranking.md')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
PRESERVE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
WEAK_CLEAN = ["BLiMP", "GlobalPIQA", "COMPS", "Reading"]


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
            rec["weak_clean_mean"] = mean(float(rec[k]) for k in WEAK_CLEAN)
        rows.append(rec)
    return rows


def best(rows: List[Dict[str, Any]], target: str, key: str = "equal7_full_eval") -> Dict[str, Any] | None:
    xs = [r for r in rows if r.get("target") == target and r.get("complete") and r.get(key) is not None]
    return max(xs, key=lambda r: float(r[key]), default=None)


def diff(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not a or not b:
        return None
    out: Dict[str, Any] = {"a": a.get("target"), "a_checkpoint": a.get("checkpoint"), "b": b.get("target"), "b_checkpoint": b.get("checkpoint")}
    for k in ["equal7_full_eval", "weak_clean_mean"] + KEYS:
        if a.get(k) is not None and b.get(k) is not None:
            out[f"delta_{k}"] = float(a[k]) - float(b[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--top_k", type=int, default=30)
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.glob("*_trajectory_summary.json")):
        rows += rows_from_summary(p, "acquisition_curriculum")
    rows += rows_from_summary(_public_path('experiments/archive/compact_experience/data/trajectory_screen/clean_qwen_seed43022_trajectory_summary.json'), "clean_qwen_reference")
    complete = [r for r in rows if r.get("complete")]
    ranked = sorted(complete, key=lambda r: float(r["equal7_full_eval"]), reverse=True)
    targets = sorted({r["target"] for r in complete})
    best_by_target = {t: best(complete, t) for t in targets}
    a2 = best_by_target.get("acq_A2_full")
    a0 = best_by_target.get("acq_A0_baseline_order")
    a0p = best_by_target.get("acq_A0p_repeated_random")
    a1 = best_by_target.get("acq_A1_frequency_only")
    a3 = best_by_target.get("acq_A3_random_label")
    a4 = best_by_target.get("acq_A4_reverse")
    clean_ref = best_by_target.get("clean_qwen_seed43022")
    payload = {
        "status": "ACQUISITION_CURRICULUM_TRAJECTORY_RANKING",
        "non_leakage_statement": "Ranks only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading. AoA/CDI words, child curves, AoA outputs, AoA predictions, and SuperGLUE are not used.",
        "root": str(root),
        "num_rows": len(rows),
        "num_complete_rows": len(complete),
        "best_by_target": best_by_target,
        "top_rows": ranked[: args.top_k],
        "reference_clean_best": clean_ref,
        "a2_minus_contemporaneous_a0_best": diff(a2, a0),
        "a2_minus_repeated_random_a0p_best": diff(a2, a0p),
        "a2_minus_a1_best": diff(a2, a1),
        "a2_minus_a3_best": diff(a2, a3),
        "a2_minus_a4_reverse_best": diff(a2, a4),
        "a2_minus_historical_clean_reference_best": diff(a2, clean_ref),
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research acquisition-curriculum trajectory ranking", "", payload["non_leakage_statement"], "", f"Complete rows: {len(complete)} / {len(rows)}", "", "## Top rows"]
    for i, r in enumerate(ranked[: args.top_k], 1):
        lines.append(f"{i}. `{r['target']}` `{r['checkpoint']}` equal7={float(r['equal7_full_eval']):.6f} weak_clean_mean={float(r.get('weak_clean_mean', 0.0)):.6f} BLiMP={r.get('BLiMP')} Supplement={r.get('Supplement')} EWoK={r.get('EWoK')} Entity={r.get('Entity')} COMPS={r.get('COMPS')} GlobalPIQA={r.get('GlobalPIQA')} Reading={r.get('Reading')}")
    lines += ["", "## Key deltas", json.dumps({k: payload[k] for k in ["a2_minus_contemporaneous_a0_best", "a2_minus_repeated_random_a0p_best", "a2_minus_a1_best", "a2_minus_a3_best", "a2_minus_a4_reverse_best", "a2_minus_historical_clean_reference_best"]}, indent=2, ensure_ascii=False)]
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "best_by_target": best_by_target, "top_rows": ranked[: min(5, len(ranked))]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
