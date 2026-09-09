#!/usr/bin/env python3
"""Prepare frozen official-geometry full-eval targets from no-AoA ranking.

This script reads only the research official-geometry trajectory ranking, whose ranker
uses BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading and excludes SuperGLUE,
AoA, CDI words, child curves, AoA predictions, and AoA scores. It emits a target list
for corrected full evaluation, where SuperGLUE and AoA are final measurements only.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
from typing import Any, Dict, List, Optional

ROOT = _public_path('experiments/archive/compact_experience')
RANKING = _public_path('experiments/archive/compact_experience/data/official_geometry_trajectory_ranking.json')
OUT = _public_path('experiments/archive/compact_experience/data/official_geometry_full_eval_targets.json')
NOTE = _public_path('research/notes/compact_experience/official_geometry_full_eval_targets.md')
FAMILIES = [
    "official160_b256",
    "official_cap120geom_b256",
    "official_cap120geom_b286_lr2515",
    "official_cap120geom_b288_lr2515",
    "official_cap120geom_b295_lr2442",
    "official160_b223_lr2809",
]


def load(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def target_name(row: Dict[str, Any]) -> str:
    ckpt = str(row["checkpoint"])
    if not ckpt.startswith("chck_") or not ckpt.endswith("M"):
        raise ValueError(f"unexpected checkpoint name: {ckpt}")
    return f"{row['target']}_{ckpt.removeprefix('chck_')}"


def best_for(ranking: Dict[str, Any], family: str) -> Optional[Dict[str, Any]]:
    rec = (ranking.get("best_by_target") or {}).get(family)
    return rec if isinstance(rec, dict) else None


def row_key(row: Dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("target")), str(row.get("checkpoint")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default=str(RANKING))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--top_k_any", type=int, default=2, help="Also include this many globally top rows from the no-AoA ranking.")
    args = ap.parse_args()
    ranking_path = pathlib.Path(args.ranking)
    if not ranking_path.exists():
        raise SystemExit(f"missing no-AoA ranking: {ranking_path}")
    ranking = load(ranking_path)

    selected: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    # Include each trained official geometry family's own no-AoA best checkpoint.
    for fam in FAMILIES:
        rec = best_for(ranking, fam)
        if rec and row_key(rec) not in seen:
            selected.append({"reason": f"best_{fam}_equal7_no_aoa", "row": rec})
            seen.add(row_key(rec))

    # Include the globally strongest rows as additional frozen candidates, still no-AoA chosen.
    for rec in ranking.get("top_rows", [])[: max(0, args.top_k_any)]:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("target")) not in FAMILIES:
            continue
        if row_key(rec) not in seen:
            selected.append({"reason": "global_top_equal7_no_aoa", "row": rec})
            seen.add(row_key(rec))

    targets = [target_name(x["row"]) for x in selected]
    payload = {
        "status": "OFFICIAL_GEOMETRY_FULL_EVAL_TARGETS",
        "non_leakage_statement": "Targets selected only from no-AoA trajectory ranking over BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading; SuperGLUE and AoA are added later as final measurements.",
        "ranking": str(ranking_path),
        "targets": targets,
        "selected": [dict(x, target=target_name(x["row"])) for x in selected],
        "reference_clean_best": ranking.get("reference_clean_best"),
        "reference_cap120_control_best": ranking.get("reference_cap120_control_best"),
        "reference_cap120_treatment_best": ranking.get("reference_cap120_treatment_best"),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research official-geometry full-eval target freeze",
        "",
        payload["non_leakage_statement"],
        "",
        f"Ranking source: `{ranking_path}`",
        "",
        "## Selected targets",
    ]
    for item, target in zip(selected, targets):
        r = item["row"]
        lines.append(
            f"- `{target}` from `{r.get('target')}` `{r.get('checkpoint')}` reason={item['reason']} "
            f"equal7={r.get('equal7_full_eval')} R={r.get('recovery_R')} "
            f"BLiMP={r.get('BLiMP')} Supplement={r.get('Supplement')} EWoK={r.get('EWoK')} "
            f"Entity={r.get('Entity')} COMPS={r.get('COMPS')} GlobalPIQA={r.get('GlobalPIQA')} Reading={r.get('Reading')}"
        )
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "note": str(NOTE), "targets": targets}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
