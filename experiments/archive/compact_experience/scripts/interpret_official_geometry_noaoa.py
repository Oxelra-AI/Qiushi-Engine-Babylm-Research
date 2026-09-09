#!/usr/bin/env python3
"""Interpret official-geometry no-AoA trajectory ranking.

Reads only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading ranking outputs.
It does not read SuperGLUE, AoA, CDI words, child curves, AoA predictions, or AoA scores.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
from statistics import mean
from typing import Any, Dict, Iterable, Optional

ROOT = _public_path('experiments/archive/compact_experience')
RANKING = _public_path('experiments/archive/compact_experience/data/official_geometry_trajectory_ranking.json')
OUT = _public_path('experiments/archive/compact_experience/data/official_geometry_noaoa_interpretation.json')
NOTE = _public_path('research/notes/compact_experience/official_geometry_noaoa_interpretation.md')
KEYS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RECOVERY = ["BLiMP", "EWoK", "COMPS", "Reading"]
CLEAN_OVERALL_9 = 41.34429066479573
VISIBLE_LEADER = 41.8


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def f(x: Any) -> Optional[float]:
    return None if x is None else float(x)


def mean_fields(row: Dict[str, Any], fields: Iterable[str]) -> Optional[float]:
    vals = [f(row.get(k)) for k in fields]
    if any(v is None for v in vals):
        return None
    return mean(v for v in vals if v is not None)


def best(ranking: Dict[str, Any], target: str) -> Optional[Dict[str, Any]]:
    rec = (ranking.get("best_by_target") or {}).get(target)
    return rec if isinstance(rec, dict) else None


def diff(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"contrast": name}
    if not a or not b:
        out["available"] = False
        return out
    out["available"] = True
    out["a"] = {"target": a.get("target"), "checkpoint": a.get("checkpoint")}
    out["b"] = {"target": b.get("target"), "checkpoint": b.get("checkpoint")}
    for k in ["equal7_full_eval", "recovery_R"] + KEYS7:
        av = f(a.get(k))
        bv = f(b.get(k))
        if av is not None and bv is not None:
            out[f"delta_{k}"] = av - bv
    if f(a.get("equal7_full_eval")) is not None:
        eq = float(a["equal7_full_eval"])
        out["required_superglue_plus_aoa_to_reach_41p8"] = 9 * VISIBLE_LEADER - 7 * eq
        out["required_superglue_plus_aoa_to_match_clean_overall9"] = 9 * CLEAN_OVERALL_9 - 7 * eq
    return out


def profile(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {"available": False}
    rec = {"available": True, "target": row.get("target"), "checkpoint": row.get("checkpoint")}
    rec["equal7"] = f(row.get("equal7_full_eval"))
    rec["recovery_R"] = mean_fields(row, RECOVERY)
    for k in KEYS7:
        rec[k] = f(row.get(k))
    if rec["equal7"] is not None:
        rec["required_superglue_plus_aoa_to_reach_41p8"] = 9 * VISIBLE_LEADER - 7 * float(rec["equal7"])
        rec["required_superglue_plus_aoa_to_match_clean_overall9"] = 9 * CLEAN_OVERALL_9 - 7 * float(rec["equal7"])
    return rec


def main() -> None:
    if not RANKING.exists():
        raise SystemExit(f"missing ranking: {RANKING}")
    ranking = read_json(RANKING)
    g0 = best(ranking, "official160_b256")
    g3 = best(ranking, "official_cap120geom_b256")
    clean = best(ranking, "clean_qwen_seed43022") or ranking.get("reference_clean_best")
    top = ranking.get("top_rows") or []
    top_official = [r for r in top if isinstance(r, dict) and r.get("target") in {"official160_b256", "official_cap120geom_b256"}]
    payload = {
        "status": "OFFICIAL_GEOMETRY_NOAOA_INTERPRETATION",
        "scope": "BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading only; SuperGLUE and AoA not read or used.",
        "ranking": str(RANKING),
        "clean_complete_overall9_reference": CLEAN_OVERALL_9,
        "visible_leader_overall_reference": VISIBLE_LEADER,
        "profiles": {
            "official160_b256_best": profile(g0),
            "official_cap120geom_b256_best": profile(g3),
            "clean_qwen_seed43022_best": profile(clean if isinstance(clean, dict) else None),
        },
        "contrasts": [
            diff(g3, g0, "cap120geom_b256_minus_official160_b256"),
            diff(g3, clean if isinstance(clean, dict) else None, "cap120geom_b256_minus_clean_qwen_seed43022"),
            diff(g0, clean if isinstance(clean, dict) else None, "official160_b256_minus_clean_qwen_seed43022"),
        ],
        "top_official_geometry_rows": top_official[:10],
        "top_all_rows": top[:10],
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research official geometry no-AoA interpretation",
        "",
        payload["scope"],
        "",
        f"Clean complete Overall-9 reference: {CLEAN_OVERALL_9}",
        f"Visible leader Overall reference: {VISIBLE_LEADER}",
        "",
        "## Best profiles",
    ]
    for name, row in payload["profiles"].items():
        lines.append("- " + name + ": " + json.dumps(row, ensure_ascii=False, sort_keys=True))
    lines += ["", "## Contrasts"]
    for d in payload["contrasts"]:
        lines.append("- " + json.dumps(d, ensure_ascii=False, sort_keys=True))
    lines += ["", "## Top official-geometry rows"]
    for r in top_official[:10]:
        lines.append(
            f"- `{r.get('target')}` `{r.get('checkpoint')}` equal7={r.get('equal7_full_eval')} "
            f"BLiMP={r.get('BLiMP')} Supplement={r.get('Supplement')} EWoK={r.get('EWoK')} "
            f"Entity={r.get('Entity')} COMPS={r.get('COMPS')} GlobalPIQA={r.get('GlobalPIQA')} Reading={r.get('Reading')}"
        )
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "profiles": payload["profiles"], "contrasts": payload["contrasts"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
