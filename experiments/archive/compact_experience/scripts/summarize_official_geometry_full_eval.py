#!/usr/bin/env python3
"""Summarize corrected full-eval results for official-geometry candidates.

Reads finalized per-target JSONs from research official-geometry full evaluation. The
script interprets AoA only as a final measurement from already-frozen checkpoints and
uses canonical `official_overall.Overall` / `official_overall.scores` when present.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

ROOT = _public_path('experiments/archive/compact_experience')
PER_TARGET = _public_path('experiments/archive/compact_experience/data/official_geometry_full_eval_candidates/per_target')
TARGETS_JSON = _public_path('experiments/archive/compact_experience/data/official_geometry_full_eval_targets.json')
RANKING = _public_path('experiments/archive/compact_experience/data/official_geometry_trajectory_ranking.json')
CLEAN_SUMMARY = _public_path('experiments/archive/compact_experience/data/full_eval/full_eval_summary.json')
OUT = _public_path('experiments/archive/compact_experience/data/official_geometry_full_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/official_geometry_full_eval_interpretation.md')
LEADER_OVERALL = 41.8
KEYS9 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "SuperGLUE", "Reading", "AoA"]
KEYS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
RECOVERY = ["BLiMP", "EWoK", "COMPS", "Reading"]
PRESERVE = ["Supplement", "Entity", "SuperGLUE"]


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_scores(scores: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in KEYS9:
        v = scores.get(k)
        out[k] = None if v is None else float(v)
    return out


def overall_block(d: Dict[str, Any]) -> Dict[str, Any]:
    return d.get("official_overall") or {}


def extract_scores(d: Dict[str, Any]) -> Dict[str, Optional[float]]:
    o = overall_block(d)
    return normalize_scores(o.get("scores") or o.get("task_scores") or d.get("scores") or d.get("task_scores") or {})


def extract_overall(d: Dict[str, Any]) -> Optional[float]:
    o = overall_block(d)
    v = o.get("Overall")
    if v is None:
        v = o.get("official_style_overall")
    if v is None:
        v = d.get("Overall") or d.get("overall") or d.get("official_style_overall")
    return None if v is None else float(v)


def mean_available(scores: Dict[str, Optional[float]], keys: Iterable[str]) -> Optional[float]:
    vals = [scores.get(k) for k in keys]
    if any(v is None for v in vals):
        return None
    return mean(float(v) for v in vals)  # type: ignore[arg-type]


def load_clean() -> Dict[str, Any]:
    d = read_json(CLEAN_SUMMARY) if CLEAN_SUMMARY.exists() else {}
    targets = d.get("targets") if isinstance(d.get("targets"), dict) else d
    rec = targets.get("qwen_clean_aligned", {}) if isinstance(targets, dict) else {}
    if not rec:
        return {"target": "qwen_clean_aligned", "overall": 41.34429066479573, "scores": {}}
    scores = normalize_scores(rec.get("scores") or rec)
    return {"target": "qwen_clean_aligned", "overall": float(rec.get("Overall", rec.get("overall", 41.34429066479573))), "scores": scores}


def row_from_file(path: pathlib.Path) -> Dict[str, Any]:
    d = read_json(path)
    scores = extract_scores(d)
    overall = extract_overall(d)
    row: Dict[str, Any] = {
        "target": str(d.get("target") or path.stem),
        "endpoint": str(d.get("endpoint") or ""),
        "family": str(d.get("family") or ""),
        "overall": overall,
        "scores": scores,
        "equal7_from_full_eval": mean_available(scores, KEYS7),
        "recovery_R": mean_available(scores, RECOVERY),
        "preserve_P": mean_available(scores, PRESERVE),
        "complete_for_provisional_overall": overall_block(d).get("complete_for_provisional_overall"),
        "submit_ready_aoa": overall_block(d).get("submit_ready_aoa"),
        "aoa_raw_correlation": overall_block(d).get("aoa_raw_correlation"),
        "aoa_leaderboard_score": overall_block(d).get("aoa_leaderboard_score"),
    }
    if row["equal7_from_full_eval"] is not None:
        row["required_superglue_plus_aoa_to_exceed_41p8_from_equal7"] = 9 * LEADER_OVERALL - 7 * float(row["equal7_from_full_eval"])
    if overall is not None:
        row["delta_vs_visible_41p8_leader"] = overall - LEADER_OVERALL
    return row


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"a": a.get("target"), "b": b.get("target"), "a_minus_b_overall": None}
    if a.get("overall") is not None and b.get("overall") is not None:
        out["a_minus_b_overall"] = float(a["overall"]) - float(b["overall"])
    for field in ["equal7_from_full_eval", "recovery_R", "preserve_P"]:
        if a.get(field) is not None and b.get(field) is not None:
            out[f"delta_{field}"] = float(a[field]) - float(b[field])
    for k in KEYS9:
        av = (a.get("scores") or {}).get(k)
        bv = (b.get("scores") or {}).get(k)
        if av is not None and bv is not None:
            out[f"delta_{k}"] = float(av) - float(bv)
    return out


def main() -> None:
    rows = [row_from_file(p) for p in sorted(PER_TARGET.glob("*.json"))] if PER_TARGET.exists() else []
    clean = load_clean()
    ranked = sorted([r for r in rows if r.get("overall") is not None], key=lambda r: float(r["overall"]), reverse=True)
    by_target = {r["target"]: r for r in rows}
    pair_diffs: List[Dict[str, Any]] = []
    for suffix in ["10M", "20M", "30M", "40M", "50M", "60M", "70M", "75M", "80M", "85M", "90M", "95M", "100M"]:
        a = by_target.get(f"official_cap120geom_b256_{suffix}")
        b = by_target.get(f"official160_b256_{suffix}")
        if a and b:
            pair_diffs.append(diff(a, b))
    payload = {
        "status": "OFFICIAL_GEOMETRY_FULL_EVAL_SUMMARY",
        "interpretation_scope": "Reads completed corrected full-eval files for frozen no-AoA selected official-geometry checkpoints. AoA is final measurement only.",
        "visible_leader_overall": LEADER_OVERALL,
        "per_target_dir": str(PER_TARGET),
        "ranking_source": str(RANKING),
        "selected_targets_source": str(TARGETS_JSON),
        "selected_targets": read_json(TARGETS_JSON) if TARGETS_JSON.exists() else {},
        "clean_reference": clean,
        "rows": rows,
        "ranked_all_by_overall": ranked,
        "candidate_minus_clean_reference": [diff(r, clean) for r in rows if r.get("overall") is not None],
        "cap120geom_minus_official160_same_checkpoint": pair_diffs,
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research official-geometry full-eval interpretation",
        "",
        payload["interpretation_scope"],
        "",
        f"Visible leader reference: {LEADER_OVERALL:.4f}",
        f"Clean-Qwen reference: {clean.get('overall')}",
        "",
        "## Ranked official-geometry targets by corrected Overall",
    ]
    if not ranked:
        lines.append("No completed official-geometry full-eval per-target JSONs were found yet.")
    for i, r in enumerate(ranked, 1):
        s = r["scores"]
        lines.append(
            f"{i}. `{r['target']}` `{r['endpoint']}` family={r['family']} Overall={float(r['overall']):.6f} "
            f"delta_vs_41.8={float(r.get('delta_vs_visible_41p8_leader', 0.0)):.6f} "
            f"equal7={r.get('equal7_from_full_eval')} R={r.get('recovery_R')} P={r.get('preserve_P')} "
            f"SuperGLUE={s.get('SuperGLUE')} AoA={s.get('AoA')} need_SG_plus_AoA>{r.get('required_superglue_plus_aoa_to_exceed_41p8_from_equal7')}"
        )
    lines += ["", "## Candidate minus clean-Qwen reference"]
    for d in payload["candidate_minus_clean_reference"]:
        lines.append("- " + json.dumps(d, ensure_ascii=False, sort_keys=True))
    lines += ["", "## cap120 official geometry minus official160 same checkpoint"]
    for d in pair_diffs:
        lines.append("- " + json.dumps(d, ensure_ascii=False, sort_keys=True))
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "best": ranked[:1]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
