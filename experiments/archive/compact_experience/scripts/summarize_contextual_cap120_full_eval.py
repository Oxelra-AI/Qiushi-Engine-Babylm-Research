#!/usr/bin/env python3
"""Summarize corrected full-eval results for contextual cap-120 candidates.

This script is for after no-AoA trajectory selection and corrected full nine-column
evaluation. It reads finalized per-target JSONs, compares treatment/control and the
current clean-Qwen coordinate, and computes the exact remaining gap to the visible
41.8 leader. It never selects training data or checkpoints; AoA is only interpreted
as a final measurement from already-frozen full evals.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = _public_path('experiments/archive/compact_experience')
PER_TARGET = _public_path('experiments/archive/compact_experience/data/full_eval_candidates/per_target')
RANKING = _public_path('experiments/archive/compact_experience/data/contextual_cap120_trajectory_ranking.json')
TARGETS_JSON = _public_path('experiments/archive/compact_experience/data/contextual_cap120_full_eval_targets.json')
CLEAN_SUMMARY = _public_path('experiments/archive/compact_experience/data/full_eval/full_eval_summary.json')
OUT = _public_path('experiments/archive/compact_experience/data/contextual_cap120_full_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/contextual_cap120_full_eval_interpretation.md')
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


def target_scores(d: Dict[str, Any]) -> Dict[str, Optional[float]]:
    o = d.get("official_overall") or {}
    scores = o.get("scores") or o.get("task_scores") or d.get("scores") or d.get("task_scores") or {}
    return normalize_scores(scores)


def official_overall(d: Dict[str, Any]) -> Optional[float]:
    o = d.get("official_overall") or {}
    v = o.get("Overall")
    if v is None:
        v = o.get("official_style_overall")
    if v is None:
        v = d.get("Overall") or d.get("overall") or d.get("official_style_overall")
    return None if v is None else float(v)


def endpoint_name(target: str) -> Optional[str]:
    # context_cap120_treat_seed43022_75M -> chck_75M
    tail = target.rsplit("_", 1)[-1]
    if tail.endswith("M") and tail[:-1].isdigit():
        return f"chck_{tail}"
    return None


def mean_available(scores: Dict[str, Optional[float]], keys: Iterable[str]) -> Optional[float]:
    vals = [scores.get(k) for k in keys]
    if any(v is None for v in vals):
        return None
    return mean(float(v) for v in vals)  # type: ignore[arg-type]


def load_clean_best() -> Dict[str, Any]:
    if not CLEAN_SUMMARY.exists():
        return {"target": "qwen_clean_aligned", "overall": 41.34429066479573, "scores": {}}
    d = read_json(CLEAN_SUMMARY)
    # research summaries have changed structure across scorer repairs; recover flexibly.
    candidate_maps = [d]
    if isinstance(d.get("targets"), dict):
        candidate_maps.append(d["targets"])
    for mapping in candidate_maps:
        for key in ["qwen_clean_aligned", "qwen", "aligned"]:
            rec = mapping.get(key)
            if isinstance(rec, dict):
                o = rec.get("official_overall") or rec
                scores = o.get("scores") or o.get("task_scores") or rec.get("scores") or rec.get("task_scores") or rec
                overall = o.get("Overall") or o.get("official_style_overall") or rec.get("Overall") or rec.get("overall") or rec.get("official_style_overall")
                if overall is not None:
                    return {"target": key, "overall": float(overall), "scores": normalize_scores(scores)}
    # fallback: scan rows/lists
    def walk(x: Any) -> Optional[Dict[str, Any]]:
        if isinstance(x, dict):
            name = str(x.get("target") or x.get("name") or "")
            if "qwen_clean_aligned" in name:
                o = x.get("official_overall") or x
                scores = o.get("scores") or o.get("task_scores") or x.get("scores") or x.get("task_scores") or x
                overall = o.get("Overall") or o.get("official_style_overall") or x.get("Overall") or x.get("overall") or x.get("official_style_overall")
                if overall is not None:
                    return {"target": name, "overall": float(overall), "scores": normalize_scores(scores)}
            for v in x.values():
                y = walk(v)
                if y:
                    return y
        elif isinstance(x, list):
            for v in x:
                y = walk(v)
                if y:
                    return y
        return None
    found = walk(d)
    return found or {"target": "qwen_clean_aligned", "overall": 41.34429066479573, "scores": {}}


def row_from_file(path: pathlib.Path) -> Dict[str, Any]:
    d = read_json(path)
    target = str(d.get("target") or path.stem)
    scores = target_scores(d)
    row: Dict[str, Any] = {
        "target": target,
        "endpoint": str(d.get("endpoint") or endpoint_name(target) or ""),
        "family": str(d.get("family") or ""),
        "overall": official_overall(d),
        "scores": scores,
        "equal7_from_full_eval": mean_available(scores, KEYS7),
        "recovery_R": mean_available(scores, RECOVERY),
        "preserve_P": mean_available(scores, PRESERVE),
        "complete_for_provisional_overall": (d.get("official_overall") or {}).get("complete_for_provisional_overall"),
        "submit_ready_aoa": (d.get("official_overall") or {}).get("submit_ready_aoa"),
        "aoa_raw_correlation": (d.get("official_overall") or {}).get("aoa_raw_correlation"),
        "aoa_leaderboard_score": (d.get("official_overall") or {}).get("aoa_leaderboard_score"),
    }
    eq7 = row["equal7_from_full_eval"]
    if eq7 is not None:
        row["required_superglue_plus_aoa_to_exceed_41p8_from_equal7"] = 9 * LEADER_OVERALL - 7 * float(eq7)
    else:
        row["required_superglue_plus_aoa_to_exceed_41p8_from_equal7"] = None
    if row["overall"] is not None:
        row["delta_vs_visible_41p8_leader"] = float(row["overall"]) - LEADER_OVERALL
    return row


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"a": a["target"], "b": b["target"], "a_minus_b_overall": None}
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
    rows: List[Dict[str, Any]] = []
    if PER_TARGET.exists():
        for p in sorted(PER_TARGET.glob("context_cap120_*.json")):
            rows.append(row_from_file(p))
    by_target = {r["target"]: r for r in rows}
    treatment_rows = [r for r in rows if "_treat_" in r["target"] and r.get("overall") is not None]
    control_rows = [r for r in rows if "_control_" in r["target"] and r.get("overall") is not None]
    ranked_treat = sorted(treatment_rows, key=lambda r: float(r["overall"]), reverse=True)
    ranked_all = sorted([r for r in rows if r.get("overall") is not None], key=lambda r: float(r["overall"]), reverse=True)
    clean = load_clean_best()

    matched_diffs: List[Dict[str, Any]] = []
    for t in treatment_rows:
        ep = t.get("endpoint")
        if not ep:
            continue
        m = ep.replace("chck_", "")
        cname = f"context_cap120_control_seed43022_{m}"
        c = by_target.get(cname)
        if c:
            matched_diffs.append(diff(t, c))
    clean_diffs = [diff(t, clean) for t in treatment_rows]

    targets = read_json(TARGETS_JSON) if TARGETS_JSON.exists() else {}
    ranking = read_json(RANKING) if RANKING.exists() else {}
    payload = {
        "status": "CONTEXTUAL_CAP120_FULL_EVAL_SUMMARY",
        "interpretation_scope": "Reads only completed corrected full-eval files for frozen no-AoA selected targets. AoA is final measurement only, not a route selector.",
        "per_target_dir": str(PER_TARGET),
        "ranking_source": str(RANKING),
        "selected_targets_source": str(TARGETS_JSON),
        "visible_leader_overall": LEADER_OVERALL,
        "clean_reference": clean,
        "selected_targets": targets,
        "no_aoa_ranking_best_treatment_equal7": ranking.get("best_treatment_equal7"),
        "rows": rows,
        "ranked_all_by_overall": ranked_all,
        "ranked_treatments_by_overall": ranked_treat,
        "matched_treatment_minus_control": matched_diffs,
        "treatment_minus_clean_reference": clean_diffs,
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research contextual cap-120 full-eval interpretation",
        "",
        payload["interpretation_scope"],
        "",
        f"Visible leader reference: {LEADER_OVERALL:.4f}",
        f"Clean-Qwen reference: {clean.get('overall')}",
        "",
        "## Ranked contextual targets by corrected Overall",
    ]
    if not ranked_all:
        lines.append("No completed contextual cap-120 full-eval per-target JSONs were found yet.")
    for i, r in enumerate(ranked_all, 1):
        s = r["scores"]
        lines.append(
            f"{i}. `{r['target']}` `{r['endpoint']}` Overall={float(r['overall']):.6f} "
            f"delta_vs_41.8={float(r.get('delta_vs_visible_41p8_leader', 0.0)):.6f} "
            f"equal7={r.get('equal7_from_full_eval')} R={r.get('recovery_R')} P={r.get('preserve_P')} "
            f"SuperGLUE={s.get('SuperGLUE')} AoA={s.get('AoA')} "
            f"need_SG_plus_AoA>{r.get('required_superglue_plus_aoa_to_exceed_41p8_from_equal7')}"
        )
    lines += ["", "## Treatment minus matched control"]
    for d in matched_diffs:
        lines.append("- " + json.dumps(d, ensure_ascii=False, sort_keys=True))
    lines += ["", "## Treatment minus clean-Qwen reference"]
    for d in clean_diffs:
        lines.append("- " + json.dumps(d, ensure_ascii=False, sort_keys=True))
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "best_treatment": ranked_treat[:1], "best_any": ranked_all[:1]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
