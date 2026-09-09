#!/usr/bin/env python3
"""Score research mask full-eval payloads with corrected AoA units."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
from babylm_official_scoring import compute_overall_from_tasks, normalize_aoa_record  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
PER = _public_path('experiments/archive/compact_experience/data/mask_full_eval/per_target')
OUT = _public_path('experiments/archive/compact_experience/data/mask_full_eval/mask_full_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/mask_full_eval_summary.md')
BASELINE = 41.34429066479573
LEADER = 41.8


def score_target(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", {})
    if isinstance(tasks.get("AoA"), dict):
        tasks["AoA"] = normalize_aoa_record(tasks["AoA"])
    oo = payload.get("official_overall")
    if not isinstance(oo, dict) or oo.get("Overall") is None:
        oo = compute_overall_from_tasks(tasks)
    sg = tasks.get("SuperGLUE", {})
    return {
        "target": payload.get("target"),
        "endpoint": payload.get("endpoint"),
        "model_path": payload.get("model_path"),
        "overall": oo.get("Overall"),
        "complete": oo.get("complete_for_provisional_overall"),
        "submit_ready_aoa": oo.get("submit_ready_aoa"),
        "submit_ready_overall": oo.get("submit_ready_overall"),
        "aoa_status": oo.get("aoa_status"),
        "aoa_raw_correlation": oo.get("aoa_raw_correlation"),
        "aoa_leaderboard_score": oo.get("aoa_leaderboard_score"),
        "superglue_mean": sg.get("superglue_mean") if isinstance(sg, dict) else None,
        "task_scores": oo.get("scores"),
        "delta_vs_clean_best": (oo.get("Overall") - BASELINE) if oo.get("Overall") is not None else None,
        "delta_vs_visible_leader": (oo.get("Overall") - LEADER) if oo.get("Overall") is not None else None,
        "payload_path": str(path),
    }


def main() -> None:
    if not PER.exists():
        raise FileNotFoundError(PER)
    rows = [score_target(p) for p in sorted(PER.glob("*.json"))]
    rows.sort(key=lambda r: (-10**9 if r["overall"] is None else -float(r["overall"])))
    payload = {"status": "MASK_FULL_EVAL_SUMMARY", "baseline_clean_qwen_100M": BASELINE, "visible_leader": LEADER, "candidates": rows}
    _public_path('experiments/archive/compact_experience/data/mask_full_eval').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research mask full evaluation summary", "", f"Clean-Qwen complete frontier: {BASELINE:.6f}; visible leader: {LEADER:.3f}.", "",
        "| target | endpoint | Overall | Δclean | Δleader | SuperGLUE | AoA raw | AoA lb | submit-ready | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r.get("task_scores") or {}
        lines.append(f"| {r['target']} | {r['endpoint']} | {r['overall']} | {r['delta_vs_clean_best']} | {r['delta_vs_visible_leader']} | {r['superglue_mean']} | {r['aoa_raw_correlation']} | {r['aoa_leaderboard_score']} | {r['submit_ready_overall']} | {s.get('BLiMP')} | {s.get('Supplement')} | {s.get('EWoK')} | {s.get('Entity')} | {s.get('COMPS')} | {s.get('GlobalPIQA')} | {s.get('Reading')} |")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "candidates": [{"target": r["target"], "overall": r["overall"], "delta_vs_visible_leader": r["delta_vs_visible_leader"], "submit_ready_overall": r["submit_ready_overall"]} for r in rows]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
