#!/usr/bin/env python3
"""Score research cluster-tail full-eval candidate payloads with corrected AoA units."""
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
PER = _public_path('experiments/archive/compact_experience/data/cluster_tail_full_eval/per_target')
OUT = _public_path('experiments/archive/compact_experience/data/cluster_tail_full_eval/cluster_tail_full_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/cluster_tail_full_eval_summary.md')
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
        "prefill_source": payload.get("prefill_source"),
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
    }


def main() -> None:
    if not PER.exists():
        raise FileNotFoundError(PER)
    rows = [score_target(p) for p in sorted(PER.glob("*.json"))]
    rows.sort(key=lambda r: (-10**9 if r["overall"] is None else -float(r["overall"])))
    payload = {
        "status": "CLUSTER_TAIL_FULL_EVAL_SUMMARY",
        "baseline_clean_qwen_100M": BASELINE,
        "visible_leader": LEADER,
        "candidates": rows,
        "note": "Corrected AoA units. For continuation artifacts lacking the full official AoA checkpoint ladder, AoA is provisional 0.0 and submit_ready_aoa is false.",
    }
    _public_path('experiments/archive/compact_experience/data/cluster_tail_full_eval').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research cluster-tail full evaluation summary",
        "",
        f"Clean-Qwen complete frontier: {BASELINE:.6f}; visible leader: {LEADER:.3f}.",
        "",
        "| target | endpoint | Overall | Δclean | Δleader | SuperGLUE | AoA lb | AoA status | submit-ready AoA | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r.get("task_scores") or {}
        lines.append(
            f"| {r['target']} | {r['endpoint']} | {r['overall'] if r['overall'] is not None else None} | {r['delta_vs_clean_best'] if r['delta_vs_clean_best'] is not None else None} | {r['delta_vs_visible_leader'] if r['delta_vs_visible_leader'] is not None else None} | {r['superglue_mean']} | {r['aoa_leaderboard_score']} | {r['aoa_status']} | {r['submit_ready_aoa']} | {s.get('BLiMP')} | {s.get('Supplement')} | {s.get('EWoK')} | {s.get('Entity')} | {s.get('COMPS')} | {s.get('GlobalPIQA')} | {s.get('Reading')} |"
        )
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "candidates": [{"target": r["target"], "overall": r["overall"], "delta_vs_clean_best": r["delta_vs_clean_best"], "delta_vs_visible_leader": r["delta_vs_visible_leader"]} for r in rows]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
