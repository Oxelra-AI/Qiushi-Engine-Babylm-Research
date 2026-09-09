#!/usr/bin/env python3
"""Score dynamic endpoint full-eval payloads with corrected BabyLM Overall units."""
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
PER = _public_path('experiments/archive/compact_experience/data/custom_endpoint_full_eval/per_target')
OUT = _public_path('experiments/archive/compact_experience/data/custom_endpoint_full_eval/custom_endpoint_full_eval_summary.json')
NOTE = _public_path('research/notes/compact_experience/custom_endpoint_full_eval_summary.md')
VISIBLE_LEADER = 41.8
CLEAN43022 = 41.34429066479573
CLEAN43122 = 40.65005195633467


def endpoint_m(endpoint: str | None) -> int | None:
    if not endpoint or not endpoint.startswith("chck_") or not endpoint.endswith("M"):
        return None
    try:
        return int(endpoint[len("chck_"):-1])
    except Exception:
        return None


def score_target(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", {})
    if isinstance(tasks.get("AoA"), dict):
        tasks["AoA"] = normalize_aoa_record(tasks["AoA"])
    oo = payload.get("official_overall")
    if not isinstance(oo, dict) or oo.get("Overall") is None:
        oo = compute_overall_from_tasks(tasks)
    ep_m = endpoint_m(payload.get("endpoint"))
    endpoint_frozen = ep_m is not None and ep_m < 100
    if endpoint_frozen and isinstance(oo, dict):
        oo["aoa_status"] = "endpoint_frozen_official_code_compatible_aoa_done" if oo.get("aoa_status") == "official_aoa_done" else oo.get("aoa_status")
        oo["submit_ready_aoa"] = False
        oo["submit_ready_overall"] = False
        payload["official_overall"] = oo
        payload["endpoint_frozen_submission_caveat"] = "For endpoints earlier than 100M, later strict-small AoA names may be plateaued at endpoint weights. This is official-code-compatible endpoint-frozen measurement, not an ordinary submission-ready complete training ladder unless official server accepts it."
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sg = tasks.get("SuperGLUE", {})
    return {
        "target": payload.get("target"),
        "endpoint": payload.get("endpoint"),
        "endpoint_frozen": endpoint_frozen,
        "model_path": payload.get("model_path"),
        "model_root": payload.get("model_root"),
        "overall": oo.get("Overall"),
        "complete_for_provisional_overall": oo.get("complete_for_provisional_overall"),
        "submit_ready_aoa": oo.get("submit_ready_aoa"),
        "submit_ready_overall": oo.get("submit_ready_overall"),
        "aoa_status": oo.get("aoa_status"),
        "aoa_raw_correlation": oo.get("aoa_raw_correlation"),
        "aoa_leaderboard_score": oo.get("aoa_leaderboard_score"),
        "superglue_mean": sg.get("superglue_mean") if isinstance(sg, dict) else None,
        "task_scores": oo.get("scores"),
        "delta_vs_clean43022": (oo.get("Overall") - CLEAN43022) if oo.get("Overall") is not None else None,
        "delta_vs_clean43122": (oo.get("Overall") - CLEAN43122) if oo.get("Overall") is not None else None,
        "delta_vs_visible_leader": (oo.get("Overall") - VISIBLE_LEADER) if oo.get("Overall") is not None else None,
        "payload_path": str(path),
    }


def main() -> None:
    if not PER.exists():
        raise FileNotFoundError(PER)
    rows = [score_target(p) for p in sorted(PER.glob("*.json"))]
    rows.sort(key=lambda r: (-1e9 if r["overall"] is None else -float(r["overall"])))
    payload = {"status": "CUSTOM_ENDPOINT_FULL_EVAL_SUMMARY", "visible_leader": VISIBLE_LEADER, "clean43022_overall": CLEAN43022, "clean43122_overall": CLEAN43122, "candidates": rows}
    _public_path('experiments/archive/compact_experience/data/custom_endpoint_full_eval').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research custom endpoint full evaluation", "",
        f"Visible leader: {VISIBLE_LEADER:.3f}; clean seed43022: {CLEAN43022:.6f}; clean seed43122: {CLEAN43122:.6f}.", "",
        "| target | endpoint | Overall | Δleader | SuperGLUE | AoA raw | AoA lb | submit-ready | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r.get("task_scores") or {}
        lines.append(f"| {r['target']} | {r['endpoint']} | {r['overall']} | {r['delta_vs_visible_leader']} | {r['superglue_mean']} | {r['aoa_raw_correlation']} | {r['aoa_leaderboard_score']} | {r['submit_ready_overall']} | {s.get('BLiMP')} | {s.get('Supplement')} | {s.get('EWoK')} | {s.get('Entity')} | {s.get('COMPS')} | {s.get('GlobalPIQA')} | {s.get('Reading')} |")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "candidates": [{"target": r["target"], "endpoint": r["endpoint"], "overall": r["overall"], "delta_vs_visible_leader": r["delta_vs_visible_leader"], "submit_ready_overall": r["submit_ready_overall"]} for r in rows]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
