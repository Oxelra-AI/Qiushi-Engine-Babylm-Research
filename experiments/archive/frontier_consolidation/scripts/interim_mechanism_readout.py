#!/usr/bin/env python3
"""research: compact interim readout after repairing CPU-safe scoring.

Reads only already-produced files.  It records the pre-declared mechanism note,
the CPU-safe worker proof, the newly measured MAX-repeat 100M EWoK value, and
the 80M sampled EWoK/Entity margin pilot.  It does not run evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
OUT = WS / "data/interim_mechanism_readout"

MAX_VIEW_100 = WS / "data/dose_ladder_stable_eval/eval/per_target/dose_max_view_chck_100M.json"
MAX_REPEAT_100 = WS / "data/dose_ladder_stable_eval/eval/per_target/dose_max_repeat_chck_100M.json"
PILOT_SUMMARY = WS / "data/ewok_entity_margin_pilot80/margin_readout_summary.json"
CPU_PROBE = WS / "data/cpu_safe_scoring_worker/cpu_probe_result.json"
PRED_NOTE = WS / "notes/binding_content_trade_predeclared_predictions.md"
WORKER = WS / "scripts/cpu_safe_scoring_worker.py"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def task_score(payload: dict[str, Any], col: str) -> float | None:
    rec = (payload.get("tasks") or {}).get(col)
    if not isinstance(rec, dict):
        return None
    if col == "Reading":
        scores = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
        val = scores.get("Reading") if scores else rec.get("score")
    else:
        val = rec.get("score")
    return float(val) if finite(val) else None


def ready_cols(payload: dict[str, Any]) -> list[str]:
    out = []
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]:
        rec = (payload.get("tasks") or {}).get(c)
        if isinstance(rec, dict) and rec.get("returncode") == 0 and finite(task_score(payload, c)):
            out.append(c)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    view = read_json(MAX_VIEW_100)
    repeat = read_json(MAX_REPEAT_100)
    pilot = read_json(PILOT_SUMMARY)
    probe = read_json(CPU_PROBE)
    ewok_view = task_score(view, "EWoK")
    ewok_repeat = task_score(repeat, "EWoK")
    pilot_delta = []
    for rec in pilot.get("delta_summaries", []):
        if rec.get("checkpoint") == "chck_80M" and rec.get("family") in {"EWoK", "Entity"}:
            pilot_delta.append({k: rec.get(k) for k in ["dose_name", "family", "n", "mean", "median", "frac_positive", "binary_delta_mean"]})
    summary = {
        "status": "INTERIM_MECHANISM_READOUT_COMPLETE",
        "created_utc": now(),
        "predeclared_prediction_note": rel(PRED_NOTE),
        "cpu_safe_worker": rel(WORKER),
        "cpu_probe": probe,
        "max_repeat_100m_current_ready_cols": ready_cols(repeat),
        "max_repeat_100m_missing_cols": [c for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"] if c not in ready_cols(repeat)],
        "max_100m_ewok": {
            "view": ewok_view,
            "repeat": ewok_repeat,
            "view_minus_repeat": (ewok_view - ewok_repeat) if ewok_view is not None and ewok_repeat is not None else None,
            "interpretation": "The newly measured repeat EWoK row makes full-exposure MAX EWoK currently slightly view-positive, unlike the negative average over the 10M-80M common window; the remaining Entity/COMPS/Reading columns are still needed for the late full-exposure fixed-budget readout.",
        },
        "margin_pilot80": {
            "summary_path": rel(PILOT_SUMMARY),
            "sample": {"ewok_per_domain": pilot.get("ewok_per_domain"), "entity_per_group": pilot.get("entity_per_group"), "ewok_rows_selected": 55, "entity_rows_selected": 90},
            "delta_summaries": pilot_delta,
            "interpretation": "In the small 80M balanced sample, Entity V-R margins are positive at all three doses and largest at MAX, while EWoK V-R margin moves from positive at 1x to near zero at 1.82x and negative at MAX. This supports using a larger sampled trajectory as a graded signed-family test, but it is not a full-dataset result.",
        },
        "running_cpu_tasks_started_by_step267": {
            "s267_t32_tool1": "finish remaining MAX-repeat 100M Entity/COMPS/Reading with CPU-safe official worker",
            "s267_t33_tool1": "finish RoBERTa clean 50M-100M stable-family remainder with CPU-safe official worker; 30M/40M COMPS+Reading still need later resumption",
            "s267_t38_tool1": "sampled EWoK/Entity continuous-margin 10M-80M dose ladder",
        },
        "still_needed_after_these_tasks": [
            "harvest and recompute dose decomposition after MAX-repeat 100M completes",
            "complete RoBERTa clean 30M/40M missing COMPS and Reading if not already recovered",
            "evaluate second-basin MAX view/repeat after trainings finish",
            "evaluate MAX breadth after training finishes and compare stable families plus duplicate-sensitive/entity-binding panels",
        ],
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    out_json = OUT / "interim_mechanism_readout_summary.json"
    out_md = OUT / "interim_mechanism_readout_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research interim mechanism readout",
        "",
        f"Pre-declared mechanism note: `{rel(PRED_NOTE)}`",
        f"CPU-safe worker: `{rel(WORKER)}`; probe status {probe.get('status')}, cpu_assert_seen={probe.get('cpu_assert_seen')}",
        "",
        "## MAX 100M EWoK now measured",
        f"- MAX view EWoK 100M: {ewok_view}",
        f"- MAX repeat EWoK 100M: {ewok_repeat}",
        f"- V-R: {summary['max_100m_ewok']['view_minus_repeat']}",
        f"- Current repeat-ready columns: {summary['max_repeat_100m_current_ready_cols']}; missing {summary['max_repeat_100m_missing_cols']}",
        "",
        "## 80M sampled margin pilot",
    ]
    for rec in pilot_delta:
        lines.append(f"- {rec['dose_name']} {rec['family']}: n={rec['n']} mean={rec['mean']} median={rec['median']} frac_positive={rec['frac_positive']} binary_delta_mean={rec['binary_delta_mean']}")
    lines += ["", summary["margin_pilot80"]["interpretation"], "", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary_json": rel(out_json), "summary_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
