#!/usr/bin/env python3
"""research: build fast-screen payload wrapper for dense-focus seed62065 replication.

Seed62065 used the same dense unchanged-Qwen focus policy as seed62064 but a different
training seed.  This script wraps the research/research fast-screen predictions into the
same per-target-like payload shape used in research, so the standard research transition
comparator can compare seed62065 against coherent86 and seed62064 on identical fast
items.  This is not an official full-eval payload and has no SuperGLUE/AoA.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from typing import Any, Dict

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
OUT_ROOT = _public_path('experiments/archive/functional_learning/data/dense_seed62065_fast_payloads')
STRICT_INITIAL_MODEL_STUDIES = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
FAST = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval')
FULL = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')

EVAL = _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_eval/cheap7/unchanged_correspondence_focus_weighted_u0080/cheap7/update_0080_eval.json')
OUTPUT = _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_eval/cheap7/unchanged_correspondence_focus_weighted_u0080/cheap7/outputs/update_0080')
MODEL = _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080')

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
DATA_PATHS = {
    "BLiMP": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/blimp_fast'),
    "Supplement": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/supplement_fast'),
    "EWoK": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast'),
    "Entity": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/entity_tracking_fast'),
    "COMPS": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/comps'),
    "GlobalPIQA_parallel": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/global_piqa_parallel'),
    "GlobalPIQA_nonparallel": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/global_piqa_nonparallel'),
    "Reading": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval/reading/reading_data.csv'),
}
TASK_NAMES = {
    "BLiMP": "blimp",
    "Supplement": "blimp",
    "EWoK": "ewok",
    "Entity": "entity_tracking",
    "COMPS": "comps",
    "GlobalPIQA_parallel": "global_piqa_parallel",
    "GlobalPIQA_nonparallel": "global_piqa_nonparallel",
    "Reading": "reading",
}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def one_match(root: pathlib.Path, column: str, filename: str) -> pathlib.Path:
    hits = sorted((root / column).rglob(filename))
    if len(hits) != 1:
        raise RuntimeError(f"Expected one {filename} for {column} under {root}, found {len(hits)}: {[rel(h) for h in hits[:8]]}")
    return hits[0]


def make_payload() -> Dict[str, Any]:
    ev = load_json(EVAL)
    scores = ev.get("scores") or {}
    tasks: Dict[str, Any] = {}
    for col in COLUMNS:
        pred = one_match(OUTPUT, col, "predictions.json")
        report = one_match(OUTPUT, col, "report.txt" if col == "Reading" else "best_temperature_report.txt")
        rec: Dict[str, Any] = {
            "column": col,
            "task": TASK_NAMES[col],
            "data_path": rel(DATA_PATHS[col]),
            "predictions": rel(pred),
            "report": rel(report),
            "fast_screen_not_official_full_eval": True,
            "source_eval_json": rel(EVAL),
        }
        if col == "Reading":
            rec["scores"] = {
                "Reading_eye": scores.get("Reading_eye"),
                "Reading_self_paced": scores.get("Reading_self_paced"),
                "Reading": scores.get("Reading"),
            }
        else:
            rec["score"] = scores.get(col)
        tasks[col] = rec
    return {
        "target": "dense_focus_seed62065_u0080_fast",
        "description": "Dense unchanged-Qwen focus seed62065 update80 fixed-policy replication research fast-screen payload wrapper",
        "family": "fast_screen_payload_wrapper",
        "model_path": rel(MODEL),
        "endpoint": "fast_screen_wrapped_existing_outputs",
        "created_utc": now(),
        "tasks": tasks,
        "scores": scores,
        "source_eval_json": rel(EVAL),
        "output_root": rel(OUTPUT),
        "interpretation": "Wrapped research/research fast-screen predictions for item-level seed replication comparison. This is not the official full-eval coordinate and has no SuperGLUE or AoA.",
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = make_payload()
    payload_path = _public_path('experiments/archive/functional_learning/data/dense_seed62065_fast_payloads/dense_focus_seed62065_u0080_fast_payload.json')
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "status": "DENSE_SEED62065_FAST_PAYLOAD_READY",
        "created_utc": now(),
        "payload": rel(payload_path),
        "model_path": rel(MODEL),
        "eval_json": rel(EVAL),
        "predictions": {c: payload["tasks"][c]["predictions"] for c in COLUMNS},
        "scores": payload.get("scores"),
        "note": "Fast-screen replication payload only; compare official validation separately.",
    }
    (_public_path('experiments/archive/functional_learning/data/dense_seed62065_fast_payloads/manifest.json')).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
