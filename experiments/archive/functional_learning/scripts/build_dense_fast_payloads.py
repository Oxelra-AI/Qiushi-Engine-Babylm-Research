#!/usr/bin/env python3
"""research: build per-target-like payloads for dense-focus fast-screen predictions.

The official-style evaluator is running for the dense seed62064 checkpoint. The
available fast
screen already saved prediction files for coherent86 and dense focus; this script
wraps those files into a per_target-like JSON shape with explicit data paths so the
existing research transition comparator can make an item-level fast-screen table.

These payloads are NOT official full-eval payloads.  They use the fast_eval data for
BLiMP/Supplement/EWoK/Entity/GlobalPIQA and full_eval COMPS exactly as research did.
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
OUT_ROOT = _public_path('experiments/archive/functional_learning/data/dense_fast_payloads')
STRICT_INITIAL_MODEL_STUDIES = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
FAST = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/fast_eval')
FULL = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')

PARENT_EVAL = _public_path('experiments/archive/functional_learning/data/common_eval/coherent86_alpha075/coherent86_alpha075_eval.json')
PARENT_OUTPUT = _public_path('experiments/archive/functional_learning/data/common_eval/coherent86_alpha075/outputs/coherent86_alpha075')
DENSE_EVAL = _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_eval/cheap7/unchanged_correspondence_focus_weighted_u0080/cheap7/update_0080_eval.json')
DENSE_OUTPUT = _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_eval/cheap7/unchanged_correspondence_focus_weighted_u0080/cheap7/outputs/update_0080')
DENSE_MODEL = _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080')
PARENT_MODEL = _public_path('models/frontier')

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
        raise RuntimeError(f"Expected one {filename} for {column} under {root}, found {len(hits)}: {[rel(h) for h in hits[:5]]}")
    return hits[0]


def make_payload(label: str, eval_path: pathlib.Path, output_root: pathlib.Path, model_path: pathlib.Path, description: str) -> Dict[str, Any]:
    ev = load_json(eval_path)
    scores = ev.get("scores") or {}
    tasks: Dict[str, Any] = {}
    for col in COLUMNS:
        pred = one_match(output_root, col, "predictions.json")
        report_name = "report.txt" if col == "Reading" else "best_temperature_report.txt"
        report = one_match(output_root, col, report_name)
        rec: Dict[str, Any] = {
            "column": col,
            "task": TASK_NAMES[col],
            "data_path": rel(DATA_PATHS[col]),
            "predictions": rel(pred),
            "report": rel(report),
            "fast_screen_not_official_full_eval": True,
            "source_eval_json": rel(eval_path),
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
    payload: Dict[str, Any] = {
        "target": label,
        "description": description,
        "family": "fast_screen_payload_wrapper",
        "model_path": rel(model_path),
        "endpoint": "fast_screen_wrapped_existing_outputs",
        "created_utc": now(),
        "tasks": tasks,
        "scores": scores,
        "source_eval_json": rel(eval_path),
        "output_root": rel(output_root),
        "interpretation": "Wrapped research/research fast-screen predictions for immediate item-level comparison. This is not the official full-eval coordinate and has no SuperGLUE or AoA.",
    }
    return payload


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    parent = make_payload(
        "coherent86_alpha075_fast",
        PARENT_EVAL,
        PARENT_OUTPUT,
        PARENT_MODEL,
        "coherent86/v4 alpha0.75 research fast-screen payload wrapper",
    )
    dense = make_payload(
        "dense_focus_seed62064_u0080_fast",
        DENSE_EVAL,
        DENSE_OUTPUT,
        DENSE_MODEL,
        "Dense unchanged-Qwen focus seed62064 update80 research fast-screen payload wrapper",
    )
    parent_path = _public_path('experiments/archive/functional_learning/data/dense_fast_payloads/coherent86_alpha075_fast_payload.json')
    dense_path = _public_path('experiments/archive/functional_learning/data/dense_fast_payloads/dense_focus_seed62064_u0080_fast_payload.json')
    parent_path.write_text(json.dumps(parent, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dense_path.write_text(json.dumps(dense, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "status": "DENSE_FAST_PAYLOADS_READY",
        "created_utc": now(),
        "parent_payload": rel(parent_path),
        "dense_payload": rel(dense_path),
        "parent_predictions": {c: parent["tasks"][c]["predictions"] for c in COLUMNS},
        "dense_predictions": {c: dense["tasks"][c]["predictions"] for c in COLUMNS},
        "parent_scores": parent.get("scores"),
        "dense_scores": dense.get("scores"),
        "note": "Use these only for fast_eval item-level comparison while official-compatible dense evaluation is running.",
    }
    manifest_path = _public_path('experiments/archive/functional_learning/data/dense_fast_payloads/manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
