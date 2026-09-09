#!/usr/bin/env python3
"""research: materialize a truthful prediction carrier for the 86M shuffled private-tail endpoint.

This script uses only existing outputs from the already-scored 4M shuffled tail:
- zero-shot / reading predictions from research cheap7 evaluation
- repeat SuperGLUE predictions from research repeat evaluation
- scalar AoA=0.0, matching the current public validator's accepted score form

It deliberately does NOT borrow the protected scale1.75 chck_82M AoA histories or fast_eval_results,
because the 86M branch has a different checkpoint history. Missing fast histories are accepted by
the current Strict-Small validator; they should be reported as absent, not silently copied.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SPACE_REPO = _public_path('experiments/archive/frontier_consolidation/data/live_leaderboard_space_repo')
CHEAP_PAYLOAD = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json')
CHEAP_SUMMARY = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_summary/frozen82_tail4M_shuffled_summary.json')
SG_PAYLOAD = _public_path('experiments/archive/frontier_consolidation/data/repeat_shuffled_tail_superglue_eval/per_target/repeat_frozen82_tail4M_shuffled_superglue.json')
SG_SUMMARY = _public_path('experiments/archive/frontier_consolidation/data/repeat_shuffled_tail_superglue_summary/repeat_frozen82_tail4M_shuffled_superglue_superglue_summary.json')
VALIDATION = _public_path('experiments/archive/frontier_consolidation/data/shuffled_tail_endpoint_validation/shuffled_tail_endpoint_validation.json')
PROTECTED_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
MODEL_FILE = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final/model.safetensors')
OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/truthful_shuffled86_carrier')
CARRIER = _public_path('experiments/archive/frontier_consolidation/data/truthful_shuffled86_carrier/all_full_preds_truthful_shuffled86_mlm.json')

MAP_CHEAP = {
    "BLiMP": "blimp",
    "Supplement": "blimp_supplement",
    "EWoK": "ewok",
    "Entity": "entity_tracking_filtered",
    "COMPS": "comps",
    "GlobalPIQA_parallel": "global_piqa_parallel",
    "GlobalPIQA_nonparallel": "global_piqa_nonparallel",
    "Reading": "reading",
}
CHEAP7_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
OVERALL_COLUMNS = CHEAP7_COLUMNS + ["SuperGLUE", "AoA"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prediction_payload(path: pathlib.Path, expected_single_key: str | None = None) -> dict[str, Any]:
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Prediction payload is not an object: {path}")
    if expected_single_key is not None and expected_single_key not in obj:
        # Some official paths already hold the direct task object; keep a useful error.
        raise KeyError(f"Expected key {expected_single_key!r} not present in {path}; keys={list(obj)[:10]}")
    return obj


def import_validator():
    sys.path.insert(0, str(SPACE_REPO))
    # Use normal import after putting the downloaded Space root on sys.path.
    from src.submission.check_validity import is_valid_predictions  # type: ignore
    return is_valid_predictions


def materialize() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cheap_payload = read_json(CHEAP_PAYLOAD)
    cheap_summary = read_json(CHEAP_SUMMARY)
    sg_payload = read_json(SG_PAYLOAD)
    sg_summary = read_json(SG_SUMMARY)
    validation = read_json(VALIDATION)
    protected = read_json(PROTECTED_VERIFY)

    carrier: dict[str, Any] = {}
    source_files: dict[str, str] = {}

    # Zero-shot/text/reading predictions.
    tasks = cheap_payload.get("tasks", {})
    for column, out_key in MAP_CHEAP.items():
        if column not in tasks:
            raise KeyError(f"Missing cheap task {column}")
        pred_path = ROOT / tasks[column]["predictions"]
        pred_obj = load_prediction_payload(pred_path)
        carrier[out_key] = pred_obj
        source_files[out_key] = rel(pred_path)

    # SuperGLUE predictions: merge per-subtask JSON files under top-level glue.
    sg_tasks = sg_payload.get("tasks", {}).get("SuperGLUE", {}).get("tasks", [])
    if not sg_tasks:
        raise RuntimeError("No SuperGLUE subtask records found")
    glue: dict[str, Any] = {}
    for rec in sg_tasks:
        subtask = str(rec["task"])
        pred_path = ROOT / rec["predictions"]
        pred_obj = load_prediction_payload(pred_path, expected_single_key=subtask)
        glue[subtask] = pred_obj[subtask]
        source_files[f"glue/{subtask}"] = rel(pred_path)
    carrier["glue"] = glue

    # Truthful 86M AoA representation: scalar zero, no borrowed AoA histories.
    carrier["aoa"] = {"aoa": 0.0}
    # Truthful fast-history representation: omit fast_eval_results rather than borrowing chck_82M history.

    CARRIER.write_text(json.dumps(carrier, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    carrier_sha = sha256_file(CARRIER)
    model_sha = sha256_file(MODEL_FILE)

    # Current public Strict-Small validator accepts this full prediction carrier.
    is_valid_predictions = import_validator()
    valid, message = is_valid_predictions(str(CARRIER), "strict-small")

    cheap_scores = {k: float(v) for k, v in cheap_summary["scores"].items() if v is not None}
    sg_score = float(sg_summary["superglue"])
    scores = {
        "BLiMP": cheap_scores["BLiMP"],
        "Supplement": cheap_scores["Supplement"],
        "EWoK": cheap_scores["EWoK"],
        "Entity": cheap_scores["Entity"],
        "COMPS": cheap_scores["COMPS"],
        "GlobalPIQA": cheap_scores["GlobalPIQA"],
        "Reading": cheap_scores["Reading"],
        "SuperGLUE": sg_score,
        "AoA": 0.0,
    }
    overall = float(mean(scores[k] for k in OVERALL_COLUMNS))
    cheap7 = float(mean(scores[k] for k in CHEAP7_COLUMNS))
    protected_overall = float(protected["score_arithmetic"]["overall_reported"])

    manifest = {
        "status": "TRUTHFUL_SHUFFLED86_CARRIER_MATERIALIZED" if valid else "CARRIER_MATERIALIZED_BUT_VALIDATION_FAILED",
        "created_utc": now(),
        "carrier_path": rel(CARRIER),
        "carrier_sha256": carrier_sha,
        "carrier_size_bytes": CARRIER.stat().st_size,
        "model_path": rel(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final')),
        "model_safetensors_sha256": model_sha,
        "expected_model_sha256_from_step139": validation.get("tensor_checks", {}).get("tail_model_safetensors_sha256"),
        "validator": {
            "space_repo": rel(SPACE_REPO),
            "track": "strict-small",
            "is_valid_predictions": bool(valid),
            "message": message,
        },
        "truthful_history_policy": {
            "aoa": "scalar_zero_only; no borrowed chck82 AoA history",
            "fast_eval_results": "omitted; no borrowed chck82 fast checkpoint history",
            "superglue": "repeat research SuperGLUE prediction payload for the same shuffled86 model",
            "zero_shot_and_reading": "research prediction payloads for the same shuffled86 model",
        },
        "score_arithmetic_candidate_native": {
            "scores": scores,
            "cheap7": cheap7,
            "overall_with_aoa0": overall,
            "protected_chck82_overall": protected_overall,
            "delta_vs_protected_chck82": float(overall - protected_overall),
            "source": {
                "cheap_summary": rel(CHEAP_SUMMARY),
                "superglue_summary": rel(SG_SUMMARY),
                "endpoint_validation": rel(VALIDATION),
                "protected_chck82": rel(PROTECTED_VERIFY),
            },
        },
        "legal_and_function_summary_from_step139": {
            "total_consumed_words": validation.get("score_and_legal", {}).get("total_consumed_words"),
            "tail_charged_words": validation.get("score_and_legal", {}).get("tail_charged_words"),
            "within_cap": validation.get("score_and_legal", {}).get("within_legal_cap"),
            "trusted_class": validation.get("trusted_load_checks", {}).get("trusted_class"),
            "parameter_count": validation.get("trusted_load_checks", {}).get("trusted_param_count"),
            "private_off_matches_protected_max_abs_diff": validation.get("trusted_load_checks", {}).get("max_abs_logits_private_off_vs_protected"),
            "private_on_vs_off_max_abs_diff": validation.get("trusted_load_checks", {}).get("max_abs_logits_private_on_vs_off"),
        },
        "source_prediction_files": source_files,
    }
    manifest_path = _public_path('experiments/archive/frontier_consolidation/data/truthful_shuffled86_carrier/truthful_shuffled86_carrier_manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# research truthful shuffled86 carrier materialization",
        "",
        f"Status: **{manifest['status']}**",
        "",
        f"Carrier: `{rel(CARRIER)}`",
        f"Carrier SHA256: `{carrier_sha}`",
        f"Model: `{rel(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_tail4M_shuffled_seed43022/hf_model/final'))}`",
        f"Model SHA256: `{model_sha}`",
        "",
        "## Truthful history policy",
        "- AoA is represented only as scalar `{'aoa': 0.0}`.",
        "- `fast_eval_results` is omitted; the protected 82M fast checkpoint history is not copied.",
        "- Full-task predictions are merged from existing research zero-shot/reading outputs and research repeat SuperGLUE outputs for the same shuffled86 model.",
        "",
        "## Candidate-native arithmetic",
        "| column | score |",
        "|---|---:|",
    ]
    for k in OVERALL_COLUMNS:
        md.append(f"| {k} | {scores[k]:.12g} |")
    md += [
        "",
        f"Cheap7: `{cheap7}`",
        f"Overall with AoA 0: `{overall}`",
        f"Delta vs protected chck82: `{overall - protected_overall:+.12f}`",
        "",
        "## Current validator",
        f"`is_valid_predictions(..., strict-small)` -> `{valid}` / `{message}`",
        "",
        "Scientific reading: this carrier is a truthful local prediction artifact for the generic shuffled private-tail endpoint hypothesis. It is not evidence for source correspondence and it should be treated as a baseline for stronger frozen-anchor fast-path consolidation, not as a mechanism by itself.",
        "",
        f"JSON: `{rel(manifest_path)}`",
    ]
    md_path = _public_path('research/documents/frontier_consolidation/data/truthful_shuffled86_carrier/truthful_shuffled86_carrier_manifest.md')
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "carrier_path": rel(CARRIER),
        "carrier_sha256": carrier_sha,
        "valid": bool(valid),
        "message": message,
        "overall_with_aoa0": overall,
        "delta_vs_chck82": overall - protected_overall,
        "manifest": rel(manifest_path),
        "manifest_md": rel(md_path),
    }, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    materialize()
