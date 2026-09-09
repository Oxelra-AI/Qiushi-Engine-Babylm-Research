#!/usr/bin/env python3
"""Materialize a truthful prediction carrier for the research/151 coherent 86M fast-path endpoint.

Uses only prediction files already produced for the same coherent endpoint:
- zero-shot / reading predictions from research cheap7 evaluation
- SuperGLUE predictions from research coherent SuperGLUE evaluation
- scalar AoA=0.0, matching the current validator-accepted form and the anchor score
- no fast_eval_results borrowed from the protected chck_82M history
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
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
CHEAP_PAYLOAD = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json')
CHEAP_SUMMARY = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_summary/fastpath4M_coherent_summary.json')
SG_PAYLOAD = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_superglue_eval/per_target/fastpath4M_coherent.json')
SG_SUMMARY = _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json')
FULL_MODEL_VALIDATION = _public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation/fastpath_full_model_validation.json')
PROTECTED_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
ORIGINAL_MODEL_FILE = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final/model.safetensors')
REPLAY_MODEL_FILE = _public_path('experiments/archive/frontier_consolidation/training/runs/replay_frozen82_fastpath4M_coherent_seed43022/hf_model/final/model.safetensors')
ORIGINAL_METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/scientific_metrics.json')
REPLAY_METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/replay_frozen82_fastpath4M_coherent_seed43022/scientific_metrics.json')
ITEM_ANALYSIS = _public_path('experiments/archive/frontier_consolidation/data/fastpath_item_family_analysis/fastpath_item_family_analysis.json')
OVERLAP_ANALYSIS = _public_path('experiments/archive/frontier_consolidation/data/fastpath_flip_overlap/fastpath_flip_overlap.json')
OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/truthful_coherent86_carrier')
CARRIER = _public_path('experiments/archive/frontier_consolidation/data/truthful_coherent86_carrier/all_full_preds_truthful_coherent86_mlm.json')

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
        raise KeyError(f"Expected key {expected_single_key!r} not present in {path}; keys={list(obj)[:10]}")
    return obj


def import_validator():
    sys.path.insert(0, str(SPACE_REPO))
    from src.submission.check_validity import is_valid_predictions  # type: ignore
    return is_valid_predictions


def same_metrics_except_elapsed(a: dict[str, Any], b: dict[str, Any]) -> bool:
    aa = {k: v for k, v in a.items() if k != "elapsed_sec"}
    bb = {k: v for k, v in b.items() if k != "elapsed_sec"}
    return aa == bb


def materialize() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cheap_payload = read_json(CHEAP_PAYLOAD)
    cheap_summary = read_json(CHEAP_SUMMARY)
    sg_payload = read_json(SG_PAYLOAD)
    sg_summary = read_json(SG_SUMMARY)
    validation = read_json(FULL_MODEL_VALIDATION)
    protected = read_json(PROTECTED_VERIFY)
    item = read_json(ITEM_ANALYSIS)
    overlap = read_json(OVERLAP_ANALYSIS)
    orig_metrics = read_json(ORIGINAL_METRICS)
    replay_metrics = read_json(REPLAY_METRICS)

    carrier: dict[str, Any] = {}
    source_files: dict[str, str] = {}

    tasks = cheap_payload.get("tasks", {})
    for column, out_key in MAP_CHEAP.items():
        if column not in tasks:
            raise KeyError(f"Missing cheap task {column}")
        pred_path = ROOT / tasks[column]["predictions"]
        pred_obj = load_prediction_payload(pred_path)
        carrier[out_key] = pred_obj
        source_files[out_key] = rel(pred_path)

    sg_tasks = sg_payload.get("tasks", {}).get("SuperGLUE", {}).get("tasks", [])
    if not sg_tasks:
        raise RuntimeError("No SuperGLUE subtask records found in coherent SG payload")
    glue: dict[str, Any] = {}
    for rec in sg_tasks:
        subtask = str(rec["task"])
        pred_path = ROOT / rec["predictions"]
        pred_obj = load_prediction_payload(pred_path, expected_single_key=subtask)
        glue[subtask] = pred_obj[subtask]
        source_files[f"glue/{subtask}"] = rel(pred_path)
    carrier["glue"] = glue

    carrier["aoa"] = {"aoa": 0.0}
    # No fast_eval_results: the private-tail branch has a nonstandard post-82M history.

    CARRIER.write_text(json.dumps(carrier, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    carrier_sha = sha256_file(CARRIER)
    original_sha = sha256_file(ORIGINAL_MODEL_FILE)
    replay_sha = sha256_file(REPLAY_MODEL_FILE)
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
    protected_scores = protected["score_arithmetic"]["scores"]
    protected_overall = float(protected["score_arithmetic"]["overall_reported"])
    protected_cheap7 = float(mean(float(protected_scores[k]) for k in CHEAP7_COLUMNS))

    coherent_vs_anchor = item.get("comparisons", {}).get("coherent_minus_chck82", {})
    coherent_vs_shuffled = item.get("comparisons", {}).get("coherent_minus_shuffled86", {})
    all_overlap = overlap.get("summary_subsets", {}).get("all_discrete", {})
    ewok_overlap = overlap.get("summary_subsets", {}).get("EWoK_fragile_tagged", {})
    val_coherent = validation.get("results", {}).get("coherent", {})

    manifest = {
        "status": "TRUTHFUL_COHERENT86_CARRIER_MATERIALIZED" if valid and original_sha == replay_sha and same_metrics_except_elapsed(orig_metrics, replay_metrics) else "CARRIER_MATERIALIZED_WITH_ISSUE",
        "created_utc": now(),
        "carrier_path": rel(CARRIER),
        "carrier_sha256": carrier_sha,
        "carrier_size_bytes": CARRIER.stat().st_size,
        "model_path": rel(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final')),
        "replay_model_path": rel(_public_path('experiments/archive/frontier_consolidation/training/runs/replay_frozen82_fastpath4M_coherent_seed43022/hf_model/final')),
        "model_safetensors_sha256": original_sha,
        "replay_model_safetensors_sha256": replay_sha,
        "replay_bit_identical_to_original": original_sha == replay_sha,
        "metrics_identical_except_elapsed": same_metrics_except_elapsed(orig_metrics, replay_metrics),
        "validator": {
            "space_repo": rel(SPACE_REPO),
            "track": "strict-small",
            "is_valid_predictions": bool(valid),
            "message": message,
        },
        "truthful_history_policy": {
            "aoa": "scalar_zero_only; no borrowed chck82 AoA history",
            "fast_eval_results": "omitted; no borrowed chck82 fast checkpoint history",
            "superglue": "research coherent SuperGLUE prediction payload for the same coherent model",
            "zero_shot_and_reading": "research cheap7 prediction payloads for the same coherent model",
        },
        "score_arithmetic_candidate_native": {
            "scores": scores,
            "cheap7": cheap7,
            "overall_with_aoa0": overall,
            "protected_chck82_overall": protected_overall,
            "protected_chck82_cheap7": protected_cheap7,
            "delta_vs_protected_chck82_overall": float(overall - protected_overall),
            "delta_vs_protected_chck82_cheap7": float(cheap7 - protected_cheap7),
            "source": {
                "cheap_summary": rel(CHEAP_SUMMARY),
                "superglue_summary": rel(SG_SUMMARY),
                "full_model_validation": rel(FULL_MODEL_VALIDATION),
                "protected_chck82": rel(PROTECTED_VERIFY),
            },
        },
        "legal_and_function_summary": {
            "total_consumed_words": orig_metrics.get("total_consumed_words"),
            "tail_charged_words": orig_metrics.get("tail_charged_words"),
            "within_cap": orig_metrics.get("total_consumed_words", 10**12) <= orig_metrics.get("full_cap_words", 100000000),
            "trusted_class": val_coherent.get("trusted_class"),
            "parameter_count": val_coherent.get("trusted_total_params"),
            "private_params": val_coherent.get("private_params"),
            "max_nonprivate_diff_vs_chck82": val_coherent.get("max_nonprivate_diff_vs_chck82"),
            "private_off_matches_protected_max_abs_diff": val_coherent.get("private_off_vs_chck82_logit_maxdiff"),
            "private_on_vs_off_max_abs_diff": val_coherent.get("private_on_vs_off_logit_maxdiff_probe_text"),
        },
        "item_transition_scientific_reading": {
            "coherent_minus_chck82_aggregate": coherent_vs_anchor.get("aggregate"),
            "coherent_minus_shuffled86_aggregate": coherent_vs_shuffled.get("aggregate"),
            "all_discrete_overlap_vs_shuffled86": all_overlap,
            "ewok_fragile_overlap_vs_shuffled86": ewok_overlap,
            "interpretation": "Endpoint score is promising, but paired transitions show redistribution: coherent loses net discrete items vs chck82 and vs shuffled86, erodes EWoK/COMPS, and shares few gained decisions with shuffled86. This carrier is an endpoint candidate, not proof of a broad slow-fast learning principle.",
        },
        "source_prediction_files": source_files,
    }
    manifest_path = _public_path('experiments/archive/frontier_consolidation/data/truthful_coherent86_carrier/truthful_coherent86_carrier_manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# research truthful coherent86 carrier materialization",
        "",
        f"Status: **{manifest['status']}**",
        "",
        f"Carrier: `{rel(CARRIER)}`",
        f"Carrier SHA256: `{carrier_sha}`",
        f"Model: `{rel(_public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final'))}`",
        f"Model SHA256: `{original_sha}`",
        f"Replay model SHA256: `{replay_sha}`",
        f"Replay bit-identical: `{original_sha == replay_sha}`",
        "",
        "## Truthful history policy",
        "- AoA is represented only as scalar `{'aoa': 0.0}`.",
        "- `fast_eval_results` is omitted; the protected 82M fast checkpoint history is not copied.",
        "- Full-task predictions are merged from research zero-shot/reading outputs and research SuperGLUE outputs for the same coherent model.",
        "",
        "## Candidate-native arithmetic",
        "| column | score | delta vs chck82 |",
        "|---|---:|---:|",
    ]
    for k in OVERALL_COLUMNS:
        delta = scores[k] - float(protected_scores[k])
        md.append(f"| {k} | {scores[k]:.12g} | {delta:+.12g} |")
    md += [
        "",
        f"Cheap7: `{cheap7}` (delta `{cheap7 - protected_cheap7:+.12f}`)",
        f"Overall with AoA 0: `{overall}` (delta `{overall - protected_overall:+.12f}`)",
        "",
        "## Validator",
        f"`is_valid_predictions(..., strict-small)` -> `{valid}` / `{message}`",
        "",
        "## Scientific reading",
        "This is a truthful local prediction carrier for the coherent private-only replay endpoint candidate. It is stronger numerically than the submitted 82M anchor, but the item-transition and overlap analyses show family redistribution rather than a broad added-decision mechanism.",
        "",
        f"JSON: `{rel(manifest_path)}`",
    ]
    md_path = _public_path('research/documents/frontier_consolidation/data/truthful_coherent86_carrier/truthful_coherent86_carrier_manifest.md')
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "carrier_path": rel(CARRIER),
        "carrier_sha256": carrier_sha,
        "valid": bool(valid),
        "message": message,
        "model_sha256": original_sha,
        "replay_sha256": replay_sha,
        "overall_with_aoa0": overall,
        "delta_vs_chck82": overall - protected_overall,
        "manifest": rel(manifest_path),
        "manifest_md": rel(md_path),
    }, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    materialize()
