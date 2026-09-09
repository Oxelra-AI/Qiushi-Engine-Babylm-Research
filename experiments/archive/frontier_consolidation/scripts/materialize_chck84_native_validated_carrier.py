#!/usr/bin/env python3
"""research: build the chck_84M prediction carrier using native prediction validation only.

The carrier is assembled from endpoint-native prediction blocks:
- seven cheap-task blocks from the selected chck_84M evaluation payload,
- SuperGLUE blocks from the chck_84M SuperGLUE run,
- scalar AoA=0.0.

No fast_eval_results block is copied, no scorer requiring an alternate dataset schema is run,
and no leaderboard submission or model upload is performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SPACE_REPO = _public_path('experiments/archive/frontier_consolidation/data/live_leaderboard_space_repo')
EVAL_CACHE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/chck82_hf_public_bundle/live_submit_dry_run')

CHEAP_PAYLOAD = _public_path('experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/eval/per_target/scale1p75_seed43022_reference_chck_84M.json')
SG_PAYLOAD = _public_path('experiments/archive/frontier_consolidation/data/reference_chck84_superglue_eval/per_target/scale1p75_seed43022_reference_chck84_superglue.json')
SG_SUMMARY = _public_path('experiments/archive/frontier_consolidation/data/reference_chck84_superglue_summary/scale1p75_seed43022_reference_chck84_superglue_superglue_summary.json')
ENDPOINT_DECISION = _public_path('experiments/archive/frontier_consolidation/data/chck84_endpoint_decision_integrator/chck84_endpoint_decision_integrated_9defb15f.json')
MODEL_CARRIER = _public_path('experiments/archive/frontier_consolidation/data/chck84_hf_public_bundle/chck84_hf_public_bundle_validation.json')
PROTECTED_CHCK82 = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
IDENTITY = _public_path('experiments/archive/frontier_consolidation/data/checkpoint_identity_audit/chck_84M_identity_audit.json')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/chck84_native_validated_full_carrier')

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
        return str(p.resolve().relative_to(_public_path('.')))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) if compact else json.dumps(obj, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def load_prediction_payload(path: pathlib.Path, expected_single_key: str | None = None) -> dict[str, Any]:
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Prediction payload is not a JSON object: {path}")
    if expected_single_key is not None and expected_single_key not in obj:
        raise KeyError(f"Expected top-level key {expected_single_key!r} missing from {path}; keys={list(obj)[:12]}")
    return obj


def prediction_block_counts(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {"type": type(block).__name__}
    total = 0
    samples: dict[str, int] = {}
    for key, val in block.items():
        if isinstance(val, dict) and isinstance(val.get("predictions"), list):
            n = len(val["predictions"])
            total += n
            if len(samples) < 8:
                samples[key] = n
    return {"n_subkeys": len(block), "total_prediction_rows": total, "sample_subkey_row_counts": samples}


def import_native_prediction_validator():
    # The saved Space code reads path constants at import time.
    os.environ["HF_HOME"] = str(_public_path('experiments/archive/frontier_consolidation/data/chck82_hf_public_bundle/live_submit_dry_run'))
    if str(SPACE_REPO) not in sys.path:
        sys.path.insert(0, str(SPACE_REPO))
    from src.submission.check_validity import is_valid_predictions  # type: ignore
    return is_valid_predictions


def build_carrier() -> dict[str, Any]:
    cheap_payload = read_json(CHEAP_PAYLOAD)
    sg_payload = read_json(SG_PAYLOAD)

    carrier: dict[str, Any] = {}
    source_prediction_files: dict[str, dict[str, Any]] = {}

    for column, out_key in MAP_CHEAP.items():
        rec = cheap_payload.get("tasks", {}).get(column)
        if not isinstance(rec, dict) or not rec.get("predictions"):
            raise KeyError(f"chck_84M cheap payload missing prediction path for {column}")
        pred_path = ROOT / rec["predictions"]
        pred_obj = load_prediction_payload(pred_path)
        carrier[out_key] = pred_obj
        source_prediction_files[out_key] = {
            **file_record(pred_path),
            "source_column": column,
            "report_score": rec.get("score") if rec.get("score") is not None else rec.get("scores"),
        }

    sg_records = sg_payload.get("tasks", {}).get("SuperGLUE", {}).get("tasks", [])
    if not sg_records:
        raise RuntimeError("SuperGLUE payload has no task records")
    glue: dict[str, Any] = {}
    for rec in sg_records:
        task = str(rec["task"])
        pred_path = ROOT / rec["predictions"]
        pred_obj = load_prediction_payload(pred_path, expected_single_key=task)
        glue[task] = pred_obj[task]
        source_prediction_files[f"glue/{task}"] = {
            **file_record(pred_path),
            "source_column": "SuperGLUE",
            "metric_for_valid": rec.get("metric_for_valid"),
        }
    carrier["glue"] = glue
    carrier["aoa"] = {"aoa": 0.0}
    return {"carrier": carrier, "source_prediction_files": source_prediction_files}


def report_score_table(cheap_payload: dict[str, Any], sg_summary: dict[str, Any], endpoint: dict[str, Any], protected: dict[str, Any]) -> dict[str, Any]:
    cheap_scores = endpoint.get("created_from", {}).get("cheap_scores") or endpoint.get("chck84", {}).get("scores")
    if not isinstance(cheap_scores, dict):
        cheap_scores = {
            "BLiMP": cheap_payload["tasks"]["BLiMP"]["score"],
            "Supplement": cheap_payload["tasks"]["Supplement"]["score"],
            "EWoK": cheap_payload["tasks"]["EWoK"]["score"],
            "Entity": cheap_payload["tasks"]["Entity"]["score"],
            "COMPS": cheap_payload["tasks"]["COMPS"]["score"],
            "GlobalPIQA": (float(cheap_payload["tasks"]["GlobalPIQA_parallel"]["score"]) + float(cheap_payload["tasks"]["GlobalPIQA_nonparallel"]["score"])) / 2.0,
            "Reading": cheap_payload["tasks"]["Reading"]["scores"]["Reading"],
        }
    scores = {k: float(cheap_scores[k]) for k in CHEAP7_COLUMNS}
    scores["SuperGLUE"] = float(sg_summary["superglue"])
    scores["AoA"] = 0.0
    cheap7 = float(mean(scores[k] for k in CHEAP7_COLUMNS))
    overall = float(mean(scores[k] for k in OVERALL_COLUMNS))

    protected_scores = {k: float(v) for k, v in protected["score_arithmetic"]["scores"].items() if v is not None}
    protected_cheap7 = float(mean(protected_scores[k] for k in CHEAP7_COLUMNS))
    protected_overall = float(protected["score_arithmetic"]["overall_reported"])
    return {
        "scores": scores,
        "cheap7": cheap7,
        "overall_with_aoa0": overall,
        "deltas_vs_protected_chck82": {
            "cheap7": cheap7 - protected_cheap7,
            "superglue": scores["SuperGLUE"] - protected_scores["SuperGLUE"],
            "overall_with_aoa0": overall - protected_overall,
        },
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    carrier_path = out_dir / "all_full_preds_truthful_chck84_mlm.json"
    manifest_path = out_dir / "chck84_truthful_full_carrier_manifest.json"
    manifest_md = out_dir / "chck84_truthful_full_carrier_manifest.md"

    built = build_carrier()
    carrier = built["carrier"]
    write_json(carrier_path, carrier, compact=True)

    cheap_payload = read_json(CHEAP_PAYLOAD)
    sg_summary = read_json(SG_SUMMARY)
    endpoint = read_json(ENDPOINT_DECISION)
    model_carrier = read_json(MODEL_CARRIER)
    protected = read_json(PROTECTED_CHCK82)
    identity = read_json(IDENTITY)

    native_validator = import_native_prediction_validator()
    valid, message = native_validator(str(carrier_path), "strict-small")

    report_scores = report_score_table(cheap_payload, sg_summary, endpoint, protected)
    carrier_sha = sha256_file(carrier_path)
    block_counts = {k: prediction_block_counts(v) for k, v in carrier.items() if k != "aoa"}
    status = "CHCK84_NATIVE_VALIDATED_FULL_CARRIER_READY" if valid else "CHCK84_NATIVE_VALIDATED_FULL_CARRIER_REJECTED"

    manifest: dict[str, Any] = {
        "status": status,
        "created_utc": now(),
        "carrier_path": rel(carrier_path),
        "carrier_sha256": carrier_sha,
        "carrier_size_bytes": carrier_path.stat().st_size,
        "native_prediction_validation": {
            "space_repo": rel(SPACE_REPO),
            "track": "strict-small",
            "is_valid_predictions": bool(valid),
            "message": message,
            "eval_cache_root": rel(EVAL_CACHE_ROOT),
            "note": "Only the prediction-shape validator is run here. Scores are taken from the existing official-compatible local evaluation records because the direct Space scorer expects a different text-task dataset schema in this local snapshot.",
        },
        "prediction_contents": {
            "cheap_predictions": rel(CHEAP_PAYLOAD),
            "superglue_predictions": rel(SG_PAYLOAD),
            "aoa": "scalar_zero_only",
            "fast_eval_results": "omitted; no checkpoint-history block is copied from protected chck_82M or any other model",
        },
        "score_arithmetic_from_existing_official_compatible_records": report_scores,
        "model_identity": {
            "endpoint": "chck_84M",
            "checkpoint": identity.get("checkpoint") or identity.get("model_dir") or model_carrier.get("required_paths", {}).get("source_checkpoint", {}).get("path"),
            "model_safetensors_sha256": model_carrier.get("expected", {}).get("model_safetensors_sha256"),
            "tokenizer_json_sha256": model_carrier.get("expected", {}).get("tokenizer_json_sha256"),
            "trusted_params": model_carrier.get("expected", {}).get("params_trusted"),
            "public_hf_repo": model_carrier.get("repo_id_requested"),
            "public_hf_revision": (model_carrier.get("public_upload_validation") or {}).get("revision") or model_carrier.get("public_revision"),
        },
        "source_prediction_files": built["source_prediction_files"],
        "carrier_block_counts": block_counts,
        "source_files": {
            "cheap_payload": file_record(CHEAP_PAYLOAD),
            "superglue_payload": file_record(SG_PAYLOAD),
            "superglue_summary": file_record(SG_SUMMARY),
            "endpoint_decision": file_record(ENDPOINT_DECISION),
            "model_carrier": file_record(MODEL_CARRIER),
            "protected_chck82": file_record(PROTECTED_CHCK82),
            "identity": file_record(IDENTITY),
        },
        "scientific_use": "This submission package preserves the exact legal chck_84M endpoint predictions for reproducibility and submission preparation. It is not a new model result and it does not establish a transferable learning principle.",
        "leaderboard_submission_performed": False,
        "model_upload_performed": False,
    }
    write_json(manifest_path, manifest)

    lines = [
        "# research chck_84M native-validated full-evaluation carrier",
        "",
        f"Status: **{status}**",
        "",
        f"Carrier: `{rel(carrier_path)}`",
        f"Carrier SHA256: `{carrier_sha}`",
        f"Native prediction validation: `{valid}` / `{message}`",
        "",
        "## Score arithmetic from existing official-compatible records",
        "",
        "| column | score |",
        "|---|---:|",
    ]
    for col in OVERALL_COLUMNS:
        lines.append(f"| {col} | {report_scores['scores'][col]:.12g} |")
    lines.extend([
        "",
        f"Cheap7: `{report_scores['cheap7']}`; Overall(AoA0): `{report_scores['overall_with_aoa0']}`.",
        f"Delta vs protected chck_82M in Overall(AoA0): `{report_scores['deltas_vs_protected_chck82']['overall_with_aoa0']:+.12f}`.",
        "",
        "## Contents",
        "",
        "- Seven cheap-task prediction blocks come from the chck_84M selected evaluation payload.",
        "- SuperGLUE prediction blocks come from the chck_84M SuperGLUE-only payload.",
        "- `aoa` is scalar `0.0`.",
        "- No `fast_eval_results` block is included.",
        "",
        "## Scientific use",
        "",
        manifest["scientific_use"],
        "",
        f"Manifest JSON: `{rel(manifest_path)}`",
    ])
    manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": status,
        "carrier_path": rel(carrier_path),
        "carrier_sha256": carrier_sha,
        "valid": bool(valid),
        "message": message,
        "overall_with_aoa0": report_scores["overall_with_aoa0"],
        "delta_vs_chck82_overall": report_scores["deltas_vs_protected_chck82"]["overall_with_aoa0"],
        "manifest": rel(manifest_path),
        "leaderboard_submission_performed": False,
        "model_upload_performed": False,
    }, indent=2, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    materialize(args)


if __name__ == "__main__":
    main()
