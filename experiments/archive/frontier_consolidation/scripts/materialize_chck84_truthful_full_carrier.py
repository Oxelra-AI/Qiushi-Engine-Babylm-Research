#!/usr/bin/env python3
"""research: materialize a truthful full-evaluation prediction carrier for chck_84M.

This merges only predictions produced for the exact chck_84M endpoint:
- seven cheap-task prediction blocks from the selected chck_84M evaluation payload,
- SuperGLUE prediction blocks from the chck_84M SuperGLUE-only payload,
- scalar AoA=0.0, matching the current Space behavior for missing or scalar AoA.

It deliberately does not copy fast_eval_results from any other checkpoint history and
it does not submit to the BabyLM leaderboard.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
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
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/chck84_truthful_full_carrier')

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
    if compact:
        path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: pathlib.Path) -> dict[str, Any]:
    return {"path": rel(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256_file(path) if path.exists() else None}


def load_prediction_payload(path: pathlib.Path, expected_single_key: str | None = None) -> dict[str, Any]:
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Prediction payload is not an object: {path}")
    if expected_single_key is not None and expected_single_key not in obj:
        raise KeyError(f"Expected key {expected_single_key!r} missing from {path}; keys={list(obj)[:12]}")
    return obj


def prediction_block_counts(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {"type": type(block).__name__}
    counts: dict[str, Any] = {"n_subkeys": len(block)}
    total = 0
    samples: dict[str, int] = {}
    for key, val in block.items():
        n = None
        if isinstance(val, dict) and isinstance(val.get("predictions"), list):
            n = len(val["predictions"])
        if isinstance(n, int):
            total += n
            if len(samples) < 8:
                samples[key] = n
    counts["total_prediction_rows"] = total
    counts["sample_subkey_row_counts"] = samples
    return counts


def configure_space_imports() -> None:
    # The saved Space scoring code derives EVAL_DATASETS_PATH from HF_HOME at import time.
    os.environ["HF_HOME"] = str(_public_path('experiments/archive/frontier_consolidation/data/chck82_hf_public_bundle/live_submit_dry_run'))
    if str(SPACE_REPO) not in sys.path:
        sys.path.insert(0, str(SPACE_REPO))


def import_space_functions():
    configure_space_imports()
    from src.submission.check_validity import is_valid_predictions  # type: ignore
    from src.submission.eval_submission import evaluate_submission  # type: ignore
    return is_valid_predictions, evaluate_submission


def mean_numeric_score(d: Any) -> float:
    if not isinstance(d, dict):
        return 0.0
    vals = []
    for v in d.values():
        if v is None:
            vals.append(0.0)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    if not vals:
        return 0.0
    return float(mean(vals) * 100.0)


def space_display_scores(processed: dict[str, Any]) -> dict[str, float]:
    blimp = mean_numeric_score(processed.get("blimp"))
    supplement = mean_numeric_score(processed.get("blimp_supplement"))
    ewok = mean_numeric_score(processed.get("ewok"))
    entity = mean_numeric_score(processed.get("entity_tracking")) if processed.get("entity_tracking_filtered") else 0.0
    comps = mean_numeric_score(processed.get("comps"))
    gpiqa = (mean_numeric_score(processed.get("global_piqa_parallel")) + mean_numeric_score(processed.get("global_piqa_nonparallel"))) / 2.0
    reading = mean_numeric_score(processed.get("reading"))
    glue = mean_numeric_score(processed.get("glue"))
    # The current read_evals logic sets AoA to zero unless aoa_surprisals is present.
    aoa = mean_numeric_score(processed.get("aoa")) if processed.get("aoa_surprisals") else 0.0
    return {
        "BLiMP": blimp,
        "Supplement": supplement,
        "EWoK": ewok,
        "Entity": entity,
        "COMPS": comps,
        "GlobalPIQA": gpiqa,
        "Reading": reading,
        "SuperGLUE": glue,
        "AoA": aoa,
    }


def build_carrier() -> dict[str, Any]:
    cheap_payload = read_json(CHEAP_PAYLOAD)
    sg_payload = read_json(SG_PAYLOAD)

    carrier: dict[str, Any] = {}
    source_prediction_files: dict[str, dict[str, Any]] = {}

    for column, out_key in MAP_CHEAP.items():
        rec = cheap_payload.get("tasks", {}).get(column)
        if not isinstance(rec, dict) or not rec.get("predictions"):
            raise KeyError(f"cheap chck_84M payload missing prediction path for {column}")
        pred_path = ROOT / rec["predictions"]
        pred_obj = load_prediction_payload(pred_path)
        carrier[out_key] = pred_obj
        source_prediction_files[out_key] = {**file_record(pred_path), "source_column": column, "report_score": rec.get("score") or rec.get("scores")}

    sg_records = sg_payload.get("tasks", {}).get("SuperGLUE", {}).get("tasks", [])
    if not sg_records:
        raise RuntimeError("SuperGLUE payload has no task records")
    glue: dict[str, Any] = {}
    for rec in sg_records:
        task = str(rec["task"])
        pred_path = ROOT / rec["predictions"]
        pred_obj = load_prediction_payload(pred_path, expected_single_key=task)
        glue[task] = pred_obj[task]
        source_prediction_files[f"glue/{task}"] = {**file_record(pred_path), "source_column": "SuperGLUE", "metric_for_valid": rec.get("metric_for_valid")}
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
    eval_processed_path = out_dir / "space_eval_processed_results.json"

    built = build_carrier()
    carrier = built["carrier"]
    write_json(carrier_path, carrier, compact=True)

    cheap_payload = read_json(CHEAP_PAYLOAD)
    sg_summary = read_json(SG_SUMMARY)
    endpoint = read_json(ENDPOINT_DECISION)
    model_carrier = read_json(MODEL_CARRIER)
    protected = read_json(PROTECTED_CHCK82)
    identity = read_json(IDENTITY)

    is_valid_predictions, evaluate_submission = import_space_functions()
    valid, message = is_valid_predictions(str(carrier_path), "strict-small")
    processed = evaluate_submission(str(carrier_path), multilingual=False)
    write_json(eval_processed_path, processed)
    display_scores = space_display_scores(processed)
    display_cheap7 = float(mean(display_scores[k] for k in CHEAP7_COLUMNS))
    display_overall = float(mean(display_scores[k] for k in OVERALL_COLUMNS))

    report_scores = report_score_table(cheap_payload, sg_summary, endpoint, protected)
    score_delta_raw_minus_report = {k: display_scores[k] - report_scores["scores"][k] for k in OVERALL_COLUMNS}
    max_abs_report_delta = max(abs(v) for v in score_delta_raw_minus_report.values())

    carrier_sha = sha256_file(carrier_path)
    block_counts = {k: prediction_block_counts(v) for k, v in carrier.items() if k != "aoa"}
    status = "CHCK84_TRUTHFUL_FULL_CARRIER_READY" if valid else "CHCK84_TRUTHFUL_FULL_CARRIER_NOT_ACCEPTED_BY_LOCAL_SPACE_CHECK"

    manifest: dict[str, Any] = {
        "status": status,
        "created_utc": now(),
        "carrier_path": rel(carrier_path),
        "carrier_sha256": carrier_sha,
        "carrier_size_bytes": carrier_path.stat().st_size,
        "validator": {
            "space_repo": rel(SPACE_REPO),
            "track": "strict-small",
            "is_valid_predictions": bool(valid),
            "message": message,
            "eval_cache_root": rel(EVAL_CACHE_ROOT),
        },
        "truthful_prediction_policy": {
            "cheap_predictions": rel(CHEAP_PAYLOAD),
            "superglue_predictions": rel(SG_PAYLOAD),
            "aoa": "scalar_zero_only",
            "fast_eval_results": "omitted; no checkpoint-history block is copied from protected chck_82M or any other model",
        },
        "score_arithmetic_from_report_summaries": report_scores,
        "score_arithmetic_from_space_eval_submission": {
            "scores": display_scores,
            "cheap7": display_cheap7,
            "overall_with_aoa0": display_overall,
            "delta_vs_report_summary_by_column": score_delta_raw_minus_report,
            "max_abs_column_delta_vs_report_summary": max_abs_report_delta,
            "overall_delta_vs_report_summary": display_overall - report_scores["overall_with_aoa0"],
            "cheap7_delta_vs_report_summary": display_cheap7 - report_scores["cheap7"],
        },
        "model_identity": {
            "endpoint": "chck_84M",
            "model_dir": identity.get("checkpoint") or identity.get("model_dir") or model_carrier.get("required_paths", {}).get("source_checkpoint", {}).get("path"),
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
            "space_eval_processed_results": file_record(eval_processed_path),
        },
        "scientific_use": "This is a full-evaluation prediction package for the exact legal chck_84M endpoint. It supports reproducibility and submission preparation, but it lacks fast_eval_results and it does not establish a transferable learning principle.",
        "leaderboard_submission_performed": False,
        "model_upload_performed": False,
    }
    write_json(manifest_path, manifest)

    lines = [
        "# research chck_84M truthful full-evaluation carrier",
        "",
        f"Status: **{status}**",
        "",
        f"Carrier: `{rel(carrier_path)}`",
        f"Carrier SHA256: `{carrier_sha}`",
        f"Local Space prediction check: `{valid}` / `{message}`",
        "",
        "## Score arithmetic",
        "",
        "Two score views are recorded: report-parsed local summaries already used in research, and a direct pass through the saved live-Space `eval_submission.py` scorer using the local eval-datasets snapshot.",
        "",
        "| column | report-summary score | Space-scorer score | Space minus report |",
        "|---|---:|---:|---:|",
    ]
    for col in OVERALL_COLUMNS:
        lines.append(f"| {col} | {report_scores['scores'][col]:.12g} | {display_scores[col]:.12g} | {score_delta_raw_minus_report[col]:+.12g} |")
    lines.extend([
        "",
        f"Report-summary cheap7: `{report_scores['cheap7']}`; Overall(AoA0): `{report_scores['overall_with_aoa0']}`.",
        f"Space-scorer cheap7: `{display_cheap7}`; Overall(AoA0): `{display_overall}`.",
        f"Delta vs protected chck_82M in report-summary arithmetic: `{report_scores['deltas_vs_protected_chck82']['overall_with_aoa0']:+.12f}` Overall.",
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
        "report_overall_with_aoa0": report_scores["overall_with_aoa0"],
        "space_scorer_overall_with_aoa0": display_overall,
        "report_delta_vs_chck82_overall": report_scores["deltas_vs_protected_chck82"]["overall_with_aoa0"],
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
