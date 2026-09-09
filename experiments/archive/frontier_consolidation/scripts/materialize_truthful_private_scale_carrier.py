#!/usr/bin/env python3
"""Materialize a truthful carrier for a no-training private-scale endpoint.

The endpoint is a materialized version of the coherent86 frozen-anchor private-path
model with a changed config-only `private_adapter_scale`.  This script merges only
prediction files produced for that exact materialized endpoint:
- zero-shot / reading predictions from its cheap7 eval payload
- SuperGLUE predictions from its SuperGLUE-only eval payload
- scalar AoA=0.0
- no fast_eval_results copied from the protected 82M history

It validates the resulting carrier using the current local copy of the live Space
validator.  It does not submit to the leaderboard.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
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
PROTECTED_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
PANEL_JSON = _public_path('experiments/archive/frontier_consolidation/data/private_scale_panel_analysis/private_scale_panel_analysis.json')
DEFAULT_OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/truthful_private_scale_carriers')

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


def get_model_identity(model_dir: pathlib.Path) -> dict[str, Any]:
    cfg = read_json(model_dir / "config.json")
    model_file = model_dir / "model.safetensors"
    return {
        "model_dir": rel(model_dir),
        "config_private_adapter_scale": cfg.get("private_adapter_scale"),
        "config_private_adapter_enabled": cfg.get("private_adapter_enabled"),
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "auto_map": cfg.get("auto_map"),
        "model_safetensors_sha256": sha256_file(model_file),
        "model_safetensors_size_bytes": model_file.stat().st_size,
        "config_sha256": sha256_file(model_dir / "config.json"),
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    out_root = pathlib.Path(args.out_root) / args.label
    out_root.mkdir(parents=True, exist_ok=True)
    carrier_path = out_root / f"all_full_preds_truthful_{args.label}_mlm.json"
    manifest_path = out_root / f"truthful_{args.label}_carrier_manifest.json"
    manifest_md = out_root / f"truthful_{args.label}_carrier_manifest.md"

    cheap_payload = read_json(pathlib.Path(args.cheap_payload))
    cheap_summary = read_json(pathlib.Path(args.cheap_summary))
    sg_payload = read_json(pathlib.Path(args.superglue_payload))
    sg_summary = read_json(pathlib.Path(args.superglue_summary))
    protected = read_json(PROTECTED_VERIFY)
    panel = read_json(PANEL_JSON) if PANEL_JSON.exists() else {}
    model_identity = get_model_identity(pathlib.Path(args.model_dir))

    carrier: dict[str, Any] = {}
    source_prediction_files: dict[str, str] = {}
    for column, out_key in MAP_CHEAP.items():
        rec = cheap_payload.get("tasks", {}).get(column)
        if not rec:
            raise KeyError(f"cheap payload missing task {column}")
        pred_path = ROOT / rec["predictions"]
        carrier[out_key] = load_prediction_payload(pred_path)
        source_prediction_files[out_key] = rel(pred_path)

    sg_task = sg_payload.get("tasks", {}).get("SuperGLUE", {})
    sg_records = sg_task.get("tasks", [])
    if not sg_records:
        raise RuntimeError("SuperGLUE payload has no subtask records")
    glue: dict[str, Any] = {}
    for rec in sg_records:
        task = str(rec["task"])
        pred_path = ROOT / rec["predictions"]
        pred_obj = load_prediction_payload(pred_path, expected_single_key=task)
        glue[task] = pred_obj[task]
        source_prediction_files[f"glue/{task}"] = rel(pred_path)
    carrier["glue"] = glue
    carrier["aoa"] = {"aoa": 0.0}

    carrier_path.write_text(json.dumps(carrier, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    carrier_sha = sha256_file(carrier_path)
    is_valid_predictions = import_validator()
    valid, message = is_valid_predictions(str(carrier_path), "strict-small")

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
    cheap7 = float(mean(scores[k] for k in CHEAP7_COLUMNS))
    overall = float(mean(scores[k] for k in OVERALL_COLUMNS))
    protected_scores = {k: float(v) for k, v in protected["score_arithmetic"]["scores"].items() if v is not None}
    protected_overall = float(protected["score_arithmetic"]["overall_reported"])
    protected_cheap7 = float(mean(protected_scores[k] for k in CHEAP7_COLUMNS))

    panel_arm = panel.get("score_table", {}).get(args.panel_arm, {}) if isinstance(panel, dict) else {}
    panel_comp = panel.get("comparisons", {}).get(f"{args.panel_arm}_minus_chck82_anchor", {}) if isinstance(panel, dict) else {}

    status = "TRUTHFUL_PRIVATE_SCALE_CARRIER_MATERIALIZED" if valid else "CARRIER_VALIDATION_FAILED"
    manifest = {
        "status": status,
        "created_utc": now(),
        "label": args.label,
        "panel_arm": args.panel_arm,
        "carrier_path": rel(carrier_path),
        "carrier_sha256": carrier_sha,
        "carrier_size_bytes": carrier_path.stat().st_size,
        "model_identity": model_identity,
        "validator": {
            "space_repo": rel(SPACE_REPO),
            "track": "strict-small",
            "is_valid_predictions": bool(valid),
            "message": message,
        },
        "truthful_history_policy": {
            "aoa": "scalar_zero_only",
            "fast_eval_results": "omitted; no borrowed protected chck82 fast history",
            "cheap_predictions": rel(args.cheap_payload),
            "superglue_predictions": rel(args.superglue_payload),
        },
        "score_arithmetic_candidate_native": {
            "scores": scores,
            "cheap7": cheap7,
            "overall_with_aoa0": overall,
            "deltas_vs_protected_chck82": {
                "cheap7": cheap7 - protected_cheap7,
                "superglue": sg_score - protected_scores["SuperGLUE"],
                "overall_with_aoa0": overall - protected_overall,
            },
        },
        "panel_reading": {
            "panel_arm_score_table": panel_arm,
            "comparison_vs_chck82_anchor_aggregate": panel_comp.get("aggregate"),
            "comparison_vs_chck82_anchor_focus": panel_comp.get("focus"),
            "interpretation": "No-training private-scale carrier. Endpoint arithmetic may improve, but the mechanism remains amplitude-controlled redistribution unless item evidence shows broad positive, stable decision addition.",
        },
        "source_files": {
            "cheap_summary": rel(args.cheap_summary),
            "superglue_summary": rel(args.superglue_summary),
            "protected_chck82": rel(PROTECTED_VERIFY),
            "panel_analysis": rel(PANEL_JSON),
            "source_prediction_files": source_prediction_files,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# Truthful private-scale carrier — {args.label}",
        "",
        f"Status: **{status}**",
        "",
        f"Carrier: `{rel(carrier_path)}`",
        f"Carrier SHA256: `{carrier_sha}`",
        f"Model: `{model_identity['model_dir']}`",
        f"Model SHA256: `{model_identity['model_safetensors_sha256']}`",
        f"Private adapter scale: `{model_identity['config_private_adapter_scale']}`",
        "",
        "## Score arithmetic with AoA=0",
        "",
        "| column | score | delta vs protected chck82 |",
        "|---|---:|---:|",
    ]
    for col in OVERALL_COLUMNS:
        lines.append(f"| {col} | {scores[col]:.12g} | {scores[col] - protected_scores[col]:+.12g} |")
    lines += [
        "",
        f"Cheap7: `{cheap7}` (delta `{cheap7 - protected_cheap7:+.12f}`)",
        f"Overall(AoA0): `{overall}` (delta `{overall - protected_overall:+.12f}`)",
        "",
        "## Truthful history policy",
        "- AoA is scalar zero only.",
        "- No `fast_eval_results` are copied.",
        "- Cheap and SuperGLUE predictions come from the exact materialized alpha endpoint.",
        "",
        "## Panel reading",
        f"Panel arm: `{args.panel_arm}`",
        f"Vs-anchor aggregate: `{panel_comp.get('aggregate')}`",
        "",
        manifest["panel_reading"]["interpretation"],
        "",
        f"JSON: `{rel(manifest_path)}`",
    ]
    manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "label": args.label, "carrier": rel(carrier_path), "manifest": rel(manifest_path), "overall_with_aoa0": overall, "delta_vs_chck82": overall - protected_overall, "valid": valid}, indent=2), flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--panel-arm", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cheap-payload", required=True)
    ap.add_argument("--cheap-summary", required=True)
    ap.add_argument("--superglue-payload", required=True)
    ap.add_argument("--superglue-summary", required=True)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = ap.parse_args()
    materialize(args)


if __name__ == "__main__":
    main()
