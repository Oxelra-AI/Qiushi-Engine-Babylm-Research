#!/usr/bin/env python3
"""research: partial official profile and SuperGLUE thresholds for clean seed62065.

Clean seed62065 now has full official zero-shot/Reading plus measured AoA, but
SuperGLUE is still running.  This script computes the non-SuperGLUE contribution
and the SuperGLUE values required for the endpoint to beat coherent86, the dense
references, and clean seed62064 under the same guarded coordinate.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
BASE_PATH = _public_path('experiments/archive/functional_learning/scripts/same_coordinate_all_candidates.py')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/clean_seed65_threshold_profile')
COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
KNOWN_NON_SG = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "AoA"]


def rel(path: pathlib.Path | str | None) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def import_table():
    spec = importlib.util.spec_from_file_location("same_coordinate_all_candidates_import", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def comp(rec: Dict[str, Any]) -> Dict[str, Any]:
    return ((rec.get("overall_computation") or {}).get("components") or {})


def overall(rec: Dict[str, Any]) -> Optional[float]:
    val = (rec.get("overall_computation") or {}).get("Overall")
    if val is None:
        return None
    return float(val)


def threshold_for_target(non_sg_sum: float, target_overall: float) -> float:
    # (non_sg_sum + sg) / 9 > target_overall. Return equality threshold.
    return 9.0 * float(target_overall) - non_sg_sum


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    table = import_table().build_result(args.out_dir)
    models = table.get("models") or {}
    clean65 = models["clean_pres_lambda1_eval_seed62065"]
    clean65_components = comp(clean65)
    non_sg_components = {k: clean65_components.get(k) for k in KNOWN_NON_SG}
    if any(v is None for v in non_sg_components.values()):
        raise SystemExit(f"clean65 non-SuperGLUE components incomplete: {non_sg_components}")
    non_sg_sum = sum(float(v) for v in non_sg_components.values())
    partial_mean_if_sg_zero = non_sg_sum / 9.0
    non_sg_mean_over_known = non_sg_sum / len(KNOWN_NON_SG)

    targets = {}
    for name in ["coherent86", "dense_seed62064", "dense_seed62065", "clean_pres_lambda1_eval_seed62064"]:
        rec = models[name]
        ov = overall(rec)
        targets[name] = {
            "target_overall": ov,
            "required_superglue_for_equal_overall": threshold_for_target(non_sg_sum, float(ov)) if ov is not None else None,
            "component_deltas_non_sg_clean65_minus_target": {k: float(clean65_components[k]) - float(comp(rec)[k]) for k in KNOWN_NON_SG},
            "non_sg_sum_delta_clean65_minus_target": non_sg_sum - sum(float(comp(rec)[k]) for k in KNOWN_NON_SG),
        }

    sg_reference_values = {}
    for sg_name in ["coherent86", "dense_seed62064", "dense_seed62065", "clean_pres_lambda1_eval_seed62064"]:
        sg = comp(models[sg_name]).get("SuperGLUE")
        if sg is not None:
            hypothetical = (non_sg_sum + float(sg)) / 9.0
            sg_reference_values[f"if_clean65_superglue_equals_{sg_name}"] = {
                "superglue": float(sg),
                "overall": hypothetical,
                "delta_vs_coherent86": hypothetical - float(overall(models["coherent86"])),
                "delta_vs_clean_seed62064": hypothetical - float(overall(models["clean_pres_lambda1_eval_seed62064"])),
            }

    # Sensitivity with clean65 GlobalPIQA equalized to parent, as used in research.
    coherent_components = comp(models["coherent86"])
    non_sg_sum_gp_zeroed = sum(float(coherent_components["GlobalPIQA"] if k == "GlobalPIQA" else clean65_components[k]) for k in KNOWN_NON_SG)
    gp_zero_thresholds = {
        name: threshold_for_target(non_sg_sum_gp_zeroed, float(overall(models[name])))
        for name in ["coherent86", "dense_seed62064", "dense_seed62065", "clean_pres_lambda1_eval_seed62064"]
    }

    result = {
        "status": "CLEAN_SEED65_THRESHOLD_PROFILE",
        "created_utc": now(),
        "source_table_script": rel(BASE_PATH),
        "source_zero_reading": table["extended_model_specs"]["clean_pres_lambda1_eval_seed62065"]["zero_reading"],
        "source_aoa": table["extended_model_specs"]["clean_pres_lambda1_eval_seed62065"]["aoa"],
        "clean65_non_superglue_components": non_sg_components,
        "clean65_non_superglue_sum": non_sg_sum,
        "clean65_partial_mean_if_superglue_zero": partial_mean_if_sg_zero,
        "clean65_non_superglue_mean_over_8_known_components": non_sg_mean_over_known,
        "thresholds": targets,
        "hypothetical_reference_superglue_values": sg_reference_values,
        "globalpiqa_zeroed_non_superglue_sum": non_sg_sum_gp_zeroed,
        "globalpiqa_zeroed_required_superglue_for_equal_overall": gp_zero_thresholds,
        "scientific_reading": (
            "Clean seed62065 has already replicated the seed62064 zero/Reading trade shape on the official-sized tasks: grammar/knowledge remain below coherent86, Entity and GlobalPIQA remain above it. "
            "The pending SuperGLUE value decides whether this becomes a complete official replication, because AoA is measured zero and the remaining components are fixed."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "clean_seed65_threshold_profile.json"
    out_md = args.out_dir / "clean_seed65_threshold_profile.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines: List[str] = ["# research clean seed62065 threshold profile\n\n"]
    lines.append("Clean seed62065 has full official zero-shot/Reading and measured AoA; SuperGLUE is still pending.\n\n")
    lines.append("## Known non-SuperGLUE components\n\n")
    lines.append(json.dumps(non_sg_components, indent=2, ensure_ascii=False) + "\n\n")
    lines.append(f"Known-component sum: `{non_sg_sum}`; mean over 8 known components: `{non_sg_mean_over_known}`.\n\n")
    lines.append("## Required SuperGLUE thresholds\n\n")
    for name, t in targets.items():
        lines.append(f"- Equal `{name}` Overall requires SuperGLUE `{t['required_superglue_for_equal_overall']}`.\n")
    lines.append("\n## Hypothetical reference SuperGLUE values\n\n")
    lines.append(json.dumps(sg_reference_values, indent=2, ensure_ascii=False) + "\n\n")
    lines.append("## Scientific reading\n\n")
    lines.append(result["scientific_reading"] + "\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "non_sg_sum": non_sg_sum,
        "threshold_equal_coherent86_sg": targets["coherent86"]["required_superglue_for_equal_overall"],
        "threshold_equal_clean_seed64_sg": targets["clean_pres_lambda1_eval_seed62064"]["required_superglue_for_equal_overall"],
        "hypothetical_if_sg_equals_seed64": sg_reference_values.get("if_clean65_superglue_equals_clean_pres_lambda1_eval_seed62064"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
