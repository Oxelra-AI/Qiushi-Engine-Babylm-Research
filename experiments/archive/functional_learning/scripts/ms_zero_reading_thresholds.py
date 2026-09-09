#!/usr/bin/env python3
"""research: threshold arithmetic for pending exact (M,S) zero-shot/Reading.

The exact `(M,S)` acquisition-only endpoint has measured AoA and completed repaired
SuperGLUE, but its official zero-shot/Reading complement is running.  This script
quantifies what the pending seven-component zero/Reading block must contribute to
beat coherent86 and clean seed62064, and evaluates transparent scenarios using only
completed official payloads.  It does not infer the pending `(M,S)` zero/Reading values.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import time
from typing import Any, Dict, List

ROOT = _public_path('.')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/ms_zero_reading_thresholds')
BASE_TABLE = _public_path('experiments/archive/functional_learning/data/same_coordinate_all_candidates_after_ms_superglue/same_coordinate_all_candidates.json')
DIRECT_SG = _public_path('experiments/archive/functional_learning/data/superglue_ms_direct_profile/superglue_ms_direct_profile.json')
ZERO_COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ALL_COMPONENTS = ZERO_COMPONENTS + ["SuperGLUE", "AoA"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def comp(model: Dict[str, Any]) -> Dict[str, float]:
    c = model.get("overall_computation", {}).get("components") or {}
    return {k: float(c[k]) for k in ALL_COMPONENTS if c.get(k) is not None}


def zsum(components: Dict[str, float]) -> float:
    return sum(components[k] for k in ZERO_COMPONENTS)


def overall_from(zero_sum: float, superglue: float, aoa: float = 0.0) -> float:
    return (zero_sum + superglue + aoa) / 9.0


def build() -> Dict[str, Any]:
    table = load_json(BASE_TABLE)
    direct = load_json(DIRECT_SG)
    models = table["models"]
    completed = {name: comp(models[name]) for name in ["coherent86", "dense_seed62064", "dense_seed62065", "clean_pres_lambda1_eval_seed62064"]}
    completed_zero_sums = {name: zsum(vals) for name, vals in completed.items()}
    completed_overalls = {name: float(models[name]["overall_computation"]["Overall"]) for name in completed}

    ms_superglue = float(direct["interpretation"]["superglue_means"]["ms_acquisition_seed62064_MS"])
    ms_aoa = 0.0

    thresholds: Dict[str, Any] = {}
    for name, ov in completed_overalls.items():
        req_zero_sum = 9.0 * ov - ms_superglue - ms_aoa
        thresholds[f"equal_{name}"] = {
            "target_overall": ov,
            "required_zero_reading_sum": req_zero_sum,
            "required_zero_reading_mean_over_7": req_zero_sum / 7.0,
            "difference_from_coherent86_zero_sum": req_zero_sum - completed_zero_sums["coherent86"],
            "difference_from_dense_seed62064_zero_sum": req_zero_sum - completed_zero_sums["dense_seed62064"],
            "difference_from_clean_seed62064_zero_sum": req_zero_sum - completed_zero_sums["clean_pres_lambda1_eval_seed62064"],
        }

    scenarios: Dict[str, Any] = {}
    for name, zs in completed_zero_sums.items():
        ov = overall_from(zs, ms_superglue, ms_aoa)
        scenarios[f"ms_zero_reading_equals_{name}"] = {
            "zero_reading_source_model": name,
            "zero_reading_sum": zs,
            "overall_with_ms_superglue_and_aoa": ov,
            "delta_vs_coherent86": ov - completed_overalls["coherent86"],
            "delta_vs_clean_seed62064": ov - completed_overalls["clean_pres_lambda1_eval_seed62064"],
            "delta_vs_dense_seed62064": ov - completed_overalls["dense_seed62064"],
            "delta_vs_dense_seed62065": ov - completed_overalls["dense_seed62065"],
        }

    # Also include clean seed62065 official zero/Reading as a scenario if present in table.
    clean65 = models.get("clean_pres_lambda1_eval_seed62065")
    if clean65 and clean65.get("zero_reading", {}).get("complete"):
        zscores = clean65["zero_reading"]["scores"]
        zs = sum(float(zscores[k]) for k in ZERO_COMPONENTS)
        ov = overall_from(zs, ms_superglue, ms_aoa)
        scenarios["ms_zero_reading_equals_clean_seed62065_zero_reading"] = {
            "zero_reading_source_model": "clean_pres_lambda1_eval_seed62065",
            "zero_reading_sum": zs,
            "overall_with_ms_superglue_and_aoa": ov,
            "delta_vs_coherent86": ov - completed_overalls["coherent86"],
            "delta_vs_clean_seed62064": ov - completed_overalls["clean_pres_lambda1_eval_seed62064"],
            "delta_vs_dense_seed62064": ov - completed_overalls["dense_seed62064"],
            "delta_vs_dense_seed62065": ov - completed_overalls["dense_seed62065"],
        }
        completed_zero_sums["clean_pres_lambda1_eval_seed62065_zero_reading_only"] = zs

    # Decompose clean64-vs-M,S direct requirement in intuitive units.
    clean64_zero_sum = completed_zero_sums["clean_pres_lambda1_eval_seed62064"]
    clean64_overall = completed_overalls["clean_pres_lambda1_eval_seed62064"]
    required_to_equal_clean = thresholds["equal_clean_pres_lambda1_eval_seed62064"]["required_zero_reading_sum"]
    direct_gap_if_ms_zero_equals_clean64 = scenarios["ms_zero_reading_equals_clean_pres_lambda1_eval_seed62064"]["delta_vs_clean_seed62064"]

    result = {
        "status": "MS_ZERO_READING_THRESHOLDS",
        "created_utc": now(),
        "purpose": "Prepare the exact acquisition-only `(M,S)` Overall interpretation before its official zero/Reading job finishes, without inferring pending values.",
        "base_table": rel(BASE_TABLE),
        "direct_superglue_profile": rel(DIRECT_SG),
        "zero_reading_components": ZERO_COMPONENTS,
        "known_ms_components": {"SuperGLUE": ms_superglue, "AoA": ms_aoa},
        "completed_zero_reading_sums": completed_zero_sums,
        "completed_overalls": completed_overalls,
        "thresholds": thresholds,
        "scenarios": scenarios,
        "key_reading": {
            "ms_superglue_vs_coherent86": ms_superglue - completed["coherent86"]["SuperGLUE"],
            "ms_superglue_vs_clean64": ms_superglue - completed["clean_pres_lambda1_eval_seed62064"]["SuperGLUE"],
            "zero_sum_needed_above_clean64_zero_sum_to_equal_clean64": required_to_equal_clean - clean64_zero_sum,
            "overall_gap_if_ms_zero_reading_equals_clean64_zero_reading": direct_gap_if_ms_zero_equals_clean64,
            "zero_sum_needed_above_coherent86_zero_sum_to_equal_coherent86": thresholds["equal_coherent86"]["difference_from_coherent86_zero_sum"],
        },
    }
    return result


def fmt(x: Any, nd: int = 6) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_md(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research thresholds for pending exact `(M,S)` zero-shot/Reading\n\n")
    lines.append("The exact acquisition-only `(M,S)` endpoint has completed repaired-AutoModel SuperGLUE and measured AoA. Its official zero-shot/Reading run is still pending. This file gives threshold arithmetic only; it does not infer the pending scores.\n\n")
    lines.append("## Known `(M,S)` components\n\n")
    lines.append(f"- SuperGLUE: {fmt(result['known_ms_components']['SuperGLUE'])}\n")
    lines.append(f"- AoA: {fmt(result['known_ms_components']['AoA'])}\n\n")
    lines.append("## Seven-component zero/Reading sums already observed\n\n")
    for k, v in result["completed_zero_reading_sums"].items():
        lines.append(f"- {k}: sum {fmt(v)}, mean-over-7 {fmt(v/7.0)}\n")
    lines.append("\n## Required zero/Reading sums for exact `(M,S)`\n\n")
    lines.append("| target to equal | target Overall | required zero/Reading sum | required mean over 7 | above coherent86 zero sum | above dense64 zero sum | above clean64 zero sum |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for k, d in result["thresholds"].items():
        label = k.replace("equal_", "")
        lines.append(f"| {label} | {fmt(d['target_overall'])} | {fmt(d['required_zero_reading_sum'])} | {fmt(d['required_zero_reading_mean_over_7'])} | {fmt(d['difference_from_coherent86_zero_sum'])} | {fmt(d['difference_from_dense_seed62064_zero_sum'])} | {fmt(d['difference_from_clean_seed62064_zero_sum'])} |\n")
    lines.append("\n## Transparent scenarios using completed official zero/Reading blocks\n\n")
    lines.append("| scenario | Overall with `(M,S)` SG/AoA | delta vs coherent86 | delta vs clean64 | delta vs dense64 | delta vs dense65 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for k, d in result["scenarios"].items():
        lines.append(f"| {k} | {fmt(d['overall_with_ms_superglue_and_aoa'])} | {fmt(d['delta_vs_coherent86'])} | {fmt(d['delta_vs_clean_seed62064'])} | {fmt(d['delta_vs_dense_seed62064'])} | {fmt(d['delta_vs_dense_seed62065'])} |\n")
    kr = result["key_reading"]
    lines.append("\n## Scientific reading\n\n")
    lines.append(f"Exact `(M,S)` SuperGLUE is {fmt(kr['ms_superglue_vs_coherent86'])} below coherent86 and {fmt(kr['ms_superglue_vs_clean64'])} below clean seed62064. To equal coherent86 Overall, `(M,S)` needs its zero/Reading sum to be {fmt(kr['zero_sum_needed_above_coherent86_zero_sum_to_equal_coherent86'])} above coherent86's seven-component sum. To equal clean seed62064 Overall, it would need {fmt(kr['zero_sum_needed_above_clean64_zero_sum_to_equal_clean64'])} above clean64's already observed zero/Reading sum. Therefore if `(M,S)` zero/Reading matches clean64 exactly, clean still keeps an Overall advantage of {-kr['overall_gap_if_ms_zero_reading_equals_clean64_zero_reading']:.6f}, equal to the known SuperGLUE difference divided by nine. The direct comparison was unresolved at the time of this note because the matched zero/Reading evaluation was not yet complete and validated.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build()
    out_json = args.out_dir / "ms_zero_reading_thresholds.json"
    out_md = args.out_dir / "ms_zero_reading_thresholds.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(out_md, result)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "known_ms_components": result["known_ms_components"],
        "key_reading": result["key_reading"],
        "scenarios": result["scenarios"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
