#!/usr/bin/env python3
"""research: sensitivity and missing-method-gap analysis for clean preservation.

This script reads the guarded same-coordinate comparison after clean seed62064
SuperGLUE delivery and computes two research-facing quantities:

1. GlobalPIQA-zero sensitivity: set the candidate-parent GlobalPIQA delta to 0
   while leaving all other component deltas unchanged.  This tests whether a
   positive Overall depends entirely on a few GlobalPIQA decisions.  It is not a
   replacement for the official metric.
2. Acquisition-to-preservation gap to be closed by official evaluation: clean
   preservation is acquisition policy (M,S) plus deterministic parent anchoring;
   the fully evaluated dense controls are (M,M).  The exact acquisition-only
   (M,S) endpoint must be evaluated in the same coordinate to isolate the added
   preservation effect from target-sampling/credit-allocation differences.
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
from typing import Any, Dict, List, Optional

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_COMPARISON = _public_path('experiments/archive/functional_learning/data/same_coordinate_comparison_after_clean_superglue/same_coordinate_comparison.json')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/candidate_sensitivity_and_method_gap')
COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fnum(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v
    except Exception:
        return None


def sensitivity_for(name: str, deltas: Dict[str, Any]) -> Dict[str, Any]:
    row = deltas.get(name, {})
    cd = row.get("component_deltas_vs_coherent86") or {}
    vals = {k: fnum(cd.get(k)) for k in COMPONENTS}
    complete = row.get("complete") and all(vals[k] is not None for k in COMPONENTS)
    official_delta = fnum(row.get("delta_vs_coherent86"))
    if not complete:
        return {"model": name, "complete": False, "official_delta_vs_coherent86": official_delta, "globalpiqa_zero_delta": None, "non_globalpiqa_sum_delta": None}
    non_gp_sum = sum(float(vals[k]) for k in COMPONENTS if k != "GlobalPIQA")
    gp_zero_delta = non_gp_sum / len(COMPONENTS)
    exclude_gp_mean_delta = non_gp_sum / (len(COMPONENTS) - 1)
    return {
        "model": name,
        "complete": True,
        "official_delta_vs_coherent86": official_delta,
        "component_deltas_vs_coherent86": vals,
        "globalpiqa_delta": vals["GlobalPIQA"],
        "non_globalpiqa_sum_delta": non_gp_sum,
        "globalpiqa_zero_delta_overall_units": gp_zero_delta,
        "exclude_globalpiqa_mean_delta_over_remaining_components": exclude_gp_mean_delta,
        "positive_if_globalpiqa_zeroed": gp_zero_delta > 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", type=pathlib.Path, default=DEFAULT_COMPARISON)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    obj = json.loads(args.comparison.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    deltas = obj.get("deltas_vs_coherent86", {})
    models = ["dense_seed62064", "dense_seed62065", "clean_pres_lambda1_eval_seed62064", "clean_pres_lambda1_eval_seed62065"]
    sens = {m: sensitivity_for(m, deltas) for m in models}

    clean = sens.get("clean_pres_lambda1_eval_seed62064", {})
    dense64 = sens.get("dense_seed62064", {})
    dense65 = sens.get("dense_seed62065", {})
    result = {
        "status": "CANDIDATE_SENSITIVITY_AND_METHOD_GAP",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/candidate_sensitivity_and_method_gap.py')),
        "comparison_source": rel(args.comparison),
        "sensitivity": sens,
        "scientific_interpretation": {
            "globalpiqa_sensitivity": "Zeroing the parent-candidate GlobalPIQA delta leaves the official Overall arithmetic denominator unchanged. This does not replace the official metric and does not establish statistical reliability, but it asks whether the improvement is wholly carried by GlobalPIQA.",
            "clean_seed62064_result": f"Clean seed62064 official delta is {clean.get('official_delta_vs_coherent86')}; with GlobalPIQA delta set to zero it is {clean.get('globalpiqa_zero_delta_overall_units')} Overall units.",
            "dense_contrast": f"Dense seed62064/62065 under the same GlobalPIQA-zero sensitivity are {dense64.get('globalpiqa_zero_delta_overall_units')} and {dense65.get('globalpiqa_zero_delta_overall_units')}, respectively.",
            "method_gap": "The fully evaluated dense controls are (M,M), while clean preservation is built on (M,S) acquisition plus deterministic parent anchoring. The exact acquisition-only (M,S) endpoint must receive compatible official zero/Reading, repaired SuperGLUE, and measured AoA before attributing the SuperGLUE/Overall increment specifically to preservation.",
        },
        "pending_method_comparison_payloads": {
            "ms_acquisition_endpoint": "experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080",
            "ms_acquisition_repaired_automodel": "experiments/archive/functional_learning/data/automodel_repair_candidates/repaired_densemask_sparselabel_seed62064_u0080",
            "needed": ["official-sized zero-shot/Reading", "repaired AutoModel SuperGLUE", "measured AoA endpoint assembled with shared ancestry"],
        },
    }
    out_json = args.out_dir / "candidate_sensitivity_and_method_gap.json"
    out_md = args.out_dir / "candidate_sensitivity_and_method_gap.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines: List[str] = []
    lines.append("# research candidate sensitivity and method gap\n\n")
    lines.append(f"Source comparison: `{rel(args.comparison)}`\n\n")
    lines.append("## GlobalPIQA-zero sensitivity\n\n")
    for m, r in sens.items():
        lines.append(f"- `{m}` complete `{r.get('complete')}` official delta `{r.get('official_delta_vs_coherent86')}`; delta with GlobalPIQA contribution set to zero `{r.get('globalpiqa_zero_delta_overall_units')}`; remaining-component mean `{r.get('exclude_globalpiqa_mean_delta_over_remaining_components')}`.\n")
    lines.append("\n## Acquisition-to-preservation method gap\n\n")
    lines.append(result["scientific_interpretation"]["method_gap"] + "\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "clean_gp_zero_delta": clean.get("globalpiqa_zero_delta_overall_units")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
