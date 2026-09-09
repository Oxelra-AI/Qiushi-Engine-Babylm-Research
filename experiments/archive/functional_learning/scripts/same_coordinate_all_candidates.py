#!/usr/bin/env python3
"""research: guarded same-coordinate table for clean replication and (M,S) causal comparison.

This extends the hardened research comparison without weakening its admission rule:
a model receives an Overall only when all three compatible sources are complete:
full official-sized zero-shot/Reading, repaired-AutoModel SuperGLUE primary
metrics, and measured batched AoA with the platform-matching scorer.

The purpose is to prepare the two load-bearing unfinished comparisons:
1. clean preservation seed62065 as a fixed-policy replication of seed62064;
2. exact acquisition-only dense-mask/sparse-label (M,S) seed62064 versus clean
   seed62064, separating preservation from target-sampling/credit differences.
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
BASE_PATH = _public_path('experiments/archive/functional_learning/scripts/same_coordinate_comparison.py')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/same_coordinate_all_candidates')

EXTENDED_MODEL_SPECS: Dict[str, Dict[str, Optional[str]]] = {
    "clean_pres_lambda1_eval_seed62065": {
        "label": "fixed-policy clean eval-mode ordinary-full-row parent KL preservation replicate seed62065",
        "zero_reading": "experiments/archive/functional_learning/data/repaired_clean_seed62065_zero_reading/per_target/clean_pres_lambda1_eval_seed62065_u0080_zero_reading.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_clean_seed62065_superglue/per_target/clean_pres_lambda1_eval_seed62065_u0080.json",
        "aoa": "experiments/archive/functional_learning/data/batched_aoa_measured/clean_pres_lambda1_eval_seed62065/full/aoa_manifest.json",
    },
    "densemask_sparselabel_seed62064": {
        "label": "exact acquisition-only dense-mask/sparse-label (M,S) seed62064 endpoint",
        "zero_reading": "experiments/archive/functional_learning/data/repaired_densemask_sparselabel_zero_reading/per_target/densemask_sparselabel_seed62064_u0080_zero_reading.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_densemask_sparselabel_superglue/per_target/densemask_sparselabel_seed62064_u0080.json",
        "aoa": "experiments/archive/functional_learning/data/batched_aoa_measured/densemask_sparselabel_seed62064/full/aoa_manifest.json",
    },
}


def rel(path: pathlib.Path | str | None) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def import_base():
    spec = importlib.util.spec_from_file_location("base_same_coordinate", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def component_delta(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]], keys: List[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    ca = ((a or {}).get("overall_computation") or {}).get("components") or {}
    cb = ((b or {}).get("overall_computation") or {}).get("components") or {}
    for k in keys:
        va, vb = ca.get(k), cb.get(k)
        try:
            out[k] = float(va) - float(vb)
        except Exception:
            out[k] = None
    return out


def overall_delta(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[float]:
    try:
        va = ((a or {}).get("overall_computation") or {}).get("Overall")
        vb = ((b or {}).get("overall_computation") or {}).get("Overall")
        if va is None or vb is None:
            return None
        return float(va) - float(vb)
    except Exception:
        return None


def gp_zero_delta_vs_base(rec: Dict[str, Any], base: Dict[str, Any], components: List[str]) -> Optional[float]:
    """Overall delta if candidate's GlobalPIQA is forced to equal base.

    Uses the same denominator as the official Overall and only returns a value
    for complete records.
    """
    if not rec.get("complete_same_coordinate") or not base.get("complete_same_coordinate"):
        return None
    c = (rec.get("overall_computation") or {}).get("components") or {}
    b = (base.get("overall_computation") or {}).get("components") or {}
    try:
        total = 0.0
        for k in components:
            vc = b["GlobalPIQA"] if k == "GlobalPIQA" else c[k]
            total += float(vc) - float(b[k])
        return total / len(components)
    except Exception:
        return None


def build_result(out_dir: pathlib.Path) -> Dict[str, Any]:
    base = import_base()
    specs = dict(base.MODEL_SPECS)
    specs.update(EXTENDED_MODEL_SPECS)

    records = {name: base.model_record(name, spec) for name, spec in specs.items()}
    deltas = base.add_deltas(records)
    coherent = records.get("coherent86", {})
    clean64 = records.get("clean_pres_lambda1_eval_seed62064", {})
    clean65 = records.get("clean_pres_lambda1_eval_seed62065", {})
    ms64 = records.get("densemask_sparselabel_seed62064", {})

    method_comp = {
        "clean_seed62064_vs_exact_ms_seed62064": {
            "available": bool(clean64.get("complete_same_coordinate") and ms64.get("complete_same_coordinate")),
            "overall_delta_clean_minus_ms": overall_delta(clean64, ms64),
            "component_deltas_clean_minus_ms": component_delta(clean64, ms64, base.OVERALL_COMPONENTS),
            "interpretation": "This is the direct preservation increment only after the exact (M,S) zero/Reading, SuperGLUE, and AoA sources are complete; before then it remains an awaiting comparison.",
        },
        "clean_seed62065_replication_vs_clean_seed62064": {
            "available": bool(clean64.get("complete_same_coordinate") and clean65.get("complete_same_coordinate")),
            "overall_delta_seed65_minus_seed64": overall_delta(clean65, clean64),
            "component_deltas_seed65_minus_seed64": component_delta(clean65, clean64, base.OVERALL_COMPONENTS),
            "interpretation": "This evaluates fixed-policy seed-level official replication; mechanism/fast replication is not enough without complete official components.",
        },
    }

    sensitivity = {}
    for name, rec in records.items():
        sensitivity[name] = {
            "complete": bool(rec.get("complete_same_coordinate")),
            "official_delta_vs_coherent86": (deltas.get(name) or {}).get("delta_vs_coherent86"),
            "delta_vs_coherent86_with_GlobalPIQA_difference_zeroed": gp_zero_delta_vs_base(rec, coherent, base.OVERALL_COMPONENTS),
        }

    completed = [n for n, r in records.items() if r.get("complete_same_coordinate")]
    result = {
        "status": "SAME_COORDINATE_ALL_CANDIDATES",
        "created_utc": now(),
        "coordinate": "full official-sized zero-shot/Reading + repaired AutoModel SuperGLUE primary metrics + measured batched AoA; no fast-screen substitutions",
        "source_base_script": rel(BASE_PATH),
        "extended_model_specs": EXTENDED_MODEL_SPECS,
        "historical_platform_records": base.HISTORICAL_PLATFORM_RECORDS,
        "models": records,
        "deltas_vs_coherent86": deltas,
        "method_comparisons": method_comp,
        "globalpiqa_zero_sensitivity": sensitivity,
        "interpretation": {
            "completed_models": completed,
            "clean_seed62064_state": "complete" if clean64.get("complete_same_coordinate") else "incomplete",
            "clean_seed62065_state": "complete" if clean65.get("complete_same_coordinate") else "awaiting official zero-shot/Reading and/or repaired SuperGLUE payloads; AoA is already measured if its AoA record is complete",
            "ms_seed62064_state": "complete" if ms64.get("complete_same_coordinate") else "awaiting official zero-shot/Reading and/or repaired SuperGLUE payloads; AoA is already measured if its AoA record is complete",
            "no_hybrid_rule": "Partial live payloads, fast screens, and repaired-loading-incompatible historical scores cannot enter Overall arithmetic.",
        },
    }
    return result


def write_markdown(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research same-coordinate table for replication and (M,S) comparison\n\n")
    lines.append("Only complete compatible sources enter Overall arithmetic: full official-sized zero-shot/Reading, repaired AutoModel SuperGLUE primary metrics, and measured batched AoA.\n\n")
    lines.append("## Model summary\n\n")
    deltas = result.get("deltas_vs_coherent86", {})
    for name, rec in result.get("models", {}).items():
        comp = rec.get("overall_computation", {})
        d = deltas.get(name, {})
        lines.append(f"### {name}\n")
        lines.append(f"- complete same-coordinate: `{rec.get('complete_same_coordinate')}`\n")
        lines.append(f"- Overall: `{comp.get('Overall')}`\n")
        lines.append(f"- delta vs coherent86: `{d.get('delta_vs_coherent86')}`\n")
        lines.append(f"- components: `{json.dumps(comp.get('components', {}), ensure_ascii=False)}`\n")
        errs = rec.get("errors") or []
        if errs:
            lines.append("- blocking errors: " + ", ".join(f"`{e}`" for e in errs[:18]) + (" ..." if len(errs) > 18 else "") + "\n")
        lines.append("\n")
    lines.append("## Method comparisons\n\n")
    lines.append(json.dumps(result.get("method_comparisons", {}), indent=2, ensure_ascii=False) + "\n\n")
    lines.append("## GlobalPIQA-zero sensitivity\n\n")
    lines.append(json.dumps(result.get("globalpiqa_zero_sensitivity", {}), indent=2, ensure_ascii=False) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build_result(args.out_dir)
    out_json = args.out_dir / "same_coordinate_all_candidates.json"
    out_md = args.out_dir / "same_coordinate_all_candidates.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "completed_models": result.get("interpretation", {}).get("completed_models"),
        "clean_seed65_state": result.get("interpretation", {}).get("clean_seed62065_state"),
        "ms_seed64_state": result.get("interpretation", {}).get("ms_seed62064_state"),
        "method_comparisons": result.get("method_comparisons"),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
