#!/usr/bin/env python3
"""research: guarded same-coordinate table with the ordinary-control rung.

This successor to research adds the verified research ordinary `inherited_wwm`
endpoint to the repaired fixed coordinate. It preserves the admission rule:
Overall is computed only when the endpoint has full official-sized zero/Reading,
repaired-AutoModel SuperGLUE primary metrics, and measured batched AoA. Fast
Cheap7 values and incomplete live payloads remain excluded.
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
BASE096_PATH = _public_path('experiments/archive/functional_learning/scripts/same_coordinate_all_candidates.py')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/same_coordinate_with_ordinary')

ORDINARY_SPEC: Dict[str, Dict[str, Optional[str]]] = {
    "ordinary_inherited_wwm_seed62064": {
        "label": "verified research ordinary inherited-WWM continuation seed62064 endpoint",
        "zero_reading": "experiments/archive/functional_learning/data/repaired_ordinary_zero_reading/per_target/ordinary_inherited_wwm_seed62064_u0080_zero_reading.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_ordinary_superglue/per_target/ordinary_inherited_wwm_seed62064_u0080.json",
        "aoa": "experiments/archive/functional_learning/data/batched_aoa_measured/ordinary_inherited_wwm_seed62064/full/aoa_manifest.json",
    }
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


def import_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
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


def component_partial_delta(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]], keys: List[str]) -> Dict[str, Optional[float]]:
    """Delta for all currently present component values, without implying Overall."""
    out: Dict[str, Optional[float]] = {}
    ca = ((a or {}).get("overall_computation") or {}).get("components") or {}
    cb = ((b or {}).get("overall_computation") or {}).get("components") or {}
    for k in keys:
        va, vb = ca.get(k), cb.get(k)
        if va is None or vb is None:
            out[k] = None
        else:
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


def present_component_sum(rec: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    comps = ((rec or {}).get("overall_computation") or {}).get("components") or {}
    present = []
    missing = []
    total = 0.0
    for k in keys:
        v = comps.get(k)
        if v is None:
            missing.append(k)
        else:
            try:
                total += float(v)
                present.append(k)
            except Exception:
                missing.append(k)
    return {"present_components": present, "missing_components": missing, "sum_present": total, "n_present": len(present)}


def gp_zero_delta_vs_base(rec: Dict[str, Any], base: Dict[str, Any], components: List[str]) -> Optional[float]:
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
    s96 = import_module(BASE096_PATH, "imported_step096_same_coordinate")
    base = s96.import_base()
    specs = dict(base.MODEL_SPECS)
    specs.update(s96.EXTENDED_MODEL_SPECS)
    specs.update(ORDINARY_SPEC)

    records = {name: base.model_record(name, spec) for name, spec in specs.items()}
    deltas = base.add_deltas(records)
    components = list(base.OVERALL_COMPONENTS)
    coherent = records.get("coherent86", {})
    ordinary = records.get("ordinary_inherited_wwm_seed62064", {})
    ms64 = records.get("densemask_sparselabel_seed62064", {})
    clean64 = records.get("clean_pres_lambda1_eval_seed62064", {})
    clean65 = records.get("clean_pres_lambda1_eval_seed62065", {})

    method_comp = {
        "ordinary_seed62064_vs_coherent86": {
            "available": bool(ordinary.get("complete_same_coordinate") and coherent.get("complete_same_coordinate")),
            "overall_delta_ordinary_minus_coherent86": overall_delta(ordinary, coherent),
            "component_deltas_ordinary_minus_coherent86": component_partial_delta(ordinary, coherent, components),
            "present_component_summary": present_component_sum(ordinary, components),
            "interpretation": "Parent -> ordinary measures additional ordinary WWM training on the same unchanged-Qwen rows. Until SuperGLUE and AoA are both present, this is only a partial component comparison.",
        },
        "ordinary_seed62064_vs_exact_ms_seed62064": {
            "available": bool(ordinary.get("complete_same_coordinate") and ms64.get("complete_same_coordinate")),
            "overall_delta_ordinary_minus_ms": overall_delta(ordinary, ms64),
            "component_deltas_ordinary_minus_ms": component_partial_delta(ordinary, ms64, components),
            "interpretation": "Ordinary -> (M,S) changes input corruption and target/credit allocation together, so it complements rather than replaces the sharper (S,S)->(M,S) masking contrast.",
        },
        "clean_seed62064_vs_exact_ms_seed62064": {
            "available": bool(clean64.get("complete_same_coordinate") and ms64.get("complete_same_coordinate")),
            "overall_delta_clean_minus_ms": overall_delta(clean64, ms64),
            "component_deltas_clean_minus_ms": component_delta(clean64, ms64, components),
        },
        "clean_seed62065_replication_vs_clean_seed62064": {
            "available": bool(clean64.get("complete_same_coordinate") and clean65.get("complete_same_coordinate")),
            "overall_delta_seed65_minus_seed64": overall_delta(clean65, clean64),
            "component_deltas_seed65_minus_seed64": component_delta(clean65, clean64, components),
        },
    }

    sensitivity = {}
    for name, rec in records.items():
        sensitivity[name] = {
            "complete": bool(rec.get("complete_same_coordinate")),
            "official_delta_vs_coherent86": (deltas.get(name) or {}).get("delta_vs_coherent86"),
            "delta_vs_coherent86_with_GlobalPIQA_difference_zeroed": gp_zero_delta_vs_base(rec, coherent, components),
        }

    completed = [n for n, r in records.items() if r.get("complete_same_coordinate")]
    result = {
        "status": "SAME_COORDINATE_WITH_ORDINARY",
        "created_utc": now(),
        "coordinate": "full official-sized zero-shot/Reading + repaired AutoModel SuperGLUE primary metrics + measured batched AoA; no fast-screen substitutions",
        "source_base_script_step096": rel(BASE096_PATH),
        "ordinary_spec": ORDINARY_SPEC,
        "historical_platform_records": base.HISTORICAL_PLATFORM_RECORDS,
        "models": records,
        "deltas_vs_coherent86": deltas,
        "method_comparisons": method_comp,
        "globalpiqa_zero_sensitivity": sensitivity,
        "interpretation": {
            "completed_models": completed,
            "ordinary_state": "complete" if ordinary.get("complete_same_coordinate") else "awaiting repaired-AutoModel SuperGLUE and/or measured AoA; zero/Reading and AoA may be present independently but no Overall is admitted until all sources are complete",
            "ordinary_no_hybrid_rule": "The existing Cheap7 ordinary values and the partial zero/Reading+AoA components cannot be averaged into an Overall. Rerun this script after the running ordinary SuperGLUE task delivers.",
            "contrast_graph": [
                "coherent86 -> ordinary_inherited_wwm_seed62064: extra ordinary WWM continuation on same unchanged-Qwen rows",
                "ordinary_inherited_wwm_seed62064 -> densemask_sparselabel_seed62064: joint change in Qwen input corruption and sparse focus target/credit allocation",
                "sparse (S,S) -> densemask_sparselabel (M,S): sharper effective-input masking contrast at fixed sparse labels/focus weighting",
                "densemask_sparselabel_seed62064 -> clean_pres_lambda1_eval_seed62064: ordinary full-row parent KL plus counted preservation presentations",
            ],
        },
    }
    return result


def write_markdown(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research same-coordinate table with ordinary-control rung\n\n")
    lines.append("Only complete compatible sources enter Overall arithmetic: full official-sized zero-shot/Reading, repaired AutoModel SuperGLUE primary metrics, and measured batched AoA.\n\n")
    lines.append("## Model summary\n\n")
    deltas = result.get("deltas_vs_coherent86", {})
    for name, rec in result.get("models", {}).items():
        comp = rec.get("overall_computation", {})
        d = deltas.get(name, {})
        present = present_component_sum(rec, list(import_module(BASE096_PATH, 'imported_step096_for_md').import_base().OVERALL_COMPONENTS))
        lines.append(f"### {name}\n")
        lines.append(f"- complete same-coordinate: `{rec.get('complete_same_coordinate')}`\n")
        lines.append(f"- Overall: `{comp.get('Overall')}`\n")
        lines.append(f"- delta vs coherent86: `{d.get('delta_vs_coherent86')}`\n")
        lines.append(f"- present components: `{present}`\n")
        lines.append(f"- components: `{json.dumps(comp.get('components', {}), ensure_ascii=False)}`\n")
        errs = rec.get("errors") or []
        if errs:
            lines.append("- blocking errors: " + ", ".join(f"`{e}`" for e in errs[:18]) + (" ..." if len(errs) > 18 else "") + "\n")
        lines.append("\n")
    lines.append("## Method comparisons\n\n")
    lines.append(json.dumps(result.get("method_comparisons", {}), indent=2, ensure_ascii=False) + "\n\n")
    lines.append("## Contrast graph\n\n")
    for item in result.get("interpretation", {}).get("contrast_graph", []):
        lines.append(f"- {item}\n")
    lines.append("\n## GlobalPIQA-zero sensitivity\n\n")
    lines.append(json.dumps(result.get("globalpiqa_zero_sensitivity", {}), indent=2, ensure_ascii=False) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build_result(args.out_dir)
    out_json = args.out_dir / "same_coordinate_with_ordinary.json"
    out_md = args.out_dir / "same_coordinate_with_ordinary.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "completed_models": result.get("interpretation", {}).get("completed_models"),
        "ordinary_state": result.get("interpretation", {}).get("ordinary_state"),
        "ordinary_present_component_summary": result.get("method_comparisons", {}).get("ordinary_seed62064_vs_coherent86", {}).get("present_component_summary"),
        "ordinary_vs_coherent_component_deltas": result.get("method_comparisons", {}).get("ordinary_seed62064_vs_coherent86", {}).get("component_deltas_ordinary_minus_coherent86"),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
