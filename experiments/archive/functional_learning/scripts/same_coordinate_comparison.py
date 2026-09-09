#!/usr/bin/env python3
"""research: same-coordinate frontier comparison for clean eval-mode preservation.

Scientific purpose
------------------
The clean preservation endpoint must not be combined from fast-screen zero-shot
scores, repaired SuperGLUE scores, and measured AoA. This script builds a
single guarded comparison object from compatible sources only:

* official-sized zero-shot/Reading payloads produced by the research official
  wrapper on full-eval data;
* repaired AutoModel SuperGLUE payloads using the current primary-metric
  convention (F1 for MRPC/QQP, accuracy otherwise);
* measured batched AoA manifests with the platform-matching AoAEvaluator.

If any component is missing or incomplete, the model is marked incomplete and no
Overall is computed. Historical platform-style values are preserved only as a
separate coordinate.
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
import pathlib
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/same_coordinate_comparison')

ZERO_COLS = [
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GlobalPIQA_parallel",
    "GlobalPIQA_nonparallel",
    "Reading",
]
SG_TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
OVERALL_COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]

MODEL_SPECS: Dict[str, Dict[str, Optional[str]]] = {
    "coherent86": {
        "label": "Qiushi-BabyLM-36M-Strict-Small-v4 faithful repaired-loading reference",
        "zero_reading": "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_coherent86_eval/per_target/repaired_coherent86_alpha075.json",
        "aoa": "experiments/archive/functional_learning/data/batched_aoa_measured/full/coherent86/full/aoa_manifest.json",
    },
    "dense_seed62064": {
        "label": "dense unchanged-Qwen focus seed62064",
        "zero_reading": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62064/per_target/repaired_dense_seed62064_u0080.json",
        "aoa": "experiments/archive/functional_learning/data/batched_aoa_measured/full/dense_seed62064/full/aoa_manifest.json",
    },
    "dense_seed62065": {
        "label": "dense unchanged-Qwen focus seed62065",
        "zero_reading": "experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_dense_eval_seed62065/per_target/repaired_dense_seed62065_u0080.json",
        "aoa": "experiments/archive/functional_learning/data/batched_aoa_measured/full/dense_seed62065/full/aoa_manifest.json",
    },
    "clean_pres_lambda1_eval_seed62064": {
        "label": "clean eval-mode ordinary-full-row parent KL preservation seed62064",
        "zero_reading": "experiments/archive/functional_learning/data/repaired_clean_eval_zero_reading/per_target/clean_pres_lambda1_eval_seed62064_u0080_zero_reading.json",
        "superglue": "experiments/archive/functional_learning/data/repaired_clean_eval_superglue/per_target/clean_pres_lambda1_eval_seed62064_u0080.json",
        "aoa": "experiments/archive/functional_learning/data/batched_aoa_clean_eval_measured/clean_pres_lambda1_eval_seed62064/full/aoa_manifest.json",
    },
    "clean_pres_lambda1_eval_seed62065": {
        "label": "fixed-policy clean eval-mode preservation replicate seed62065 (official components not yet run)",
        "zero_reading": None,
        "superglue": None,
        "aoa": None,
    },
}

HISTORICAL_PLATFORM_RECORDS = {
    "coherent86_historical_platform_style_overall": {
        "Overall": 42.1210247099666,
        "interpretation": "Preserved historical uploaded/platform-style coordinate; not used for repaired-loading scientific comparisons because historical SuperGLUE used stock AutoModel and omitted private adapters.",
    }
}


def rel(p: pathlib.Path | str | None) -> Optional[str]:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path_s: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not path_s:
        return None, {"path": None, "exists": False, "sha256": None, "status": "not_configured"}
    p = ROOT / path_s if not pathlib.Path(path_s).is_absolute() else pathlib.Path(path_s)
    info = {"path": rel(p), "exists": p.exists(), "sha256": sha256_file(p)}
    if not p.exists():
        info["status"] = "missing"
        return None, info
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        info["status"] = "loaded"
        return obj, info
    except Exception as exc:
        info["status"] = "json_error"
        info["error"] = repr(exc)
        return None, info


def fnum(x: Any) -> Optional[float]:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def score_from_task(task: Dict[str, Any], col: str) -> Optional[float]:
    if col == "Reading":
        return fnum((task.get("scores") or {}).get("Reading") if isinstance(task.get("scores"), dict) else task.get("Reading"))
    return fnum(task.get("score"))


def is_full_eval_path(col: str, task: Dict[str, Any]) -> bool:
    data = str(task.get("data_path", ""))
    if col == "Reading":
        return "full_eval/reading" in data
    if col in {"GlobalPIQA_parallel", "GlobalPIQA_nonparallel"}:
        return "full_eval/global_piqa" in data or "full_eval" in data
    if col == "COMPS":
        return "full_eval/comps" in data
    return "full_eval" in data and "fast_eval" not in data


def parse_zero_reading(obj: Optional[Dict[str, Any]], info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"source": info, "complete": False, "scores": {}, "task_details": {}, "errors": []}
    if obj is None:
        out["errors"].append("payload_not_loaded")
        return out
    tasks = obj.get("tasks") or {}
    for col in ZERO_COLS:
        t = tasks.get(col)
        if not isinstance(t, dict):
            out["errors"].append(f"missing_task:{col}")
            continue
        score = score_from_task(t, col)
        ret = t.get("returncode")
        full = is_full_eval_path(col, t)
        detail = {
            "returncode": ret,
            "score": score,
            "data_path": t.get("data_path"),
            "revision_name": t.get("revision_name"),
            "full_eval_path_ok": full,
            "predictions": t.get("predictions"),
            "report": t.get("report"),
        }
        out["task_details"][col] = detail
        if ret != 0:
            out["errors"].append(f"nonzero_returncode:{col}:{ret}")
        if score is None:
            out["errors"].append(f"missing_score:{col}")
        if not full:
            out["errors"].append(f"not_full_eval_path:{col}")
        out["scores"][col] = score
    gp = None
    if out["scores"].get("GlobalPIQA_parallel") is not None and out["scores"].get("GlobalPIQA_nonparallel") is not None:
        gp = (float(out["scores"]["GlobalPIQA_parallel"]) + float(out["scores"]["GlobalPIQA_nonparallel"])) / 2.0
    out["scores"]["GlobalPIQA"] = gp
    out["model_path"] = obj.get("model_path")
    out["target"] = obj.get("target")
    out["complete"] = len(out["errors"]) == 0
    return out


def parse_superglue(obj: Optional[Dict[str, Any]], info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"source": info, "complete": False, "score": None, "primary_metric_details": [], "errors": []}
    if obj is None:
        out["errors"].append("payload_not_loaded")
        return out
    sg = (obj.get("tasks") or {}).get("SuperGLUE")
    if not isinstance(sg, dict):
        out["errors"].append("missing_superglue_task")
        return out
    subtasks = sg.get("tasks") or []
    by_task = {str(t.get("task")): t for t in subtasks if isinstance(t, dict)}
    for task in SG_TASKS:
        t = by_task.get(task)
        if not t:
            out["errors"].append(f"missing_superglue_subtask:{task}")
            continue
        if t.get("returncode") != 0:
            out["errors"].append(f"nonzero_superglue_subtask:{task}:{t.get('returncode')}")
    details = list(sg.get("superglue_primary_metric_details") or [])
    detail_tasks = {str(d.get("task")) for d in details if isinstance(d, dict)}
    for task in SG_TASKS:
        if task not in detail_tasks:
            out["errors"].append(f"missing_primary_metric_detail:{task}")
    score = fnum(sg.get("superglue_mean"))
    if score is None:
        out["errors"].append("missing_superglue_mean")
    out.update({
        "score": score,
        "coordinate": sg.get("superglue_coordinate"),
        "primary_metric_details": details,
        "legacy_accuracy_only": sg.get("superglue_mean_accuracy_only_legacy"),
        "target": obj.get("target"),
        "model_path": obj.get("model_path"),
    })
    out["complete"] = len(out["errors"]) == 0
    return out


def parse_aoa(obj: Optional[Dict[str, Any]], info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"source": info, "complete": False, "score": None, "errors": []}
    if obj is None:
        out["errors"].append("payload_not_loaded")
        return out
    payload = obj.get("aoa_payload_for_comparison") or {}
    score_obj = obj.get("score") or {}
    measured = bool(payload.get("measured", score_obj.get("measured")))
    legitimate_zero = bool(payload.get("legitimate_zero", score_obj.get("legitimate_zero")))
    n_results = fnum(payload.get("n_results") or (obj.get("assembled") or {}).get("summary", {}).get("n_results"))
    n_steps = fnum(payload.get("n_steps") or len((obj.get("assembled") or {}).get("steps") or []))
    score = fnum(payload.get("aoa_leaderboard_score", score_obj.get("aoa_leaderboard_score")))
    if not measured:
        out["errors"].append("aoa_not_measured")
    if score is None:
        out["errors"].append("aoa_score_missing")
    if int(n_results or 0) != 144090:
        out["errors"].append(f"aoa_n_results_not_144090:{n_results}")
    if int(n_steps or 0) != 18:
        out["errors"].append(f"aoa_n_steps_not_18:{n_steps}")
    if score == 0.0 and not legitimate_zero:
        out["errors"].append("zero_not_marked_legitimate")
    out.update({
        "score": score,
        "measured": measured,
        "legitimate_zero": legitimate_zero,
        "estimator_id": payload.get("estimator_id") or (obj.get("aoa_estimator") or {}).get("estimator_id"),
        "n_results": int(n_results) if n_results is not None else None,
        "n_steps": int(n_steps) if n_steps is not None else None,
        "target": obj.get("target"),
        "assembled_steps": (obj.get("assembled") or {}).get("steps"),
    })
    out["complete"] = len(out["errors"]) == 0
    return out


def combine_components(zero: Dict[str, Any], sg: Dict[str, Any], aoa: Dict[str, Any]) -> Dict[str, Any]:
    # Do not let a partial payload leak into the arithmetic surface.  Partial
    # scores remain visible inside zero_reading/superglue/aoa diagnostics, but
    # the component map used for comparison is null until the whole source is
    # complete.  This prevents a BoolQ-only or fast-screen hybrid from looking
    # like a real SuperGLUE/zero-shot component.
    zero_scores = zero.get("scores", {}) if zero.get("complete") else {}
    sg_score = sg.get("score") if sg.get("complete") else None
    aoa_score = aoa.get("score") if aoa.get("complete") else None
    components = {
        "BLiMP": zero_scores.get("BLiMP"),
        "Supplement": zero_scores.get("Supplement"),
        "EWoK": zero_scores.get("EWoK"),
        "Entity": zero_scores.get("Entity"),
        "COMPS": zero_scores.get("COMPS"),
        "SuperGLUE": sg_score,
        "GlobalPIQA": zero_scores.get("GlobalPIQA"),
        "Reading": zero_scores.get("Reading"),
        "AoA": aoa_score,
    }
    missing = [k for k, v in components.items() if v is None]
    complete = zero.get("complete") and sg.get("complete") and aoa.get("complete") and not missing
    overall = sum(float(components[k]) for k in OVERALL_COMPONENTS) / len(OVERALL_COMPONENTS) if complete else None
    return {"components": components, "missing_components": missing, "complete": bool(complete), "Overall": overall}


def model_record(name: str, spec: Dict[str, Optional[str]]) -> Dict[str, Any]:
    zero_obj, zero_info = load_json(spec.get("zero_reading"))
    sg_obj, sg_info = load_json(spec.get("superglue"))
    aoa_obj, aoa_info = load_json(spec.get("aoa"))
    zero = parse_zero_reading(zero_obj, zero_info)
    sg = parse_superglue(sg_obj, sg_info)
    aoa = parse_aoa(aoa_obj, aoa_info)
    comp = combine_components(zero, sg, aoa)
    errors = list(zero.get("errors", [])) + list(sg.get("errors", [])) + list(aoa.get("errors", [])) + [f"missing_component:{m}" for m in comp.get("missing_components", [])]
    return {
        "name": name,
        "label": spec.get("label"),
        "comparison_coordinate": "faithful repaired-loading full official-sized zero-shot/Reading + repaired AutoModel SuperGLUE primary metrics + measured batched AoA",
        "component_sources": {"zero_reading": zero_info, "superglue": sg_info, "aoa": aoa_info},
        "zero_reading": zero,
        "superglue": sg,
        "aoa": aoa,
        "overall_computation": comp,
        "complete_same_coordinate": bool(comp.get("complete")),
        "errors": errors,
    }


def add_deltas(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    base = records.get("coherent86", {}).get("overall_computation", {})
    base_components = base.get("components") or {}
    base_overall = base.get("Overall")
    out: Dict[str, Any] = {}
    for name, rec in records.items():
        comp = rec.get("overall_computation", {})
        components = comp.get("components") or {}
        cdelta = {}
        for k in OVERALL_COMPONENTS:
            a, b = fnum(components.get(k)), fnum(base_components.get(k))
            cdelta[k] = (a - b) if a is not None and b is not None else None
        overall = fnum(comp.get("Overall"))
        out[name] = {
            "overall": overall,
            "delta_vs_coherent86": (overall - base_overall) if overall is not None and base_overall is not None else None,
            "component_deltas_vs_coherent86": cdelta,
            "complete": rec.get("complete_same_coordinate"),
        }
    completed = [name for name, rec in out.items() if rec.get("complete")]
    dense_names = [n for n in ["dense_seed62064", "dense_seed62065"] if out.get(n, {}).get("complete")]
    dense_deltas = [out[n].get("delta_vs_coherent86") for n in dense_names if out[n].get("delta_vs_coherent86") is not None]
    out["summary"] = {
        "completed_models": completed,
        "dense_completed": dense_names,
        "dense_mean_delta_vs_coherent86": sum(dense_deltas) / len(dense_deltas) if dense_deltas else None,
        "clean_seed62064_complete": records.get("clean_pres_lambda1_eval_seed62064", {}).get("complete_same_coordinate", False),
        "clean_seed62064_state": "complete" if records.get("clean_pres_lambda1_eval_seed62064", {}).get("complete_same_coordinate") else "awaiting official zero-shot/Reading and/or repaired SuperGLUE payloads",
    }
    return out


def write_markdown(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research same-coordinate comparison\n\n")
    lines.append("This file combines only compatible full official-sized zero-shot/Reading, repaired AutoModel SuperGLUE, and measured AoA sources. Fast-screen scores and historical stripped-AutoModel values are not mixed into this coordinate.\n\n")
    lines.append("## Historical coordinate kept separate\n\n")
    lines.append(json.dumps(result.get("historical_platform_records", {}), indent=2, ensure_ascii=False) + "\n\n")
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
            lines.append("- blocking errors: " + ", ".join(f"`{e}`" for e in errs[:16]) + (" ..." if len(errs) > 16 else "") + "\n")
        lines.append("\n")
    lines.append("## Delta table\n\n")
    lines.append(json.dumps(deltas, indent=2, ensure_ascii=False) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = {name: model_record(name, spec) for name, spec in MODEL_SPECS.items()}
    deltas = add_deltas(records)
    result = {
        "status": "SAME_COORDINATE_COMPARISON",
        "created_utc": now(),
        "coordinate": "full official-sized zero-shot/Reading + repaired AutoModel SuperGLUE primary metrics + measured batched AoA; no fast-screen substitutions",
        "historical_platform_records": HISTORICAL_PLATFORM_RECORDS,
        "models": records,
        "deltas_vs_coherent86": deltas,
        "interpretation": {
            "clean_seed62064_current_state": deltas.get("summary", {}).get("clean_seed62064_state"),
            "dense_status": "dense62064/62065 remain the completed same-coordinate trade-shaped positive reference while clean seed62064 official components are pending",
            "no_hybrid_rule": "A model with missing zero-shot/Reading or SuperGLUE payloads is incomplete even if fast-screen or partial files exist.",
        },
    }
    out_json = args.out_dir / "same_coordinate_comparison.json"
    out_md = args.out_dir / "same_coordinate_comparison.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "completed_models": deltas.get("summary", {}).get("completed_models"), "clean_state": deltas.get("summary", {}).get("clean_seed62064_state")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
