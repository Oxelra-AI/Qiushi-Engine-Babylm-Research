#!/usr/bin/env python3
"""research: aggregate split paired-seed evaluation components from separate evaluation paths.

This is a source reconciliation tool, not a scorer.  It admits independently run
component payloads only when the expected file exists and the payload reports a
valid completed component.  It can combine official-entry outputs
and research single-component outputs, keeping provenance for each column so
paired-seed conclusions are not separated from their measurement source.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from statistics import mean
from typing import Any, Dict, Iterable, Optional

ROOT = _public_path('.')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/split_eval_aggregation')

ZERO_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
SG_TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
SG_PRIMARY = {"boolq": "accuracy", "multirc": "accuracy", "rte": "accuracy", "wsc": "accuracy", "mrpc": "f1", "qqp": "f1", "mnli": "accuracy"}
ENDPOINT_LABELS = {
    "o62065": "ordinary_inherited_wwm_seed62065",
    "ms62065": "ms_acquisition_seed62065",
}

# Known single-source component path candidates, in priority order.  Official
# entry points are preferred when both exist for the same endpoint/component; independent
# research payloads are kept as independent/cross-check evidence.
def a01_zero_paths(endpoint: str, col: str) -> list[pathlib.Path]:
    paths = []
    if endpoint == "ms62065" and col == "BLiMP":
        paths.append(_public_path('experiments/archive/functional_learning/data/parallel_eval/ms62065_BLiMP/result_BLiMP.json'))
    paths.append(ROOT / f"experiments/archive/functional_learning/data/parallel_zero_reading/{endpoint}_{col}/result_{col}.json")
    return paths


def a02_zero_paths(endpoint: str, col: str) -> list[pathlib.Path]:
    return [ROOT / f"experiments/archive/relation_learning/data/{endpoint}_{col}/{endpoint}/with_special/{col}_component_payload.json"]


def a01_sg_paths(endpoint: str, task: str) -> list[pathlib.Path]:
    return [ROOT / f"experiments/archive/functional_learning/data/parallel_superglue/{endpoint}_{task}/superglue_task_result.json"]


def a02_sg_paths(endpoint: str, task: str) -> list[pathlib.Path]:
    return [ROOT / f"experiments/archive/relation_learning/data/superglue_single_tasks/{endpoint}/ftseed42/{task}/superglue_single_payload.json"]


def aoa_paths(endpoint: str) -> list[pathlib.Path]:
    paths = [ROOT / f"experiments/archive/functional_learning/data/batched_aoa_measured/{endpoint}/full/aoa_manifest.json"]
    paths.append(ROOT / f"experiments/archive/relation_learning/data/aoa_measured/{endpoint}/full/aoa_manifest.json")
    return paths


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_exists_rel(path_text: Any) -> bool:
    if not path_text:
        return False
    p = pathlib.Path(str(path_text))
    if not p.is_absolute():
        p = ROOT / p
    return p.exists()


def parse_zero_payload(path: pathlib.Path, source: str, endpoint: str, col: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    data = load_json(path)
    score = data.get("score")
    valid = False
    if data.get("status") in {"ZERO_READING_COMPONENT_DONE", "COLUMN_EVAL_DONE"}:
        valid = bool(data.get("valid_for_component_aggregation", True) and score is not None)
        task_data = data.get("task_data") or {}
        if isinstance(task_data, dict):
            valid = valid and task_data.get("returncode") == 0
    elif data.get("status") == "SINGLE_COMPONENT_DONE":
        valid = bool(score is not None and data.get("column") == col and data.get("label") == endpoint)
        ident = data.get("model_identity") or {}
        valid = valid and ident.get("loaded_class") == "FrozenSlowPrivateDebertaV2ForMaskedLM" and int(ident.get("private_params", -1)) == 995584
    else:
        valid = False
    pred = None
    report = None
    if isinstance(data.get("task_data"), dict):
        pred = data["task_data"].get("predictions")
        report = data["task_data"].get("report")
    if isinstance(data.get("record"), dict):
        pred = data["record"].get("predictions", pred)
        report = data["record"].get("report", report)
    return {
        "source": source,
        "path": rel(path),
        "status": data.get("status"),
        "endpoint": endpoint,
        "column": col,
        "score": float(score) if score is not None else None,
        "valid": bool(valid),
        "predictions": pred,
        "predictions_exists": file_exists_rel(pred) if pred else None,
        "report": report,
        "report_exists": file_exists_rel(report) if report else None,
        "gpu": data.get("physical_gpu_arg", data.get("physical_gpu")),
        "gpu_snapshot_indices": (((data.get("gpu_snapshot") or {}).get("this_pid_physical_gpu_indices")) if isinstance(data.get("gpu_snapshot"), dict) else None),
        "elapsed_sec": data.get("elapsed_sec") or ((data.get("task_data") or {}).get("elapsed_sec") if isinstance(data.get("task_data"), dict) else None),
    }


def parse_sg_payload(path: pathlib.Path, source: str, endpoint: str, task: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    data = load_json(path)
    if data.get("status") == "SUPERGLUE_TASK_DONE":
        score = data.get("primary_score")
        metric = data.get("primary_metric")
        valid = bool(data.get("valid_for_primary_superglue_aggregation") and score is not None and metric == SG_PRIMARY[task] and data.get("returncode") == 0)
        pred = data.get("predictions")
        results_txt = data.get("results_txt")
        gpu_indices = None
        gpu = data.get("physical_gpu_arg")
    elif data.get("status") == "SUPERGLUE_SINGLE_DONE":
        score = data.get("primary_score")
        metric = data.get("primary_metric")
        valid = bool(score is not None and metric == SG_PRIMARY[task] and data.get("task") == task and data.get("label") == endpoint)
        pred = data.get("predictions")
        results_txt = data.get("results_txt")
        gpu_indices = (data.get("child_snapshot") or {}).get("child_physical_gpu_indices") if isinstance(data.get("child_snapshot"), dict) else None
        gpu = data.get("requested_physical_gpu")
    else:
        score = data.get("primary_score")
        metric = data.get("primary_metric")
        valid = False
        pred = data.get("predictions")
        results_txt = data.get("results_txt")
        gpu_indices = None
        gpu = None
    return {
        "source": source,
        "path": rel(path),
        "status": data.get("status"),
        "endpoint": endpoint,
        "task": task,
        "primary_metric": metric,
        "primary_score": float(score) if score is not None else None,
        "valid": bool(valid),
        "predictions": pred,
        "predictions_exists": file_exists_rel(pred) if pred else None,
        "results_txt": results_txt,
        "results_txt_exists": file_exists_rel(results_txt) if results_txt else None,
        "gpu": gpu,
        "gpu_snapshot_indices": gpu_indices,
        "elapsed_sec": data.get("elapsed_sec"),
    }


def parse_aoa_manifest(path: pathlib.Path, source: str, endpoint: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    data = load_json(path)
    score = data.get("score") or {}
    assembled = data.get("assembled") or {}
    summary = assembled.get("summary") or {}
    valid = bool(data.get("complete_measured_evidence") and score.get("measured") is True and score.get("aoa_leaderboard_score") is not None)
    return {
        "source": source,
        "path": rel(path),
        "status": data.get("status"),
        "endpoint": endpoint,
        "aoa": float(score.get("aoa_leaderboard_score")) if score.get("aoa_leaderboard_score") is not None else None,
        "raw": score.get("aoa_raw_correlation"),
        "valid": valid,
        "complete_measured_evidence": data.get("complete_measured_evidence"),
        "n_results": summary.get("n_results"),
        "n_finite": summary.get("n_finite"),
        "missing_steps": summary.get("missing_steps"),
        "steps_with_wrong_counts": summary.get("steps_with_wrong_counts"),
    }


def pick_valid(records: Iterable[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    vals = [r for r in records if r and r.get("valid")]
    if not vals:
        return None
    # Prefer official-entry payloads over the independent scorer if both exist.
    vals.sort(key=lambda r: (0 if str(r.get("source", "")).startswith("A01") else 1, str(r.get("path"))))
    return vals[0]


def endpoint_status(endpoint: str) -> Dict[str, Any]:
    zero: Dict[str, Any] = {}
    for col in ZERO_COLS:
        recs = []
        for p in a01_zero_paths(endpoint, col):
            recs.append(parse_zero_payload(p, "A01_step109_or_step110", endpoint, col))
        for p in a02_zero_paths(endpoint, col):
            recs.append(parse_zero_payload(p, "A02_step127", endpoint, col))
        zero[col] = {"chosen": pick_valid(recs), "all_found": [r for r in recs if r is not None]}
    sg: Dict[str, Any] = {}
    for task in SG_TASKS:
        recs = []
        for p in a01_sg_paths(endpoint, task):
            recs.append(parse_sg_payload(p, "A01_step110", endpoint, task))
        for p in a02_sg_paths(endpoint, task):
            recs.append(parse_sg_payload(p, "A02_step127", endpoint, task))
        sg[task] = {"chosen": pick_valid(recs), "all_found": [r for r in recs if r is not None]}
    arecs = []
    for i, p in enumerate(aoa_paths(endpoint)):
        arecs.append(parse_aoa_manifest(p, "A01_step110" if i == 0 else "A02_step127_or_later", endpoint))
    aoa = {"chosen": pick_valid(arecs), "all_found": [r for r in arecs if r is not None]}

    scores = {k: None for k in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "SuperGLUE", "AoA"]}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        ch = zero[col]["chosen"]
        if ch:
            scores[col] = ch["score"]
    gp1 = zero["GlobalPIQA_parallel"]["chosen"]
    gp2 = zero["GlobalPIQA_nonparallel"]["chosen"]
    if gp1 and gp2:
        scores["GlobalPIQA"] = (gp1["score"] + gp2["score"]) / 2.0
    rd = zero["Reading"]["chosen"]
    if rd:
        scores["Reading"] = rd["score"]
    sg_chosen = [sg[t]["chosen"] for t in SG_TASKS]
    if all(sg_chosen):
        scores["SuperGLUE"] = mean(float(r["primary_score"]) for r in sg_chosen if r)
    if aoa["chosen"]:
        scores["AoA"] = aoa["chosen"]["aoa"]
    complete = all(v is not None for v in scores.values())
    return {
        "endpoint": endpoint,
        "endpoint_label": ENDPOINT_LABELS.get(endpoint),
        "zero_reading": zero,
        "superglue_tasks": sg,
        "aoa": aoa,
        "scores": scores,
        "complete_for_overall": complete,
        "Overall": mean(float(v) for v in scores.values()) if complete else None,
        "missing_scores": [k for k, v in scores.items() if v is None],
        "globalpiqa_subscores": {
            "parallel": gp1["score"] if gp1 else None,
            "nonparallel": gp2["score"] if gp2 else None,
        },
        "superglue_task_scores": {t: (sg[t]["chosen"] or {}).get("primary_score") for t in SG_TASKS},
    }


def write_md(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research split paired-seed evaluation aggregation\n\n")
    lines.append(f"Created: `{result['created_utc']}`\n\n")
    lines.append("This file reconciles independently executed component payloads. It does not alter scores or rerun evaluation. Official-entry payloads are preferred when both evaluation paths produce the same component; all found records remain listed in JSON.\n\n")
    for endpoint, rec in result["endpoints"].items():
        lines.append(f"## {endpoint} ({rec.get('endpoint_label')})\n\n")
        lines.append(f"Complete for Overall: `{rec['complete_for_overall']}`; Overall: `{rec.get('Overall')}`; missing: `{rec['missing_scores']}`\n\n")
        lines.append("| component | score | source | path |\n|---|---:|---|---|\n")
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
            ch = rec["zero_reading"][col]["chosen"]
            lines.append(f"| {col} | {ch.get('score') if ch else '-'} | {ch.get('source') if ch else '-'} | `{ch.get('path') if ch else ''}` |\n")
        lines.append(f"| GlobalPIQA_parallel | {rec['globalpiqa_subscores']['parallel']} | {(rec['zero_reading']['GlobalPIQA_parallel']['chosen'] or {}).get('source','-')} | `{(rec['zero_reading']['GlobalPIQA_parallel']['chosen'] or {}).get('path','')}` |\n")
        lines.append(f"| GlobalPIQA_nonparallel | {rec['globalpiqa_subscores']['nonparallel']} | {(rec['zero_reading']['GlobalPIQA_nonparallel']['chosen'] or {}).get('source','-')} | `{(rec['zero_reading']['GlobalPIQA_nonparallel']['chosen'] or {}).get('path','')}` |\n")
        lines.append(f"| GlobalPIQA mean | {rec['scores']['GlobalPIQA']} | derived | - |\n")
        rd = rec["zero_reading"]["Reading"]["chosen"]
        lines.append(f"| Reading | {rd.get('score') if rd else '-'} | {rd.get('source') if rd else '-'} | `{rd.get('path') if rd else ''}` |\n")
        lines.append(f"| AoA | {rec['scores']['AoA']} | {(rec['aoa']['chosen'] or {}).get('source','-')} | `{(rec['aoa']['chosen'] or {}).get('path','')}` |\n")
        lines.append(f"| SuperGLUE mean | {rec['scores']['SuperGLUE']} | derived from tasks | - |\n\n")
        lines.append("SuperGLUE tasks: " + ", ".join(f"{t}={rec['superglue_task_scores'][t]}" for t in SG_TASKS) + "\n\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "SPLIT_EVAL_AGGREGATION",
        "created_utc": now(),
        "endpoints": {ep: endpoint_status(ep) for ep in ["o62065", "ms62065"]},
    }
    out_json = _public_path('experiments/archive/functional_learning/data/split_eval_aggregation/split_eval_status.json')
    out_md = _public_path('research/documents/functional_learning/data/split_eval_aggregation/split_eval_status.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_md(out_md, result)
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "complete": {ep: result["endpoints"][ep]["complete_for_overall"] for ep in result["endpoints"]},
        "missing": {ep: result["endpoints"][ep]["missing_scores"] for ep in result["endpoints"]},
        "overall": {ep: result["endpoints"][ep]["Overall"] for ep in result["endpoints"]},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
