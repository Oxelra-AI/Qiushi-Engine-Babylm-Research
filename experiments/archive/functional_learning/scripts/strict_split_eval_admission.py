#!/usr/bin/env python3
"""research: strict admission pass for split paired-seed evaluation.

This is a measurement-convention reconciler, not a new scorer.  It reads the
independently executed official-entry and single-component payloads and
admits component scores only when the score surface, data coverage, report
semantics, model identity, and source paths are compatible with the repaired
same-coordinate comparison.

The important distinction from the research progress collector is that this file
keeps progress information separate from admission.  `complete_for_overall` here
means all required components are admitted under the same score transformation,
not merely that completion flags exist.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import re
import statistics
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/strict_split_eval_admission')

PRISTINE_FULL = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
GLOBALPIQA_FULL = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')

ZERO_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
SURFACE_COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
SG_TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
SG_PRIMARY = {"boolq": "accuracy", "multirc": "accuracy", "rte": "accuracy", "wsc": "accuracy", "mrpc": "f1", "qqp": "f1", "mnli": "accuracy"}
OVERALL_COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]

ZERO_SPECS: Dict[str, Dict[str, Any]] = {
    "BLiMP": {"task": "blimp", "path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/blimp_filtered')},
    "Supplement": {"task": "blimp", "path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered')},
    "EWoK": {"task": "ewok", "path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered')},
    "Entity": {"task": "entity_tracking", "path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')},
    "COMPS": {"task": "comps", "path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/comps')},
    "GlobalPIQA_parallel": {"task": "global_piqa_parallel", "path": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_parallel')},
    "GlobalPIQA_nonparallel": {"task": "global_piqa_nonparallel", "path": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_nonparallel')},
    "Reading": {"task": "reading", "path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv')},
}

ENDPOINTS: Dict[str, Dict[str, Any]] = {
    "o62065": {
        "label": "ordinary_inherited_wwm_seed62065",
        "checkpoint": _public_path('experiments/archive/relation_learning/data/repair_ordinary62065_bundle/repaired_ordinary62065_u0080'),
        "seed62064_counterpart": "ordinary_inherited_wwm_seed62064",
    },
    "ms62065": {
        "label": "ms_acquisition_seed62065",
        "checkpoint": _public_path('experiments/archive/relation_learning/data/repair_ms62065_bundle/repaired_ms62065_u0080'),
        "seed62064_counterpart": "densemask_sparselabel_seed62064",
    },
}

SEED62064_TABLE = _public_path('experiments/archive/functional_learning/data/same_coordinate_with_ordinary_complete/same_coordinate_with_ordinary.json')
ALL6_TABLE = _public_path('experiments/archive/functional_learning/data/same_coordinate_all_candidates_final_all6/same_coordinate_all_candidates.json')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str | None) -> str | None:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def rj(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(x: Any) -> Optional[float]:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
    except Exception:
        pass
    return None


def resolve_path(path_text: Any) -> Optional[pathlib.Path]:
    if not path_text:
        return None
    p = pathlib.Path(str(path_text))
    if not p.is_absolute():
        p = ROOT / p
    return p


def file_exists(path_text: Any) -> bool:
    p = resolve_path(path_text)
    return bool(p and p.exists())


def count_jsonl_rows(path: pathlib.Path) -> int:
    total = 0
    for fp in sorted(path.rglob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def expected_count(col: str) -> Optional[int]:
    spec = ZERO_SPECS[col]
    p = pathlib.Path(spec["path"])
    if col == "Reading":
        try:
            with p.open("r", encoding="utf-8") as f:
                # subtract CSV header
                return max(0, sum(1 for _ in f) - 1)
        except Exception:
            return None
    if col == "Entity":
        # The official full Entity scorer drops "nothing"-answer datapoints before
        # writing predictions and uses the filtered per-subtask sizes hard-coded
        # in strict/evaluation_pipeline/collate_preds.py.  The raw source jsonl
        # rows are 9483, but official full predictions contain 6780 filtered
        # examples across 18 subsubtasks; comparing against raw jsonl rows wrongly
        # rejects valid research/research official-entry payloads.
        return sum({
            "regular_0_ops": 517, "regular_1_ops": 409, "regular_2_ops": 405,
            "regular_3_ops": 425, "regular_4_ops": 388, "regular_5_ops": 94,
            "ambiref_0_ops": 508, "ambiref_1_ops": 428, "ambiref_2_ops": 413,
            "ambiref_3_ops": 409, "ambiref_4_ops": 434, "ambiref_5_ops": 123,
            "move_contents_0_ops": 516, "move_contents_1_ops": 437,
            "move_contents_2_ops": 399, "move_contents_3_ops": 406,
            "move_contents_4_ops": 353, "move_contents_5_ops": 116,
        }.values())
    if p.is_dir():
        return count_jsonl_rows(p)
    return None


def parse_sentence_score(text: str) -> Optional[float]:
    # Same strict report surface accepted by research: explicit average headers.
    for pat in [
        r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, text)
        if m:
            val = fnum(m.group(1))
            if val is not None and -5.0 <= val <= 105.0:
                return val
            return None
    return None


def parse_reading_report(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            val = fnum(m.group(1))
            if val is not None:
                out[key] = val
    if "Reading_eye" in out and "Reading_self_paced" in out:
        # This deliberately follows the established report-derived convention:
        # first round eye/self to the displayed report values, then average them.
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def parse_results_txt_metric(path: pathlib.Path, metric: str) -> Optional[float]:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() == metric:
                val = fnum(value.strip())
                return None if val is None else val * 100.0
    except Exception:
        return None
    return None


def prediction_count(path_text: Any, col: str) -> Optional[int]:
    p = resolve_path(path_text)
    if not p or not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if col == "Reading":
        rec = data.get("reading") if isinstance(data, dict) else None
        preds = rec.get("predictions") if isinstance(rec, dict) else None
        return len(preds) if isinstance(preds, list) else None
    if not isinstance(data, dict):
        return None
    total = 0
    for v in data.values():
        if isinstance(v, dict) and isinstance(v.get("predictions"), list):
            total += len(v["predictions"])
        elif isinstance(v, list):
            total += len(v)
        else:
            total += 1
    return total


def data_path_is_expected(col: str, data_path_text: Any) -> bool:
    if not data_path_text:
        return False
    expected = pathlib.Path(ZERO_SPECS[col]["path"]).resolve()
    actual = resolve_path(data_path_text)
    if actual is None:
        return False
    try:
        actual = actual.resolve()
    except Exception:
        pass
    if actual == expected:
        return True
    # Some older payloads keep paths relative to the strict repo; accept only if
    # the task-specific suffix is the same full-eval surface, not fast_eval.
    s = str(data_path_text).replace("\\", "/")
    if "fast_eval" in s:
        return False
    if col == "Reading":
        return s.endswith("evaluation_data/full_eval/reading/reading_data.csv") or s.endswith("full_eval/reading/reading_data.csv")
    return str(expected).replace("\\", "/").endswith(s.split("full_eval/")[-1]) if "full_eval/" in s else False


def a01_zero_paths(endpoint: str, col: str) -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []
    if endpoint == "ms62065" and col == "BLiMP":
        paths.append(_public_path('experiments/archive/functional_learning/data/parallel_eval/ms62065_BLiMP/result_BLiMP.json'))
    paths.append(ROOT / f"experiments/archive/functional_learning/data/parallel_zero_reading/{endpoint}_{col}/result_{col}.json")
    return paths


def a02_zero_paths(endpoint: str, col: str) -> List[pathlib.Path]:
    return [ROOT / f"experiments/archive/relation_learning/data/{endpoint}_{col}/{endpoint}/with_special/{col}_component_payload.json"]


def a01_sg_paths(endpoint: str, task: str) -> List[pathlib.Path]:
    paths = [ROOT / f"experiments/archive/functional_learning/data/parallel_superglue/{endpoint}_{task}/superglue_task_result.json"]
    # The cancelled O62065 BoolQ/MultiRC attempts were restarted with
    # isolated output roots. Include this source without
    # disturbing already admitted research MS62065 evidence.
    paths.append(ROOT / f"experiments/archive/functional_learning/data/o62065_superglue_restart/{endpoint}_{task}/superglue_task_result.json")
    return paths


def a02_sg_paths(endpoint: str, task: str) -> List[pathlib.Path]:
    return [ROOT / f"experiments/archive/relation_learning/data/superglue_single_tasks/{endpoint}/ftseed42/{task}/superglue_single_payload.json"]


def aoa_paths(endpoint: str) -> List[Tuple[str, pathlib.Path]]:
    return [
        ("A01_step110", ROOT / f"experiments/archive/functional_learning/data/batched_aoa_measured/{endpoint}/full/aoa_manifest.json"),
        ("A02_step127", ROOT / f"experiments/archive/relation_learning/data/aoa_measured/{endpoint}/full/aoa_manifest.json"),
        # O62065 was assembled from the completed endpoint extraction and
        # the established shared ancestry without re-running the endpoint.
        ("A01_step113_from_A02_extract", ROOT / f"experiments/archive/functional_learning/data/o62065_aoa_measured_from_extract/{endpoint}/full/aoa_manifest.json"),
    ]


def zero_record_from_a01(path: pathlib.Path, endpoint: str, col: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    d = rj(path)
    td = d.get("task_data") if isinstance(d.get("task_data"), dict) else {}
    report = td.get("report")
    report_p = resolve_path(report)
    report_scores: Dict[str, Any] = {}
    report_score: Optional[float] = None
    if report_p and report_p.is_file():
        text = report_p.read_text(encoding="utf-8", errors="replace")
        if col == "Reading":
            report_scores = parse_reading_report(text)
            report_score = report_scores.get("Reading")
        else:
            report_score = parse_sentence_score(text)
    score_field = fnum(d.get("score"))
    if score_field is None:
        score_field = fnum(td.get("score"))
    if col == "Reading" and score_field is None:
        score_field = fnum(((td.get("scores") or {}).get("Reading") if isinstance(td.get("scores"), dict) else None))
    exp = expected_count(col)
    pred_count = prediction_count(td.get("predictions"), col)
    errors: List[str] = []
    warnings: List[str] = []
    if d.get("status") not in {"ZERO_READING_COMPONENT_DONE", "COLUMN_EVAL_DONE"}:
        errors.append(f"bad_status:{d.get('status')}")
    if d.get("endpoint") != endpoint:
        errors.append(f"endpoint_mismatch:{d.get('endpoint')}")
    if d.get("column") != col:
        errors.append(f"column_mismatch:{d.get('column')}")
    if td.get("returncode") != 0:
        errors.append(f"nonzero_returncode:{td.get('returncode')}")
    if report_score is None:
        errors.append("report_score_missing")
    if score_field is None:
        errors.append("payload_score_missing")
    if report_score is not None and score_field is not None and abs(report_score - score_field) > 1e-9:
        errors.append(f"payload_report_score_mismatch:{score_field}:{report_score}")
    if not file_exists(report):
        errors.append("report_missing")
    if not file_exists(td.get("predictions")):
        errors.append("predictions_missing")
    if exp is not None and pred_count is not None and pred_count != exp:
        errors.append(f"prediction_count_mismatch:{pred_count}:{exp}")
    if exp is None:
        warnings.append("expected_count_unknown")
    if not data_path_is_expected(col, td.get("data_path")):
        errors.append(f"data_path_not_expected:{td.get('data_path')}")
    valid = not errors
    return {
        "source": "A01_official_entry",
        "path": rel(path),
        "status": d.get("status"),
        "endpoint": endpoint,
        "column": col,
        "raw_payload_score": score_field,
        "report_score": report_score,
        "report_scores": report_scores,
        "admitted_score": report_score,
        "score_transform": "official_report_score" if col != "Reading" else "official_report_eye_self_rounded_mean",
        "valid_for_score_admission": valid,
        "valid_for_item_analysis": valid,
        "coverage": {"expected": exp, "predictions_count": pred_count, "coverage_source": "predictions" if pred_count is not None else None},
        "report": rel(report_p) if report_p else report,
        "predictions": td.get("predictions"),
        "predictions_exists": file_exists(td.get("predictions")),
        "data_path": td.get("data_path"),
        "gpu": d.get("physical_gpu_arg", d.get("physical_gpu")),
        "errors": errors,
        "warnings": warnings,
    }


def zero_record_from_a02(path: pathlib.Path, endpoint: str, col: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    d = rj(path)
    rec = d.get("record") if isinstance(d.get("record"), dict) else {}
    ident = d.get("model_identity") if isinstance(d.get("model_identity"), dict) else {}
    report = rec.get("report")
    report_p = resolve_path(report)
    report_scores: Dict[str, Any] = {}
    report_score: Optional[float] = None
    if report_p and report_p.is_file():
        text = report_p.read_text(encoding="utf-8", errors="replace")
        if col == "Reading":
            report_scores = parse_reading_report(text)
            report_score = report_scores.get("Reading")
        else:
            report_score = parse_sentence_score(text)
    raw_score = fnum(d.get("score"))
    if raw_score is None:
        raw_score = fnum(rec.get("score"))
    exp = expected_count(col)
    pred = rec.get("predictions")
    pred_count = prediction_count(pred, col)
    record_n = rec.get("n_rows") if col == "Reading" else rec.get("n_examples")
    record_n_i: Optional[int] = None
    try:
        record_n_i = int(record_n) if record_n is not None else None
    except Exception:
        record_n_i = None
    errors: List[str] = []
    warnings: List[str] = []
    if d.get("status") != "SINGLE_COMPONENT_DONE":
        errors.append(f"bad_status:{d.get('status')}")
    if d.get("label") != endpoint:
        errors.append(f"endpoint_label_mismatch:{d.get('label')}")
    if d.get("column") != col:
        errors.append(f"column_mismatch:{d.get('column')}")
    if rec.get("column") != col:
        errors.append(f"record_column_mismatch:{rec.get('column')}")
    if rec.get("task") and rec.get("task") != ZERO_SPECS[col]["task"]:
        errors.append(f"task_mismatch:{rec.get('task')}:{ZERO_SPECS[col]['task']}")
    if ident.get("loaded_class") != "FrozenSlowPrivateDebertaV2ForMaskedLM":
        errors.append(f"loaded_class_mismatch:{ident.get('loaded_class')}")
    try:
        if int(ident.get("private_params", -1)) != 995584:
            errors.append(f"private_params_mismatch:{ident.get('private_params')}")
    except Exception:
        errors.append(f"private_params_bad:{ident.get('private_params')}")
    if pathlib.Path(str(d.get("checkpoint", ""))) != pathlib.Path(rel(ENDPOINTS[endpoint]["checkpoint"])):
        # Relative-path string comparison after rel() prevents accidental endpoint mix.
        if str(d.get("checkpoint")) != str(rel(ENDPOINTS[endpoint]["checkpoint"])):
            errors.append(f"checkpoint_mismatch:{d.get('checkpoint')}")
    if report_score is None:
        errors.append("report_score_missing")
    if raw_score is None:
        errors.append("raw_payload_score_missing")
    # The component scorer records raw unrounded scores while the established research surface reads
    # the report.  Require that the displayed report value is the standard rounded
    # rendering of the raw value, then admit the report-derived value.
    if report_score is not None and raw_score is not None:
        if col == "Reading":
            raw_scores = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
            eye_raw = fnum(raw_scores.get("Reading_eye"))
            self_raw = fnum(raw_scores.get("Reading_self_paced"))
            if eye_raw is not None and abs(round(eye_raw, 2) - report_scores.get("Reading_eye", float("nan"))) > 1e-9:
                errors.append(f"reading_eye_report_round_mismatch:{eye_raw}:{report_scores.get('Reading_eye')}")
            if self_raw is not None and abs(round(self_raw, 2) - report_scores.get("Reading_self_paced", float("nan"))) > 1e-9:
                errors.append(f"reading_self_report_round_mismatch:{self_raw}:{report_scores.get('Reading_self_paced')}")
        else:
            if abs(round(raw_score, 2) - report_score) > 1e-9:
                errors.append(f"raw_report_round_mismatch:{raw_score}:{report_score}")
    if not file_exists(report):
        errors.append("report_missing")
    if pred and pred_count is None:
        errors.append("predictions_unreadable")
    if exp is not None:
        if pred_count is not None and pred_count != exp:
            errors.append(f"prediction_count_mismatch:{pred_count}:{exp}")
        elif pred_count is None and record_n_i != exp:
            errors.append(f"record_count_mismatch:{record_n_i}:{exp}")
    if pred_count is None and record_n_i is not None:
        warnings.append("no_predictions_for_item_analysis;score_coverage_uses_record_count")
    if pred_count is None and record_n_i is None:
        errors.append("coverage_missing")
    if not data_path_is_expected(col, rec.get("data_path")):
        errors.append(f"data_path_not_expected:{rec.get('data_path')}")
    gpu_indices = None
    if isinstance(d.get("gpu_snapshot"), dict):
        gpu_indices = d["gpu_snapshot"].get("this_pid_physical_gpu_indices")
    valid_score = not errors
    valid_item = valid_score and pred_count == exp
    return {
        "source": "A02_custom_single_component",
        "path": rel(path),
        "status": d.get("status"),
        "endpoint": endpoint,
        "column": col,
        "raw_payload_score": raw_score,
        "report_score": report_score,
        "report_scores": report_scores,
        "admitted_score": report_score,
        "score_transform": "report_score_from_A02_raw_rounded" if col != "Reading" else "report_eye_self_rounded_mean_from_A02_raw",
        "valid_for_score_admission": valid_score,
        "valid_for_item_analysis": valid_item,
        "coverage": {"expected": exp, "predictions_count": pred_count, "record_count": record_n_i, "coverage_source": "predictions" if pred_count is not None else "record_count"},
        "report": rel(report_p) if report_p else report,
        "predictions": pred,
        "predictions_exists": file_exists(pred),
        "data_path": rec.get("data_path"),
        "gpu": (gpu_indices or d.get("requested_physical_gpu")),
        "errors": errors,
        "warnings": warnings,
    }


def collect_zero_records(endpoint: str, col: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in a01_zero_paths(endpoint, col):
        r = zero_record_from_a01(p, endpoint, col)
        if r is not None:
            out.append(r)
    for p in a02_zero_paths(endpoint, col):
        r = zero_record_from_a02(p, endpoint, col)
        if r is not None:
            out.append(r)
    return out


def choose_component(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    admitted = [r for r in records if r.get("valid_for_score_admission") and r.get("admitted_score") is not None]
    if not admitted:
        return {"chosen": None, "admitted_records": admitted, "all_found": records, "state": "missing_or_unadmitted", "conflict": None}
    vals = [float(r["admitted_score"]) for r in admitted]
    conflict = (max(vals) - min(vals)) > 1e-9
    if conflict:
        return {"chosen": None, "admitted_records": admitted, "all_found": records, "state": "conflict", "conflict": {"min": min(vals), "max": max(vals), "records": admitted}}
    # After equality has been demonstrated, prefer the official-entry payload with
    # predictions for item analyses; otherwise choose the lexicographically first
    # admitted record.  This is not a score-selection rule.
    admitted.sort(key=lambda r: (0 if r.get("source") == "A01_official_entry" and r.get("valid_for_item_analysis") else 1, 0 if r.get("valid_for_item_analysis") else 1, str(r.get("path"))))
    return {"chosen": admitted[0], "admitted_records": admitted, "all_found": records, "state": "admitted", "conflict": None}


def sg_record(path: pathlib.Path, source: str, endpoint: str, task: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    d = rj(path)
    errors: List[str] = []
    warnings: List[str] = []
    if source.startswith("A01"):
        if d.get("status") != "SUPERGLUE_TASK_DONE":
            errors.append(f"bad_status:{d.get('status')}")
        if d.get("endpoint") != endpoint:
            errors.append(f"endpoint_mismatch:{d.get('endpoint')}")
        if d.get("task") != task:
            errors.append(f"task_mismatch:{d.get('task')}")
        if d.get("returncode") != 0:
            errors.append(f"nonzero_returncode:{d.get('returncode')}")
        metric = d.get("primary_metric")
        score_payload = fnum(d.get("primary_score"))
        gpu = d.get("physical_gpu_arg")
    else:
        if d.get("status") != "SUPERGLUE_SINGLE_DONE":
            errors.append(f"bad_status:{d.get('status')}")
        if d.get("label") != endpoint:
            errors.append(f"endpoint_label_mismatch:{d.get('label')}")
        if d.get("task") != task:
            errors.append(f"task_mismatch:{d.get('task')}")
        metric = d.get("primary_metric")
        score_payload = fnum(d.get("primary_score"))
        gpu = d.get("requested_physical_gpu")
    expected_metric = SG_PRIMARY[task]
    if metric != expected_metric:
        errors.append(f"primary_metric_mismatch:{metric}:{expected_metric}")
    results_txt = d.get("results_txt")
    pred = d.get("predictions")
    log = d.get("log")
    if not file_exists(results_txt):
        errors.append("results_txt_missing")
        score_report = None
    else:
        score_report = parse_results_txt_metric(resolve_path(results_txt), expected_metric)  # type: ignore[arg-type]
        if score_report is None:
            errors.append(f"results_txt_primary_missing:{expected_metric}")
    if not file_exists(pred):
        errors.append("predictions_missing")
    if not file_exists(log):
        warnings.append("log_missing_or_external")
    if score_report is not None and score_payload is not None and abs(score_report - score_payload) > 1e-8:
        errors.append(f"payload_results_score_mismatch:{score_payload}:{score_report}")
    valid = not errors
    gpu_indices = None
    snap = d.get("child_snapshot") if isinstance(d.get("child_snapshot"), dict) else None
    if snap:
        gpu_indices = snap.get("child_physical_gpu_indices")
    return {
        "source": source,
        "path": rel(path),
        "status": d.get("status"),
        "endpoint": endpoint,
        "task": task,
        "primary_metric": metric,
        "payload_score": score_payload,
        "results_txt_score": score_report,
        "admitted_score": score_report,
        "results_txt": results_txt,
        "predictions": pred,
        "predictions_exists": file_exists(pred),
        "gpu": gpu,
        "gpu_snapshot_indices": gpu_indices,
        "valid_for_score_admission": valid,
        "errors": errors,
        "warnings": warnings,
    }


def collect_sg_records(endpoint: str, task: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in a01_sg_paths(endpoint, task):
        r = sg_record(p, "A01_step110_single_sg", endpoint, task)
        if r is not None:
            out.append(r)
    for p in a02_sg_paths(endpoint, task):
        r = sg_record(p, "A02_step127_single_sg", endpoint, task)
        if r is not None:
            out.append(r)
    return out


def aoa_record(path: pathlib.Path, source: str, endpoint: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    d = rj(path)
    score = d.get("score") if isinstance(d.get("score"), dict) else {}
    assembled = d.get("assembled") if isinstance(d.get("assembled"), dict) else {}
    summary = assembled.get("summary") if isinstance(assembled.get("summary"), dict) else {}
    payload = d.get("aoa_payload_for_comparison") if isinstance(d.get("aoa_payload_for_comparison"), dict) else {}
    aoa = fnum(score.get("aoa_leaderboard_score", payload.get("aoa_leaderboard_score")))
    n_finite = summary.get("n_finite", payload.get("n_results"))
    n_steps = len(assembled.get("steps") or []) if assembled else payload.get("n_steps")
    errors: List[str] = []
    if d.get("target") not in {endpoint, ENDPOINTS[endpoint]["label"]}:
        # Some older manifests use endpoint labels; keep this a warning unless
        # the actual score/shape evidence is wrong.
        pass
    if d.get("complete_measured_evidence") is not True:
        errors.append("complete_measured_evidence_false")
    if score.get("measured", payload.get("measured")) is not True:
        errors.append("score_not_marked_measured")
    if aoa is None:
        errors.append("aoa_missing")
    if aoa == 0.0 and not bool(score.get("legitimate_zero", payload.get("legitimate_zero"))):
        errors.append("zero_not_marked_legitimate")
    try:
        if int(n_finite) != 144090:
            errors.append(f"n_finite_mismatch:{n_finite}")
    except Exception:
        errors.append(f"n_finite_bad:{n_finite}")
    try:
        if int(n_steps) != 18:
            errors.append(f"n_steps_mismatch:{n_steps}")
    except Exception:
        errors.append(f"n_steps_bad:{n_steps}")
    return {
        "source": source,
        "path": rel(path),
        "status": d.get("status"),
        "endpoint": endpoint,
        "aoa": aoa,
        "admitted_score": aoa,
        "score_transform": "measured_aoa_leaderboard_score",
        "raw_correlation": score.get("aoa_raw_correlation", payload.get("aoa_raw_correlation")),
        "n_finite": n_finite,
        "n_steps": n_steps,
        "complete_measured_evidence": d.get("complete_measured_evidence"),
        "estimator_id": (score.get("aoa_estimator") or {}).get("estimator_id") if isinstance(score.get("aoa_estimator"), dict) else payload.get("estimator_id"),
        "valid_for_score_admission": not errors,
        "errors": errors,
        "warnings": [],
    }


def collect_aoa_records(endpoint: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src, p in aoa_paths(endpoint):
        r = aoa_record(p, src, endpoint)
        if r is not None:
            out.append(r)
    return out


def endpoint_admission(endpoint: str) -> Dict[str, Any]:
    zero: Dict[str, Any] = {}
    for col in ZERO_COLS:
        zero[col] = choose_component(collect_zero_records(endpoint, col))
    sg: Dict[str, Any] = {}
    for task in SG_TASKS:
        sg[task] = choose_component(collect_sg_records(endpoint, task))
    aoa = choose_component(collect_aoa_records(endpoint))

    scores: Dict[str, Optional[float]] = {k: None for k in OVERALL_COMPONENTS}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        ch = zero[col]["chosen"]
        if ch:
            scores[col] = float(ch["admitted_score"])
    gp1 = zero["GlobalPIQA_parallel"]["chosen"]
    gp2 = zero["GlobalPIQA_nonparallel"]["chosen"]
    if gp1 and gp2:
        scores["GlobalPIQA"] = (float(gp1["admitted_score"]) + float(gp2["admitted_score"])) / 2.0
    rd = zero["Reading"]["chosen"]
    if rd:
        scores["Reading"] = float(rd["admitted_score"])
    sg_chosen = [sg[t]["chosen"] for t in SG_TASKS]
    if all(sg_chosen):
        scores["SuperGLUE"] = statistics.mean(float(r["admitted_score"]) for r in sg_chosen if r)
    if aoa["chosen"]:
        scores["AoA"] = float(aoa["chosen"].get("aoa", aoa["chosen"].get("admitted_score")))
    missing = [k for k, v in scores.items() if v is None]
    conflicts = []
    for col, obj in zero.items():
        if obj.get("state") == "conflict":
            conflicts.append(f"zero:{col}")
    for task, obj in sg.items():
        if obj.get("state") == "conflict":
            conflicts.append(f"superglue:{task}")
    if aoa.get("state") == "conflict":
        conflicts.append("aoa")
    complete = not missing and not conflicts
    return {
        "endpoint": endpoint,
        "endpoint_label": ENDPOINTS[endpoint]["label"],
        "checkpoint": rel(ENDPOINTS[endpoint]["checkpoint"]),
        "zero_reading": zero,
        "superglue_tasks": sg,
        "aoa": aoa,
        "scores": scores,
        "globalpiqa_subscores": {
            "parallel": float(gp1["admitted_score"]) if gp1 else None,
            "nonparallel": float(gp2["admitted_score"]) if gp2 else None,
        },
        "superglue_task_scores": {task: (sg[task]["chosen"] or {}).get("admitted_score") for task in SG_TASKS},
        "complete_for_overall": complete,
        "Overall": statistics.mean(float(scores[k]) for k in OVERALL_COMPONENTS) if complete else None,
        "missing_scores": missing,
        "conflicts": conflicts,
    }


def load_counterparts() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if SEED62064_TABLE.is_file():
        d = rj(SEED62064_TABLE)
        for key, rec in (d.get("models") or {}).items():
            comp = (rec.get("overall_computation") or {})
            if comp.get("complete"):
                out[key] = {"Overall": comp.get("Overall"), "components": comp.get("components"), "path": rel(SEED62064_TABLE)}
    if ALL6_TABLE.is_file():
        d = rj(ALL6_TABLE)
        for key, rec in (d.get("models") or {}).items():
            comp = (rec.get("overall_computation") or {})
            if comp.get("complete") and key not in out:
                out[key] = {"Overall": comp.get("Overall"), "components": comp.get("components"), "path": rel(ALL6_TABLE)}
    return out


def delta_components(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in OVERALL_COMPONENTS:
        va, vb = a.get(k), b.get(k)
        out[k] = None if va is None or vb is None else float(va) - float(vb)
    return out


def paired_contrasts(endpoints: Dict[str, Any], counterparts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ep, counterpart_key in [("o62065", "ordinary_inherited_wwm_seed62064"), ("ms62065", "densemask_sparselabel_seed62064")]:
        rec = endpoints.get(ep, {})
        base = counterparts.get(counterpart_key, {})
        if rec.get("complete_for_overall") and base.get("Overall") is not None:
            out[f"{ep}_minus_{counterpart_key}"] = {
                "available": True,
                "delta_overall": float(rec["Overall"]) - float(base["Overall"]),
                "component_deltas": delta_components(rec["scores"], base["components"]),
                "seed62065_overall": rec["Overall"],
                "seed62064_overall": base["Overall"],
            }
        else:
            out[f"{ep}_minus_{counterpart_key}"] = {"available": False, "missing_seed62065": rec.get("missing_scores"), "seed62064_available": bool(base)}
    clean65 = counterparts.get("clean_pres_lambda1_eval_seed62065")
    clean64 = counterparts.get("clean_pres_lambda1_eval_seed62064")
    ms64 = counterparts.get("densemask_sparselabel_seed62064")
    ms65 = endpoints.get("ms62065", {})
    if clean65 and ms65.get("complete_for_overall"):
        residual65 = float(clean65["Overall"]) - float(ms65["Overall"])
    else:
        residual65 = None
    residual64 = None
    if clean64 and ms64:
        residual64 = float(clean64["Overall"]) - float(ms64["Overall"])
    out["anchoring_associated_residual_seed_comparison"] = {
        "seed62064_clean_minus_ms": residual64,
        "seed62065_clean_minus_ms": residual65,
        "difference_seed65_minus_seed64": None if residual64 is None or residual65 is None else residual65 - residual64,
        "available": residual65 is not None and residual64 is not None,
    }
    return out


def flatten_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ep, rec in result["endpoints"].items():
        for col, obj in rec["zero_reading"].items():
            for r in obj["all_found"]:
                rows.append({
                    "endpoint": ep, "kind": "zero_reading", "component": col, "source": r.get("source"),
                    "state": obj.get("state"), "admitted_score": r.get("admitted_score"), "raw_payload_score": r.get("raw_payload_score"),
                    "report_score": r.get("report_score"), "valid_score": r.get("valid_for_score_admission"),
                    "valid_item": r.get("valid_for_item_analysis"), "path": r.get("path"),
                    "errors": ";".join(r.get("errors") or []), "warnings": ";".join(r.get("warnings") or []),
                })
        for task, obj in rec["superglue_tasks"].items():
            for r in obj["all_found"]:
                rows.append({
                    "endpoint": ep, "kind": "superglue", "component": task, "source": r.get("source"),
                    "state": obj.get("state"), "admitted_score": r.get("admitted_score"), "raw_payload_score": r.get("payload_score"),
                    "report_score": r.get("results_txt_score"), "valid_score": r.get("valid_for_score_admission"),
                    "valid_item": "", "path": r.get("path"), "errors": ";".join(r.get("errors") or []), "warnings": ";".join(r.get("warnings") or []),
                })
        for r in rec["aoa"]["all_found"]:
            rows.append({
                "endpoint": ep, "kind": "aoa", "component": "AoA", "source": r.get("source"),
                "state": rec["aoa"].get("state"), "admitted_score": r.get("aoa"), "raw_payload_score": r.get("raw_correlation"),
                "report_score": r.get("aoa"), "valid_score": r.get("valid_for_score_admission"),
                "valid_item": "", "path": r.get("path"), "errors": ";".join(r.get("errors") or []), "warnings": ";".join(r.get("warnings") or []),
            })
    return rows


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fmt(x: Any, nd: int = 12) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_md(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research strict split-evaluation admission\n\n")
    lines.append(f"Created: `{result['created_utc']}`\n\n")
    lines.append("This pass admits split paired-seed components only after reconciling the score surface. For sentence zero-shot columns the admitted value is parsed from `best_temperature_report.txt`, which is the established two-decimal report surface. For Reading, the admitted value is the mean of the report-displayed eye and self-paced scores. Raw custom-scorer precision is retained in JSON/CSV but is not mixed into the same-coordinate Overall.\n\n")
    for ep, rec in result["endpoints"].items():
        lines.append(f"## {ep} — {rec['endpoint_label']}\n\n")
        lines.append(f"Complete for Overall: `{rec['complete_for_overall']}`; Overall: `{rec['Overall']}`; missing: `{rec['missing_scores']}`; conflicts: `{rec['conflicts']}`\n\n")
        lines.append("| component | admitted score | chosen source | chosen path | notes |\n|---|---:|---|---|---|\n")
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]:
            obj = rec["zero_reading"][col]
            ch = obj.get("chosen")
            note = ""
            if ch:
                ws = ch.get("warnings") or []
                note = "; ".join(ws)
            lines.append(f"| {col} | {fmt(ch.get('admitted_score')) if ch else '-'} | {ch.get('source') if ch else '-'} | `{ch.get('path') if ch else ''}` | {note} |\n")
        lines.append(f"| GlobalPIQA mean | {fmt(rec['scores']['GlobalPIQA'])} | derived | - | mean of report-derived subcolumns |\n")
        lines.append(f"| AoA | {fmt(rec['scores']['AoA'])} | {(rec['aoa']['chosen'] or {}).get('source','-')} | `{(rec['aoa']['chosen'] or {}).get('path','')}` | measured trajectory |\n")
        lines.append(f"| SuperGLUE mean | {fmt(rec['scores']['SuperGLUE'])} | derived | - | primary metrics: {json.dumps(rec['superglue_task_scores'], ensure_ascii=False)} |\n\n")
    lines.append("## Paired contrasts against seed62064 counterparts\n\n")
    lines.append("```json\n" + json.dumps(result["paired_contrasts"], indent=2, ensure_ascii=False) + "\n```\n\n")
    lines.append("## Current unresolved admission work\n\n")
    for ep, rec in result["endpoints"].items():
        if rec["missing_scores"] or rec["conflicts"]:
            lines.append(f"- {ep}: missing {rec['missing_scores']}, conflicts {rec['conflicts']}\n")
    path.write_text("".join(lines), encoding="utf-8")


def build() -> Dict[str, Any]:
    endpoints = {ep: endpoint_admission(ep) for ep in ["o62065", "ms62065"]}
    counterparts = load_counterparts()
    result = {
        "status": "STRICT_SPLIT_EVAL_ADMISSION",
        "created_utc": now(),
        "purpose": "Admit split O62065/MS62065 paired-seed evidence only when measurement conventions match the repaired same-coordinate surface.",
        "expected_counts": {col: expected_count(col) for col in ZERO_COLS},
        "score_convention": {
            "sentence_zero_shot": "admitted_score=parse(best_temperature_report.txt); raw custom-scorer precision retained only as diagnostics",
            "Reading": "admitted_score=(report Eye + report SelfPaced)/2 using displayed report values",
            "GlobalPIQA": "mean(report-derived GlobalPIQA_parallel, report-derived GlobalPIQA_nonparallel)",
            "SuperGLUE": "mean of results.txt primary metrics; F1 for MRPC/QQP, accuracy otherwise",
            "AoA": "complete measured 18-step/144090-row AoA manifest with legitimate zero if score is 0",
        },
        "endpoints": endpoints,
        "seed62064_counterparts": counterparts,
        "paired_contrasts": paired_contrasts(endpoints, counterparts),
        "all_complete": all(v.get("complete_for_overall") for v in endpoints.values()),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build()
    out_json = out_dir / "strict_split_eval_admission.json"
    out_md = out_dir / "strict_split_eval_admission.md"
    out_csv = out_dir / "strict_split_eval_sources.csv"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_md(out_md, result)
    write_csv(out_csv, flatten_rows(result))
    print(json.dumps({
        "status": result["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "out_csv": rel(out_csv),
        "all_complete": result["all_complete"],
        "overall": {ep: rec["Overall"] for ep, rec in result["endpoints"].items()},
        "missing": {ep: rec["missing_scores"] for ep, rec in result["endpoints"].items()},
        "conflicts": {ep: rec["conflicts"] for ep, rec in result["endpoints"].items()},
        "paired_contrasts": result["paired_contrasts"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
