#!/usr/bin/env python3
"""Strict scorer for research/052 mixed-objective true-100M full evaluations.

The scorer is intentionally stricter than older endpoint scorers because the mixed-
objective result is a central route test.  It accepts exactly two per-target JSONs,
recomputes Overall from the current task records, checks AoA ladder integrity, and
binds the payloads to the model hashes recorded by launch preflight.
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
import sys
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = _public_path('experiments/archive/compact_experience/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
from babylm_official_scoring import compute_overall_from_tasks, normalize_aoa_record  # noqa: E402

WORKSPACE = _public_path('experiments/archive/compact_experience')
DEFAULT_OUTROOT = _public_path('experiments/archive/compact_experience/data/mixed_objective_full_eval')
CLEAN_PAYLOAD = _public_path('experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned.json')
LEADERBOARD_TOP20 = _public_path('experiments/archive/compact_experience/data/babylm2026_surface/strict_small_top20.json')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
CALC_RESULTS = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/calculate_results_from_pred.py')
PRINT_TABLE = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/scripts/print_results_table.py')

EXPECTED: Dict[str, Dict[str, str]] = {
    "mixed_causal50_100M": {
        "run_name": "mixed_causal50_qwen_seed43022",
        "endpoint": "chck_100M",
        "intended_causal_fraction": "0.5",
    },
    "mixed_causal15_100M": {
        "run_name": "mixed_causal15_qwen_seed43022",
        "endpoint": "chck_100M",
        "intended_causal_fraction": "0.15",
    },
}
AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{10*i}M" for i in range(1, 11)]
COLS7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
SUPERGLUE_TASKS = ["boolq", "multirc", "rte", "wsc", "mrpc", "qqp", "mnli"]
CLEAN_OVERALL = 41.34429066479573
CLEAN_EQUAL7 = 43.112857142857145
VISIBLE_LEADER = 41.8


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def p_resolve(pathlike: Any) -> pathlib.Path:
    return pathlib.Path(str(pathlike)).expanduser().resolve()


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_equal7(scores: dict) -> float | None:
    if not all(scores.get(c) is not None for c in COLS7):
        return None
    vals = [float(scores[c]) for c in COLS7]
    if not all(math.isfinite(v) for v in vals):
        return None
    return sum(vals) / len(vals)


def load_clean_reference() -> dict:
    payload = load_json(CLEAN_PAYLOAD)
    tasks = payload.get("tasks", {})
    if isinstance(tasks.get("AoA"), dict):
        tasks["AoA"] = normalize_aoa_record(tasks["AoA"])
    oo = compute_overall_from_tasks(tasks)
    aoa = tasks.get("AoA", {}) if isinstance(tasks, dict) else {}
    scores = oo.get("scores") or {}
    eq7 = compute_equal7(scores)
    failures: List[str] = []
    if abs(float(oo.get("Overall")) - CLEAN_OVERALL) > 1e-9:
        failures.append(f"clean Overall recompute {oo.get('Overall')} != {CLEAN_OVERALL}")
    if eq7 is None or abs(eq7 - CLEAN_EQUAL7) > 1e-9:
        failures.append(f"clean equal7 recompute {eq7} != {CLEAN_EQUAL7}")
    if not (isinstance(aoa, dict) and aoa.get("status") == "official_aoa_done" and aoa.get("num_steps") == 19):
        failures.append("clean-Qwen AoA is not recorded as official_aoa_done with 19 steps")
    if failures:
        raise RuntimeError({"clean_reference_validation_failed": failures, "payload": str(CLEAN_PAYLOAD)})
    return {
        "payload_path": str(CLEAN_PAYLOAD),
        "model_path": payload.get("model_path"),
        "Overall": oo.get("Overall"),
        "equal7": eq7,
        "scores": scores,
        "aoa_status": aoa.get("status") if isinstance(aoa, dict) else None,
        "aoa_raw_correlation": aoa.get("aoa_raw_correlation") if isinstance(aoa, dict) else None,
        "aoa_num_steps": aoa.get("num_steps") if isinstance(aoa, dict) else None,
    }


def load_leader_reference() -> dict:
    rows = load_json(LEADERBOARD_TOP20)
    leader = rows[0]
    if leader.get("Model_plain") != "wwm_curriculum_simplification_40k":
        raise RuntimeError({"unexpected_leader_row": leader})
    return {
        "source_path": str(LEADERBOARD_TOP20),
        "model": leader.get("Model_plain"),
        "hf_repo": leader.get("HF Repo"),
        "overall_displayed": leader.get("Overall Average"),
        "row": leader,
        "display_precision_note": "leaderboard scrape stores displayed rounded values; 41.8 should not be treated as high-precision server arithmetic",
    }


def validate_payload_identity(target: str, payload: dict, preflight: dict | None) -> Tuple[dict, List[str]]:
    exp = EXPECTED[target]
    run_dir = p_resolve(payload.get("run_dir"))
    model_path = p_resolve(payload.get("model_path"))
    expected_run = (_public_path('experiments/archive/compact_experience/training/runs') / exp["run_name"]).resolve()
    expected_model = expected_run / "hf_model" / exp["endpoint"]
    failures: List[str] = []
    if payload.get("target") != target:
        failures.append(f"payload target {payload.get('target')} != {target}")
    if payload.get("endpoint") != exp["endpoint"]:
        failures.append(f"endpoint {payload.get('endpoint')} != {exp['endpoint']}")
    if run_dir != expected_run:
        failures.append(f"run_dir {run_dir} != {expected_run}")
    if model_path != expected_model.resolve():
        failures.append(f"model_path {model_path} != {expected_model.resolve()}")
    model_file = model_path / "model.safetensors"
    if not model_file.exists() or model_file.stat().st_size <= 0:
        failures.append(f"missing or empty endpoint weights {model_file}")
        endpoint_sha = None
    else:
        endpoint_sha = sha256_file(model_file)
    pf_target = (preflight or {}).get("targets", {}).get(target, {}) if isinstance(preflight, dict) else {}
    pf_sha = pf_target.get("endpoint_model_safetensors_sha256")
    if pf_sha and endpoint_sha and pf_sha != endpoint_sha:
        failures.append("endpoint model hash differs from launch preflight")
    metrics_path = expected_run / "scientific_metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    if metrics.get("status") != "MIXED_OBJECTIVE_MANIFEST":
        failures.append(f"metrics status {metrics.get('status')}")
    if metrics.get("actual_word_exposure") != 100000000:
        failures.append(f"actual_word_exposure {metrics.get('actual_word_exposure')}")
    if metrics.get("actual_steps") != metrics.get("planned_steps"):
        failures.append(f"actual_steps {metrics.get('actual_steps')} != planned_steps {metrics.get('planned_steps')}")
    if abs(float(metrics.get("causal_fraction", -1.0)) - float(exp["intended_causal_fraction"])) > 1e-12:
        failures.append(f"intended causal_fraction {metrics.get('causal_fraction')}")
    saved = [x.get("name") for x in metrics.get("saved_checkpoints", []) if isinstance(x, dict)]
    if len(saved) != 100:
        failures.append(f"saved checkpoint count {len(saved)} != 100")
    missing = [s for s in AOA_STEPS if s not in saved]
    if missing:
        failures.append(f"metrics missing required AoA checkpoint names {missing}")
    return {
        "expected_run_dir": str(expected_run),
        "expected_model_path": str(expected_model),
        "endpoint_model_safetensors_sha256": endpoint_sha,
        "preflight_endpoint_model_safetensors_sha256": pf_sha,
        "metrics_path": str(metrics_path),
        "training_metrics": {
            k: metrics.get(k) for k in [
                "status", "data", "data_sha256", "metadata_sha256", "file_words", "file_rows",
                "selected_word_exposure", "planned_steps", "effective_steps", "actual_word_exposure",
                "actual_steps", "causal_fraction", "realized_causal_batch_fraction", "mlm_batches",
                "causal_batches", "mlm_targets", "causal_targets", "mlm_loss_last", "causal_loss_last",
                "word_embedding_sha256", "parameter_count", "objective_policy", "causal_definition",
                "mlm_definition", "word_count_policy",
            ] if k in metrics
        },
    }, failures


def validate_tasks(target: str, tasks: dict, official_overall: dict) -> Tuple[dict, List[str]]:
    failures: List[str] = []
    for col in ZERO_COLS:
        rec = tasks.get(col)
        if not isinstance(rec, dict):
            failures.append(f"missing {col}")
            continue
        if rec.get("returncode") != 0:
            failures.append(f"{col} returncode {rec.get('returncode')}")
        if not finite_number(rec.get("score")):
            failures.append(f"{col} nonfinite/missing score {rec.get('score')}")
    reading = tasks.get("Reading")
    if not isinstance(reading, dict) or reading.get("returncode") != 0 or not finite_number((reading.get("scores") or {}).get("Reading")):
        failures.append("Reading missing, failed, or lacks finite combined score")

    sg = tasks.get("SuperGLUE")
    sg_detail: Dict[str, Any] = {}
    if not isinstance(sg, dict):
        failures.append("missing SuperGLUE")
    else:
        sg_tasks = sg.get("tasks") or []
        by_task = {r.get("task"): r for r in sg_tasks if isinstance(r, dict)}
        if set(by_task) != set(SUPERGLUE_TASKS):
            failures.append(f"SuperGLUE task set {sorted(by_task)} != {SUPERGLUE_TASKS}")
        accs = []
        for task in SUPERGLUE_TASKS:
            rec = by_task.get(task)
            if not isinstance(rec, dict):
                continue
            if rec.get("returncode") != 0:
                failures.append(f"SuperGLUE {task} returncode {rec.get('returncode')}")
            if not finite_number(rec.get("accuracy")):
                failures.append(f"SuperGLUE {task} missing finite accuracy")
            else:
                accs.append(float(rec["accuracy"]))
        if len(accs) == len(SUPERGLUE_TASKS):
            recomputed = sum(accs) / len(accs)
            sg_detail["superglue_accuracy_mean_recomputed"] = recomputed
            if not finite_number(sg.get("superglue_mean")) or abs(float(sg.get("superglue_mean")) - recomputed) > 1e-9:
                failures.append(f"SuperGLUE mean {sg.get('superglue_mean')} != recomputed accuracy mean {recomputed}")
        sg_detail["source_equivalence"] = {
            "local_column_used_here": "unweighted mean of validation accuracy over boolq, multirc, rte, wsc, mrpc, qqp, mnli",
            "current_official_code_evidence": "evaluation_pipeline/calculate_results_from_pred.py::_calculate_glue_results lines 232-249 computes correct/total accuracy per GLUE subtask and returns the unweighted mean; scripts/print_results_table.py reports MRPC/QQP F1 for markdown tables, so local result should be described as official-style until server submission but is internally consistent with calculate_results_from_pred.py.",
            "calculate_results_sha256": sha256_file(CALC_RESULTS) if CALC_RESULTS.exists() else None,
            "print_results_table_sha256": sha256_file(PRINT_TABLE) if PRINT_TABLE.exists() else None,
        }

    aoa = tasks.get("AoA")
    aoa_detail: Dict[str, Any] = {}
    if not isinstance(aoa, dict):
        failures.append("missing AoA")
    else:
        aoa = normalize_aoa_record(aoa)
        tasks["AoA"] = aoa
        aoa_detail = {
            "status": aoa.get("status"),
            "helper_status": aoa.get("helper_status"),
            "num_steps": aoa.get("num_steps"),
            "expected_steps": aoa.get("expected_steps"),
            "missing_steps": aoa.get("missing_steps"),
            "unexpected_steps": aoa.get("unexpected_steps"),
            "row_count_values": aoa.get("row_count_values"),
            "finite_surprisals": aoa.get("finite_surprisals"),
            "aoa_raw_correlation": aoa.get("aoa_raw_correlation"),
            "aoa_leaderboard_score": aoa.get("aoa_leaderboard_score"),
            "out_json": aoa.get("out_json"),
            "surprisal_path": aoa.get("surprisal_path"),
            "score_path": aoa.get("score_path"),
        }
        if aoa.get("status") != "official_aoa_done":
            failures.append(f"AoA status {aoa.get('status')}")
        if aoa.get("num_steps") != 19:
            failures.append(f"AoA num_steps {aoa.get('num_steps')} != 19")
        if aoa.get("expected_steps") != AOA_STEPS:
            failures.append("AoA expected_steps mismatch")
        if aoa.get("missing_steps") not in ([], None):
            failures.append(f"AoA missing_steps {aoa.get('missing_steps')}")
        if aoa.get("unexpected_steps") not in ([], None):
            failures.append(f"AoA unexpected_steps {aoa.get('unexpected_steps')}")
        row_counts = aoa.get("row_count_values") or []
        if len(row_counts) != 1 or int(row_counts[0]) <= 0:
            failures.append(f"AoA row_count_values {row_counts}")
        if aoa.get("finite_surprisals") is not True:
            failures.append(f"AoA finite_surprisals {aoa.get('finite_surprisals')}")
        if not finite_number(aoa.get("aoa_raw_correlation")) or not finite_number(aoa.get("aoa_leaderboard_score")):
            failures.append("AoA raw or leaderboard score nonfinite")

    scores = official_overall.get("scores") or {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]:
        if not finite_number(scores.get(col)):
            failures.append(f"official_overall score {col} missing/nonfinite")
    if official_overall.get("complete_for_provisional_overall") is not True:
        failures.append("official_overall not complete")
    if official_overall.get("submit_ready_aoa") is not True:
        failures.append("submit_ready_aoa is not true")
    if official_overall.get("submit_ready_overall") is not True:
        failures.append("submit_ready_overall is not true")
    if not finite_number(official_overall.get("Overall")):
        failures.append("Overall missing/nonfinite")
    return {"SuperGLUE": sg_detail, "AoA": aoa_detail}, failures


def score_one(target: str, path: pathlib.Path, preflight: dict | None) -> Tuple[dict, List[str]]:
    payload = load_json(path)
    tasks = payload.get("tasks", {})
    if isinstance(tasks.get("AoA"), dict):
        tasks["AoA"] = normalize_aoa_record(tasks["AoA"])
    old_oo = payload.get("official_overall") if isinstance(payload.get("official_overall"), dict) else None
    recomputed = compute_overall_from_tasks(tasks)
    aoa = tasks.get("AoA", {})
    recomputed["aoa_status"] = aoa.get("status") if isinstance(aoa, dict) else None
    recomputed["submit_ready_aoa"] = bool(isinstance(aoa, dict) and aoa.get("status") == "official_aoa_done")
    recomputed["submit_ready_overall"] = bool(recomputed.get("submit_ready_aoa") and recomputed.get("complete_for_provisional_overall"))
    if isinstance(aoa, dict):
        recomputed["aoa_raw_correlation"] = aoa.get("aoa_raw_correlation")
        recomputed["aoa_leaderboard_score"] = aoa.get("aoa_leaderboard_score")
    payload["official_overall_prior_to_step52_rescore"] = old_oo
    payload["official_overall"] = recomputed
    payload["rescore_note"] = "Overall recomputed unconditionally from current task records after AoA normalization; old cached Overall is not trusted."
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    identity, ident_fail = validate_payload_identity(target, payload, preflight)
    validation_detail, task_fail = validate_tasks(target, tasks, recomputed)
    scores = recomputed.get("scores") or {}
    equal7 = compute_equal7(scores)
    row = {
        "target": target,
        "payload_path": str(path),
        "model_path": payload.get("model_path"),
        "run_dir": payload.get("run_dir"),
        "endpoint": payload.get("endpoint"),
        "identity": identity,
        "overall": recomputed.get("Overall"),
        "equal7_noaoa": equal7,
        "delta_overall_vs_clean_qwen": (float(recomputed.get("Overall")) - CLEAN_OVERALL) if finite_number(recomputed.get("Overall")) else None,
        "delta_overall_vs_visible_leader_displayed": (float(recomputed.get("Overall")) - VISIBLE_LEADER) if finite_number(recomputed.get("Overall")) else None,
        "delta_equal7_vs_clean_qwen": (equal7 - CLEAN_EQUAL7) if equal7 is not None else None,
        "complete_for_provisional_overall": recomputed.get("complete_for_provisional_overall"),
        "submit_ready_aoa": recomputed.get("submit_ready_aoa"),
        "submit_ready_overall": recomputed.get("submit_ready_overall"),
        "aoa_status": recomputed.get("aoa_status"),
        "aoa_raw_correlation": recomputed.get("aoa_raw_correlation"),
        "aoa_leaderboard_score": recomputed.get("aoa_leaderboard_score"),
        "superglue_mean": (tasks.get("SuperGLUE") or {}).get("superglue_mean") if isinstance(tasks.get("SuperGLUE"), dict) else None,
        "scores": scores,
        "validation_detail": validation_detail,
    }
    failures = ident_fail + task_fail
    if failures:
        row["validation_failures"] = failures
    return row, failures


def write_note(outroot: pathlib.Path, summary: dict) -> pathlib.Path:
    note = outroot / "mixed_objective_full_eval_summary.md"
    rows = summary.get("candidates", [])
    lines = [
        "# research mixed-objective true-100M full evaluation", "",
        f"Output root: `{outroot}`", "",
        f"Clean-Qwen reference recomputed from `{CLEAN_PAYLOAD}`: Overall {summary['clean_qwen_reference']['Overall']:.12f}, equal7 {summary['clean_qwen_reference']['equal7']:.12f}.",
        f"Visible leader reference from `{LEADERBOARD_TOP20}`: {summary['visible_leader_reference']['model']} displayed Overall {summary['visible_leader_reference']['overall_displayed']} (rounded display).", "",
        "AoA is used only as a terminal aggregate readout from the frozen complete checkpoint ladder. Both causal fractions were fixed before evaluation; no endpoint or fraction was selected from AoA.", "",
        "SuperGLUE in this local scorer is the unweighted mean of validation accuracies over BoolQ, MultiRC, RTE, WSC, MRPC, QQP, and MNLI, matching `calculate_results_from_pred.py::_calculate_glue_results`; `print_results_table.py` has a separate markdown-table metric map that uses F1 for MRPC/QQP and is recorded as a remaining server-equivalence caveat.", "",
        "| target | Overall | Δ clean | Δ visible leader | equal7 | Δ clean equal7 | SuperGLUE | AoA raw | AoA lb | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | valid |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        s = r.get("scores") or {}
        valid = "yes" if not r.get("validation_failures") else "NO: " + "; ".join(r.get("validation_failures", [])[:4])
        lines.append(
            f"| {r['target']} | {r.get('overall')} | {r.get('delta_overall_vs_clean_qwen')} | {r.get('delta_overall_vs_visible_leader_displayed')} | "
            f"{r.get('equal7_noaoa')} | {r.get('delta_equal7_vs_clean_qwen')} | {r.get('superglue_mean')} | {r.get('aoa_raw_correlation')} | {r.get('aoa_leaderboard_score')} | "
            f"{s.get('BLiMP')} | {s.get('Supplement')} | {s.get('EWoK')} | {s.get('Entity')} | {s.get('COMPS')} | {s.get('GlobalPIQA')} | {s.get('Reading')} | {valid} |"
        )
    lines.extend(["", "## Interpretation scope", "", summary["interpretation_scope"], ""])
    note.write_text("\n".join(lines), encoding="utf-8")
    return note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(DEFAULT_OUTROOT), help="Root containing per_target/*.json from this launch.")
    args = ap.parse_args()
    outroot = pathlib.Path(args.out_root).resolve()
    per = outroot / "per_target"
    if not per.exists():
        raise FileNotFoundError(per)
    preflight_path = outroot / "preflight.json"
    preflight = load_json(preflight_path) if preflight_path.exists() else None
    clean_ref = load_clean_reference()
    leader_ref = load_leader_reference()

    expected_files = {target: per / f"{target}.json" for target in EXPECTED}
    existing_json = sorted(p.name for p in per.glob("*.json"))
    expected_names = sorted(p.name for p in expected_files.values())
    if existing_json != expected_names:
        raise RuntimeError({"unexpected_per_target_jsons": existing_json, "expected": expected_names, "per": str(per)})

    rows: List[dict] = []
    all_failures: Dict[str, List[str]] = {}
    for target, path in expected_files.items():
        row, failures = score_one(target, path, preflight)
        rows.append(row)
        if failures:
            all_failures[target] = failures
    rows.sort(key=lambda r: (-1e9 if r["overall"] is None else -float(r["overall"])))
    summary = {
        "status": "MIXED_OBJECTIVE_FULL_EVAL_SUMMARY_STRICT" if not all_failures else "MIXED_OBJECTIVE_FULL_EVAL_INVALID",
        "out_root": str(outroot),
        "preflight_path": str(preflight_path) if preflight_path.exists() else None,
        "interpretation_scope": "Scores apply first to the full mixed-objective package: causal visibility, uncorrupted inputs, dense next-token targets, fewer MLM updates, and changed gradient statistics. Target density and objective complementarity require a separate matched control before a deeper mechanism assignment.",
        "aoa_use_policy": "AoA is recorded only as terminal aggregate full-evaluation evidence from frozen true-100M ladders, not as a training, endpoint, or fraction selection signal.",
        "clean_qwen_reference": clean_ref,
        "visible_leader_reference": leader_ref,
        "candidates": rows,
        "validation_failures": all_failures,
    }
    out = outroot / "mixed_objective_full_eval_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = write_note(outroot, summary)
    print(json.dumps({
        "out": str(out),
        "note": str(note),
        "status": summary["status"],
        "candidates": [
            {
                "target": r["target"],
                "overall": r["overall"],
                "delta_vs_clean": r["delta_overall_vs_clean_qwen"],
                "delta_vs_visible_leader_displayed": r["delta_overall_vs_visible_leader_displayed"],
                "equal7": r["equal7_noaoa"],
                "aoa_raw": r["aoa_raw_correlation"],
                "valid": not bool(r.get("validation_failures")),
            }
            for r in rows
        ],
    }, indent=2), flush=True)
    if all_failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
