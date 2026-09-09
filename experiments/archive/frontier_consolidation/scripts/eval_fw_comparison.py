#!/usr/bin/env python3
"""Evaluate FW compact-view vs whole-sentence source-breadth comparison.

This script uses the official-compatible evaluator and parses its
per-target JSON through the same `tasks` schema used by the proven research
rephase reader. It is intentionally restricted to the cheap zero-shot/reading
surface needed to decide the data-mechanism comparison before any full official
endpoint work.

Usage after the paired trainings have produced checkpoints:
  python3 experiments/archive/frontier_consolidation/scripts/eval_fw_comparison.py \
    --checkpoint 70M --checkpoint 80M --gpu 0

Parser validation only:
  python3 experiments/archive/frontier_consolidation/scripts/eval_fw_comparison.py --validate-parser
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
EVAL_SCRIPT = WORKSPACE / "scripts/evaluate_compliant_endpoint.py"
OUT_ROOT = WORKSPACE / "data/fw_comparison_eval"
COLLATE_ROOT = WORKSPACE / "data/fw_comparison_collate"

CHEAP_COLUMNS = [
    "BLiMP",
    "Supplement",
    "EWoK",
    "Entity",
    "COMPS",
    "GlobalPIQA_parallel",
    "GlobalPIQA_nonparallel",
    "Reading",
]
SCALAR_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

ARMS = {
    "compact_view": WORKSPACE / "training/runs/fw_compact_view_shared16k_seed43022",
    "source_breadth": WORKSPACE / "training/runs/fw_source_breadth_shared16k_seed43022",
}

DECISION_DELTA = 0.3
DECISIVE_EARLY_DELTA = 1.0
DEFAULT_ARMS = ["compact_view", "source_breadth"]
SUMMARY_JSON = OUT_ROOT / "fw_comparison_eval_summary.json"

PARSER_VALIDATION_JSON = WORKSPACE / "data/rephase_eval/per_target/matched_lr_chck_90M.json"
PARSER_VALIDATION_EXPECTED = {
    "BLiMP": 66.48,
    "Supplement": 60.7,
    "EWoK": 51.02,
    "Entity": 27.4,
    "COMPS": 51.84,
    "GlobalPIQA_parallel": 30.1,
    "GlobalPIQA_nonparallel": 44.0,
    "GlobalPIQA": 37.05,
    "Reading_eye": 11.43,
    "Reading_self_paced": 5.84,
    "Reading": 8.635,
    "cheap7": 43.3036,
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def scalar_score(rec: dict[str, Any], *, nested_key: str | None = None) -> float | None:
    if rec.get("accuracy") is not None:
        return float(rec["accuracy"])
    if rec.get("score") is not None:
        return float(rec["score"])
    nested = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
    if nested_key and nested.get(nested_key) is not None:
        return float(nested[nested_key])
    return None


def parse_scores_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse research per-target JSON into the scalar cheap-surface scores."""
    tasks = payload.get("tasks", {})
    scores: dict[str, Any] = {}
    missing: list[str] = []

    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        val = scalar_score(tasks.get(col, {}))
        if val is None:
            missing.append(col)
        else:
            scores[col] = val

    gpiqa_parallel = scalar_score(tasks.get("GlobalPIQA_parallel", {}))
    gpiqa_nonparallel = scalar_score(tasks.get("GlobalPIQA_nonparallel", {}))
    if gpiqa_parallel is None:
        missing.append("GlobalPIQA_parallel")
    else:
        scores["GlobalPIQA_parallel"] = gpiqa_parallel
    if gpiqa_nonparallel is None:
        missing.append("GlobalPIQA_nonparallel")
    else:
        scores["GlobalPIQA_nonparallel"] = gpiqa_nonparallel
    if gpiqa_parallel is not None and gpiqa_nonparallel is not None:
        scores["GlobalPIQA"] = (gpiqa_parallel + gpiqa_nonparallel) / 2.0

    reading_rec = tasks.get("Reading", {})
    nested_reading = reading_rec.get("scores") if isinstance(reading_rec.get("scores"), dict) else {}
    reading_eye = reading_rec.get("reading_eye", nested_reading.get("Reading_eye"))
    reading_sp = reading_rec.get("reading_self_paced", nested_reading.get("Reading_self_paced"))
    reading = reading_rec.get("reading", nested_reading.get("Reading"))
    if reading_eye is not None:
        scores["Reading_eye"] = float(reading_eye)
    if reading_sp is not None:
        scores["Reading_self_paced"] = float(reading_sp)
    if reading is None:
        missing.append("Reading")
    else:
        scores["Reading"] = float(reading)

    if all(k in scores for k in SCALAR_COLUMNS):
        scores["cheap7"] = round(sum(float(scores[k]) for k in SCALAR_COLUMNS) / 7.0, 4)
    else:
        scores["cheap7"] = None

    return {"scores": scores, "missing_scores": missing}


def parse_target_scores(target_json: pathlib.Path) -> dict[str, Any]:
    if not target_json.exists():
        return {"status": "NO_OUTPUT", "per_target_json": str(target_json)}
    payload = json.loads(target_json.read_text(encoding="utf-8"))
    parsed = parse_scores_from_payload(payload)
    status = "OK" if not parsed["missing_scores"] and parsed["scores"].get("cheap7") is not None else "INCOMPLETE_SCORES"
    return {
        "status": status,
        "per_target_json": str(target_json),
        "scores": parsed["scores"],
        "missing_scores": parsed["missing_scores"],
        "official_overall_payload": payload.get("official_overall"),
    }


def validate_parser() -> dict[str, Any]:
    parsed = parse_target_scores(PARSER_VALIDATION_JSON)
    result = {
        "status": "PARSER_VALIDATION_OK",
        "source_json": str(PARSER_VALIDATION_JSON),
        "max_abs_error": None,
        "errors": {},
        "parsed_scores": parsed.get("scores", {}),
        "expected_scores": PARSER_VALIDATION_EXPECTED,
    }
    if parsed.get("status") != "OK":
        result["status"] = "PARSER_VALIDATION_FAILED"
        result["reason"] = parsed
        return result
    max_err = 0.0
    errors: dict[str, float] = {}
    scores = parsed.get("scores", {})
    for key, expected in PARSER_VALIDATION_EXPECTED.items():
        got = scores.get(key)
        if got is None:
            errors[key] = float("inf")
            max_err = float("inf")
            continue
        err = abs(float(got) - float(expected))
        errors[key] = err
        if err > max_err:
            max_err = err
    result["max_abs_error"] = max_err
    result["errors"] = errors
    if not (max_err <= 5e-4):
        result["status"] = "PARSER_VALIDATION_FAILED"
    return result


def run_eval(arm_name: str, ckpt_name: str, gpu: int, force: bool = False) -> dict[str, Any]:
    """Run cheap evaluation for one arm+checkpoint and parse research output."""
    run_dir = ARMS[arm_name]
    endpoint = f"chck_{ckpt_name}"
    target = f"step086_{arm_name}_{endpoint}"
    model_path = run_dir / "hf_model" / endpoint
    target_json = OUT_ROOT / "per_target" / f"{target}.json"

    if not model_path.exists():
        return {
            "label": f"{arm_name}_{ckpt_name}",
            "arm": arm_name,
            "checkpoint": ckpt_name,
            "status": "MISSING_CHECKPOINT",
            "model_path": str(model_path),
        }

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--arm",
        "reinvest",
        "--target",
        target,
        "--run-dir",
        str(run_dir),
        "--endpoint",
        endpoint,
        "--out-root",
        str(OUT_ROOT),
        "--collate-root",
        str(COLLATE_ROOT),
        "--gpu",
        str(gpu),
        "--columns",
        *CHEAP_COLUMNS,
    ]
    if force:
        cmd.append("--force")

    print(f"\n{'=' * 72}\nEvaluating {arm_name} {endpoint}\n{'=' * 72}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), text=True, capture_output=True, timeout=7200)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-1500:], flush=True)
        print(proc.stderr[-2500:], flush=True)
        return {
            "label": f"{arm_name}_{ckpt_name}",
            "arm": arm_name,
            "checkpoint": ckpt_name,
            "status": "EVAL_FAILED",
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 1),
            "stdout_tail": proc.stdout[-2500:],
            "stderr_tail": proc.stderr[-2500:],
            "command": cmd,
        }

    parsed = parse_target_scores(target_json)
    parsed.update({
        "label": f"{arm_name}_{ckpt_name}",
        "arm": arm_name,
        "checkpoint": ckpt_name,
        "elapsed_sec": round(elapsed, 1),
        "model_path": str(model_path),
        "command": cmd,
    })
    if parsed.get("status") == "OK":
        print(f"{arm_name}_{ckpt_name}: cheap7={parsed['scores']['cheap7']:.4f}", flush=True)
    else:
        print(f"{arm_name}_{ckpt_name}: {parsed.get('status')} {parsed.get('missing_scores')}", flush=True)
    return parsed
def checkpoint_sort_key(ckpt: str) -> tuple[int, str]:
    text = str(ckpt)
    if text.endswith("M") and text[:-1].isdigit():
        return (int(text[:-1]), text)
    return (10**9, text)


def merge_ordered_checkpoints(existing: list[str], requested: list[str]) -> list[str]:
    merged = list(dict.fromkeys(list(existing) + list(requested)))
    return sorted(merged, key=checkpoint_sort_key)


def load_existing_summary() -> dict[str, Any] | None:
    if not SUMMARY_JSON.exists():
        return None
    try:
        return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "EXISTING_SUMMARY_UNREADABLE", "error": repr(exc), "results": {}, "checkpoints": []}



def compare_checkpoint(results: dict[str, dict[str, Any]], ckpt: str) -> dict[str, Any] | None:
    compact = results.get(f"compact_view_{ckpt}")
    breadth = results.get(f"source_breadth_{ckpt}")
    if not compact or not breadth:
        return None
    if compact.get("status") != "OK" or breadth.get("status") != "OK":
        return {
            "checkpoint": ckpt,
            "status": "NOT_COMPARABLE",
            "compact_status": compact.get("status"),
            "breadth_status": breadth.get("status"),
        }

    cs = compact["scores"]
    bs = breadth["scores"]
    deltas = {col: round(float(cs[col]) - float(bs[col]), 4) for col in SCALAR_COLUMNS}
    delta = round(float(cs["cheap7"]) - float(bs["cheap7"]), 4)
    if delta >= DECISION_DELTA:
        decision = "compact_view_better_on_predeclared_rule"
    elif delta <= -DECISION_DELTA:
        decision = "source_breadth_better_on_predeclared_rule"
    else:
        decision = "too_close_next_source_repeat_test"
    return {
        "checkpoint": ckpt,
        "status": "OK",
        "compact_cheap7": cs["cheap7"],
        "breadth_cheap7": bs["cheap7"],
        "cheap7_delta_compact_minus_breadth": delta,
        "column_deltas": deltas,
        "decision_rule": {
            "compact_view_better_if_delta_ge": DECISION_DELTA,
            "source_breadth_better_if_delta_le": -DECISION_DELTA,
            "result": decision,
        },
        "early_stop_note": (
            "absolute_80M_delta_exceeds_1_point" if ckpt == "80M" and abs(delta) > DECISIVE_EARLY_DELTA else None
        ),
    }


def write_markdown(summary: dict[str, Any]) -> pathlib.Path:
    lines = [
        "# research FW compact-view vs whole-sentence source-breadth evaluation",
        "",
        f"Created UTC: {summary['created_utc']}",
        "",
        "| Checkpoint | Arm | BLiMP | Suppl | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ckpt in summary["checkpoints"]:
        for arm in ["compact_view", "source_breadth"]:
            r = summary["results"].get(f"{arm}_{ckpt}", {})
            if r.get("status") != "OK":
                lines.append(f"| {ckpt} | {arm} | {r.get('status')} | | | | | | | |")
                continue
            s = r["scores"]
            lines.append(
                f"| {ckpt} | {arm} | {s['BLiMP']:.2f} | {s['Supplement']:.2f} | {s['EWoK']:.2f} | "
                f"{s['Entity']:.2f} | {s['COMPS']:.2f} | {s['GlobalPIQA']:.2f} | {s['Reading']:.3f} | {s['cheap7']:.4f} |"
            )
    lines.append("")
    lines.append("## Compact minus breadth deltas")
    lines.append("")
    lines.append("| Checkpoint | Δcheap7 | ΔBLiMP | ΔSuppl | ΔEWoK | ΔEntity | ΔCOMPS | ΔGPIQA | ΔReading | Result |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for ckpt, comp in summary["comparisons"].items():
        if comp.get("status") != "OK":
            lines.append(f"| {ckpt} | {comp.get('status')} | | | | | | | | |")
            continue
        d = comp["column_deltas"]
        lines.append(
            f"| {ckpt} | {comp['cheap7_delta_compact_minus_breadth']:+.4f} | {d['BLiMP']:+.2f} | "
            f"{d['Supplement']:+.2f} | {d['EWoK']:+.2f} | {d['Entity']:+.2f} | {d['COMPS']:+.2f} | "
            f"{d['GlobalPIQA']:+.2f} | {d['Reading']:+.3f} | {comp['decision_rule']['result']} |"
        )
    out_md = OUT_ROOT / "fw_comparison_eval_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", default=None, help="Checkpoint names without chck_: 70M, 80M, 90M, 100M")
    ap.add_argument("--arm", action="append", choices=DEFAULT_ARMS, default=None, help="Evaluate only selected arm(s); default evaluates both")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--validate-parser", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    parser_validation = validate_parser()
    validation_path = OUT_ROOT / "parser_validation.json"
    validation_path.write_text(json.dumps(parser_validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"parser_validation": parser_validation["status"], "path": str(validation_path), "max_abs_error": parser_validation.get("max_abs_error")}, indent=2), flush=True)
    if parser_validation["status"] != "PARSER_VALIDATION_OK":
        raise SystemExit(2)
    if args.validate_parser:
        return

    requested_checkpoints = args.checkpoint or ["70M", "80M"]
    selected_arms = args.arm or DEFAULT_ARMS

    existing_summary = load_existing_summary()
    all_results: dict[str, dict[str, Any]] = {}
    existing_checkpoints: list[str] = []
    if existing_summary:
        existing_checkpoints = list(existing_summary.get("checkpoints", []))
        all_results.update(existing_summary.get("results", {}))

    for ckpt in requested_checkpoints:
        for arm in selected_arms:
            result = run_eval(arm, ckpt, args.gpu, args.force)
            all_results[f"{arm}_{ckpt}"] = result

    checkpoints = merge_ordered_checkpoints(existing_checkpoints, requested_checkpoints)
    comparisons: dict[str, Any] = {}
    for ckpt in checkpoints:
        comp = compare_checkpoint(all_results, ckpt)
        if comp is not None:
            comparisons[ckpt] = comp

    summary = {
        "status": "FW_COMPARISON_EVAL",
        "created_utc": now_utc(),
        "checkpoints": checkpoints,
        "arms": {k: str(v) for k, v in ARMS.items()},
        "columns": CHEAP_COLUMNS,
        "cheap7_columns": SCALAR_COLUMNS,
        "parser_validation": parser_validation,
        "results": all_results,
        "comparisons": comparisons,
        "selected_arms_this_run": selected_arms,
        "requested_checkpoints_this_run": requested_checkpoints,
        "merged_existing_summary": str(SUMMARY_JSON) if existing_summary else None,
    }

    out_json = SUMMARY_JSON
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md = write_markdown(summary)

    print("\nSummary JSON:", out_json, flush=True)
    print("Summary MD:", out_md, flush=True)
    for ckpt, comp in comparisons.items():
        if comp.get("status") == "OK":
            print(
                f"{ckpt}: compact-breadth cheap7 delta = {comp['cheap7_delta_compact_minus_breadth']:+.4f}; "
                f"{comp['decision_rule']['result']}",
                flush=True,
            )

if __name__ == "__main__":
    main()
