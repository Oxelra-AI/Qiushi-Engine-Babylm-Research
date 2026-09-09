#!/usr/bin/env python3
"""Post-delivery driver for compliant-tokenizer BabyLM endpoints.

Run this only after a retrain task has delivered a terminal result and the run
directory is ready.  It is not a wait/poll script.

The driver keeps tokenizer coordinates separated by accepting explicit target,
run-dir, output roots, and expected tokenizer SHA.  This matters after research,
where the original legal research tokenizer was found to produce <unk> on scored
Supplement newlines and a same-pool byte-alphabet tokenizer repair was prepared.
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
EVAL_HARNESS = WORKSPACE / "scripts/evaluate_compliant_endpoint.py"
INSPECTOR = WORKSPACE / "scripts/inspect_compliant_retrain.py"
PROJECTOR = WORKSPACE / "scripts/project_compliant_eval_continuation.py"
SUPPLEMENT_SLICER = WORKSPACE / "scripts/supplement_prediction_slices.py"
COLLATOR = USER_ROOT / "experiments/archive/representation_and_objectives/scripts/stage_pristine_collate.py"
DEFAULT_FULL_ROOT = WORKSPACE / "data/compliant_full_eval"
DEFAULT_COLLATE_ROOT = WORKSPACE / "data/compliant_pristine_collate"
DEFAULT_OUT = WORKSPACE / "data/compliant_postdelivery_driver"
FROZEN_REINVEST_POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
FROZEN_REINVEST_TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
FROZEN_REINVEST_META = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
FROZEN_REINVEST_POOL_SHA256 = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
FROZEN_REINVEST_TRAIN_SHA256 = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
FROZEN_REINVEST_META_SHA256 = "92aa4c00b09d201a09da27e7e16677643bc7cda321471e1ec46332600339b3e4"
FROZEN_REINVEST_POOL_ROWS = 64_740
FROZEN_REINVEST_TRAIN_ROWS = 647_400
FROZEN_REINVEST_POOL_WORDS = 10_000_000
FROZEN_REINVEST_TRAIN_WORDS = 100_000_000
FROZEN_REINVEST_PASSES = 10

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
EXPENSIVE_FINAL_COLUMNS = ["SuperGLUE", "AoA"]
DEFAULT_RUNS = {
    "reinvest": STUDY / "training/runs/complianttok_reinvest_seed43022_r2",
    "clean_qwen": STUDY / "training/runs/complianttok_cleanqwen_seed43022_r2",
}
DEFAULT_TARGETS = {
    "reinvest": "complianttok_reinvest_seed43022",
    "clean_qwen": "complianttok_clean_qwen_seed43022",
}


def run_cmd(cmd: list[str], *, timeout: int, dry_run: bool) -> dict[str, Any]:
    rec: dict[str, Any] = {"cmd": cmd, "timeout": timeout, "dry_run": dry_run, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if dry_run:
        rec.update({"returncode": None, "stdout_tail": "", "stderr_tail": ""})
        return rec
    p = subprocess.run(cmd, cwd=str(USER_ROOT), text=True, capture_output=True, timeout=timeout)
    rec.update({
        "returncode": p.returncode,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stdout_tail": p.stdout[-6000:],
        "stderr_tail": p.stderr[-6000:],
    })
    if p.returncode != 0:
        raise RuntimeError(json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_arg(path: pathlib.Path | str) -> str:
    """Return a command path portable from USER_ROOT when possible."""
    p = pathlib.Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(USER_ROOT))
        except ValueError:
            return str(p)
    return str(p)

def recorded_prediction_for_column(full_root: pathlib.Path, target: str, per_target: dict[str, Any], column: str) -> pathlib.Path:
    """Return the prediction path recorded by the per-target JSON, with target isolation checks.

    Do not fall back to a filesystem 'latest predictions.json' search: stale or
    partial files under the same output root could otherwise enter pristine
    collation after an interrupted evaluation. The evaluation harness writes the
    authoritative prediction path in tasks[column]['predictions']; require it.
    """
    tasks = per_target.get("tasks", {})
    rec = tasks.get(column) if isinstance(tasks, dict) else None
    if not isinstance(rec, dict) or not rec.get("predictions"):
        raise FileNotFoundError(f"Per-target JSON for {target} has no recorded predictions path for {column}")
    p_raw = pathlib.Path(str(rec["predictions"]))
    p = p_raw if p_raw.is_absolute() else USER_ROOT / p_raw
    if not p.exists():
        raise FileNotFoundError(f"Recorded {column} predictions do not exist: {p}")
    try:
        rel_parts = p.resolve().relative_to((full_root / "official_outputs" / target / column).resolve()).parts
        if "predictions.json" not in rel_parts[-1:]:
            raise ValueError
    except Exception as exc:
        raise RuntimeError(f"Recorded {column} predictions are not under the isolated target/column output root: {p}") from exc
    return p.resolve()


def recorded_prediction_for_ewok(full_root: pathlib.Path, target: str, per_target: dict[str, Any]) -> pathlib.Path:
    try:
        return recorded_prediction_for_column(full_root, target, per_target, "EWoK")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}; cannot pristine-collate") from exc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["reinvest", "clean_qwen"])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--decision-target", type=float, default=41.8)
    ap.add_argument("--target", default="", help="Evaluation target name; use a distinct name for byte-alphabet tokenizer runs.")
    ap.add_argument("--run-dir", default="", help="Delivered/fresh run directory; default uses research old-tokenizer run directories.")
    ap.add_argument("--expected-tokenizer", default="", help="Tokenizer directory expected by the inspector.")
    ap.add_argument("--expected-tokenizer-sha256", default="", help="Tokenizer JSON SHA expected by the inspector.")
    ap.add_argument("--full-root", default=str(DEFAULT_FULL_ROOT))
    ap.add_argument("--collate-root", default=str(DEFAULT_COLLATE_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--inspection-out-dir", default="", help="Optional target-specific inspector output directory to avoid concurrent endpoint races.")
    ap.add_argument("--dry-run", action="store_true", help="Write intended commands without running inspection/evaluation/collation.")
    ap.add_argument("--force", action="store_true", help="Pass --force to the evaluation harness.")
    ap.add_argument("--cheap-only", action="store_true", help="Run only cheap columns and projection; do not launch SuperGLUE/AoA even if warranted.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target or DEFAULT_TARGETS[args.arm]
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else DEFAULT_RUNS[args.arm]
    full_root = pathlib.Path(args.full_root)
    collate_root = pathlib.Path(args.collate_root)
    out_dir = pathlib.Path(args.out_dir)
    inspection_out_dir = pathlib.Path(args.inspection_out_dir) if args.inspection_out_dir else (WORKSPACE / "data/compliant_retrain_inspection" / target)
    inspection_json = inspection_out_dir / "compliant_retrain_inspection.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []

    inspect_cmd = [
        sys.executable,
        path_arg(INSPECTOR),
        "--arms", path_arg(run_dir),
        "--out-dir", path_arg(inspection_out_dir),
        "--expected-train-file", path_arg(FROZEN_REINVEST_TRAIN_100M),
        "--expected-train-sha256", FROZEN_REINVEST_TRAIN_SHA256,
        "--expected-pool-10m", path_arg(FROZEN_REINVEST_POOL_10M),
        "--expected-pool-sha256", FROZEN_REINVEST_POOL_SHA256,
        "--expected-meta", path_arg(FROZEN_REINVEST_META),
        "--expected-meta-sha256", FROZEN_REINVEST_META_SHA256,
        "--expected-pool-words", str(FROZEN_REINVEST_POOL_WORDS),
        "--expected-train-words", str(FROZEN_REINVEST_TRAIN_WORDS),
        "--expected-pool-rows", str(FROZEN_REINVEST_POOL_ROWS),
        "--expected-train-rows", str(FROZEN_REINVEST_TRAIN_ROWS),
        "--expected-passes", str(FROZEN_REINVEST_PASSES),
        "--verify-corpus-content",
    ]
    if args.expected_tokenizer:
        inspect_cmd += ["--expected-tokenizer", args.expected_tokenizer]
    if args.expected_tokenizer_sha256:
        inspect_cmd += ["--expected-tokenizer-sha256", args.expected_tokenizer_sha256]
    events.append({"stage": "inspect", **run_cmd(inspect_cmd, timeout=1800, dry_run=args.dry_run)})

    if not args.dry_run:
        inspection = load_json(inspection_json)
        recs = inspection.get("records", [])
        arm_rec = recs[0] if recs else {}
        if not arm_rec.get("complete_100m_compliant_retrain"):
            out = {
                "status": "POSTDELIVERY_STOP_INCOMPLETE_RETRAIN",
                "arm": args.arm,
                "target": target,
                "run_dir": str(run_dir),
                "inspection_json": str(inspection_json),
                "inspection_record": arm_rec,
                "events": events,
                "interpretation": "The delivered retrain directory is not a complete 100M endpoint under the expected tokenizer; do not evaluate or infer model quality.",
            }
            out_path = out_dir / f"{target}_postdelivery_driver.json"
            out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({"status": out["status"], "out_json": str(out_path)}, indent=2, ensure_ascii=False), flush=True)
            return

    eval_cmd = [
        sys.executable, path_arg(EVAL_HARNESS),
        "--arm", args.arm,
        "--target", target,
        "--run-dir", path_arg(run_dir),
        "--out-root", path_arg(full_root),
        "--collate-root", path_arg(collate_root),
        "--gpu", str(args.gpu),
        "--columns", *CHEAP_COLUMNS,
    ]
    if args.force:
        eval_cmd.append("--force")
    events.append({"stage": "cheap_columns", **run_cmd(eval_cmd, timeout=8 * 3600, dry_run=args.dry_run)})


    per_target = full_root / "per_target" / f"{target}.json"
    if args.dry_run:
        supp_pred = pathlib.Path("<dry-run-supplement-predictions>")
    else:
        cheap_payload = load_json(per_target)
        supp_pred = recorded_prediction_for_column(full_root, target, cheap_payload, "Supplement")
    slice_cmd = [
        sys.executable, path_arg(SUPPLEMENT_SLICER),
        "--predictions", path_arg(supp_pred),
        "--tag", target,
        "--out-dir", path_arg(full_root / "posthoc_supplement_slices"),
    ]
    events.append({"stage": "supplement_prediction_slices", **run_cmd(slice_cmd, timeout=600, dry_run=args.dry_run)})
    per_target = full_root / "per_target" / f"{target}.json"
    projection_json = per_target.with_name(per_target.stem + "_continuation_policy.json")
    project_cmd = [sys.executable, path_arg(PROJECTOR), "--per-target-json", path_arg(per_target), "--decision-target", str(args.decision_target), "--out", path_arg(projection_json)]
    events.append({"stage": "continuation_projection", **run_cmd(project_cmd, timeout=600, dry_run=args.dry_run)})

    hard_stop = False
    projection: dict[str, Any] = {}
    recommended = "dry_run_no_projection_read"
    if not args.dry_run:
        projection = load_json(projection_json)
        policy = projection.get("policy", {}) if isinstance(projection, dict) else {}
        hard_stop = bool(policy.get("hard_stop_remaining_evaluation"))
        recommended = str(policy.get("recommended_action"))

    if hard_stop:
        out = {
            "status": "POSTDELIVERY_STOP_BY_HARD_UPPER_BOUND",
            "arm": args.arm,
            "target": target,
            "run_dir": str(run_dir),
            "per_target_json": str(per_target),
            "projection_json": str(projection_json),
            "projection": projection,
            "events": events,
            "interpretation": "Known columns plus favorable upper-bound missing columns cannot reach the decision target; remaining expensive evaluation is not warranted.",
        }
        out_path = out_dir / f"{target}_postdelivery_driver.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": out["status"], "out_json": str(out_path), "recommended_action": recommended}, indent=2, ensure_ascii=False), flush=True)
        return

    if args.cheap_only:
        out = {
            "status": "POSTDELIVERY_CHEAP_ONLY_DONE",
            "arm": args.arm,
            "target": target,
            "run_dir": str(run_dir),
            "per_target_json": str(per_target),
            "projection_json": str(projection_json),
            "projection": projection,
            "events": events,
            "interpretation": "Cheap columns are complete. Do not treat this as a full official result; finish SuperGLUE/AoA/collation for the reinvest endpoint unless hard-stop arithmetic later proves impossible.",
        }
        out_path = out_dir / f"{target}_postdelivery_driver.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": out["status"], "out_json": str(out_path), "recommended_action": recommended}, indent=2, ensure_ascii=False), flush=True)
        return

    final_eval_cmd = [
        sys.executable, path_arg(EVAL_HARNESS),
        "--arm", args.arm,
        "--target", target,
        "--run-dir", path_arg(run_dir),
        "--out-root", path_arg(full_root),
        "--collate-root", path_arg(collate_root),
        "--gpu", str(args.gpu),
        "--columns", *EXPENSIVE_FINAL_COLUMNS,
    ]
    if args.force:
        final_eval_cmd.append("--force")
    events.append({"stage": "superglue_aoa", **run_cmd(final_eval_cmd, timeout=24 * 3600, dry_run=args.dry_run)})

    if args.dry_run:
        ewok_pred = pathlib.Path("<dry-run-ewok-predictions>")
    else:
        per_payload = load_json(per_target)
        ewok_pred = recorded_prediction_for_ewok(full_root, target, per_payload)

    model_root = run_dir / "hf_model"
    collate_out = collate_root / target
    collate_cmd = [
        sys.executable, path_arg(COLLATOR),
        "--full-root", path_arg(full_root),
        "--target", target,
        "--model-root", path_arg(model_root),
        "--out-dir", path_arg(collate_out),
        "--pristine-ewok-predictions", path_arg(ewok_pred),
        "--aoa-dir", path_arg(full_root / "aoa_outputs" / target),
        "--endpoint", "chck_100M",
        "--tag", target,
    ]
    events.append({"stage": "pristine_collate", **run_cmd(collate_cmd, timeout=1800, dry_run=args.dry_run)})

    summary_path = collate_out / f"pristine_collate_{target}_summary.json"
    summary = load_json(summary_path) if (summary_path.exists() and not args.dry_run) else None
    score_summary = summary.get("score_summary") if isinstance(summary, dict) else None
    official = score_summary.get("official_overall") if isinstance(score_summary, dict) else None
    collated_overall = official.get("Overall") if isinstance(official, dict) else None
    collated_columns = official.get("scores") if isinstance(official, dict) else None
    out = {
        "status": "POSTDELIVERY_FULL_EVAL_AND_COLLATE_DONE" if not args.dry_run else "POSTDELIVERY_DRY_RUN",
        "arm": args.arm,
        "target": target,
        "run_dir": str(run_dir),
        "inspection_json": str(inspection_json),
        "per_target_json": str(per_target),
        "projection_json": str(projection_json),
        "collate_summary_json": str(summary_path),
        "collated_overall": collated_overall,
        "collated_columns": collated_columns,
        "events": events,
        "interpretation": "For reinvest, DONE status is the end-to-end official-compatible result under the specified legal tokenizer coordinate. For clean_qwen, it remains a fixed-tokenizer scientific control.",
    }
    out_path = out_dir / f"{target}_postdelivery_driver.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_path), "collate_summary_json": str(summary_path), "collated_overall": collated_overall}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
