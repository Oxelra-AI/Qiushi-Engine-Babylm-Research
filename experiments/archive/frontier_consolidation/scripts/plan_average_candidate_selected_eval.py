#!/usr/bin/env python3
"""research: dry plan for selected cheap-task scoring of a late-weight average.

This does not run evaluation.  It creates a run-dir view with `hf_model/chck_84M`
pointing to an already-built same-trajectory average candidate so the existing
research selected MLM evaluator can score it later without code changes.  The
pseudo-endpoint name intentionally uses chck_84M because the average is centered on
80/82/84 and should only be compared to the reference late window; exposure remains
provenance, not new training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
DEFAULT_OUT = WORKSPACE / "data" / "average_candidate_selected_eval_plan"
DEFAULT_CANDIDATE = WORKSPACE / "data/late_weight_average_scaffold/candidates/reference_scale1p75_seed43022__center_80_82_84_uniform"
SELECTED_EVAL_SCRIPT = WORKSPACE / "scripts/selected_mlm_checkpoint_eval.py"


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pseudo-endpoint", default="chck_84M")
    ap.add_argument("--label", default="reference_scale1p75_seed43022_avg80_82_84_uniform")
    ap.add_argument("--materialize-copy", action="store_true", help="copy instead of symlink candidate files")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_view = args.out_dir / "run_view"
    endpoint_dir = run_view / "hf_model" / args.pseudo_endpoint
    if endpoint_dir.exists() or endpoint_dir.is_symlink():
        if endpoint_dir.is_symlink() or endpoint_dir.is_file():
            endpoint_dir.unlink()
        else:
            shutil.rmtree(endpoint_dir)
    endpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    cand = args.candidate_dir.resolve()
    if not cand.exists():
        raise FileNotFoundError(cand)
    if args.materialize_copy:
        shutil.copytree(cand, endpoint_dir)
        view_kind = "copy"
    else:
        os.symlink(cand, endpoint_dir, target_is_directory=True)
        view_kind = "symlink"
    future_out = WORKSPACE / "data/avg80_82_84_selected_eval_if_authorized"
    cmd = [
        "PYTHONDONTWRITEBYTECODE=1", "python", "-B", rel(SELECTED_EVAL_SCRIPT),
        "--run-dir", rel(run_view),
        "--gpu", "0",
        "--out-dir", rel(future_out),
        "--label", args.label,
        "--endpoints", args.pseudo_endpoint,
    ]
    dry_cmd = cmd + ["--dry-run"]
    manifest = {
        "status": "AVERAGE_CANDIDATE_SELECTED_EVAL_PLAN_READY",
        "created_utc": utc_now(),
        "candidate_dir": rel(cand),
        "candidate_manifest": rel(cand / "averaging_manifest.json"),
        "run_view": rel(run_view),
        "endpoint_view": str(endpoint_dir),
        "endpoint_link_path": str(endpoint_dir),
        "endpoint_target_resolved": rel(cand),
        "view_kind": view_kind,
        "pseudo_endpoint": args.pseudo_endpoint,
        "label": args.label,
        "selected_eval_script": rel(SELECTED_EVAL_SCRIPT),
        "dry_run_command": " ".join(dry_cmd),
        "future_selected_eval_command_if_authorized": " ".join(cmd),
        "future_out_dir_if_authorized": rel(future_out),
        "official_evaluation_performed": False,
        "leaderboard_submission_performed": False,
        "scientific_use": "Run the future command only after delivered seed43122/A01 evidence makes same-trajectory stabilization the right target. The average carries no new training exposure; the pseudo-endpoint is a scoring wrapper label, not a chronological checkpoint.",
    }
    write_json(args.out_dir / "average_candidate_selected_eval_plan.json", manifest)
    md = [
        "# research average-candidate selected-eval dry plan\n\n",
        "No evaluation, upload, or submission was run. This only prepares a run-dir view for a possible later one-endpoint selected cheap-task score.\n\n",
        f"- Candidate: `{manifest['candidate_dir']}`\n",
        f"- Endpoint link path: `{manifest['endpoint_link_path']}` ({view_kind})\n",
        f"- Endpoint target: `{manifest['endpoint_target_resolved']}`\n",
        f"- Dry command: `{manifest['dry_run_command']}`\n",
        f"- Future command if authorized: `{manifest['future_selected_eval_command_if_authorized']}`\n",
        f"- Scientific use: {manifest['scientific_use']}\n\n",
        f"JSON: `{rel(args.out_dir / 'average_candidate_selected_eval_plan.json')}`\n",
    ]
    (args.out_dir / "average_candidate_selected_eval_plan.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "out_json": rel(args.out_dir / "average_candidate_selected_eval_plan.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
