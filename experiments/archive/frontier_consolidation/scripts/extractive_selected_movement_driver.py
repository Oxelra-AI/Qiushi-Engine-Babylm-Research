#!/usr/bin/env python3
"""research: CPU driver for post-result selected item movement readings.

After extractive_selected_eval_panel.py has produced per-target JSON files,
this driver runs the existing research movement reader for extractive-vs-compact
checkpoint pairs.  In --plan-only mode it only reports which comparisons are ready.
It never trains, scores models, runs SuperGLUE/AoA, uploads, or submits.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_PANEL_DIR = WS / "data/extractive_selected_eval_panel"
ROWS_CSV = DEFAULT_PANEL_DIR / "extractive_selected_panel_rows.csv"
MOVEMENT_READER = WS / "scripts/selected_prediction_movement_reader.py"
DEFAULT_OUT = WS / "data/extractive_selected_movement_ready_driver"
EXTRACTIVE_ARMS = ["extractive_balanced", "extractive_wide"]
REFERENCE_ARM = "legal_compact"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve(path_s: str | None) -> Path | None:
    if not path_s:
        return None
    p = Path(path_s)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return ROOT / p


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def per_target_from_summary(summary_path_s: str | None) -> Path | None:
    p = resolve(summary_path_s)
    if p is None or not p.exists():
        return None
    obj = read_json(p)
    if isinstance(obj, dict) and "record" in obj and isinstance(obj["record"], dict):
        pt = obj["record"].get("per_target")
        return resolve(pt) if pt else None
    if isinstance(obj, dict) and "tasks" in obj:
        return p
    return None


def load_rows(rows_csv: Path) -> list[dict[str, str]]:
    if not rows_csv.exists():
        return []
    with rows_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_plan(rows_csv: Path) -> dict[str, Any]:
    rows = load_rows(rows_csv)
    by: dict[tuple[str, str], dict[str, str]] = {(r.get("arm", ""), r.get("checkpoint", "")): r for r in rows}
    comparisons: list[dict[str, Any]] = []
    cks = sorted({r.get("checkpoint", "") for r in rows if r.get("checkpoint")}, key=lambda x: int(x.split("_")[1][:-1]) if x.startswith("chck_") and x.endswith("M") else 10**9)
    for ck in cks:
        ref = by.get((REFERENCE_ARM, ck))
        ref_pt = per_target_from_summary(ref.get("summary_path")) if ref else None
        for arm in EXTRACTIVE_ARMS:
            row = by.get((arm, ck))
            arm_pt = per_target_from_summary(row.get("summary_path")) if row else None
            comparisons.append({
                "arm": arm,
                "checkpoint": ck,
                "reference_arm": REFERENCE_ARM,
                "reference_per_target": str(ref_pt) if ref_pt else None,
                "extractive_per_target": str(arm_pt) if arm_pt else None,
                "ready": bool(ref_pt and arm_pt and ref_pt.exists() and arm_pt.exists()),
            })
    # If no extractive rows exist yet, expose intended late checkpoints from the panel summary when available.
    summary_path = rows_csv.parent / "extractive_selected_panel_summary.json"
    if not comparisons and summary_path.exists():
        s = read_json(summary_path)
        for ck in s.get("checkpoints", []):
            for arm in EXTRACTIVE_ARMS:
                comparisons.append({
                    "arm": arm,
                    "checkpoint": ck,
                    "reference_arm": REFERENCE_ARM,
                    "reference_per_target": None,
                    "extractive_per_target": None,
                    "ready": False,
                })
    return {
        "status": "EXTRACTIVE_SELECTED_MOVEMENT_PLAN",
        "created_utc": now(),
        "rows_csv": str(rows_csv),
        "movement_reader": str(MOVEMENT_READER),
        "reference_arm": REFERENCE_ARM,
        "extractive_arms": EXTRACTIVE_ARMS,
        "comparisons": comparisons,
        "ready_count": sum(1 for c in comparisons if c["ready"]),
        "meaning": "CPU item-movement reading of already evaluated selected predictions; use only after selected per-target files exist.",
        "no_training_selected_eval_upload_aoa_or_leaderboard": True,
    }


def run_comparisons(plan: dict[str, Any], out_dir: Path, max_examples: int, force: bool) -> list[dict[str, Any]]:
    if not MOVEMENT_READER.exists():
        raise FileNotFoundError(MOVEMENT_READER)
    results: list[dict[str, Any]] = []
    for cmp in plan["comparisons"]:
        if not cmp.get("ready"):
            continue
        arm = cmp["arm"]
        ck = cmp["checkpoint"]
        out = out_dir / arm / ck
        done = out / "selected_prediction_movement.json"
        if done.exists() and not force:
            results.append({"arm": arm, "checkpoint": ck, "status": "exists", "out_json": str(done)})
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-B", str(MOVEMENT_READER),
            "--left-per-target", cmp["reference_per_target"],
            "--right-per-target", cmp["extractive_per_target"],
            "--left-label", REFERENCE_ARM,
            "--right-label", arm,
            "--out-dir", str(out),
            "--max-examples", str(max_examples),
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
        (out / "movement_driver_command.json").write_text(json.dumps({"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}, indent=2) + "\n", encoding="utf-8")
        results.append({"arm": arm, "checkpoint": ck, "returncode": proc.returncode, "out_json": str(done), "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]})
        if proc.returncode != 0:
            break
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-examples", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rows_csv = args.panel_dir / "extractive_selected_panel_rows.csv"
    plan = build_plan(rows_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.plan_only:
        plan["run_results"] = run_comparisons(plan, args.out_dir, args.max_examples, args.force)
    (args.out_dir / "movement_driver_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "out": str(args.out_dir / "movement_driver_plan.json"), "ready_count": plan["ready_count"], "n_comparisons": len(plan["comparisons"]), "ran": not args.plan_only}, indent=2), flush=True)


if __name__ == "__main__":
    main()
