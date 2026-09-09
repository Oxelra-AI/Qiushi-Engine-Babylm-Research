#!/usr/bin/env python3
"""research: interval driver for DeBERTa architecture-interaction selected predictions.

After `architecture_interaction_selected_panel.py` has produced per-target files,
this CPU-only driver runs `selected_pair_interval_analyzer.py` in direct mode for:
  full_repeat -> full_compact
  nodis_repeat -> nodis_compact
at each requested checkpoint. It also writes a plan when prediction files are missing.

It does not train, score, run SuperGLUE/AoA, upload, or submit.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
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
DEFAULT_PANEL = WS / "data/architecture_interaction_selected_panel"
DEFAULT_OUT = WS / "data/architecture_interaction_intervals"
ANALYZER = WS / "scripts/selected_pair_interval_analyzer.py"
PAIRS = [
    ("full_repeat", "full_compact", "full_compact_minus_repeat"),
    ("nodis_repeat", "nodis_compact", "nodis_compact_minus_repeat"),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(p: str | Path | None) -> Path | None:
    if p is None or str(p) == "":
        return None
    q = Path(str(p))
    if q.is_absolute():
        return q
    if q.exists():
        return q
    return ROOT / q


def load_panel_rows(panel_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summary = panel_dir / "architecture_interaction_selected_panel_summary.json"
    if not summary.exists():
        return {}
    obj = read_json(summary)
    return {(r.get("arm"), r.get("checkpoint")): r for r in obj.get("rows", [])}


def build_comparisons(panel_dir: Path, checkpoints: list[str]) -> list[dict[str, Any]]:
    rows = load_panel_rows(panel_dir)
    comps: list[dict[str, Any]] = []
    for ck in checkpoints:
        for left, right, label in PAIRS:
            lpt = resolve(rows.get((left, ck), {}).get("per_target"))
            rpt = resolve(rows.get((right, ck), {}).get("per_target"))
            comps.append({
                "checkpoint": ck,
                "label": label,
                "left_arm": left,
                "right_arm": right,
                "left_per_target": str(lpt) if lpt else None,
                "right_per_target": str(rpt) if rpt else None,
                "ready": bool(lpt and rpt and lpt.exists() and rpt.exists()),
            })
    return comps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoints", nargs="+", default=["chck_80M", "chck_100M"])
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=300)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comps = build_comparisons(args.panel_dir, args.checkpoints)
    results: list[dict[str, Any]] = []
    if not args.plan_only:
        for c in comps:
            if not c["ready"]:
                results.append({**c, "status": "missing"})
                continue
            subdir = args.out_dir / c["label"] / c["checkpoint"]
            done = subdir / "selected_pair_interval_analysis.json"
            if done.exists() and not args.force:
                results.append({**c, "status": "exists", "out_json": str(done)})
                continue
            cmd = [
                sys.executable, str(ANALYZER),
                "--left-per-target", c["left_per_target"],
                "--right-per-target", c["right_per_target"],
                "--left-label", c["left_arm"],
                "--right-label", c["right_arm"],
                "--out-dir", str(subdir),
                "--n-bootstrap", str(args.n_bootstrap),
            ]
            proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
            rec = {**c, "status": "ran", "returncode": proc.returncode, "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-2000:], "out_json": str(done)}
            results.append(rec)
            if proc.returncode != 0:
                break
    payload = {
        "status": "ARCHITECTURE_INTERACTION_INTERVAL_PLAN",
        "created_utc": now(),
        "panel_dir": str(args.panel_dir),
        "out_dir": str(args.out_dir),
        "comparisons": comps,
        "ready_count": sum(1 for c in comps if c["ready"]),
        "run_results": results,
        "meaning": "CPU item/subtask intervals for compact-minus-repeat within full and no-disentangle variants; combine with selected panel for the interaction reading.",
        "boundary": "No training, selected scoring, SuperGLUE, AoA, upload, or leaderboard submission.",
    }
    (args.out_dir / "interval_driver_plan.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(args.out_dir / "interval_driver_plan.json"), "ready_count": payload["ready_count"], "ran": not args.plan_only}, indent=2), flush=True)
    if any(r.get("returncode") not in (None, 0) for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
