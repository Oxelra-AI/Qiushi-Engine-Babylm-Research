#!/usr/bin/env python3
"""research GPU0 sequential launcher for the architecture-interaction experiment.

Runs exactly the two repeat-side cells that share one GPU:
1. full_p2c_c2p_abs / repeat: missing legal full-DeBERTa repeat reference.
2. no_disentangle_abs / repeat: repeat side of the c2p+p2c removal contrast.

The compact no_disentangle_abs cell is launched separately on GPU1.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import subprocess
import sys
import time

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
SCRIPT = _public_path('experiments/archive/frontier_consolidation/scripts/train_deberta_positional_ablation.py')
RUNS = [
    {
        "variant": "full_p2c_c2p_abs",
        "data_arm": "repeat",
        "gpu": "0",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022'),
    },
    {
        "variant": "no_disentangle_abs",
        "data_arm": "repeat",
        "gpu": "0",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/no_disentangle_abs_repeat_deberta100M_seed43022'),
    },
]
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/architecture_interaction_launch')
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = _public_path('experiments/archive/frontier_consolidation/data/architecture_interaction_launch/gpu0_fullrepeat_then_nodisrepeat_launcher.jsonl')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_event(obj: dict) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time_utc": now(), **obj}, ensure_ascii=False) + "\n")


def main() -> int:
    if not SCRIPT.exists():
        raise FileNotFoundError(SCRIPT)
    for spec in RUNS:
        rd = spec["run_dir"]
        if rd.exists() and any(rd.iterdir()):
            raise RuntimeError(f"refusing to overwrite nonempty run_dir: {rd}")
    write_event({"event": "sequential_start", "runs": [{**s, "run_dir": str(s["run_dir"])} for s in RUNS]})
    for i, spec in enumerate(RUNS, 1):
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--variant", spec["variant"],
            "--data-arm", spec["data_arm"],
            "--gpu", spec["gpu"],
            "--run-dir", str(spec["run_dir"]),
            "--count-words",
        ]
        write_event({"event": "run_start", "index": i, "cmd": cmd})
        proc = subprocess.run(cmd, cwd=str(USER_ROOT), text=True)
        write_event({"event": "run_finished", "index": i, "returncode": proc.returncode, "run_dir": str(spec["run_dir"])})
        if proc.returncode != 0:
            write_event({"event": "sequential_abort", "failed_index": i})
            return proc.returncode
    write_event({"event": "sequential_done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
