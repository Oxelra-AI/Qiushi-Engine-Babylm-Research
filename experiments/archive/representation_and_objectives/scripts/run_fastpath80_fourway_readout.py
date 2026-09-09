#!/usr/bin/env python3
"""Run the fixed research four-way readout for the frozen80 fast-path experiment."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, pathlib, subprocess, sys, time
ROOT = _public_path('.')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/fastpath80_fourway_readout')
PATHS = {
    "anchor": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/anchor80.json'),
    "coherent": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/coherent80.json'),
    "shuffled": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/shuffled80.json'),
    "ordinary": _public_path('experiments/archive/representation_and_objectives/data/frozen80_eval/per_target/ordinary84.json'),
}
SCRIPT = _public_path('experiments/archive/representation_and_objectives/scripts/task_balanced_readout_75e45868.py')

def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)

missing = {k: rel(v) for k,v in PATHS.items() if not v.exists()}
if missing:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = {"status":"MISSING_INPUTS","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),"missing":missing,"paths":{k:rel(v) for k,v in PATHS.items()}}
    (_public_path('experiments/archive/representation_and_objectives/data/fastpath80_fourway_readout/fastpath80_fourway_launcher_status.json')).write_text(json.dumps(status, indent=2)+"\n")
    print(json.dumps(status, indent=2))
    sys.exit(2)
cmd = [
    sys.executable, "-B", str(SCRIPT),
    "--anchor-payload", str(PATHS["anchor"]),
    "--coherent-payload", str(PATHS["coherent"]),
    "--shuffled-payload", str(PATHS["shuffled"]),
    "--ordinary-payload", str(PATHS["ordinary"]),
    "--out-dir", str(OUT_DIR),
]
print(json.dumps({"event":"running_step192_readout","cmd":[rel(pathlib.Path(x)) if x.startswith(str(ROOT)) else x for x in cmd]}), flush=True)
proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
(_public_path('experiments/archive/representation_and_objectives/data/fastpath80_fourway_readout/fastpath80_fourway_readout_stdout.log')).write_text(proc.stdout)
(_public_path('experiments/archive/representation_and_objectives/data/fastpath80_fourway_readout/fastpath80_fourway_readout_stderr.log')).write_text(proc.stderr)
if proc.returncode != 0:
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    sys.exit(proc.returncode)
print(proc.stdout, end="")
