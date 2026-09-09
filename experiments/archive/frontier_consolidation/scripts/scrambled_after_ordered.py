#!/usr/bin/env python3
"""research: Wait for ordered arm to finish, then launch scrambled arm on GPU0.

The scrambled arm OOM'd on GPU1 (residual memory), so we run it sequentially
on GPU0 after the ordered arm completes.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, os, subprocess, sys, time
from pathlib import Path

USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/scrambled_after_ordered.py')
for _ in range(10):
    if (_public_path("experiments")).exists():
        break
    USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts')

ORDERED_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_ordered_40M_seed43022')
SCRAMBLED_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022')
LAUNCHER = _public_path('experiments/archive/frontier_consolidation/scripts/compact_order_mechanism_experiment.py')

# Wait for ordered arm to produce scientific_metrics.json (training completion marker)
print("Waiting for ordered arm to complete...")
while True:
    metrics = _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_ordered_40M_seed43022/scientific_metrics.json')
    if metrics.exists():
        data = json.loads(metrics.read_text())
        cum = data.get("cum_words_exposed", 0)
        print(f"Ordered arm completed: cum_words={cum}")
        break
    # Also check training_log for progress
    log = _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_ordered_40M_seed43022/training_log.jsonl')
    if log.exists():
        lines = log.read_text().strip().split('\n')
        if lines and lines[-1]:
            last = json.loads(lines[-1])
            print(f"  Ordered at step {last.get('step','?')}, loss={last.get('loss','?'):.4f}" if isinstance(last.get('loss'), (int,float)) else f"  Ordered at step {last.get('step','?')}")
    time.sleep(30)

# Brief pause to ensure GPU0 memory is freed
print("Waiting 10s for GPU0 cleanup...")
time.sleep(10)

# Verify GPU0 is available without importing torch in this supervisor process.
# Importing torch here can create a small CUDA context before the child trainer starts;
# use nvidia-smi text telemetry instead and let the child own the CUDA process.
try:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    print("GPU snapshot before scrambled launch:")
    print(q.stdout.strip())
except Exception as e:
    print(f"GPU0 check warning: {e}")

# Clean previous failed scrambled run contents, but do not remove the run directory itself:
# The trainer accepts an existing empty output directory.
SCRAMBLED_RUN.mkdir(parents=True, exist_ok=True)
for p in sorted(SCRAMBLED_RUN.rglob("*"), key=lambda x: len(x.parts), reverse=True):
    if p.is_file() or p.is_symlink():
        p.unlink()
    elif p.is_dir():
        try:
            p.rmdir()
        except OSError:
            pass

# Launch scrambled arm on GPU0
print(f"\nLaunching scrambled arm on GPU0...")
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = "0"
cmd = [sys.executable, str(LAUNCHER), "--arm", "scrambled"]
proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env)

if proc.returncode == 0:
    print("\n✓ Scrambled arm completed successfully on GPU0")
    # Verify output
    if (_public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022/scientific_metrics.json')).exists():
        data = json.loads((_public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022/scientific_metrics.json')).read_text())
        print(f"  cum_words={data.get('cum_words_exposed',0)}, checkpoints={data.get('checkpoints','?')}")
    if (_public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022/hf_model/chck_40M')).exists():
        print(f"  ✓ chck_40M exists")
    if (_public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022/hf_model/chck_20M')).exists():
        print(f"  ✓ chck_20M exists")
else:
    print(f"\n✗ Scrambled arm failed with return code {proc.returncode}")
    sys.exit(proc.returncode)
