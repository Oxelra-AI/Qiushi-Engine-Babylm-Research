#!/usr/bin/env python3
"""research: Launch one-step GPU smoke for innovation-biased trainer.

Runs the innovation-biased trainer for exactly one step to verify:
- Model builds on GPU
- Innovation-biased masking works with example_id lookup
- Loss computes and backward pass succeeds
- No crash in the training loop

This is NOT a training run. It validates the implementation end-to-end.
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

def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"

TRAINER = WORKSPACE / "scripts/innovation_biased_trainer.py"
TRAIN_100M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
TOKENIZER = WORKSPACE / "data/compliant_tokenizer"
INNOVATION_MAP = WORKSPACE / "data/innovation_group_map/innovation_group_map.json"
OUT_DIR = WORKSPACE / "data/gpu_smoke"

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cmd = [
        sys.executable, str(TRAINER),
        "--output_dir", str(OUT_DIR / "smoke_run"),
        "--example_jsonl", str(TRAIN_100M),
        "--tokenizer_path", str(TOKENIZER),
        "--innovation_map", str(INNOVATION_MAP),
        "--max_word_exposure", "1000078",  # lands on example boundary
        "--checkpoint_words", "500000",
        "--batch_size", "256",
        "--p_innov", "0.5",
        "--p_copy", "0.0",
        "--mask_prob", "0.15",
        "--seed", "43",
        "--extra_init_seed", "43022",
        "--train_rng_seed", "43023",
        "--log_every", "5",
    ]
    print(json.dumps({"event": "launch", "cmd": " ".join(cmd)}), flush=True)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(WORKSPACE),
    )

    stdout_tail = result.stdout[-3000:] if result.stdout else ""
    stderr_tail = result.stderr[-2000:] if result.stderr else ""

    outcome = {
        "status": "GPU_SMOKE",
        "returncode": result.returncode,
        "elapsed_sec": round(time.time() - t0, 2),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }

    (OUT_DIR / "gpu_smoke_result.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )

    # Check for key success indicators in stdout
    success = result.returncode == 0 and '"event": "done"' in result.stdout
    outcome["smoke_passed"] = success

    print(json.dumps({
        "status": outcome["status"],
        "smoke_passed": success,
        "returncode": result.returncode,
        "elapsed_sec": outcome["elapsed_sec"],
        "stdout_has_done": '"event": "done"' in result.stdout,
    }, indent=2), flush=True)

    if not success:
        print("STDOUT TAIL:", stdout_tail[-1000:], file=sys.stderr)
        print("STDERR TAIL:", stderr_tail[-1000:], file=sys.stderr)


if __name__ == "__main__":
    main()
