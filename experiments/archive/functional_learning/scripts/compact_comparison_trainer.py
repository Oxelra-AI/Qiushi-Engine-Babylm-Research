#!/usr/bin/env python3
"""research: Compact-view learner comparison trainer.

Trains the coherent86 parent on matched tail variants to test whether selective
compaction + reinvestment improves learning.  Each arm uses the same corrected
bridge trainer infrastructure (word-paced macro-updates, deterministic row-keyed
WWM, trusted private-adapter loading, private-only optimization).

Arms:
  reference     - already trained (research), no action needed
  compact_unspent  - compacted tail, saved words removed (fewer total words)
  compact_reinvest - compacted tail, saved words spent on new source (same total)
  compact_neutral  - compacted tail, saved words recycled as copies (same total)

The critical control: for the same selected source IDs that receive compaction,
the reference arm uses the inherited rewrite.  This prevents confounding selection
(easier/cleaner pairs) with transformation (compaction).
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
TRAINER = _public_path('experiments/archive/functional_learning/scripts/corrected_bridge_trainer.py')
EVALUATOR = _public_path('experiments/archive/functional_learning/scripts/bridge_eval.py')

DEFAULT_REFERENCE = _public_path('experiments/archive/functional_learning/data/reference_tail_corrected_ordinary/checkpoints/update_0354')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def query_gpu_memory() -> List[Dict[str, Any]]:
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
           "--format=csv,noheader,nounits"]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed")
    rows = []
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4:
            rows.append({"index": int(parts[0]), "memory_used_mb": int(parts[2]),
                          "memory_total_mb": int(parts[3])})
    return rows


def choose_idle_gpu(max_used_mb: int) -> Optional[Dict[str, Any]]:
    rows = query_gpu_memory()
    idle = [r for r in rows if r["memory_used_mb"] <= max_used_mb]
    if not idle:
        return None
    idle.sort(key=lambda r: (r["memory_used_mb"], r["index"]))
    return idle[0]


def wait_for_gpu(max_used_mb: int, interval: float, max_wait: float) -> Dict[str, Any]:
    start = time.time()
    while True:
        selected = choose_idle_gpu(max_used_mb)
        if selected:
            return selected
        if time.time() - start > max_wait:
            raise TimeoutError(f"No idle GPU after {max_wait}s")
        time.sleep(interval)


def train_arm(arm_name: str, tail_jsonl: pathlib.Path, out_dir: pathlib.Path,
              gpu: int, train_seed: int, checkpoint_every: int,
              micro_batch: int, loss_mode: str) -> int:
    """Train one compact-comparison arm using the corrected bridge trainer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        sys.executable, str(TRAINER),
        "--overlay-jsonl", str(tail_jsonl),
        "--arm", "ordinary_wwm",
        "--out-dir", str(out_dir),
        "--gpu", "0",  # Mapped by CUDA_VISIBLE_DEVICES
        "--micro-batch", str(micro_batch),
        "--checkpoint-every", str(checkpoint_every),
        "--train-seed", str(train_seed),
        "--loss-mode", loss_mode,
    ]

    launch = {
        "status": f"COMPACT_COMPARISON_LAUNCH_{arm_name.upper()}",
        "arm": arm_name,
        "tail_jsonl": rel(tail_jsonl),
        "out_dir": rel(out_dir),
        "gpu": gpu,
        "train_seed": train_seed,
        "cmd": [str(c) for c in cmd],
    }
    (out_dir / "launch.json").write_text(
        json.dumps(launch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(launch, ensure_ascii=False), flush=True)

    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Save last 40 lines of stdout
    for line in proc.stdout.splitlines()[-40:]:
        print(line, flush=True)
    if proc.stderr:
        for line in proc.stderr.splitlines()[-5:]:
            print(f"STDERR: {line}", file=sys.stderr, flush=True)

    return proc.returncode


def eval_arm(arm_name: str, checkpoint_path: pathlib.Path, out_dir: pathlib.Path,
             gpu: int, eval_relation: bool = True, eval_cheap7: bool = True) -> int:
    """Evaluate one arm checkpoint with relation probes and Cheap7."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [sys.executable, str(EVALUATOR),
           "--model-path", str(checkpoint_path),
           "--out-dir", str(out_dir),
           "--gpu", "0"]
    if eval_relation:
        cmd.append("--eval-relation")
        cmd.append("--include-parent")
    if eval_cheap7:
        cmd.extend(["--eval-cheap7", "--cheap7-columns", "Cheap7"])
    cmd.append("--force")

    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines()[-20:]:
        print(line, flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=[
        "compact_unspent", "compact_reinvest", "compact_neutral", "all"])
    ap.add_argument("--materialization-dir", type=pathlib.Path, required=True,
                    help="Directory containing the materialized tails (e.g., compact_reinvest/tail.jsonl)")
    ap.add_argument("--out-root", type=pathlib.Path,
                    default=_public_path('experiments/archive/functional_learning/data/compact_comparison'))
    ap.add_argument("--train-seed", type=int, default=56056)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--loss-mode", default="explicit_word_fraction")
    ap.add_argument("--idle-memory-mb", type=int, default=2500)
    ap.add_argument("--max-wait-sec", type=float, default=3600.0)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    mat_dir = args.materialization_dir
    arms = [args.arm] if args.arm != "all" else [
        "compact_unspent", "compact_reinvest", "compact_neutral"]

    results = {}
    for arm in arms:
        tail_path = mat_dir / arm / "tail.jsonl"
        if not tail_path.exists():
            print(json.dumps({"event": "skip_missing_tail", "arm": arm,
                               "path": rel(tail_path)}), flush=True)
            continue

        out_dir = args.out_root / arm
        print(json.dumps({"event": "arm_start", "arm": arm,
                           "tail": rel(tail_path)}), flush=True)

        # Wait for GPU
        selected = wait_for_gpu(args.idle_memory_mb, 60.0, args.max_wait_sec)
        gpu = selected["index"]

        # Train
        t0 = time.time()
        rc = train_arm(arm, tail_path, out_dir, gpu, args.train_seed,
                       args.checkpoint_every, args.micro_batch, args.loss_mode)
        elapsed = round(time.time() - t0, 1)

        arm_result = {
            "arm": arm, "train_rc": rc, "train_elapsed_sec": elapsed,
            "out_dir": rel(out_dir), "tail": rel(tail_path),
        }

        if rc == 0 and not args.skip_eval:
            # Find last checkpoint
            ckpt_dir = out_dir / "checkpoints"
            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.iterdir())
                if ckpts:
                    last_ckpt = ckpts[-1]
                    eval_dir = args.out_root / f"{arm}_eval"
                    eval_rc = eval_arm(arm, last_ckpt, eval_dir, gpu)
                    arm_result["eval_rc"] = eval_rc
                    arm_result["eval_dir"] = rel(eval_dir)
                    arm_result["eval_checkpoint"] = rel(last_ckpt)

        results[arm] = arm_result
        print(json.dumps({"event": "arm_done", **arm_result}), flush=True)

    summary = {
        "status": "COMPACT_COMPARISON_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": results,
        "reference_checkpoint": rel(DEFAULT_REFERENCE),
        "materialization_dir": rel(mat_dir),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
