#!/usr/bin/env python3
"""Train seed43222 CLEAN in an alternate run directory.

The scientific recipe and data stream match the
standard preflight; only the output directory differs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = _public_path('experiments/archive/relation_learning/scripts/train_seed43222_clean_parallel.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
sys.path.insert(0, str(_public_path('experiments/archive/relation_learning/scripts')))
import train_seed43222_arms as base  # noqa: E402

RUN_DIR = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel')
OUT = _public_path('experiments/archive/relation_learning/data/seed43222_clean_parallel')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    existing = base.summarize_metrics(RUN_DIR)
    if existing.get("ready"):
        print(json.dumps({"status": "CLEAN_PARALLEL_ALREADY_FINISHED", "run_dir": rel(RUN_DIR), "metrics": existing}, indent=2), flush=True)
        return
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise SystemExit(f"alternate clean run_dir exists and is non-empty but not complete: {RUN_DIR}")

    info = base.preflight("clean", RUN_DIR, False)
    info["parallel_reason"] = "The output path differs; the recipe and stream are unchanged."
    write_json(_public_path('experiments/archive/relation_learning/data/seed43222_clean_parallel/clean_parallel_preflight.json'), info)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_json(_public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel/clean_parallel_preflight.json'), info)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_relation_learning_step011_clean_parallel_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stream_path = pathlib.Path(info["stream"])
    if not stream_path.is_absolute():
        stream_path = ROOT / stream_path
    cmd = base.build_command("clean", RUN_DIR, stream_path)
    stdout_path = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel/train_stdout.log')
    stderr_path = _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel/train_stderr.log')
    start = now()
    start_rec = {"status": "CLEAN_PARALLEL_TRAIN_STARTING", "gpu": gpu, "run_dir": rel(RUN_DIR), "created_utc": start, "same_recipe_as_step007": True}
    print(json.dumps(start_rec, indent=2), flush=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({"event": "clean_parallel_train_start", "utc": start, "data_arm": "clean", "gpu": gpu, "recipe": base.RECIPE}, ensure_ascii=False) + "\n")
        out.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err, text=True)
    result = {
        "status": "CLEAN_PARALLEL_TRAIN_FINISHED" if proc.returncode == 0 else "CLEAN_PARALLEL_TRAIN_FAILED",
        "returncode": proc.returncode,
        "gpu": gpu,
        "started_utc": start,
        "finished_utc": now(),
        "run_dir": rel(RUN_DIR),
        "stdout_log": rel(stdout_path),
        "stderr_log": rel(stderr_path),
        "metrics": base.summarize_metrics(RUN_DIR),
        "same_recipe_as_step007": True,
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    write_json(_public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel/clean_parallel_train_result.json'), result)
    write_json(_public_path('experiments/archive/relation_learning/data/seed43222_clean_parallel/clean_parallel_train_result.json'), result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
