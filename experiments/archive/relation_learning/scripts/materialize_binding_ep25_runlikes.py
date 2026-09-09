#!/usr/bin/env python3
"""research: package the exposure-matched binding private branch as run-like endpoints.

The source checkpoint is the research ep25 chck_82M frozen-slow/fresh-private
binding model.  This script materializes reversible private-scale variants and
wraps each as a run directory consumable by the research official-compatible
cheap7 evaluator.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = pathlib.Path(".").resolve()
STUDY = ROOT / "experiments/archive" / 'relation_learning'
SOURCE_CKPT = STUDY / "data/chck82_binding_candidate_ep25/scale_0.75/checkpoint"
SOURCE_SUMMARY = STUDY / "data/chck82_binding_candidate_ep25/scale_0.75/summary.json"
SCALE_SCRIPT = ROOT / "experiments/archive/frontier_consolidation/scripts/materialize_private_scale.py"
OUT_DATA = STUDY / "data/binding_ep25_runlikes"
CHCK82_WORDS = 82_012_495
BINDING_WORDS = 3_915_850
TOTAL_WORDS = CHCK82_WORDS + BINDING_WORDS
PARAM_TOTAL = 36_458_592
PARAM_FROZEN = 35_463_008
PARAM_PRIVATE = 995_584

SCALES = [(0.50, "0p50"), (0.75, "0p75"), (1.00, "1p00")]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}\nCMD={cmd}\nSTDOUT={proc.stdout[-2000:]}\nSTDERR={proc.stderr[-4000:]}")


def scientific_metrics(alpha: float, endpoint_dir: pathlib.Path, src_summary: dict[str, Any]) -> dict[str, Any]:
    identity = src_summary.get("identity", {})
    traj = src_summary.get("trajectory", [])
    final_eval = traj[-1] if traj else {}
    return {
        "status": "BINDING_EP25_RUNLIKE",
        "created_utc": now(),
        "mode": "chck82_frozen_slow_fresh_private_binding_answer_credit_ep25",
        "endpoint": "final",
        "source_checkpoint": rel(SOURCE_CKPT),
        "source_summary": rel(SOURCE_SUMMARY),
        "private_adapter_scale": float(alpha),
        "private_adapter_enabled": True,
        "initial_consumed_words": CHCK82_WORDS,
        "binding_epoch_words": 156_634,
        "binding_epochs": 25,
        "tail_main_word_exposure": 0,
        "tail_aux_word_exposure": BINDING_WORDS,
        "tail_charged_words": BINDING_WORDS,
        "total_consumed_words": TOTAL_WORDS,
        "max_tail_charged_words_reference_coherent86": 3_992_800,
        "full_strict_small_cap_words": 100_000_000,
        "stopped_before_cap": True,
        "trainable": PARAM_PRIVATE,
        "total_params": PARAM_TOTAL,
        "private_params": PARAM_PRIVATE,
        "frozen_slow_params": PARAM_FROZEN,
        "model_identity_from_training": identity,
        "training_updates_seen": src_summary.get("updates_seen"),
        "training_lr": src_summary.get("lr"),
        "training_weight_decay": src_summary.get("weight_decay"),
        "training_train_pairs": src_summary.get("train_pairs"),
        "training_held_pairs": src_summary.get("held_pairs"),
        "in_format_final_saved_scale_eval": final_eval,
        "hf_model_final": rel(endpoint_dir),
        "notes": [
            "Run-like wrapper for official-compatible evaluation only; no additional training in this packaging step.",
            "The underlying ep25 branch was trained with answer-only balanced binding credit and no ordinary suffix replay or private-off preservation pressure.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not SOURCE_CKPT.exists():
        raise FileNotFoundError(SOURCE_CKPT)
    if not SOURCE_SUMMARY.exists():
        raise FileNotFoundError(SOURCE_SUMMARY)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    src_summary = read_json(SOURCE_SUMMARY)
    manifest = {
        "status": "BINDING_EP25_RUNLIKES_MATERIALIZED",
        "created_utc": now(),
        "source_checkpoint": rel(SOURCE_CKPT),
        "source_summary": rel(SOURCE_SUMMARY),
        "initial_consumed_words": CHCK82_WORDS,
        "binding_words": BINDING_WORDS,
        "total_consumed_words": TOTAL_WORDS,
        "runs": [],
    }
    for alpha, tag in SCALES:
        run_dir = STUDY / f"training/runs/binding_ep25_alpha{tag}"
        endpoint_dir = run_dir / "hf_model" / "final"
        if run_dir.exists() and args.force:
            for child in list(run_dir.iterdir()):
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        run_dir.mkdir(parents=True, exist_ok=True)
        endpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "-B", str(SCALE_SCRIPT),
            "--source", str(SOURCE_CKPT),
            "--output", str(endpoint_dir),
            "--scale", str(alpha),
            "--force",
        ])
        metrics = scientific_metrics(alpha, endpoint_dir, src_summary)
        (run_dir / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["runs"].append({
            "alpha": alpha,
            "tag": tag,
            "run_dir": rel(run_dir),
            "endpoint": "final",
            "metrics": rel(run_dir / "scientific_metrics.json"),
            "scale_manifest": rel(endpoint_dir / "private_scale_manifest.json"),
        })
    out = OUT_DATA / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": rel(out), "runs": manifest["runs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
