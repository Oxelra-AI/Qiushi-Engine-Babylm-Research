#!/usr/bin/env python3
"""research: launch a same-architecture second-seed full-DeBERTa compact/repeat pair.

Scientific purpose
------------------
research found that compact-minus-repeat item flips are weakly correlated across
architectures, but that fact is not interpretable without a same-architecture
same-data different-seed reference. This wrapper measures that reference in the
exact full-DeBERTa legal coordinate that produced the compact-view marginal.

This script deliberately changes only the model-initialization and training RNG
seeds relative to the validated research full-DeBERTa launcher. It preserves:
legal research tokenizer, exact compact/repeat 100M streams, full p2c+c2p DeBERTa,
WWM p=0.15, AdamW, LR schedule, batch256, seq256, 10M checkpoint cadence, and
100M counted-word exposure. It does not run any evaluation or upload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
BASE_SCRIPT = USER_ROOT / "experiments/archive/frontier_consolidation/scripts/train_deberta_positional_ablation.py"
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(BASE_SCRIPT)

spec = importlib.util.spec_from_file_location("train_deberta_positional_ablation", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load research launcher module from {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)  # type: ignore[union-attr]

VARIANT = "full_p2c_c2p_abs"
DEFAULT_SEED = 43
DEFAULT_EXTRA_INIT_SEED = 43122
DEFAULT_TRAIN_RNG_SEED = 43123


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_run_dir(data_arm: str, extra_init_seed: int) -> pathlib.Path:
    return base.RUNS_DIR / f"full_p2c_c2p_abs_{data_arm}_deberta100M_seed{extra_init_seed}"


def replace_arg(cmd: list[str], flag: str, value: Any) -> None:
    try:
        idx = cmd.index(flag)
    except ValueError as exc:
        raise RuntimeError(f"Base command missing expected flag {flag}") from exc
    if idx + 1 >= len(cmd):
        raise RuntimeError(f"Base command flag {flag} has no value slot")
    cmd[idx + 1] = str(value)


def build_seed_command(data_arm: str, run_dir: pathlib.Path, seed: int, extra_init_seed: int, train_rng_seed: int) -> list[str]:
    cmd = base.build_command(VARIANT, data_arm, run_dir)
    replace_arg(cmd, "--seed", seed)
    replace_arg(cmd, "--extra_init_seed", extra_init_seed)
    replace_arg(cmd, "--train_rng_seed", train_rng_seed)
    return cmd


def preflight(args: argparse.Namespace, run_dir: pathlib.Path) -> dict[str, Any]:
    status = base.preflight(VARIANT, args.data_arm, run_dir, count_words=args.count_words)
    recipe = dict(base.RECIPE)
    recipe.update({
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
    })
    status.update({
        "status": "FULL_DEBERTA_SEED_REPLICATE_PREFLIGHT_OK",
        "purpose": "Measure same-architecture same-data different-seed compact-minus-repeat reference for the research item/margin interpretation.",
        "decision_role": (
            "If this full-DeBERTa second seed preserves the family-level compact advantage well above same-architecture seed spread, "
            "compact semantic second views retain a systematic family-level component; if not, prior single-seed architecture/transfer closures "
            "must be interpreted at the resolution of basin-specific competence reallocation."
        ),
        "variant": VARIANT,
        "data_arm": args.data_arm,
        "run_dir": str(run_dir),
        "seed_triplet": {
            "seed": args.seed,
            "extra_init_seed": args.extra_init_seed,
            "train_rng_seed": args.train_rng_seed,
        },
        "only_intended_change_from_step242_seed43022_pair": "extra_init_seed/train_rng_seed moved from 43022/43023 to the requested replicate values; base seed remains 43 by default to avoid changing any non-overridden coordinate.",
        "adjusted_recipe": recipe,
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
    })
    cmd = build_seed_command(args.data_arm, run_dir, args.seed, args.extra_init_seed, args.train_rng_seed)
    status["command"] = cmd
    status["cuda_visible_devices"] = str(args.gpu)
    return status


def summarize_metrics(metrics_path: pathlib.Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"metrics_exists": False}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved = metrics.get("saved_checkpoints", [])
    return {
        "metrics_exists": True,
        "word_exposure": metrics.get("word_exposure"),
        "actual_training_steps": metrics.get("actual_training_steps"),
        "loss_first": metrics.get("loss_first"),
        "loss_last": metrics.get("loss_last"),
        "parameter_count": metrics.get("parameter_count"),
        "vocab_size": metrics.get("vocab_size"),
        "tokenizer_label": metrics.get("tokenizer_label"),
        "mask_prob_start": metrics.get("mask_prob_start"),
        "mask_prob_end": metrics.get("mask_prob_end"),
        "seq_length": metrics.get("seq_length"),
        "max_seq_length": metrics.get("max_seq_length"),
        "saved_checkpoints": len(saved),
        "first_checkpoint": saved[0].get("name") if saved else None,
        "last_checkpoint": saved[-1].get("name") if saved else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-arm", required=True, choices=sorted(base.STREAMS))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--extra-init-seed", type=int, default=DEFAULT_EXTRA_INIT_SEED)
    ap.add_argument("--train-rng-seed", type=int, default=DEFAULT_TRAIN_RNG_SEED)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-words", action="store_true")
    ap.add_argument("--allow-nonempty", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else default_run_dir(args.data_arm, args.extra_init_seed)
    status = preflight(args, run_dir)

    if args.dry_run:
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
        return

    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_nonempty:
        raise RuntimeError(f"run_dir exists and is non-empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_command.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["TMPDIR"] = f"/tmp/q_frontier_consolidation_step252_full_deberta_{args.data_arm}_{args.extra_init_seed}_{int(time.time())}"
    pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    started = now()
    cmd = status["command"]
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write(json.dumps({
            "event": "launcher_start",
            "started_utc": started,
            "variant": VARIANT,
            "data_arm": args.data_arm,
            "gpu": args.gpu,
            "seed": args.seed,
            "extra_init_seed": args.extra_init_seed,
            "train_rng_seed": args.train_rng_seed,
        }) + "\n")
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env, cwd=str(USER_ROOT))
    finished = now()

    metrics_path = run_dir / "scientific_metrics.json"
    result: dict[str, Any] = {
        "status": "FULL_DEBERTA_SEED_REPLICATE_TRAIN_FINISHED" if proc.returncode == 0 else "FULL_DEBERTA_SEED_REPLICATE_TRAIN_FAILED",
        "variant": VARIANT,
        "data_arm": args.data_arm,
        "gpu": args.gpu,
        "returncode": proc.returncode,
        "started_utc": started,
        "finished_utc": finished,
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "metrics_path": str(metrics_path),
        "seed": args.seed,
        "extra_init_seed": args.extra_init_seed,
        "train_rng_seed": args.train_rng_seed,
        "metrics": summarize_metrics(metrics_path),
        "no_evaluation_superglue_aoa_upload_or_leaderboard": True,
    }
    if proc.returncode != 0 and stderr_path.exists():
        result["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    (run_dir / "launcher_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
