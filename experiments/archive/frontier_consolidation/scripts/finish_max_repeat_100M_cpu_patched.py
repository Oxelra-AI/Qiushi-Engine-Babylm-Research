#!/usr/bin/env python3
"""research: true CPU-only finisher for MAX-repeat 100M stable rows.

The research evaluator's normal CLI resets CUDA_VISIBLE_DEVICES from `--gpu`, so
wrapping the CLI is not enough to keep H100s free.  This script imports research,
patches its `setup_env` function so every child subprocess sees
`CUDA_VISIBLE_DEVICES=-1`, then calls research's `run_one` for only the missing
stable columns of `dose_max_repeat_chck_100M`.

Scientific role: complete the late 100M MAX-repeat counterfactual so the
fixed-budget decomposition and full-exposure turnover are measured, not inferred.

No GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
from types import ModuleType
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PATH = WS / "scripts/evaluate_compliant_endpoint.py"
OUT_ROOT = WS / "data/dose_ladder_stable_eval/eval"
COLLATE_ROOT = WS / "data/dose_ladder_stable_eval/collate"
RUN_DIR = WS / "training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022"
TARGET = "dose_max_repeat_chck_100M"
PER_TARGET = OUT_ROOT / "per_target" / f"{TARGET}.json"
STABLE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
DEFAULT_COLUMNS = ["EWoK", "Entity", "COMPS", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def score_value(rec: dict[str, Any], col: str) -> Any:
    if col == "Reading":
        s = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
        return s.get("Reading") if s else rec.get("score")
    return rec.get("score")


def ready_cols(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for col in STABLE:
        rec = (payload.get("tasks") or {}).get(col)
        if isinstance(rec, dict) and rec.get("returncode") == 0 and finite(score_value(rec, col)):
            out.append(col)
    return out


def load_step037() -> ModuleType:
    if not PATH.exists():
        raise FileNotFoundError(PATH)
    spec = importlib.util.spec_from_file_location("cpu_patched", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def patch_cpu_env(mod: ModuleType) -> None:
    original_setup_env = mod.setup_env

    def setup_env_cpu(target: str, gpu: int) -> dict[str, str]:
        env = original_setup_env(target, gpu)
        # Hide all GPUs from the actual evaluation subprocesses.  `--gpu` remains
        # a harmless label in payloads, but torch.cuda.is_available() is false.
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Separate cache path from interrupted GPU evaluator outputs.
        cache = OUT_ROOT / "hf_cache" / f"{TARGET}_true_cpu"
        tmp = OUT_ROOT / "tmp" / f"{TARGET}_true_cpu"
        env["HF_HOME"] = str(cache.resolve())
        env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
        env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
        env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
        env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
        env["TMPDIR"] = str(tmp.resolve())
        for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
            pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
        return env

    mod.setup_env = setup_env_cpu


def build_namespace(columns: list[str], force: bool) -> argparse.Namespace:
    return argparse.Namespace(
        arm="reinvest",
        target=TARGET,
        run_dir=str(RUN_DIR),
        endpoint="chck_100M",
        out_root=str(OUT_ROOT),
        collate_root=str(COLLATE_ROOT),
        gpu=0,
        columns=columns,
        force=force,
        preflight_only=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    if not (RUN_DIR / "hf_model/chck_100M").exists():
        raise FileNotFoundError(RUN_DIR / "hf_model/chck_100M")
    if not (RUN_DIR / "scientific_metrics.json").exists():
        raise FileNotFoundError(RUN_DIR / "scientific_metrics.json")
    before = read_json(PER_TARGET) if PER_TARGET.exists() else {}
    before_ready = ready_cols(before)
    to_run = [c for c in args.columns if args.force or c not in before_ready]
    plan = {
        "status": "TRUE_CPU_MAX_REPEAT_100M_FINISHER_PLAN",
        "created_utc": now(),
        "target": TARGET,
        "run_dir": rel(RUN_DIR),
        "model_path": rel(RUN_DIR / "hf_model/chck_100M"),
        "per_target": rel(PER_TARGET),
        "before_ready_cols": before_ready,
        "requested_columns": args.columns,
        "to_run": to_run,
        "cuda_visible_devices_for_children": "-1",
        "scientific_role": "Complete the late MAX-repeat 100M stable columns so the fixed-budget identity is measured at the exposure boundary where MAX view growth has turned negative.",
        "lowest_cost_method": "Run only missing stable columns on CPU; do not run GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard; do not occupy H100 training slots.",
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    if args.plan_only or not to_run:
        print(json.dumps({**plan, "plan_only": args.plan_only, "no_new_eval_needed": not to_run}, indent=2, ensure_ascii=False), flush=True)
        return

    mod = load_step037()
    patch_cpu_env(mod)
    t0 = time.time()
    # Confirm the child environment is genuinely CPU-only before the costly run.
    env_probe = mod.setup_env(TARGET, 0)
    if env_probe.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError(f"CPU patch failed: {env_probe.get('CUDA_VISIBLE_DEVICES')}")
    print(json.dumps({"event": "true_cpu_finish_start", **plan}, indent=2, ensure_ascii=False), flush=True)
    mod.run_one(build_namespace(to_run, args.force))
    after = read_json(PER_TARGET) if PER_TARGET.exists() else {}
    after_ready = ready_cols(after)
    missing = [c for c in STABLE if c not in after_ready]
    result = {
        "status": "TRUE_CPU_MAX_REPEAT_100M_FINISHER_DONE" if not missing else "TRUE_CPU_MAX_REPEAT_100M_FINISHER_INCOMPLETE",
        "created_utc": now(),
        "elapsed_sec": round(time.time() - t0, 2),
        "target": TARGET,
        "per_target": rel(PER_TARGET),
        "before_ready_cols": before_ready,
        "after_ready_cols": after_ready,
        "missing_after": missing,
        "per_target_size": PER_TARGET.stat().st_size if PER_TARGET.exists() else 0,
        "cuda_visible_devices_for_children": "-1",
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if missing:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
