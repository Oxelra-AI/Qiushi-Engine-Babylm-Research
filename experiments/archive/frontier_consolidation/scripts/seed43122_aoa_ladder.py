#!/usr/bin/env python3
"""research official AoA on the seed43122 compact_view_reinvest checkpoint ladder.

Purpose: the +0.2868 lead of the seed43022
compact_view_reinvest endpoint over the visible 41.8 leader rests entirely on AoA
staying non-significant (mapped to 0.0). The seed43122 fast screen runs only a fast
no-AoA screen, so it cannot measure the decisive column. This script runs the
official-compatible AoA helper on the seed43122 checkpoint ladder (already saved by
the identical training recipe) to resolve whether the AoA gate reproduces across
the seed BEFORE any expensive full second-seed surface is completed.

It trains nothing, samples no data, and never uses AoA words as a pretraining
signal. It only reads an existing checkpoint ladder and runs the official AoA
extractor + AoAEvaluator, identical to research localization.

Run order:
  1. --preflight_only  (CPU; verify the 19-step ladder is complete and readable,
     confirm no write lock, report GPU state)  -- safe to run any time.
  2. (after seed43122 training finishes and a GPU is available)
     run without --preflight_only to execute the AoA pass.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
STUDY = _public_path('experiments/archive/frontier_consolidation')
USER_ROOT = _public_path('.')
OUT_ROOT_DEFAULT = _public_path('experiments/archive/frontier_consolidation/data/seed43122_aoa')
HELPER = _public_path('experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py')

# The seed43122 reinvest ladder uses train_density_arm_seed.py (identical recipe, checkpoint_words=1e6).
SEED43122_MODEL_ROOT = (
    _public_path('experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model')
)

TARGET = "density_compact_view_reinvest_seed43122"
ROLE = "second independent seed of the compact_view_reinvest SOTA-facing endpoint; official AoA on its checkpoint ladder"
AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(_public_path('.')))
    except Exception:
        return str(p)


def have_complete_ladder(model_root: pathlib.Path) -> tuple[bool, List[str]]:
    missing = [s for s in AOA_STEPS if not (model_root / s / "model.safetensors").exists()]
    return (len(missing) == 0), missing


def query_gpus() -> List[Dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    if p.returncode != 0:
        return []
    out: List[Dict[str, int]] = []
    for line in p.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4:
            try:
                out.append({"index": int(parts[0]), "used_mib": int(parts[1]), "free_mib": int(parts[2]), "util": int(parts[3])})
            except ValueError:
                pass
    return out


def choose_gpu(args: argparse.Namespace) -> int:
    if args.gpu != "auto":
        return int(args.gpu)
    deadline = time.time() + args.wait_timeout_sec
    best_seen: Optional[Dict[str, int]] = None
    while True:
        gpus = query_gpus()
        if gpus:
            best_seen = max(gpus, key=lambda g: (g["free_mib"], -g["util"]))
            candidates = [g for g in gpus if g["free_mib"] >= args.min_free_mib and g["util"] <= args.max_util]
            if candidates:
                chosen = max(candidates, key=lambda g: (g["free_mib"], -g["util"]))
                return int(chosen["index"])
        if time.time() >= deadline:
            if best_seen is not None and args.allow_timeout_best_gpu:
                return int(best_seen["index"])
            raise TimeoutError({
                "error": "no_gpu_available_for_seed43122_aoa",
                "min_free_mib": args.min_free_mib,
                "max_util": args.max_util,
                "last_gpu_state": gpus,
            })
        time.sleep(args.poll_sec)


def preflight(out_root: pathlib.Path) -> Dict[str, Any]:
    model_root = _public_path('experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model')
    ladder_ok, missing = have_complete_ladder(model_root)
    return {
        "status": "SEED43122_AOA_PREFLIGHT",
        "target": TARGET,
        "role": ROLE,
        "helper": rel(HELPER),
        "helper_exists": HELPER.exists(),
        "model_root": rel(model_root),
        "model_root_exists": model_root.exists(),
        "complete_ladder": ladder_ok,
        "missing_steps": missing,
        "expected_out_json": rel(out_root / "aoa_outputs" / TARGET / "aoa_local_ckpts.json"),
        "gpu_state_now": query_gpus(),
        "note": "Run the AoA pass only after A01 seed43122 training finished, lock released, ladder complete, and an isolated GPU is free.",
    }


def run_aoa(out_root: pathlib.Path, args: argparse.Namespace) -> Dict[str, Any]:
    model_root = _public_path('experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model')
    ok, missing = have_complete_ladder(model_root)
    target_dir = out_root / "aoa_outputs" / TARGET
    out_json = target_dir / "aoa_local_ckpts.json"
    out_note = target_dir / "aoa_local_ckpts.md"
    log = out_root / "logs" / f"{TARGET}_aoa.log"
    rec: Dict[str, Any] = {
        "target": TARGET,
        "role": ROLE,
        "model_root": rel(model_root),
        "out_json": rel(out_json),
        "out_note": rel(out_note),
        "log": rel(log),
        "complete_ladder": ok,
        "missing_steps": missing,
    }
    if not ok:
        rec["status"] = "missing_checkpoint_ladder"
        return rec
    if out_json.exists() and not args.force:
        try:
            d = json.loads(out_json.read_text(encoding="utf-8"))
            rec.update({"status": "already_done", "aoa": d.get("aoa"), "elapsed_sec": d.get("elapsed_sec")})
            return rec
        except Exception:
            pass
    gpu = choose_gpu(args)
    rec["gpu"] = gpu
    target_dir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_public_path('experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py')),
        "--model_root", str(model_root),
        "--out_dir", str(target_dir.resolve()),
        "--out_json", str(out_json.resolve()),
        "--out_note", str(out_note.resolve()),
        "--log", str(log.resolve()),
        "--gpu", str(gpu),
    ]
    started = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write("\n[research] $ " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, cwd=str(_public_path('.')), capture_output=True, text=True, timeout=args.timeout_sec)
    with log.open("a", encoding="utf-8") as f:
        f.write(p.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(p.stderr)
        f.write(f"\n[returncode={p.returncode} elapsed_sec={time.time() - started:.2f}]\n")
    rec.update({"status": "done" if p.returncode == 0 else "failed", "returncode": p.returncode, "elapsed_sec": round(time.time() - started, 3)})
    if out_json.exists():
        try:
            d = json.loads(out_json.read_text(encoding="utf-8"))
            rec["aoa"] = d.get("aoa")
            rec["num_rows"] = d.get("num_rows")
            rec["surprisal_path"] = d.get("surprisal_path")
        except Exception as exc:
            rec["read_error"] = repr(exc)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--gpu", default="auto", help="physical GPU id or auto")
    ap.add_argument("--min_free_mib", type=int, default=55000)
    ap.add_argument("--max_util", type=int, default=25)
    ap.add_argument("--poll_sec", type=int, default=45)
    ap.add_argument("--wait_timeout_sec", type=int, default=21600)
    ap.add_argument("--timeout_sec", type=int, default=3000)
    ap.add_argument("--allow_timeout_best_gpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--preflight_only", action="store_true")
    args = ap.parse_args()

    if not HELPER.exists():
        raise FileNotFoundError(HELPER)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.preflight_only:
        payload = preflight(out_root)
        (out_root / "seed43122_aoa_preflight.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return

    rec = run_aoa(out_root, args)
    payload = {
        "status": "SEED43122_AOA_DONE" if rec.get("status") == "done" else "SEED43122_AOA_INCOMPLETE",
        "record": rec,
    }
    (out_root / "seed43122_aoa_run_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
