#!/usr/bin/env python3
"""research AoA localization for already-trained BabyLM density ladders.

This script runs the official-compatible local-checkpoint AoA helper on existing
models only.  It does not train, sample data, alter a corpus, or use AoA words as
a pretraining signal.  The purpose is to localize whether the negative fitted
acquisition-timing relation appears in FineWeb replacement, generated views,
compression, or added compact source exposure.
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
RUN_BASE = _public_path('experiments/archive/frontier_consolidation/training/runs')
OUT_ROOT_DEFAULT = _public_path('experiments/archive/frontier_consolidation/data/aoa_localization')
HELPER = _public_path('experiments/archive/compact_experience/scripts/aoa_local_ckpts_for_model.py')

TARGETS: Dict[str, Dict[str, pathlib.Path | str]] = {
    "density_near_repeat": {
        "model_root": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_repeat_near_core_16k_seed43022/hf_model'),
        "role": "FineWeb near-source repetition on the clean-Qwen row-holdout base",
    },
    "density_near_view": {
        "model_root": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_near_view_core_16k_seed43022/hf_model'),
        "role": "near-length generated views against the same FineWeb near-source core",
    },
    "density_compact_repeat_core": {
        "model_root": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_repeat_compact_core_neutral_16k_seed43022/hf_model'),
        "role": "compact-core source repetition on the clean-Qwen row-holdout base",
    },
    "density_compact_view_reinvest": {
        "model_root": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model'),
        "role": "compact views with saved words reinvested into additional source-view packets",
    },
}
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
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
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
                "error": "no_gpu_available_for_aoa_localization",
                "min_free_mib": args.min_free_mib,
                "max_util": args.max_util,
                "last_gpu_state": gpus,
            })
        time.sleep(args.poll_sec)


def run_target(target: str, out_root: pathlib.Path, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = TARGETS[target]
    model_root = pathlib.Path(cfg["model_root"]).resolve()
    ok, missing = have_complete_ladder(model_root)
    target_dir = out_root / "aoa_outputs" / target
    out_json = target_dir / "aoa_local_ckpts.json"
    out_note = target_dir / "aoa_local_ckpts.md"
    log = out_root / "logs" / f"{target}_aoa.log"
    rec: Dict[str, Any] = {
        "target": target,
        "role": cfg.get("role"),
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
    p = subprocess.run(cmd, cwd=str(_public_path('.')), capture_output=True, text=True, timeout=args.per_target_timeout_sec)
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
            rec["row_count_values"] = d.get("row_count_values")
            rec["surprisal_path"] = d.get("surprisal_path")
        except Exception as exc:
            rec["read_error"] = repr(exc)
    if p.returncode != 0 and not args.keep_going:
        raise RuntimeError(rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["density_near_repeat", "density_near_view", "density_compact_repeat_core"], choices=sorted(TARGETS))
    ap.add_argument("--out_root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--gpu", default="auto", help="physical GPU id or auto")
    ap.add_argument("--min_free_mib", type=int, default=55000)
    ap.add_argument("--max_util", type=int, default=25)
    ap.add_argument("--poll_sec", type=int, default=45)
    ap.add_argument("--wait_timeout_sec", type=int, default=21600)
    ap.add_argument("--per_target_timeout_sec", type=int, default=2400)
    ap.add_argument("--allow_timeout_best_gpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep_going", action="store_true")
    ap.add_argument("--preflight_only", action="store_true", help="Check helper/targets/ladders and write a preflight summary without choosing GPU or running AoA.")
    args = ap.parse_args()

    if not HELPER.exists():
        raise FileNotFoundError(HELPER)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        preflight_records: List[Dict[str, Any]] = []
        for target in args.targets:
            cfg = TARGETS[target]
            model_root = pathlib.Path(cfg["model_root"]).resolve()
            ok, missing = have_complete_ladder(model_root)
            preflight_records.append({
                "target": target,
                "role": cfg.get("role"),
                "model_root": rel(model_root),
                "complete_ladder": ok,
                "missing_steps": missing,
                "existing_out_json": rel(out_root / "aoa_outputs" / target / "aoa_local_ckpts.json"),
                "existing_out_json_exists": (out_root / "aoa_outputs" / target / "aoa_local_ckpts.json").exists(),
            })
        payload = {
            "status": "AOA_LOCALIZATION_PREFLIGHT",
            "helper": rel(HELPER),
            "targets_requested": args.targets,
            "records": preflight_records,
            "gpu_state_now": query_gpus(),
        }
        (out_root / "aoa_localization_preflight.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return

    records: List[Dict[str, Any]] = []
    started = time.time()
    for target in args.targets:
        rec = run_target(target, out_root, args)
        records.append(rec)
        summary = {
            "status": "AOA_LOCALIZATION_RUNNING_OR_DONE",
            "targets_requested": args.targets,
            "records": records,
            "elapsed_sec": round(time.time() - started, 3),
        }
        (out_root / "aoa_localization_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    final = {
        "status": "AOA_LOCALIZATION_DONE",
        "purpose": "AoA on already-trained ladders to localize the compact-density timing inversion before any new 100M training.",
        "targets_requested": args.targets,
        "records": records,
        "elapsed_sec": round(time.time() - started, 3),
    }
    (out_root / "aoa_localization_run_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
