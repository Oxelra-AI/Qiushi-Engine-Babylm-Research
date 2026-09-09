#!/usr/bin/env python3
"""research runner: official-row-count AoA for compact_view_reinvest seeds.

Runs the repaired explicit-min_context local AoA helper on existing checkpoint
ladders only. This is the source-level repair after finding that current
`collate_preds.py` expects AOA_SIZE=8005 per checkpoint, which corresponds to
`min_context=0`, not the historical helper's hardcoded min_context=20.

Expensive-work purpose:
- seed43022: recompute the frozen 42.0868 endpoint's AoA under current row-count
  compatibility before treating the score as submission-facing.
- seed43122: decide whether the same AoA behavior reproduces on the independent
  seed before spending the much larger cost of a full second-seed official surface.
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
HELPER = _public_path('experiments/archive/frontier_consolidation/scripts/aoa_local_ckpts_minctx.py')
OUT_ROOT_DEFAULT = _public_path('experiments/archive/frontier_consolidation/data/official_rowcount_aoa_reinvest_seeds')
AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i * 10}M" for i in range(1, 11)]
TARGETS: Dict[str, Dict[str, str]] = {
    "reinvest_seed43022": {
        "model_root": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model",
        "role": "frozen 42.0868 compact_view_reinvest endpoint; recompute AoA with 8005 rows/checkpoint",
    },
    "reinvest_seed43122": {
        "model_root": "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model",
        "role": "independent seed43122 compact_view_reinvest ladder; AoA-first stability check",
    },
}


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
                "error": "no_gpu_available_for_official_rowcount_aoa",
                "min_free_mib": args.min_free_mib,
                "max_util": args.max_util,
                "last_gpu_state": gpus,
            })
        time.sleep(args.poll_sec)


def preflight(targets: List[str], out_root: pathlib.Path) -> Dict[str, Any]:
    records = []
    for target in targets:
        cfg = TARGETS[target]
        model_root = (USER_ROOT / cfg["model_root"]).resolve()
        ok, missing = have_complete_ladder(model_root)
        records.append({
            "target": target,
            "role": cfg["role"],
            "model_root": rel(model_root),
            "model_root_exists": model_root.exists(),
            "complete_ladder": ok,
            "missing_steps": missing,
            "expected_out_json": rel(out_root / target / "aoa_local_ckpts_minctx0.json"),
        })
    return {
        "status": "OFFICIAL_ROWCOUNT_AOA_PREFLIGHT",
        "helper": rel(HELPER),
        "helper_exists": HELPER.exists(),
        "min_context": 0,
        "expected_rows_per_step": 8005,
        "targets_requested": targets,
        "records": records,
        "gpu_state_now": query_gpus(),
    }


def run_target(target: str, out_root: pathlib.Path, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = TARGETS[target]
    model_root = (USER_ROOT / cfg["model_root"]).resolve()
    ok, missing = have_complete_ladder(model_root)
    target_dir = out_root / target
    out_json = target_dir / "aoa_local_ckpts_minctx0.json"
    out_note = target_dir / "aoa_local_ckpts_minctx0.md"
    log = out_root / "logs" / f"{target}.log"
    rec: Dict[str, Any] = {
        "target": target,
        "role": cfg["role"],
        "model_root": rel(model_root),
        "complete_ladder": ok,
        "missing_steps": missing,
        "out_json": rel(out_json),
        "out_note": rel(out_note),
        "log": rel(log),
    }
    if not ok:
        rec["status"] = "missing_checkpoint_ladder"
        return rec
    if out_json.exists() and not args.force:
        try:
            d = json.loads(out_json.read_text(encoding="utf-8"))
            rec.update({
                "status": "already_done",
                "aoa": d.get("aoa"),
                "curve_fitness_record": d.get("curve_fitness_record"),
                "num_rows": d.get("num_rows"),
                "row_count_values": d.get("row_count_values"),
                "elapsed_sec": d.get("elapsed_sec"),
            })
            return rec
        except Exception:
            pass
    gpu = choose_gpu(args)
    rec["gpu"] = gpu
    target_dir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_public_path('experiments/archive/frontier_consolidation/scripts/aoa_local_ckpts_minctx.py')),
        "--model_root", str(model_root),
        "--out_dir", str(target_dir.resolve()),
        "--out_json", str(out_json.resolve()),
        "--out_note", str(out_note.resolve()),
        "--log", str(log.resolve()),
        "--gpu", str(gpu),
        "--min_context", "0",
        "--expected_rows_per_step", "8005",
    ]
    started = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write("\n[research official-rowcount AoA] $ " + " ".join(cmd) + "\n")
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
            rec.update({
                "aoa": d.get("aoa"),
                "curve_fitness_record": d.get("curve_fitness_record"),
                "num_rows": d.get("num_rows"),
                "row_count_values": d.get("row_count_values"),
                "surprisal_path": d.get("surprisal_path"),
                "score_path": d.get("score_path"),
            })
        except Exception as exc:
            rec["read_error"] = repr(exc)
    if p.returncode != 0 and not args.keep_going:
        raise RuntimeError(rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["reinvest_seed43022", "reinvest_seed43122"], choices=sorted(TARGETS))
    ap.add_argument("--out_root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--min_free_mib", type=int, default=55000)
    ap.add_argument("--max_util", type=int, default=25)
    ap.add_argument("--poll_sec", type=int, default=45)
    ap.add_argument("--wait_timeout_sec", type=int, default=21600)
    ap.add_argument("--per_target_timeout_sec", type=int, default=4200)
    ap.add_argument("--allow_timeout_best_gpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep_going", action="store_true")
    ap.add_argument("--preflight_only", action="store_true")
    args = ap.parse_args()

    if not HELPER.exists():
        raise FileNotFoundError(HELPER)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        payload = preflight(args.targets, out_root)
        (out_root / "official_rowcount_aoa_preflight.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return

    records: List[Dict[str, Any]] = []
    started = time.time()
    for target in args.targets:
        rec = run_target(target, out_root, args)
        records.append(rec)
        partial = {
            "status": "OFFICIAL_ROWCOUNT_AOA_RUNNING_OR_DONE",
            "targets_requested": args.targets,
            "records": records,
            "elapsed_sec": round(time.time() - started, 3),
        }
        (out_root / "official_rowcount_aoa_run_summary.json").write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    final = {
        "status": "OFFICIAL_ROWCOUNT_AOA_DONE",
        "purpose": "Official-row-count AoA on reinvest seed43022 and seed43122 ladders before second-seed full-surface decisions.",
        "min_context": 0,
        "expected_rows_per_step": 8005,
        "targets_requested": args.targets,
        "records": records,
        "elapsed_sec": round(time.time() - started, 3),
    }
    (out_root / "official_rowcount_aoa_run_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
