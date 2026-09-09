#!/usr/bin/env python3
"""research: true CPU-only finisher for RoBERTa MAX clean stable rows.

The research RoBERTa evaluator was cancelled after producing view 10/10 and only
partial clean rows.  This script imports research's evaluator, patches its
`launch_eval` so child research processes run with `CUDA_VISIBLE_DEVICES=-1`, and
runs only missing clean checkpoints/columns.  It then writes research summary
CSVs using the existing research functions.

Scientific role: complete RoBERTa view-clean total fixed-budget transfer without
occupying H100s needed for DeBERTa second-basin and breadth training.
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
import subprocess
import sys
import time
from types import ModuleType
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PATH = WS / "scripts/wait_and_eval_roberta_viewclean_total.py"
OUT_DIR = WS / "data/roberta_viewclean_total_stable_eval"
STABLE = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading"]
CKS = [f"chck_{i}M" for i in range(10, 101, 10)]


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


def per_target(arm: str, ck: str) -> pathlib.Path:
    return OUT_DIR / "eval/per_target" / f"roberta_viewclean_total_{arm}_{ck}.json"


def ready_cols(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = read_json(path)
    except Exception:
        return []
    out = []
    for col in STABLE:
        rec = (payload.get("tasks") or {}).get(col)
        if isinstance(rec, dict) and rec.get("returncode") == 0 and finite(score_value(rec, col)):
            out.append(col)
    return out


def complete(path: pathlib.Path) -> bool:
    return set(ready_cols(path)) == set(STABLE)


def load_step260() -> ModuleType:
    if not PATH.exists():
        raise FileNotFoundError(PATH)
    spec = importlib.util.spec_from_file_location("cpu_patched", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def patch_launch_eval_cpu(mod: ModuleType) -> None:
    def launch_eval_cpu(out_dir: pathlib.Path, arm: str, ck: str, gpu: int, force: bool):
        base = mod.task_out_base(out_dir, arm, ck)
        base.mkdir(parents=True, exist_ok=True)
        log = (base / "stable_eval_stdout_cpu_step265.log").open("w", encoding="utf-8")
        cmd = mod.eval_command(out_dir, arm, ck, gpu, force)
        # Append only stable columns already encoded by research eval_command.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cache = out_dir / "eval/hf_cache" / f"cpu_roberta_{arm}_{ck}"
        tmp = out_dir / "eval/tmp" / f"cpu_roberta_{arm}_{ck}"
        env["HF_HOME"] = str(cache.resolve())
        env["HF_HUB_CACHE"] = str((cache / "hub").resolve())
        env["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
        env["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
        env["HF_DATASETS_CACHE"] = str((cache / "datasets").resolve())
        env["TMPDIR"] = str(tmp.resolve())
        for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
            pathlib.Path(env[key]).mkdir(parents=True, exist_ok=True)
        log.write(json.dumps({"event": "launch_eval_cpu", "utc": now(), "arm": arm, "checkpoint": ck, "cmd": cmd, "CUDA_VISIBLE_DEVICES": "-1"}) + "\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return p, log

    mod.launch_eval = launch_eval_cpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["clean", "view"], default="clean")
    ap.add_argument("--checkpoints", nargs="*", default=CKS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    before = {ck: ready_cols(per_target(args.arm, ck)) for ck in args.checkpoints}
    pending = [ck for ck in args.checkpoints if args.force or not complete(per_target(args.arm, ck))]
    plan = {
        "status": "ROBERTA_CLEAN_TRUE_CPU_FINISHER_PLAN",
        "created_utc": now(),
        "arm": args.arm,
        "out_dir": rel(OUT_DIR),
        "before_ready_cols": before,
        "pending_checkpoints": pending,
        "workers": args.workers,
        "cuda_visible_devices_for_children": "-1",
        "scientific_role": "Complete RoBERTa MAX view-clean stable-family contrast as architecture-transfer evidence without occupying H100 training slots.",
        "lowest_cost_method": "Only missing stable checkpoint evaluations are run; child processes see CUDA hidden; no GlobalPIQA/SuperGLUE/AoA is run.",
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    if args.plan_only or not pending:
        print(json.dumps({**plan, "plan_only": args.plan_only, "no_new_eval_needed": not pending}, indent=2, ensure_ascii=False), flush=True)
        return

    mod = load_step260()
    patch_launch_eval_cpu(mod)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    active: list[tuple[str, Any, Any]] = []
    done: list[dict[str, Any]] = []
    problems: list[str] = []
    queue = list(pending)
    print(json.dumps({"event": "roberta_cpu_finish_start", **plan}, indent=2, ensure_ascii=False), flush=True)
    while queue or active:
        while queue and len(active) < max(1, args.workers):
            ck = queue.pop(0)
            p, log = mod.launch_eval(out_dir, args.arm, ck, 0, args.force)
            active.append((ck, p, log))
            print(json.dumps({"event": "roberta_cpu_eval_launched", "arm": args.arm, "checkpoint": ck, "remaining": len(queue)}, ensure_ascii=False), flush=True)
        still = []
        for ck, p, log in active:
            rc = p.poll()
            if rc is None:
                still.append((ck, p, log))
                continue
            log.close()
            ok, probs = mod.per_target_ok(mod.out_per_target(out_dir, args.arm, ck))
            done.append({"checkpoint": ck, "returncode": rc, "ok": ok, "problems": probs})
            if rc != 0 or not ok:
                problems.extend([f"{args.arm} {ck} rc={rc} {p}" for p in probs] or [f"{args.arm} {ck} rc={rc}"])
            print(json.dumps({"event": "roberta_cpu_eval_finished", "arm": args.arm, "checkpoint": ck, "returncode": rc, "ok": ok}, ensure_ascii=False), flush=True)
        active = still
        if active:
            time.sleep(10)
    # Rebuild research rows and summaries using its own functions.
    rows = []
    for arm in ["view", "clean"]:
        for ck in CKS:
            p = mod.out_per_target(out_dir, arm, ck)
            ok, _ = mod.per_target_ok(p)
            if ok:
                rows.append(mod.summarize_payload(p, arm, ck, "cpu_finish" if arm == args.arm else "existing"))
    rows = sorted(rows, key=lambda r: (str(r.get("arm")), int(r.get("words") or 0)))
    summary = mod.write_summary(out_dir, rows, {"source": "true_cpu_roberta_finish", "pending_initial": pending}, problems)
    after = {ck: ready_cols(per_target(args.arm, ck)) for ck in args.checkpoints}
    result = {
        "status": "ROBERTA_TRUE_CPU_FINISHER_DONE" if not problems else "ROBERTA_TRUE_CPU_FINISHER_DONE_WITH_PROBLEMS",
        "created_utc": now(),
        "arm": args.arm,
        "done": done,
        "problem_count": len(problems),
        "problems": problems[:20],
        "after_ready_cols": after,
        "summary": summary,
        "cuda_visible_devices_for_children": "-1",
        "no_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if problems:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
