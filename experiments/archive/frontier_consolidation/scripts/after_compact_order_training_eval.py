#!/usr/bin/env python3
"""research queued post-training evaluator for compact ordered-vs-scrambled arms.

Waits until both research arms have finished real 40M training, verifies finite logs and
checkpoint presence, evaluates only the selected cheap BabyLM columns at chck_20M and
chck_40M, runs the source-absent compact-channel probe, and integrates both surfaces.
No SuperGLUE, AoA, upload, or leaderboard submission is performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/after_compact_order_training_eval.py')
for _ in range(12):
    if (_public_path("experiments")).exists():
        break
    USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WS = _public_path('experiments/archive/frontier_consolidation')
RUN_ROOTS = {
    "ordered": _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_ordered_40M_seed43022'),
    "scrambled": _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022'),
}
CHECKPOINTS = ["chck_20M", "chck_40M"]
EVAL_WRAPPER = _public_path('experiments/archive/frontier_consolidation/scripts/eval_custom_checkpoint.py')
CHANNEL_PROBE = _public_path('experiments/archive/frontier_consolidation/scripts/compact_order_channel_probe.py')
INTERPRETER = _public_path('experiments/archive/frontier_consolidation/scripts/integrate_compact_order_results.py')
DEFAULT_EVAL_DIR = _public_path('experiments/archive/frontier_consolidation/data/compact_order_selected_eval')
DEFAULT_CHANNEL_DIR = _public_path('experiments/archive/frontier_consolidation/data/compact_order_channel_probe')
DEFAULT_INTEGRATED_DIR = _public_path('experiments/archive/frontier_consolidation/data/compact_order_integrated_readout')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_training(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"run_dir": str(run_dir)}
    mpath = run_dir / "scientific_metrics.json"
    if not mpath.exists():
        out["status"] = "missing_scientific_metrics"
        return out
    metrics = read_json(mpath)
    out["status"] = "has_scientific_metrics"
    out["scientific_metrics_keys"] = sorted(metrics.keys())
    for k in ["selected_for_training_words", "cum_words_exposed", "max_word_exposure", "num_consumed_examples", "param_count", "data_source_type", "example_jsonl_label"]:
        if k in metrics:
            out[k] = metrics[k]
    # Trainer variants use either checkpoints or saved_checkpoints lists.
    ck = metrics.get("checkpoints") or metrics.get("saved_checkpoints")
    if ck is not None:
        out["checkpoints_recorded"] = ck
    logp = run_dir / "training_log.jsonl"
    if logp.exists():
        rows = [json.loads(l) for l in logp.read_text(encoding="utf-8").splitlines() if l.strip()]
        out["training_log_rows"] = len(rows)
        if rows:
            out["first_log"] = rows[0]
            out["last_log"] = rows[-1]
            out["loss_finite_all_rows"] = all(math.isfinite(float(r.get("loss", float("nan")))) for r in rows)
            tail = rows[-min(50, len(rows)):]
            vals = [float(r["loss"]) for r in tail if "loss" in r]
            out["tail50_loss_mean"] = mean(vals) if vals else None
            out["last_cumulative_word_exposure"] = rows[-1].get("cumulative_word_exposure")
    out["checkpoint_files"] = {ck: (run_dir / "hf_model" / ck / "model.safetensors").exists() for ck in CHECKPOINTS}
    return out


def training_ready(run_dir: Path) -> bool:
    mpath = run_dir / "scientific_metrics.json"
    if not mpath.exists():
        return False
    for ck in CHECKPOINTS:
        if not (run_dir / "hf_model" / ck / "model.safetensors").exists():
            return False
    logp = run_dir / "training_log.jsonl"
    if not logp.exists() or logp.stat().st_size <= 0:
        return False
    return True


def wait_for_training(timeout_sec: int, sleep_sec: int) -> dict[str, Any]:
    t0 = time.time()
    last_print = 0.0
    while True:
        ready = {arm: training_ready(run) for arm, run in RUN_ROOTS.items()}
        if all(ready.values()):
            return {"status": "ready", "elapsed_wait_sec": round(time.time() - t0, 1), "ready": ready}
        if time.time() - t0 > timeout_sec:
            return {"status": "timeout_waiting_for_training", "elapsed_wait_sec": round(time.time() - t0, 1), "ready": ready, "summaries": {a: summarize_training(r) for a, r in RUN_ROOTS.items()}}
        if time.time() - last_print >= max(30, sleep_sec):
            print(json.dumps({"event": "waiting_for_training", "utc": now(), "ready": ready, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)
            last_print = time.time()
        time.sleep(sleep_sec)


def run_one_eval(arm: str, ck: str, gpu: int, eval_dir: Path, force: bool) -> dict[str, Any]:
    target = f"compact_order_{arm}_{ck}"
    out_base = eval_dir / arm / ck
    cmd = [
        sys.executable, "-B", str(EVAL_WRAPPER),
        "--run-dir", str(RUN_ROOTS[arm]),
        "--endpoint", ck,
        "--target", target,
        "--out-base", str(out_base),
        "--gpu", str(gpu),
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_dir = eval_dir / "orchestrator_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{target}_gpu{gpu}_stdout.log"
    stderr_log = log_dir / f"{target}_gpu{gpu}_stderr.log"
    print(json.dumps({"event": "eval_start", "utc": now(), "target": target, "gpu": gpu, "cmd": cmd}), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    rec: dict[str, Any] = {
        "arm": arm, "checkpoint": ck, "target": target, "gpu": gpu, "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 1), "stdout_log": str(stdout_log), "stderr_log": str(stderr_log),
        "out_base": str(out_base),
    }
    summary_path = out_base / f"{target}_summary.json"
    if proc.returncode == 0 and summary_path.exists():
        rec["summary_path"] = str(summary_path)
        try:
            payload = read_json(summary_path)
            rec["cheap7"] = payload.get("record", {}).get("cheap7")
            rec["scores"] = payload.get("record", {}).get("scores")
            rec["per_target"] = payload.get("record", {}).get("per_target")
        except Exception as e:
            rec["summary_read_error"] = str(e)
    else:
        rec["stdout_tail"] = proc.stdout[-3000:]
        rec["stderr_tail"] = proc.stderr[-3000:]
    print(json.dumps({"event": "eval_done", "utc": now(), **{k: rec.get(k) for k in ["target", "gpu", "returncode", "cheap7", "elapsed_sec"]}}), flush=True)
    return rec


def run_eval_panel(eval_dir: Path, force: bool) -> dict[str, Any]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("ordered", "chck_20M", 0), ("scrambled", "chck_20M", 1), ("ordered", "chck_40M", 0), ("scrambled", "chck_40M", 1)]
    # Run two waves so each GPU has one job at a time.  If GPU1 fails (e.g. residual memory), retry that job on GPU0 after the wave.
    results = []
    for wave in [jobs[:2], jobs[2:]]:
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(run_one_eval, arm, ck, gpu, eval_dir, force): (arm, ck, gpu) for arm, ck, gpu in wave}
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
        retry = [r for r in results if r["returncode"] != 0 and not r.get("retried") and (r["arm"], r["checkpoint"], r["gpu"]) in wave]
        for r in retry:
            print(json.dumps({"event": "eval_retry_on_gpu0", "utc": now(), "arm": r["arm"], "checkpoint": r["checkpoint"], "prev_gpu": r["gpu"]}), flush=True)
            rr = run_one_eval(r["arm"], r["checkpoint"], 0, eval_dir, True)
            rr["retried_from"] = r
            rr["retried"] = True
            results.append(rr)
    ok = all(any(r["arm"] == arm and r["checkpoint"] == ck and r["returncode"] == 0 for r in results) for arm in RUN_ROOTS for ck in CHECKPOINTS)
    return {"status": "selected_eval_complete" if ok else "selected_eval_incomplete", "records": results}


def run_channel_probe(channel_dir: Path, force: bool) -> dict[str, Any]:
    channel_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", str(CHANNEL_PROBE), "--out-dir", str(channel_dir), "--device", "cuda", "--batch-size", "96", "--per-category", "4096", "--n-boot", "1000", "--checkpoints", *CHECKPOINTS]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_dir = channel_dir / "driver_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "channel_probe_start", "utc": now(), "cmd": cmd}), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    (log_dir / "channel_probe_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "channel_probe_stderr.log").write_text(proc.stderr, encoding="utf-8")
    rec: dict[str, Any] = {"returncode": proc.returncode, "elapsed_sec": round(elapsed, 1), "stdout_log": str(log_dir / "channel_probe_stdout.log"), "stderr_log": str(log_dir / "channel_probe_stderr.log"), "out_json": str(channel_dir / "compact_order_channel_probe.json")}
    if proc.returncode != 0:
        rec["stdout_tail"] = proc.stdout[-3000:]
        rec["stderr_tail"] = proc.stderr[-3000:]
    print(json.dumps({"event": "channel_probe_done", "utc": now(), "returncode": proc.returncode, "elapsed_sec": round(elapsed, 1)}), flush=True)
    return rec


def run_interpreter(eval_dir: Path, channel_dir: Path, integrated_dir: Path, force: bool) -> dict[str, Any]:
    integrated_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-B", str(INTERPRETER), "--eval-dir", str(eval_dir), "--channel-dir", str(channel_dir), "--out-dir", str(integrated_dir)]
    if force:
        cmd.append("--force")
    print(json.dumps({"event": "integrate_start", "utc": now(), "cmd": cmd}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), capture_output=True, text=True)
    (integrated_dir / "integrator_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (integrated_dir / "integrator_stderr.log").write_text(proc.stderr, encoding="utf-8")
    rec: dict[str, Any] = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:], "out_json": str(integrated_dir / "compact_order_integrated_readout.json")}
    if proc.returncode == 0:
        try:
            payload = read_json(integrated_dir / "compact_order_integrated_readout.json")
            rec["decision_readout"] = payload.get("decision_readout")
        except Exception as e:
            rec["read_error"] = str(e)
    print(json.dumps({"event": "integrate_done", "utc": now(), "returncode": proc.returncode}), flush=True)
    return rec


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    ap.add_argument("--channel-dir", type=Path, default=DEFAULT_CHANNEL_DIR)
    ap.add_argument("--integrated-dir", type=Path, default=DEFAULT_INTEGRATED_DIR)
    ap.add_argument("--wait-timeout-sec", type=int, default=9000)
    ap.add_argument("--sleep-sec", type=int, default=60)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "AFTER_TRAINING_EVAL_PLAN",
        "meaning": "Minimum post-training evidence for compact ordered-vs-scrambled: selected cheap official-compatible scores for 2 arms x 2 checkpoints plus fixed source-absent channel probe; no SuperGLUE/AoA/upload/submission.",
        "run_roots": {k: str(v) for k, v in RUN_ROOTS.items()},
        "checkpoints": CHECKPOINTS,
        "eval_dir": str(args.eval_dir),
        "channel_dir": str(args.channel_dir),
        "integrated_dir": str(args.integrated_dir),
        "eval_wrapper": str(EVAL_WRAPPER),
        "channel_probe": str(CHANNEL_PROBE),
        "interpreter": str(INTERPRETER),
        "training_summaries_at_start": {a: summarize_training(r) for a, r in RUN_ROOTS.items()},
    }
    write_json(args.eval_dir / "queued_eval_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry:
        return
    wait = wait_for_training(args.wait_timeout_sec, args.sleep_sec)
    write_json(args.eval_dir / "training_wait_result.json", wait)
    if wait.get("status") != "ready":
        raise RuntimeError(f"training not ready: {wait}")
    train_summaries = {a: summarize_training(r) for a, r in RUN_ROOTS.items()}
    write_json(args.eval_dir / "training_completion_summary.json", train_summaries)
    eval_panel = run_eval_panel(args.eval_dir, args.force)
    write_json(args.eval_dir / "selected_eval_panel_summary.json", eval_panel)
    if eval_panel.get("status") != "selected_eval_complete":
        raise RuntimeError("selected eval panel incomplete; see selected_eval_panel_summary.json")
    channel = run_channel_probe(args.channel_dir, args.force)
    write_json(args.channel_dir / "channel_probe_driver_summary.json", channel)
    if channel.get("returncode") != 0:
        raise RuntimeError("channel probe failed; see channel_probe_driver_summary.json")
    integ = run_interpreter(args.eval_dir, args.channel_dir, args.integrated_dir, args.force)
    write_json(args.integrated_dir / "integrator_driver_summary.json", integ)
    if integ.get("returncode") != 0:
        raise RuntimeError("integrator failed; see integrator_driver_summary.json")
    print(json.dumps({"status": "AFTER_TRAINING_EVAL_COMPLETE", "eval_dir": str(args.eval_dir), "channel_dir": str(args.channel_dir), "integrated_dir": str(args.integrated_dir), "decision_readout": integ.get("decision_readout")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
