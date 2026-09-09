#!/usr/bin/env python3
"""research: coherent-unsplit no-special trainer control.

The coherent-unsplit-special arm differed from historical coherent86 in both input
special-token exposure and the research trainer implementation.  This control runs the
same research word-paced trainer on the exact coherent86 suffix rows but with
add_special_tokens=False, then evaluates cheap7 through the same research wrapper used
for the format screen.
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
import shutil
import subprocess
import sys
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
TRAINER = _public_path('experiments/archive/relation_learning/scripts/word_paced_format_replay_trainer_control.py')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_tail_eval_one.py')
STREAM = _public_path('experiments/archive/relation_learning/data/format_control_streams/coherent_unsplit_replay_3992800w.jsonl')
LEASH = _public_path('experiments/archive/relation_learning/data/held_coherent_sets/held_coherent_leash_256rows.jsonl')
READOUT = _public_path('experiments/archive/relation_learning/data/held_coherent_sets/held_coherent_readout_256rows.jsonl')
TRAIN_ROOT = _public_path('experiments/archive/relation_learning/data/coherent_no_special_control/train')
EVAL_ROOT = _public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval')
WAIT_ON = _public_path('experiments/archive/relation_learning/data/format_train_eval_sequences/gpu0/sequence_summary.json')
INITIAL_CONSUMED_WORDS = 82_012_495
LR_TOTAL_STEPS = 455


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_path(path: pathlib.Path, timeout: float, interval: float) -> None:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waited {timeout}s for {path}")
        time.sleep(interval)


def run_cmd(label: str, cmd: list[str], cwd: pathlib.Path, env: dict[str, str], log_dir: pathlib.Path, timeout: float) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "start", "label": label, "cmd": [str(x) for x in cmd], "utc": now()}), flush=True)
    t0 = time.time()
    p = subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, text=True, capture_output=True, timeout=timeout)
    (log_dir / f"{label}_stdout.log").write_text(p.stdout, encoding="utf-8")
    (log_dir / f"{label}_stderr.log").write_text(p.stderr, encoding="utf-8")
    rec = {
        "label": label,
        "returncode": p.returncode,
        "elapsed_sec": round(time.time() - t0, 1),
        "stdout_log": rel(log_dir / f"{label}_stdout.log"),
        "stderr_log": rel(log_dir / f"{label}_stderr.log"),
    }
    print(json.dumps({"event": "finished", **rec, "stdout_tail": p.stdout[-1000:], "stderr_tail": p.stderr[-1000:]}, ensure_ascii=False), flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"{label} failed rc={p.returncode}")
    return rec


def read_training_log(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def make_runlike(train_dir: pathlib.Path, label: str) -> pathlib.Path:
    summary = read_json(train_dir / "summary.json")
    config = read_json(train_dir / "train_config.json")
    logs = read_training_log(train_dir / "training_log.jsonl")
    first = logs[0] if logs else summary.get("first_update", {})
    last = logs[-1] if logs else summary.get("last_update", {})
    model_src = train_dir / "alpha_0.75"
    if not (model_src / "model.safetensors").exists():
        model_src = train_dir / "checkpoint"
    runlike = _public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval/runlike') / label
    hf_final = runlike / "hf_model" / "final"
    if hf_final.exists():
        shutil.rmtree(hf_final)
    hf_final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_src, hf_final)
    total_words = int(summary.get("total_words") or config.get("total_words_scheduled") or 3_992_800)
    metrics = {
        "status": "COHERENT_NO_SPECIAL_CONTROL_ENDPOINT",
        "mode": "trainer_coherent_unsplit_no_special_control",
        "replay_mode": config.get("arm_label", label),
        "endpoint": rel(model_src),
        "initial_consumed_words": INITIAL_CONSUMED_WORDS,
        "skip_rows": 530944,
        "max_tail_charged_words": total_words,
        "tail_main_word_exposure": total_words,
        "tail_aux_word_exposure": 0,
        "tail_charged_words": total_words,
        "total_consumed_words": INITIAL_CONSUMED_WORDS + total_words,
        "updates": int(summary.get("updates") or len(logs)),
        "schedule_total": LR_TOTAL_STEPS,
        "stopped_before_cap": False,
        "trainable": "private_adapter_only",
        "total_params": config.get("total_params"),
        "private_params": config.get("trainable_params"),
        "frozen_slow_params": None if config.get("total_params") is None or config.get("trainable_params") is None else int(config["total_params"]) - int(config["trainable_params"]),
        "first_main_loss": first.get("main_ce"),
        "final_main_loss": last.get("main_ce"),
        "mean_main_loss": mean([float(x["main_ce"]) for x in logs]) if logs else None,
        "main_loss_batches": len(logs),
        "first_neutral_loss": first.get("readout_neutral_kl"),
        "final_neutral_loss": last.get("readout_neutral_kl"),
        "mean_neutral_loss": mean([float(x.get("readout_neutral_kl", 0.0)) for x in logs]) if logs else None,
        "first_leash_neutral_loss": first.get("leash_neutral_kl"),
        "final_leash_neutral_loss": last.get("leash_neutral_kl"),
        "mean_leash_neutral_loss": mean([float(x.get("leash_neutral_kl", 0.0)) for x in logs]) if logs else None,
        "deterministic_neutrality_eval_mode": True,
        "first_target_ratio": first.get("target_ratio"),
        "final_target_ratio": last.get("target_ratio"),
        "mean_target_ratio": mean([float(x.get("target_ratio", 0.0)) for x in logs]) if logs else None,
        "add_special_tokens": bool(config.get("add_special_tokens")),
        "first_update": first,
        "last_update": last,
        "source_train_dir": rel(train_dir),
        "created_utc": now(),
        "scientific_note": "Exact coherent86 suffix rows trained through research word-paced machinery with add_special_tokens=False; controls trainer change versus special-token exposure.",
    }
    (runlike / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return runlike


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[98097, 98098])
    ap.add_argument("--wait-timeout", type=float, default=21600.0)
    ap.add_argument("--wait-interval", type=float, default=60.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    if WAIT_ON.exists():
        wait_record = {"waited_for_gpu0_sequence": False, "path": rel(WAIT_ON), "status": read_json(WAIT_ON).get("status")}
    else:
        wait_for_path(WAIT_ON, args.wait_timeout, args.wait_interval)
        wait_record = {"waited_for_gpu0_sequence": True, "path": rel(WAIT_ON), "status": read_json(WAIT_ON).get("status")}
    if wait_record["status"] != "FORMAT_TRAIN_EVAL_SEQUENCE_DONE":
        raise RuntimeError(f"Prerequisite sequence did not finish cleanly: {wait_record}")

    records = []
    for seed in args.seeds:
        label = f"coherent_unsplit_no_special_seed{seed}_alpha0p75"
        train_dir = TRAIN_ROOT / f"seed{seed}"
        if args.force and train_dir.exists():
            shutil.rmtree(train_dir)
        if not (train_dir / "summary.json").exists():
            cmd = [
                sys.executable, "-B", str(TRAINER),
                "--stream", str(STREAM),
                "--out-dir", str(train_dir),
                "--arm-label", f"coherent_unsplit_no_special_seed{seed}",
                "--leash-coherent-stream", str(LEASH),
                "--readout-coherent-stream", str(READOUT),
                "--gpu", str(args.gpu),
                "--seed", str(seed),
                "--private-scale", "1.0",
                "--alpha-endpoint", "0.75",
                "--neutral-lambda", "1.0",
                "--no-add-special-tokens",
                "--log-every", "10",
            ]
            run_cmd(f"train_{seed}", cmd, ROOT, env, _public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval/launcher_logs'), timeout=3600)
        runlike = make_runlike(train_dir, label)
        out_root = _public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval/cheap7') / label
        collate_root = _public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval/collate') / label
        summary_root = _public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval/summary')
        cmd = [
            sys.executable, "-B", str(EVALUATOR),
            "--run-dir", str(runlike),
            "--target", label,
            "--endpoint", "final",
            "--out-root", str(out_root),
            "--collate-root", str(collate_root),
            "--summary-root", str(summary_root),
            "--gpu", str(args.gpu),
        ]
        if args.force:
            cmd.append("--force")
        eval_rec = run_cmd(f"eval_{seed}", cmd, ROOT, env, _public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval/launcher_logs'), timeout=9000)
        summary_path = summary_root / f"{label}_summary.json"
        rec = {"seed": seed, "label": label, "train_dir": rel(train_dir), "runlike": rel(runlike), **eval_rec}
        if summary_path.exists():
            s = read_json(summary_path)
            rec["summary_path"] = rel(summary_path)
            rec["cheap7"] = s.get("cheap7")
            rec["cheap7_delta_vs_chck82"] = s.get("cheap7_delta_vs_chck82")
            rec["scores"] = s.get("scores")
            rec["deltas_vs_chck82"] = s.get("deltas_vs_chck82")
        records.append(rec)
    summary = {"status": "COHERENT_NO_SPECIAL_CONTROL_SEQUENCE_DONE", "created_utc": now(), "wait_record": wait_record, "records": records}
    (_public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval/sequence_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
