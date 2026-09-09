#!/usr/bin/env python3
"""research quick integrity reader for repaired factorial RoBERTa training arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS = {
    "hs": "experiments/archive/representation_and_objectives/training/runs/factorial_hs_roberta_100M_accum_mb16",
    "ls": "experiments/archive/representation_and_objectives/training/runs/factorial_ls_roberta_100M_accum_mb16",
    "hd": "experiments/archive/representation_and_objectives/training/runs/factorial_hd_roberta_100M_accum_mb16",
    "ld": "experiments/archive/representation_and_objectives/training/runs/factorial_ld_roberta_100M_accum_mb16",
}
CHECKPOINTS = ["chck_20M", "chck_40M", "chck_60M", "chck_80M", "chck_100M"]
EXPECTED = {
    "word_exposure": 99999910,
    "target_word_exposure": 99999910,
    "actual_training_steps": 2529,
    "planned_training_steps": 2529,
    "optimizer_steps": 2529,
    "batch_size": 256,
    "effective_optimizer_batch_size": 256,
    "micro_batch_size": 16,
    "parameter_count": 30528064,
    "seed": 43,
    "extra_init_seed": 43022,
    "train_rng_seed": 43023,
}


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def arm_report(arm: str) -> tuple[dict[str, Any], list[str]]:
    run = Path(RUNS[arm])
    problems: list[str] = []
    rep: dict[str, Any] = {"arm": arm, "run_dir": str(run), "metrics_exists": False, "checkpoints": {}}
    mpath = run / "scientific_metrics.json"
    if not mpath.exists():
        problems.append(f"{arm}: missing scientific_metrics.json")
    else:
        rep["metrics_exists"] = True
        m = read_json(mpath)
        for k in sorted(set(EXPECTED) | {"status", "loss_first", "loss_last", "elapsed_sec", "example_jsonl", "example_jsonl_label"}):
            if k in m:
                rep[k] = m[k]
        for k, v in EXPECTED.items():
            if m.get(k) != v:
                problems.append(f"{arm}: {k}={m.get(k)} expected {v}")
        if m.get("status") != "ROBERTA_ACCUM_TRAINING_DONE":
            problems.append(f"{arm}: status={m.get('status')} not done")
    log = run / "training_log.jsonl"
    if log.exists():
        lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rep["training_log_lines"] = len(lines)
        if lines:
            rep["first_log"] = json.loads(lines[0])
            rep["last_log"] = json.loads(lines[-1])
    else:
        problems.append(f"{arm}: missing training_log.jsonl")
    for ck in CHECKPOINTS:
        model = run / "hf_model" / ck / "model.safetensors"
        rep["checkpoints"][ck] = {"model": str(model), "exists": model.exists(), "size": model.stat().st_size if model.exists() else 0}
        if not model.exists() or model.stat().st_size <= 0:
            problems.append(f"{arm}: missing/empty {ck}")
    return rep, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["hs", "ls", "hd", "ld"], choices=list(RUNS))
    ap.add_argument("--out", type=Path, default=Path("experiments/archive/representation_and_objectives/data/factorial_training_integrity/factorial_training_integrity.json"))
    args = ap.parse_args()
    reports = {}
    problems: list[str] = []
    for arm in args.arms:
        rep, probs = arm_report(arm)
        reports[arm] = rep
        problems.extend(probs)
    payload = {"status": "FACTORIAL_TRAINING_INTEGRITY", "arms": args.arms, "reports": reports, "problems": problems, "ready": not problems}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "ready": payload["ready"], "problems": len(problems), "out": str(args.out)}, indent=2), flush=True)
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
