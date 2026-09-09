#!/usr/bin/env python3
"""research smoke/equivalence harness for v3 directional causal trainer.

Runs four small two-epoch jobs over the same compact pair arm:
  ff, fr, rr, rf.
Then checks:
  * each run exits and writes a manifest/log/checkpoints;
  * it has many optimizer updates, not a one-step parse-only pass;
  * ff and fr have bit-identical epoch_1 model hashes and identical first-epoch
    numeric log records except elapsed time;
  * rr and rf likewise match through epoch_1;
  * ff and fr diverge by final epoch, and rr/rf diverge by final epoch, showing the
    schedule actually changes the second epoch.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = pathlib.Path(".")
TRAINER = ROOT / "experiments/archive/representation_and_objectives/scripts/pair_aware_causal_trainer_v3.py"
PAIR = ROOT / "experiments/archive/representation_and_objectives/data/prediction_geometry_scaffold_v2/compact_oneway.jsonl"
FILLER = ROOT / "experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/filler_rows.jsonl"
TOK = ROOT / "experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer"
OUT = ROOT / "experiments/archive/representation_and_objectives/data/v3_directional_smoke"

SCHEDULES = ["ff", "fr", "rr", "rf"]


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_log(path: pathlib.Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def scrub_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in e.items() if k != "elapsed_sec"}


def first_epoch_records(log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [scrub_entry(e) for e in log if e["epoch"] == 1]


def run_schedule(sched: str) -> Dict[str, Any]:
    out = OUT / f"compact_{sched}"
    cmd = [
        sys.executable,
        "-B",
        str(TRAINER),
        "--arm_pair_jsonl",
        str(PAIR),
        "--filler_jsonl",
        str(FILLER),
        "--tokenizer_dir",
        str(TOK),
        "--output_dir",
        str(out),
        "--schedule",
        sched,
        "--n_epochs",
        "2",
        "--batch_size",
        "64",
        "--grad_accum",
        "1",
        "--max_pairs",
        "512",
        "--max_filler_words",
        "50000",
        "--log_every",
        "1",
        "--save_epochs",
        "1,2",
        "--device",
        "cuda:0",
        "--overwrite",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    (out / "harness_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out / "harness_stderr.log").write_text(proc.stderr, encoding="utf-8")
    result = {"schedule": sched, "returncode": proc.returncode, "elapsed_sec": time.time() - t0, "out": str(out)}
    if proc.returncode != 0:
        result["stdout_tail"] = proc.stdout[-4000:]
        result["stderr_tail"] = proc.stderr[-4000:]
        return result
    manifest = load_json(out / "training_manifest.json")
    config = load_json(out / "train_config.json")
    log = load_log(out / "training_log.jsonl")
    result.update(
        {
            "manifest": manifest,
            "config": config,
            "log_records": len(log),
            "first_log": log[0] if log else None,
            "last_log": log[-1] if log else None,
            "epoch1_hash": manifest["checkpoint_hashes"].get("epoch_1"),
            "epoch2_hash": manifest["checkpoint_hashes"].get("epoch_2"),
            "final_hash": manifest["checkpoint_hashes"].get("final"),
        }
    )
    return result


def pair_check(a_name: str, b_name: str, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    a_dir = pathlib.Path(results[a_name]["out"])
    b_dir = pathlib.Path(results[b_name]["out"])
    a_log = load_log(a_dir / "training_log.jsonl")
    b_log = load_log(b_dir / "training_log.jsonl")
    a_first = first_epoch_records(a_log)
    b_first = first_epoch_records(b_log)
    same_first_records = a_first == b_first
    same_epoch1_hash = results[a_name].get("epoch1_hash") == results[b_name].get("epoch1_hash")
    same_final_hash = results[a_name].get("final_hash") == results[b_name].get("final_hash")
    first_mismatch = None
    if not same_first_records:
        for i, (x, y) in enumerate(zip(a_first, b_first)):
            if x != y:
                first_mismatch = {"index": i, a_name: x, b_name: y}
                break
        if first_mismatch is None and len(a_first) != len(b_first):
            first_mismatch = {"len_a": len(a_first), "len_b": len(b_first)}
    return {
        "pair": [a_name, b_name],
        "same_first_epoch_records_excluding_elapsed": same_first_records,
        "same_epoch1_model_hash": same_epoch1_hash,
        "same_final_model_hash": same_final_hash,
        "first_epoch_record_count": [len(a_first), len(b_first)],
        "first_mismatch": first_mismatch,
        "epoch1_hashes": [results[a_name].get("epoch1_hash"), results[b_name].get("epoch1_hash")],
        "final_hashes": [results[a_name].get("final_hash"), results[b_name].get("final_hash")],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    for sched in SCHEDULES:
        print(f"Running smoke schedule {sched}...", flush=True)
        results[sched] = run_schedule(sched)
        print(json.dumps({k: v for k, v in results[sched].items() if k not in {"manifest", "config"}}, indent=2), flush=True)
        if results[sched]["returncode"] != 0:
            break

    all_returned = all(results.get(s, {}).get("returncode") == 0 for s in SCHEDULES)
    checks: Dict[str, Any] = {}
    ok = False
    if all_returned:
        checks["ff_fr_forward_prefix"] = pair_check("ff", "fr", results)
        checks["rr_rf_reverse_prefix"] = pair_check("rr", "rf", results)
        min_steps = min(results[s]["manifest"]["steps"] for s in SCHEDULES)
        first_epoch_step_counts = {s: results[s]["manifest"]["epoch_summaries"][0]["last_step"] for s in SCHEDULES}
        ok = (
            min_steps >= 10
            and checks["ff_fr_forward_prefix"]["same_first_epoch_records_excluding_elapsed"]
            and checks["ff_fr_forward_prefix"]["same_epoch1_model_hash"]
            and not checks["ff_fr_forward_prefix"]["same_final_model_hash"]
            and checks["rr_rf_reverse_prefix"]["same_first_epoch_records_excluding_elapsed"]
            and checks["rr_rf_reverse_prefix"]["same_epoch1_model_hash"]
            and not checks["rr_rf_reverse_prefix"]["same_final_model_hash"]
        )
        checks["min_steps"] = min_steps
        checks["first_epoch_step_counts"] = first_epoch_step_counts

    summary = {
        "status": "V3_DIRECTIONAL_SMOKE_PASS" if ok else "V3_DIRECTIONAL_SMOKE_FAIL",
        "all_runs_returned_zero": all_returned,
        "output_root": str(OUT),
        "results": {k: {kk: vv for kk, vv in v.items() if kk not in {"manifest", "config"}} for k, v in results.items()},
        "checks": checks,
    }
    (OUT / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
