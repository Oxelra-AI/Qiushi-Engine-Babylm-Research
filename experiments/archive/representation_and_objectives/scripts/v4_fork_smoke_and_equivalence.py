#!/usr/bin/env python3
"""research v4 smoke harness: prove forked first-epoch identity before full GPU runs."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List

TRAINER = pathlib.Path("experiments/archive/representation_and_objectives/scripts/pair_aware_causal_epoch_fork_v4.py")
PAIR = pathlib.Path("experiments/archive/representation_and_objectives/data/prediction_geometry_scaffold_v2/compact_oneway.jsonl")
FILLER = pathlib.Path("experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/filler_rows.jsonl")
TOK = pathlib.Path("experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer")
OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/v4_fork_smoke")


def load_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def load_log(p: pathlib.Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def scrub_log(log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in e.items() if k != "elapsed_sec"} for e in log]


def run(cmd: List[str], out_dir: pathlib.Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run(cmd, text=True, capture_output=True)
    (out_dir / "harness_stdout.log").write_text(p.stdout, encoding="utf-8")
    (out_dir / "harness_stderr.log").write_text(p.stderr, encoding="utf-8")
    result = {"returncode": p.returncode, "elapsed_sec": time.time() - t0, "out": str(out_dir)}
    if p.returncode != 0:
        result["stdout_tail"] = p.stdout[-4000:]
        result["stderr_tail"] = p.stderr[-4000:]
    else:
        result["manifest"] = load_json(out_dir / "training_manifest.json")
        result["log"] = load_log(out_dir / "training_log.jsonl")
    return result


def common_args(out_dir: pathlib.Path, phase: str, direction: str, extra: List[str] | None = None) -> List[str]:
    cmd = [
        sys.executable,
        "-B",
        str(TRAINER),
        "--phase",
        phase,
        "--direction",
        direction,
        "--arm_pair_jsonl",
        str(PAIR),
        "--filler_jsonl",
        str(FILLER),
        "--tokenizer_dir",
        str(TOK),
        "--output_dir",
        str(out_dir),
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
        "--device",
        "cuda:0",
        "--overwrite",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    # Build one forward and one reverse prefix.
    specs = [
        ("prefix_forward", common_args(OUT / "prefix_forward", "prefix", "forward")),
        ("prefix_reverse", common_args(OUT / "prefix_reverse", "prefix", "reverse")),
    ]
    for name, cmd in specs:
        print(f"Running {name}...", flush=True)
        results[name] = run(cmd, OUT / name)
        print(json.dumps({k: v for k, v in results[name].items() if k not in {"manifest", "log"}}, indent=2), flush=True)
        if results[name]["returncode"] != 0:
            break

    if all(results.get(x, {}).get("returncode") == 0 for x in ["prefix_forward", "prefix_reverse"]):
        f_state = pathlib.Path(results["prefix_forward"]["manifest"]["state_file"])
        r_state = pathlib.Path(results["prefix_reverse"]["manifest"]["state_file"])
        cont_specs = [
            ("ff", common_args(OUT / "ff", "continue", "forward", ["--resume_state", str(f_state)])),
            ("fr", common_args(OUT / "fr", "continue", "reverse", ["--resume_state", str(f_state)])),
            ("rr", common_args(OUT / "rr", "continue", "reverse", ["--resume_state", str(r_state)])),
            ("rf", common_args(OUT / "rf", "continue", "forward", ["--resume_state", str(r_state)])),
        ]
        for name, cmd in cont_specs:
            print(f"Running continuation {name}...", flush=True)
            results[name] = run(cmd, OUT / name)
            print(json.dumps({k: v for k, v in results[name].items() if k not in {"manifest", "log"}}, indent=2), flush=True)
            if results[name]["returncode"] != 0:
                break

    all_ok = all(results.get(x, {}).get("returncode") == 0 for x in ["prefix_forward", "prefix_reverse", "ff", "fr", "rr", "rf"])
    checks: Dict[str, Any] = {}
    ok = False
    if all_ok:
        pf = results["prefix_forward"]["manifest"]
        pr = results["prefix_reverse"]["manifest"]
        ff = results["ff"]["manifest"]
        fr = results["fr"]["manifest"]
        rr = results["rr"]["manifest"]
        rf = results["rf"]["manifest"]
        checks["prefix_forward_log_records"] = len(results["prefix_forward"]["log"])
        checks["prefix_reverse_log_records"] = len(results["prefix_reverse"]["log"])
        checks["forward_prefix_hash_equals_in_ff_fr"] = (
            pf["model_hash"] == ff["prefix_info"]["prefix_model_hash"] == fr["prefix_info"]["prefix_model_hash"]
        )
        checks["reverse_prefix_hash_equals_in_rr_rf"] = (
            pr["model_hash"] == rr["prefix_info"]["prefix_model_hash"] == rf["prefix_info"]["prefix_model_hash"]
        )
        checks["ff_fr_same_start_step"] = ff["prefix_info"]["prefix_global_step"] == fr["prefix_info"]["prefix_global_step"]
        checks["rr_rf_same_start_step"] = rr["prefix_info"]["prefix_global_step"] == rf["prefix_info"]["prefix_global_step"]
        checks["ff_fr_final_diverge"] = ff["model_hash"] != fr["model_hash"]
        checks["rr_rf_final_diverge"] = rr["model_hash"] != rf["model_hash"]
        checks["continuation_steps"] = {k: results[k]["manifest"]["global_step"] for k in ["ff", "fr", "rr", "rf"]}
        min_steps = min(checks["continuation_steps"].values())
        ok = (
            checks["prefix_forward_log_records"] >= 10
            and checks["prefix_reverse_log_records"] >= 10
            and checks["forward_prefix_hash_equals_in_ff_fr"]
            and checks["reverse_prefix_hash_equals_in_rr_rf"]
            and checks["ff_fr_same_start_step"]
            and checks["rr_rf_same_start_step"]
            and checks["ff_fr_final_diverge"]
            and checks["rr_rf_final_diverge"]
            and min_steps >= 20
        )

    summary = {
        "status": "V4_FORK_SMOKE_PASS" if ok else "V4_FORK_SMOKE_FAIL",
        "output_root": str(OUT),
        "all_ok": all_ok,
        "checks": checks,
        "results": {
            k: {
                "returncode": v.get("returncode"),
                "elapsed_sec": v.get("elapsed_sec"),
                "out": v.get("out"),
                "manifest_summary": None
                if "manifest" not in v
                else {
                    "status": v["manifest"].get("status"),
                    "direction": v["manifest"].get("direction"),
                    "global_step": v["manifest"].get("global_step"),
                    "model_hash": v["manifest"].get("model_hash"),
                    "state_file": v["manifest"].get("state_file"),
                    "prefix_info": v["manifest"].get("prefix_info"),
                },
            }
            for k, v in results.items()
        },
    }
    (OUT / "fork_smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
