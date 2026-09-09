#!/usr/bin/env python3
"""research: run broad Cheap7 screens for corrected bridge checkpoints.

This wrapper is only orchestration around the strict research evaluator.  It evaluates
scientifically informative bridge states under the repaired trusted-loader path and
collates the scores with the already known coherent86 and isolated-specialist
references.  The suite is deliberately separate from relation-only acquisition
control so broad-competence evidence can proceed while the acquisition-control task
runs asynchronously.
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
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
EVAL = _public_path('experiments/archive/functional_learning/scripts/bridge_eval.py')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/bridge_cheap7_suite')
TARGETS = [
    {
        "name": "ordinary_wwm_update0354",
        "path": "experiments/archive/functional_learning/data/bridge_ordinary_wwm_corrected/checkpoints/update_0354",
        "why": "endpoint of relation-row WWM policy; relation probe shows no operation acquisition",
    },
    {
        "name": "answer_allocation_update0150",
        "path": "experiments/archive/functional_learning/data/bridge_answer_allocation_corrected/checkpoints/update_0150",
        "why": "largest early neutral full-source count and bridge alpha-beta split before endpoint saturation; Entity alone was low and full Cheap7 tests broad cost",
    },
    {
        "name": "answer_allocation_update0200",
        "path": "experiments/archive/functional_learning/data/bridge_answer_allocation_corrected/checkpoints/update_0200",
        "why": "near transition where alpha continues falling while self-update has already weakened; tests whether any broad score peak precedes endpoint",
    },
    {
        "name": "answer_allocation_update0354",
        "path": "experiments/archive/functional_learning/data/bridge_answer_allocation_corrected/checkpoints/update_0354",
        "why": "endpoint of answer-allocation mixed policy; relation probe shows replacement-preference reduction but no complete operation",
    },
]
REFERENCES = {
    "coherent86_alpha0p75": {
        "official_style_overall": 42.1210247099666,
        "fast_entity": 27.78,
        "fast_equal_valid_mean": 44.56428571428571,
        "note": "trusted inherited reference; official-style Overall and fast-screen mean are different metrics",
    },
    "isolated_relation_specialist_seed40040": {
        "fast_entity": 27.48,
        "fast_equal_valid_mean": 43.346428571428575,
        "note": "answer-only relation specialist acquired selection but lost broad fast-screen competence",
        "source": "experiments/archive/functional_learning/data/saved_state_cheap7_entity/seed40040_saved_state_eval.json",
    },
    "ordinary_step027_continuation": {
        "fast_equal_valid_mean": 44.30785714285714,
        "note": "historical ordinary continuation, not same-trainer reference for research",
    },
}


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def run_target(target: Dict[str, str], out_root: pathlib.Path, gpu: int, force: bool) -> Dict[str, Any]:
    target_dir = out_root / target["name"]
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        rel(EVAL),
        "--model-path", target["path"],
        "--out-dir", rel(target_dir),
        "--eval-cheap7",
        "--cheap7-columns", "Cheap7",
        "--gpu", str(int(gpu)),
    ]
    if force:
        cmd.append("--force")
    print(json.dumps({"event": "cheap7_target_start", "target": target["name"], "cmd": cmd}), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    rec: Dict[str, Any] = {
        "target": target,
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "elapsed_sec": round(time.time() - t0, 1),
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-3000:],
        "out_dir": rel(target_dir),
    }
    summary_path = target_dir / "bridge_eval_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rec["summary_path"] = rel(summary_path)
        rec["bridge_eval_summary"] = summary
        try:
            rec["scores"] = summary["results"][0]["cheap7"]["scores"]
            rec["identity"] = summary["results"][0]["cheap7"].get("identity") or summary["results"][0]["cheap7"].get("strict_identity")
        except Exception as exc:
            rec["score_parse_error"] = repr(exc)
    if proc.returncode != 0:
        print(json.dumps({"event": "cheap7_target_failed", "target": target["name"], "returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:]}, ensure_ascii=False), flush=True)
    else:
        print(json.dumps({"event": "cheap7_target_done", "target": target["name"], "scores": rec.get("scores")}, ensure_ascii=False), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    suite = {
        "status": "BRIDGE_CHEAP7_SUITE_RUNNING",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_question": "Do the corrected bridge policies preserve or improve broad BabyLM fast-screen competence while relation probes show weak or absent operation acquisition?",
        "strict_evaluator": rel(EVAL),
        "targets": TARGETS,
        "references": REFERENCES,
        "results": [],
    }
    partial_path = out_root / "bridge_cheap7_suite_partial.json"
    partial_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    exit_code = 0
    for target in TARGETS:
        rec = run_target(target, out_root, int(args.gpu), bool(args.force))
        suite["results"].append(rec)
        partial_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if int(rec.get("returncode", 1)) != 0:
            exit_code = int(rec.get("returncode", 1)) or 1
            break
    suite["status"] = "BRIDGE_CHEAP7_SUITE_DONE" if exit_code == 0 else "BRIDGE_CHEAP7_SUITE_FAILED"
    suite["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    suite["n_success"] = sum(1 for r in suite["results"] if int(r.get("returncode", 1)) == 0)
    suite["n_targets"] = len(TARGETS)
    suite["scientific_use"] = "Compare broad fast-screen competence of bridge states to coherent86, isolated specialist, and matched reference when available; this is not official submission evidence."
    summary_path = out_root / "bridge_cheap7_suite_summary.json"
    summary_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": suite["status"], "summary": rel(summary_path), "n_success": suite["n_success"], "n_targets": suite["n_targets"]}, ensure_ascii=False), flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
