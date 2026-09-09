#!/usr/bin/env python3
"""research: evaluate early real-stream policy checkpoints.

Runs two scientific readouts for checkpoints produced by `real_stream_train.py`:

* common-target source-following probe from research, to see whether the policy moves
  the compact/source-conditioned diagnostic in the same direction as concentrated
  research;
* fast BabyLM Cheap7 screen, to catch broad-preservation damage before any full-tail
  or official-coordinate investment.

This is an early prefix comparison, not a BabyLM endpoint.
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
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
TRAIN_ROOT = _public_path('experiments/archive/functional_learning/data/real_stream_policy_comparison')
OUT_ROOT = _public_path('experiments/archive/functional_learning/data/real_stream_policy_eval')
COMMON_PROBE = _public_path('experiments/archive/functional_learning/scripts/common_target_probe.py')
BRIDGE_EVAL = _public_path('experiments/archive/functional_learning/scripts/bridge_eval.py')
REF_CHEAP7 = _public_path('experiments/archive/functional_learning/data/reference_tail_cheap7/bridge_eval_summary.json')
SUMMARY = _public_path('experiments/archive/functional_learning/data/allocation_balanced_synthesis/summary.json')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def run(cmd: List[str], env: Dict[str, str], cwd: pathlib.Path = ROOT, timeout: int | None = None) -> Dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "elapsed_sec": round(time.time() - t0, 1),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def read_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_scores(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    obj = read_json(path)
    try:
        return obj["results"][0]["cheap7"]["scores"]
    except Exception:
        return obj.get("cheap7_scores", {}) or obj.get("scores", {}) or {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-root", type=pathlib.Path, default=TRAIN_ROOT)
    ap.add_argument("--out-root", type=pathlib.Path, default=OUT_ROOT)
    ap.add_argument("--update", type=int, default=80)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--skip-cheap7", action="store_true")
    ap.add_argument("--common-device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = str((args.out_root / "hf_cache").resolve())
    env["TRANSFORMERS_CACHE"] = str((args.out_root / "hf_cache/transformers").resolve())
    env["HF_MODULES_CACHE"] = str((args.out_root / "hf_cache/modules").resolve())
    (args.out_root / "hf_cache/modules").mkdir(parents=True, exist_ok=True)

    ckpts = {
        "realstream_inherited_wwm80": args.train_root / "inherited_wwm" / "checkpoints" / f"update_{int(args.update):04d}",
        "realstream_correspondence_focus80": args.train_root / "correspondence_focus" / "checkpoints" / f"update_{int(args.update):04d}",
    }
    missing = {k: rel(v) for k, v in ckpts.items() if not v.exists()}
    if missing:
        print(json.dumps({"status": "EVAL_MISSING_CHECKPOINTS", "missing": missing}, indent=2), flush=True)
        sys.exit(2)

    common_out = args.out_root / "common_target"
    common_cmd = [
        sys.executable, rel(COMMON_PROBE),
        "--out-dir", rel(common_out),
        "--device", args.common_device,
        "--gpu", "0",
        "--include-compact-wordmatched",
    ]
    for name, path in ckpts.items():
        common_cmd += ["--extra-model", f"{name}={rel(path)}"]
    common_res = run(common_cmd, env)

    cheap_results = []
    if not args.skip_cheap7:
        for name, path in ckpts.items():
            out_dir = args.out_root / "cheap7" / name
            cmd = [
                sys.executable, rel(BRIDGE_EVAL),
                "--model-path", rel(path),
                "--out-dir", rel(out_dir),
                "--eval-cheap7",
                "--cheap7-columns", "Cheap7",
                "--gpu", "0",
            ]
            if args.force:
                cmd.append("--force")
            res = run(cmd, env)
            scores = extract_scores(out_dir / "bridge_eval_summary.json")
            cheap_results.append({"name": name, "checkpoint": rel(path), "run": res, "scores": scores, "summary_path": rel(out_dir / "bridge_eval_summary.json")})
            if res["returncode"] != 0:
                break

    common_summary_path = common_out / "summary.json"
    common_extract = {}
    if common_summary_path.exists():
        obj = read_json(common_summary_path)
        common_extract = {
            "deltas_vs_parent": obj.get("deltas_vs_parent", {}),
            "model_summaries": obj.get("model_summaries", {}),
            "summary_path": rel(common_summary_path),
        }

    reference_scores = extract_scores(REF_CHEAP7)
    summary = {
        "status": "REAL_STREAM_POLICY_EVAL_DONE" if common_res["returncode"] == 0 and all(x["run"]["returncode"] == 0 for x in cheap_results) else "REAL_STREAM_POLICY_EVAL_INCOMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_root": rel(args.train_root),
        "checkpoints": {k: rel(v) for k, v in ckpts.items()},
        "common_target_run": common_res,
        "common_target": common_extract,
        "cheap7_results": cheap_results,
        "reference_tail_cheap7_scores": reference_scores,
        "coherent86_fast_reference": {"equal_valid_mean": 44.56428571428571, "Entity": 27.78},
        "interpretation": "Prefix readout only. A correspondence-focus advantage over inherited WWM on common-target without Cheap7 damage would justify longer real-stream training; a broad-score decline or no diagnostic movement argues against this policy/schedule/objective mixture, not against compaction in general.",
    }
    (args.out_root / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if common_res["returncode"] != 0 or any(x["run"]["returncode"] != 0 for x in cheap_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
