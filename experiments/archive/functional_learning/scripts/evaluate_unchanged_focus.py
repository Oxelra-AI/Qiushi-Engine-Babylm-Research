#!/usr/bin/env python3
"""research: evaluate unchanged-tail real-stream objective checkpoints.

This evaluator keeps the research common-target bank as one mechanistic readout, but
also measures the paired qwen material actually trained in the unchanged-tail run:
masked-token reconstruction on source-visible current-rewrite second views.  The
broad Cheap7 fast screen is included to judge practical promise without treating the
small common-target bank as the sole decision maker.
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
COMMON_PROBE = _public_path('experiments/archive/functional_learning/scripts/common_target_probe.py')
BRIDGE_EVAL = _public_path('experiments/archive/functional_learning/scripts/bridge_eval.py')
QWEN_PROBE = _public_path('experiments/archive/functional_learning/scripts/qwen_view_surface_probe.py')
DEFAULT_TRAIN_ROOT = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_eval')
REF_CHEAP7 = _public_path('experiments/archive/functional_learning/data/reference_tail_cheap7/bridge_eval_summary.json')


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
    ap.add_argument("--train-root", type=pathlib.Path, default=DEFAULT_TRAIN_ROOT)
    ap.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--objectives", nargs="+", default=["inherited_wwm", "correspondence_focus_pooled", "correspondence_focus_weighted"])
    ap.add_argument("--update", type=int, default=80)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-cheap7", action="store_true")
    ap.add_argument("--skip-common", action="store_true")
    ap.add_argument("--skip-qwen-probe", action="store_true")
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

    ckpts = {f"unchanged_{obj}_u{int(args.update):04d}": args.train_root / obj / "checkpoints" / f"update_{int(args.update):04d}" for obj in args.objectives}
    missing = {k: rel(v) for k, v in ckpts.items() if not v.exists()}
    if missing:
        print(json.dumps({"status": "EVAL_MISSING_CHECKPOINTS", "missing": missing}, indent=2), flush=True)
        sys.exit(2)

    common_res: Dict[str, Any] | None = None
    if not args.skip_common:
        common_out = args.out_root / "common_target"
        common_cmd = [
            sys.executable, rel(COMMON_PROBE),
            "--out-dir", rel(common_out),
            "--device", "cuda",
            "--gpu", "0",
            "--include-compact-wordmatched",
        ]
        for name, path in ckpts.items():
            common_cmd += ["--extra-model", f"{name}={rel(path)}"]
        common_res = run(common_cmd, env)
    common_extract = {}
    common_summary_path = args.out_root / "common_target" / "summary.json"
    if common_summary_path.exists():
        obj = read_json(common_summary_path)
        common_extract = {
            "deltas_vs_parent": obj.get("deltas_vs_parent", {}),
            "model_summaries": obj.get("model_summaries", {}),
            "summary_path": rel(common_summary_path),
        }

    qwen_res: Dict[str, Any] | None = None
    if not args.skip_qwen_probe:
        qwen_out = args.out_root / "qwen_view_surface"
        qwen_cmd = [sys.executable, rel(QWEN_PROBE), "--out-dir", rel(qwen_out), "--device", "cuda", "--gpu", "0"]
        for name, path in ckpts.items():
            qwen_cmd += ["--extra-model", f"{name}={rel(path)}"]
        qwen_res = run(qwen_cmd, env)
    qwen_extract = {}
    qwen_summary_path = args.out_root / "qwen_view_surface" / "summary.json"
    if qwen_summary_path.exists():
        qwen_extract = read_json(qwen_summary_path)

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

    ok_common = common_res is None or common_res.get("returncode") == 0
    ok_qwen = qwen_res is None or qwen_res.get("returncode") == 0
    ok_cheap = all(x["run"]["returncode"] == 0 for x in cheap_results)
    reference_scores = extract_scores(REF_CHEAP7)
    summary = {
        "status": "UNCHANGED_FOCUS_EVAL_DONE" if ok_common and ok_qwen and ok_cheap else "UNCHANGED_FOCUS_EVAL_INCOMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_root": rel(args.train_root),
        "checkpoints": {k: rel(v) for k, v in ckpts.items()},
        "common_target_run": common_res,
        "common_target": common_extract,
        "qwen_view_surface_run": qwen_res,
        "qwen_view_surface": qwen_extract,
        "cheap7_results": cheap_results,
        "reference_tail_cheap7_scores": reference_scores,
        "coherent86_fast_reference": {"equal_valid_mean": 44.56428571428571, "Entity": 27.78},
        "interpretation": "Unchanged-tail prefix readout. Use Cheap7 and qwen trained-material probes together with common-target; the common-target bank constrains one transfer mechanism but is not the only signal of practical competence.",
    }
    (args.out_root / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not (ok_common and ok_qwen and ok_cheap):
        sys.exit(1)


if __name__ == "__main__":
    main()
