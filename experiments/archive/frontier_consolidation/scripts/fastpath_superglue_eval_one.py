#!/usr/bin/env python3
"""SuperGLUE-only evaluator for research frozen-anchor fast-path replay arms.

This is a generalization of the research tail SuperGLUE evaluator: cheap7 is first
measured by `frozen82_tail_eval_one.py` for a specific research target, then
this script evaluates the missing SuperGLUE column and computes projected Overall
with AoA=0.  It does not decide submission readiness; it supplies one column for
the A2/A3/scored-null comparison.
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
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
EVAL_SCRIPT = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
CHCK82_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def chck82_ref() -> dict[str, Any]:
    j = read_json(CHCK82_VERIFY)
    scores = {k: float(v) for k, v in j["score_arithmetic"]["scores"].items() if v is not None}
    return {
        "scores": scores,
        "cheap7": mean(scores[c] for c in CHEAP_COLUMNS),
        "superglue": scores["SuperGLUE"],
        "aoa": scores["AoA"],
        "overall": float(j["score_arithmetic"]["overall_reported"]),
        "source": rel(CHCK82_VERIFY),
    }


def superglue_score(payload: dict[str, Any]) -> float | None:
    official = payload.get("official_overall", {}).get("scores", {})
    if official.get("SuperGLUE") is not None:
        return float(official["SuperGLUE"])
    tasks = payload.get("tasks", {})
    r = tasks.get("SuperGLUE")
    if isinstance(r, dict):
        if r.get("score") is not None:
            return float(r["score"])
        if r.get("superglue_mean") is not None:
            return float(r["superglue_mean"])
        if r.get("superglue_primary_metric_mean") is not None:
            return float(r["superglue_primary_metric_mean"])
    return None


def compute_overall(cheap_scores: dict[str, Any], superglue: float, aoa: float = 0.0) -> float:
    vals = [float(cheap_scores[c]) for c in CHEAP_COLUMNS] + [float(superglue), float(aoa)]
    return float(mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--cheap-summary", required=True)
    ap.add_argument("--endpoint", default="final")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--collate-root", required=True)
    ap.add_argument("--summary-root", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    out_root = pathlib.Path(args.out_root)
    collate_root = pathlib.Path(args.collate_root)
    summary_root = pathlib.Path(args.summary_root)
    cheap_summary_path = pathlib.Path(args.cheap_summary)
    out_root.mkdir(parents=True, exist_ok=True)
    collate_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)
    ref = chck82_ref()
    cheap = read_json(cheap_summary_path)
    cheap_scores = {k: float(v) for k, v in cheap["scores"].items() if v is not None}
    if any(c not in cheap_scores for c in CHEAP_COLUMNS):
        raise RuntimeError(f"cheap summary missing columns: {cheap_summary_path}")
    cheap7_value = float(cheap.get("cheap7", mean(cheap_scores[c] for c in CHEAP_COLUMNS)))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cache = summary_root / "runtime_cache" / args.target
    for k, p in {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        env[k] = str(p.resolve())

    cmd = [
        sys.executable, "-B", str(EVAL_SCRIPT),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", args.target,
        "--endpoint", args.endpoint,
        "--out-root", str(out_root),
        "--collate-root", str(collate_root),
        "--gpu", str(args.gpu),
        "--columns", "SuperGLUE",
    ]
    if args.force:
        cmd.append("--force")
    t0 = time.time()
    print(json.dumps({"event": "superglue_eval_start", "target": args.target,
                       "run_dir": rel(run_dir), "gpu": args.gpu, "cheap7": cheap7_value,
                       "chck82_overall": ref["overall"], "utc": now()}), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=7200)
    log_dir = summary_root / "logs" / args.target
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    print(json.dumps({"event": "superglue_eval_returned", "target": args.target,
                      "returncode": proc.returncode, "stdout_tail": proc.stdout[-3000:],
                      "stderr_tail": proc.stderr[-3000:]}), flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    payload_path = out_root / "per_target" / f"{args.target}.json"
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    payload = read_json(payload_path)
    sg = superglue_score(payload)
    if sg is None:
        raise RuntimeError("SuperGLUE score not found in payload")
    overall_with_aoa0 = compute_overall(cheap_scores, sg, 0.0)
    summary = {
        "status": "FASTPATH_REPLAY_SUPERGLUE_EVAL",
        "created_utc": now(),
        "target": args.target,
        "run_dir": rel(run_dir),
        "endpoint": args.endpoint,
        "cheap_summary": rel(cheap_summary_path),
        "cheap_scores": cheap_scores,
          "cheap7": cheap7_value,
        "superglue": sg,
        "aoa_assumed_for_projection": 0.0,
        "projected_overall_with_aoa0": overall_with_aoa0,
        "protected_chck82": ref,
        "deltas_vs_chck82": {
              "cheap7": float(cheap7_value - ref["cheap7"]),
            "superglue": float(sg - ref["superglue"]),
            "projected_overall_with_aoa0": float(overall_with_aoa0 - ref["overall"]),
        },
        "payload_path": rel(payload_path),
        "stdout_log": rel(log_dir / "stdout.log"),
        "stderr_log": rel(log_dir / "stderr.log"),
        "elapsed_sec": round(time.time() - t0, 1),
        "scientific_reading": "SuperGLUE completes the provisional no-AoA endpoint arithmetic for a research fast-path arm. Continuation is judged by item-retention and family turnover, not this aggregate alone.",
    }
    out_json = summary_root / f"{args.target}_superglue_summary.json"
    out_md = summary_root / f"{args.target}_superglue_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(
        f"# research fast-path SuperGLUE — {args.target}\n\n"
          f"SuperGLUE: `{sg}`; cheap7: `{cheap7_value}`.\n\n"
        f"Projected Overall with AoA=0: `{overall_with_aoa0}` "
        f"(delta vs chck82 `{overall_with_aoa0 - ref['overall']:+.6f}`).\n\n"
        f"Payload: `{rel(payload_path)}`\nJSON: `{rel(out_json)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "target": args.target,
                      "superglue": sg, "projected_overall_with_aoa0": overall_with_aoa0,
                      "delta_vs_chck82": overall_with_aoa0 - ref["overall"],
                      "summary_json": rel(out_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
