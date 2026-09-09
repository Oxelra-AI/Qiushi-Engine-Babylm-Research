#!/usr/bin/env python3
"""research: cheap7 evaluation for ordinary scale-1.75 chck_86M.

This no-training script evaluates the existing ordinary scale-1.75 checkpoint at the
same exposure coordinate as the frozen-82M + 4M private fast-path arms.  It should be
run only if coherent fast-path remains competitive after SuperGLUE, because its purpose
is to distinguish protected fast-path learning from ordinary extra exposure and from
merely avoiding a destructive spanbreak control.
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
EVAL_SCRIPT = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
CHCK82_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
COHERENT_CHEAP_SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_coherent_summary/fastpath4M_coherent_summary.json')
COHERENT_SUPERGLUE_SUMMARY = _public_path('experiments/archive/representation_and_objectives/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json')
SCALE_RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder')
METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/scientific_metrics.json')
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]


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
        "cheap7": mean(scores[c] for c in CHEAP_COLS),
        "superglue": scores["SuperGLUE"],
        "aoa": scores["AoA"],
        "overall": float(j["score_arithmetic"]["overall_reported"]),
        "source": rel(CHCK82_VERIFY),
    }


def checkpoint_record(endpoint: str) -> dict[str, Any] | None:
    if not METRICS.exists():
        return None
    m = read_json(METRICS)
    for rec in m.get("saved_checkpoints", []):
        if rec.get("name") == endpoint:
            return rec
    return None


def optional_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    return read_json(path)


def official_scores(payload: dict[str, Any]) -> dict[str, float]:
    scores = payload.get("official_overall", {}).get("scores", {})
    out = {}
    for c in CHEAP_COLS:
        if scores.get(c) is None:
            raise RuntimeError(f"Missing score {c} in payload")
        out[c] = float(scores[c])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(SCALE_RUN))
    ap.add_argument("--target", default="scale1p75_chck86_cheap7")
    ap.add_argument("--endpoint", default="chck_86M")
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
    out_root.mkdir(parents=True, exist_ok=True)
    collate_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    ref = chck82_ref()
    chk = checkpoint_record(args.endpoint)
    coherent = optional_json(COHERENT_CHEAP_SUMMARY)
    coherent_sg = optional_json(COHERENT_SUPERGLUE_SUMMARY)

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
        "--columns", *EVAL_COLUMNS,
    ]
    if args.force:
        cmd.append("--force")

    t0 = time.time()
    print(json.dumps({
        "event": "ordinary86_cheap7_eval_start",
        "target": args.target,
        "run_dir": rel(run_dir),
        "endpoint": args.endpoint,
        "checkpoint_record": chk,
        "gpu": args.gpu,
        "scientific_purpose": "exposure-matched ordinary continuation comparator for coherent frozen-anchor replay",
        "utc": now(),
    }), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=7200)
    log_dir = summary_root / "logs" / args.target
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    print(json.dumps({
        "event": "ordinary86_cheap7_eval_returned",
        "target": args.target,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-3000:],
    }), flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    payload_path = out_root / "per_target" / f"{args.target}.json"
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    payload = read_json(payload_path)
    scores = official_scores(payload)
    cheap7 = float(mean(scores[c] for c in CHEAP_COLS))
    coherent_scores = coherent.get("scores") if isinstance(coherent, dict) else None
    coherent_cheap7 = float(coherent["cheap7"]) if isinstance(coherent, dict) and coherent.get("cheap7") is not None else None
    coherent_projected = None
    if isinstance(coherent_sg, dict) and coherent_sg.get("projected_overall_with_aoa0") is not None:
        coherent_projected = float(coherent_sg["projected_overall_with_aoa0"])

    deltas_vs_coherent = None
    if coherent_scores and coherent_cheap7 is not None:
        deltas_vs_coherent = {c: float(scores[c] - float(coherent_scores[c])) for c in CHEAP_COLS}
        deltas_vs_coherent["cheap7"] = float(cheap7 - coherent_cheap7)

    ordinary_projected_if_chck82_superglue_aoa = None
    ordinary_projected_if_coherent_superglue_aoa0 = None
    if ref.get("superglue") is not None:
        ordinary_projected_if_chck82_superglue_aoa = float((sum(scores[c] for c in CHEAP_COLS) + float(ref["superglue"]) + 0.0) / 9.0)
    if isinstance(coherent_sg, dict) and coherent_sg.get("superglue") is not None:
        ordinary_projected_if_coherent_superglue_aoa0 = float((sum(scores[c] for c in CHEAP_COLS) + float(coherent_sg["superglue"]) + 0.0) / 9.0)

    summary = {
        "status": "SCALE1P75_CHCK86_CHEAP7_EVAL",
        "created_utc": now(),
        "target": args.target,
        "run_dir": rel(run_dir),
        "endpoint": args.endpoint,
        "model_path": rel(run_dir / "hf_model" / args.endpoint),
        "checkpoint_record_from_scientific_metrics": chk,
        "scores": scores,
        "cheap7": cheap7,
        "protected_chck82": ref,
        "deltas_vs_chck82": {c: float(scores[c] - float(ref["scores"][c])) for c in CHEAP_COLS} | {"cheap7": float(cheap7 - ref["cheap7"])},
        "coherent4M_reference": {
            "summary_path": rel(COHERENT_CHEAP_SUMMARY),
            "exists": bool(coherent),
            "cheap7": coherent_cheap7,
            "projected_overall_with_aoa0": coherent_projected,
            "superglue_summary": rel(COHERENT_SUPERGLUE_SUMMARY),
            "superglue_exists": bool(coherent_sg),
        },
        "deltas_vs_coherent4M": deltas_vs_coherent,
        "ordinary86_projected_overall_if_chck82_superglue_and_aoa0": ordinary_projected_if_chck82_superglue_aoa,
        "ordinary86_projected_overall_if_coherent_superglue_and_aoa0": ordinary_projected_if_coherent_superglue_aoa0,
        "payload_path": rel(payload_path),
        "stdout_log": rel(log_dir / "stdout.log"),
        "stderr_log": rel(log_dir / "stderr.log"),
        "elapsed_sec": round(time.time() - t0, 1),
        "scientific_reading": "Ordinary chck_86M is the exposure-matched no-training comparator for coherent fast-path replay. If it matches or exceeds coherent on the same item transitions, the coherent effect is ordinary extra exposure or private-tail redistribution rather than protected fast-path learning.",
    }
    out_json = summary_root / f"{args.target}_summary.json"
    out_md = summary_root / f"{args.target}_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_lines = [
        "# research ordinary scale1.75 chck_86M cheap7",
        "",
        f"Endpoint: `{rel(run_dir / 'hf_model' / args.endpoint)}`",
        f"Checkpoint record: `{chk}`",
        "",
        f"Cheap7: `{cheap7}` (delta vs protected chck82 `{cheap7 - ref['cheap7']:+.6f}`).",
        "",
        "| column | ordinary86 | chck82 | Δ vs chck82 | coherent4M | Δ vs coherent |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in CHEAP_COLS:
        co_v = float(coherent_scores[c]) if coherent_scores else None
        co_delta = None if co_v is None else scores[c] - co_v
        md_lines.append(f"| {c} | {scores[c]} | {ref['scores'][c]} | {scores[c]-ref['scores'][c]:+.3f} | {co_v} | {'' if co_delta is None else f'{co_delta:+.3f}'} |")
    md_lines += [
        "",
        f"Payload: `{rel(payload_path)}`",
        f"JSON: `{rel(out_json)}`",
    ]
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "target": args.target,
        "cheap7": cheap7,
        "cheap7_delta_vs_chck82": float(cheap7 - ref["cheap7"]),
        "cheap7_delta_vs_coherent4M": None if coherent_cheap7 is None else float(cheap7 - coherent_cheap7),
        "summary_json": rel(out_json),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
