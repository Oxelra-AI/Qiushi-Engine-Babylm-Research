#!/usr/bin/env python3
"""research: no-training adapter residual amplitude readout on key official-compatible columns.

Scientific purpose
------------------
The matched-horizon adapter128 20M checkpoint gained BLiMP/Supplement/COMPS but
lost EWoK, GlobalPIQA, and Reading. research showed GlobalPIQA damage is mostly the
direct adapter residual output, while EWoK damage is mostly the changed stock
trajectory. This script maps *inference-only* adapter output scale on the already
trained live checkpoint. It does not update weights and is not a submission-time
scorer. If an interior scale recovers GlobalPIQA/Reading without further damaging
EWoK while preserving Supplement, then future architecture work should regulate
residual amplitude during training; if no scale helps, ordinary adapter residuals
are weaker as a route.

Boundary: this uses official-compatible evaluation rows only as a mechanistic
readout of an existing 20M checkpoint. Any scale found here is not a submission
hyperparameter unless later chosen from corpus-derived validation and retrained.
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
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Optional

USER_ROOT = _public_path('.')
A01 = _public_path('experiments/archive/representation_and_objectives')
A02 = _public_path('experiments/archive/frontier_consolidation')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
A02_SHADOW = _public_path('experiments/archive/frontier_consolidation/data/adapter_scale_sweep/shadow_runs')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/adapter_scale_key_columns/eval')
COLLATE_ROOT = _public_path('experiments/archive/representation_and_objectives/data/adapter_scale_key_columns/collate')
SUMMARY_DIR = _public_path('experiments/archive/representation_and_objectives/data/adapter_scale_key_columns')

COLUMNS = ["Supplement", "EWoK", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]
KEY_COLUMNS = ["Supplement", "EWoK", "GlobalPIQA", "Reading"]
ANCHOR_STEP35 = {"Supplement": 55.45, "EWoK": 50.73, "GlobalPIQA": 34.195, "Reading": 8.67}
LIVE_SCALE1 = {"Supplement": 56.23, "EWoK": 49.49, "GlobalPIQA": 32.225, "Reading": 8.31}
SCALE0_STEP139 = {"Supplement": 56.12, "EWoK": 48.98, "GlobalPIQA": 34.165, "Reading": 8.38}


def scale_tag(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p").replace("-", "m")


def shadow_run(scale: float) -> Path:
    return A02_SHADOW / f"adapter128_scale_{scale_tag(scale)}_from_live20M"


def target_name(scale: float) -> str:
    return f"adapter128_scale_{scale_tag(scale)}_from_live20M_keycols"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_scores(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    tasks = payload.get("tasks", {})
    out: Dict[str, Optional[float]] = {}
    for c in ["Supplement", "EWoK"]:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if rec.get("score") is not None else None
    gp_vals = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if rec.get("score") is not None:
            out[c] = float(rec["score"])
            gp_vals.append(float(rec["score"]))
        else:
            out[c] = None
    out["GlobalPIQA"] = float(mean(gp_vals)) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def key_mean(scores: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [scores.get(c) for c in KEY_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def run_eval(scale: float, gpu: int, force: bool = False) -> Dict[str, Any]:
    rd = shadow_run(scale)
    if not (rd / "hf_model/chck_20M/model.safetensors").exists():
        raise FileNotFoundError(f"missing shadow checkpoint for scale {scale}: {rd}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)
    target = target_name(scale)
    per_target = _public_path('experiments/archive/representation_and_objectives/data/adapter_scale_key_columns/eval/per_target') / f"{target}.json"
    if per_target.exists() and not force:
        payload = json.loads(per_target.read_text(encoding="utf-8"))
        scores = extract_scores(payload)
        return {
            "scale": float(scale),
            "tag": scale_tag(scale),
            "target": target,
            "source": "existing_step140_per_target",
            "returncode": 0,
            "per_target": str(per_target),
            "scores": scores,
            "key4": key_mean(scores),
        }
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", "reinvest",
        "--run-dir", str(rd),
        "--target", target,
        "--endpoint", "chck_20M",
        "--out-root", str(OUT_ROOT),
        "--collate-root", str(COLLATE_ROOT),
        "--gpu", str(gpu),
        "--columns", *COLUMNS,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_dir = _public_path('experiments/archive/representation_and_objectives/data/adapter_scale_key_columns/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{target}_stdout.log"
    stderr_log = log_dir / f"{target}_stderr.log"
    print(json.dumps({"event": "scale_eval_start", "scale": scale, "target": target, "gpu": gpu, "utc": now(), "columns": COLUMNS}), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=2400)
    elapsed = time.time() - t0
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    rec: Dict[str, Any] = {
        "scale": float(scale),
        "tag": scale_tag(scale),
        "target": target,
        "run_dir": str(rd),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    if proc.returncode != 0:
        rec["stdout_tail"] = proc.stdout[-2000:]
        rec["stderr_tail"] = proc.stderr[-3000:]
        print(json.dumps({"event": "scale_eval_failed", **rec}, ensure_ascii=False), flush=True)
        return rec
    if not per_target.exists():
        rec["error"] = f"missing per_target {per_target}"
        print(json.dumps({"event": "scale_eval_missing_json", **rec}, ensure_ascii=False), flush=True)
        return rec
    payload = json.loads(per_target.read_text(encoding="utf-8"))
    scores = extract_scores(payload)
    rec["per_target"] = str(per_target)
    rec["scores"] = scores
    rec["key4"] = key_mean(scores)
    print(json.dumps({"event": "scale_eval_done", "scale": scale, "key4": rec["key4"], "scores": scores, "elapsed_sec": round(elapsed, 1)}, ensure_ascii=False), flush=True)
    return rec


def augment_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    scores = rec.get("scores") or {}
    if rec.get("key4") is None and scores:
        rec["key4"] = key_mean(scores)
    anchor_key = mean(ANCHOR_STEP35[c] for c in KEY_COLUMNS)
    live_key = mean(LIVE_SCALE1[c] for c in KEY_COLUMNS)
    rec["delta_vs_step35"] = {c: (None if scores.get(c) is None else float(scores[c] - ANCHOR_STEP35[c])) for c in KEY_COLUMNS}
    rec["delta_vs_live_scale1"] = {c: (None if scores.get(c) is None else float(scores[c] - LIVE_SCALE1[c])) for c in KEY_COLUMNS}
    if rec.get("key4") is not None:
        rec["delta_vs_step35"]["key4"] = float(rec["key4"] - anchor_key)
        rec["delta_vs_live_scale1"]["key4"] = float(rec["key4"] - live_key)
    return rec


def built_in_record(scale: float, scores: Dict[str, float], source: str) -> Dict[str, Any]:
    rec = {"scale": float(scale), "tag": scale_tag(scale), "source": source, "returncode": 0, "scores": dict(scores)}
    rec["key4"] = key_mean(rec["scores"])
    return augment_record(rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.25, 1.5])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    records = [
        built_in_record(0.0, SCALE0_STEP139, "live_adapter_disabled_at_inference"),
        built_in_record(1.0, LIVE_SCALE1, "a02_step103_live_enabled_reference"),
    ]
    for scale in args.scales:
        records.append(augment_record(run_eval(float(scale), int(args.gpu), args.force)))
    records = sorted(records, key=lambda r: float(r["scale"]))
    successes = [r for r in records if r.get("returncode") == 0 and r.get("key4") is not None]
    best_key4 = max(successes, key=lambda r: r["key4"]) if successes else None
    best_gp = max(successes, key=lambda r: (r.get("scores") or {}).get("GlobalPIQA", -1e9)) if successes else None
    anchor_key = mean(ANCHOR_STEP35[c] for c in KEY_COLUMNS)
    live_key = mean(LIVE_SCALE1[c] for c in KEY_COLUMNS)
    interpretation = []
    if best_key4:
        if best_key4["key4"] > anchor_key:
            interpretation.append("An inference-only adapter scale beats the research key4 anchor; do not use this tuned scale for submission, but amplitude control is a serious construction route.")
        elif best_key4["key4"] > live_key:
            interpretation.append("Some scale is better than live scale=1 on the harmed key columns but still below the research key4 anchor; amplitude control can reduce damage but does not rescue ordinary adapters by itself at 20M.")
        else:
            interpretation.append("No scale improves over live scale=1 on the key columns; post-hoc amplitude scaling gives no rescue signal.")
        if 0.0 < float(best_key4["scale"]) < 1.0 and best_key4["key4"] > live_key:
            interpretation.append("The best key-column point is subunit and interior, pointing to residual amplitude regulation during training rather than width expansion.")
    if best_gp and float(best_gp["scale"]) <= 0.25:
        interpretation.append("GlobalPIQA is maximized near zero adapter output, consistent with research's direct-residual damage decomposition.")
    out = {
        "status": "ADAPTER_SCALE_KEY_COLUMNS_DONE",
        "created_utc": now(),
        "boundary": "No-training mechanistic readout on official-compatible rows; not a submission hyperparameter selection rule.",
        "columns_evaluated_for_new_scales": COLUMNS,
        "key_columns": KEY_COLUMNS,
        "anchor_step35_key4": anchor_key,
        "live_scale1_key4": live_key,
        "records": records,
        "best_key4": best_key4,
        "best_globalpiqa": best_gp,
        "interpretation": interpretation,
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/adapter_scale_key_columns/adapter_scale_key_columns_summary.json')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md = _public_path('research/documents/representation_and_objectives/data/adapter_scale_key_columns/adapter_scale_key_columns_summary.md')
    lines = [
        "# research — adapter residual amplitude key-column readout",
        "",
        "Inference-only scaling of A02's already-trained live adapter128 20M checkpoint. This is a route readout, not a submission tuning rule.",
        "",
        f"research key4 anchor (Supplement/EWoK/GlobalPIQA/Reading): {anchor_key:.4f}; live scale=1 key4: {live_key:.4f}.",
        "",
        "| scale | key4 | Δkey4 vs research | Supp | EWoK | GPpar | GPnon | GP | Reading |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        sc = r.get("scores") or {}
        d = (r.get("delta_vs_step35") or {}).get("key4")
        def fmt(x: Any) -> str:
            return "" if x is None else f"{float(x):.3f}"
        lines.append(
            f"| {float(r['scale']):.2f} | {fmt(r.get('key4'))} | {'' if d is None else f'{float(d):+.3f}'} | "
            f"{fmt(sc.get('Supplement'))} | {fmt(sc.get('EWoK'))} | {fmt(sc.get('GlobalPIQA_parallel'))} | {fmt(sc.get('GlobalPIQA_nonparallel'))} | {fmt(sc.get('GlobalPIQA'))} | {fmt(sc.get('Reading'))} |"
        )
    lines += ["", "## Interpretation"] + [f"- {x}" for x in interpretation]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "summary_json": str(out_json), "summary_md": str(out_md), "best_key4_scale": None if not best_key4 else best_key4.get("scale"), "best_key4": None if not best_key4 else best_key4.get("key4")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
