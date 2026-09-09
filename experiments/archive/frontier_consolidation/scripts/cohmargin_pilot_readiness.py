#!/usr/bin/env python3
"""Readiness and comparison helper for the research coherence-margin 4M pilot.

This helper is CPU-only. It does not train or evaluate. It records whether the
pilot and matched standard scale1.75 chck_4M reference have cheap7 payloads, emits
exact isolated evaluation commands when absent, and computes matched deltas when
both payloads exist.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_pilot_readiness')
EVAL_SCRIPT = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = "BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading"

ENDPOINTS = {
    "cohmargin4M": {
        "target": "cohmargin4M_scale1p75_seed43022",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022'),
        "endpoint": "final",
        "out_root": _public_path('experiments/archive/frontier_consolidation/data/cohmargin4M_eval'),
        "collate_root": _public_path('experiments/archive/frontier_consolidation/data/cohmargin4M_collate'),
        "description": "coherence-margin MLM pilot, 4M charged including disrupted views",
    },
    "scale1p75_chck4M_ref": {
        "target": "scale1p75_standard_chck4M_ref",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder'),
        "endpoint": "chck_4M",
        "out_root": _public_path('experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_eval'),
        "collate_root": _public_path('experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_collate'),
        "description": "matched standard scale1.75 checkpoint at 4M charged/coherent exposure",
    },
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    q = pathlib.Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    scores: dict[str, float | None] = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        scores[c] = float(rec["score"]) if rec.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if rec.get("score") is not None:
            gp.append(float(rec["score"]))
    scores["GlobalPIQA"] = float(mean(gp)) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        scores["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        scores["Reading"] = float(rd["score"])
    else:
        scores["Reading"] = None
    return scores


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def endpoint_record(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    model_dir = cfg["run_dir"] / "hf_model" / cfg["endpoint"]
    payload = cfg["out_root"] / "per_target" / f"{cfg['target']}.json"
    command = (
        "PYTHONDONTWRITEBYTECODE=1 python -B "
        f"{rel(EVAL_SCRIPT)} --arm reinvest --run-dir {rel(cfg['run_dir'])} "
        f"--target {cfg['target']} --endpoint {cfg['endpoint']} "
        f"--out-root {rel(cfg['out_root'])} --collate-root {rel(cfg['collate_root'])} "
        f"--gpu <free_gpu> --columns {EVAL_COLUMNS}"
    )
    rec: dict[str, Any] = {
        "description": cfg["description"],
        "target": cfg["target"],
        "run_dir": rel(cfg["run_dir"]),
        "endpoint": cfg["endpoint"],
        "model_dir": rel(model_dir),
        "model_exists": model_dir.exists(),
        "payload_path": rel(payload),
        "payload_exists": payload.exists(),
        "eval_command_not_run": command,
    }
    if payload.exists():
        payload_obj = read_json(payload)
        scores = extract_scores(payload_obj)
        rec.update({"scores": scores, "cheap7": cheap7(scores)})
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {name: endpoint_record(name, cfg) for name, cfg in ENDPOINTS.items()}
    comparison = None
    if rows["cohmargin4M"].get("cheap7") is not None and rows["scale1p75_chck4M_ref"].get("cheap7") is not None:
        a = rows["cohmargin4M"]["scores"]
        b = rows["scale1p75_chck4M_ref"]["scores"]
        comparison = {
            "cohmargin_minus_standard_ref": {c: float(a[c] - b[c]) for c in CHEAP},
            "cheap7_delta": float(rows["cohmargin4M"]["cheap7"] - rows["scale1p75_chck4M_ref"]["cheap7"]),
            "decision_reading": "Continue only if not broadly destructive and movement is relation/state-relevant, not COMPS/BLiMP or few-GlobalPIQA redistribution.",
        }
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "rows": rows,
        "comparison": comparison,
        "policy": "CPU-only helper; no training/evaluation/upload/submission. Use eval commands after managed pilot finishes and a GPU is free.",
    }
    js = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_pilot_readiness/cohmargin_pilot_readiness.json')
    md = _public_path('research/documents/frontier_consolidation/data/cohmargin_pilot_readiness/cohmargin_pilot_readiness.md')
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research coherence-margin pilot readiness",
        "",
        f"Status: **{out['status']}**",
        "",
        "| endpoint | model exists | payload exists | cheap7 |",
        "|---|---:|---:|---:|",
    ]
    for name, rec in rows.items():
        lines.append(f"| {name} | {rec['model_exists']} | {rec['payload_exists']} | {rec.get('cheap7','NA')} |")
    lines += ["", "## Evaluation commands (not run)", ""]
    for name, rec in rows.items():
        if not rec["payload_exists"]:
            lines += [f"### {name}", "", "```bash", rec["eval_command_not_run"], "```", ""]
    if comparison is not None:
        lines += ["## Matched comparison", "", f"Cheap7 delta cohmargin - standard: `{comparison['cheap7_delta']}`", "", "| column | delta |", "|---|---:|"]
        for c, v in comparison["cohmargin_minus_standard_ref"].items():
            lines.append(f"| {c} | {v:+.6f} |")
    lines += ["", out["policy"], "", f"JSON: `{rel(js)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md), "comparison_ready": comparison is not None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
