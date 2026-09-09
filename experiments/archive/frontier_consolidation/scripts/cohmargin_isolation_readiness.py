#!/usr/bin/env python3
"""research readiness helper for coherence-margin pilot and lambda-zero isolate.

CPU-only. It reads existing pilot/reference/lambda-zero artifacts if present,
prints exact minimal commands for missing cheap7 evaluations and the lambda-zero
same-charge arm, and summarizes what comparisons are scientifically valid.
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
OUT = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_isolation_readiness')
EVAL_SCRIPT = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
TRAINER = _public_path('experiments/archive/frontier_consolidation/scripts/coherence_margin_trainer.py')
PROBE = _public_path('experiments/archive/frontier_consolidation/scripts/cohmargin_nll_gap_probe.py')
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = "BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading"

ENDPOINTS = {
    "cohmargin4M_lambda0p10": {
        "target": "cohmargin4M_scale1p75_seed43022",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_scale1p75_seed43022'),
        "endpoint": "final",
        "out_root": _public_path('experiments/archive/frontier_consolidation/data/cohmargin4M_eval'),
        "collate_root": _public_path('experiments/archive/frontier_consolidation/data/cohmargin4M_collate'),
        "role": "lambda>0 pilot under evaluation",
    },
    "scale1p75_chck4M_ref": {
        "target": "scale1p75_standard_chck4M_ref",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder'),
        "endpoint": "chck_4M",
        "out_root": _public_path('experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_eval'),
        "collate_root": _public_path('experiments/archive/frontier_consolidation/data/scale1p75_chck4M_ref_collate'),
        "role": "ordinary charged-word reference, not sufficient isolate",
    },
    "cohmargin4M_lambda0": {
        "target": "cohmargin4M_lambda0_scale1p75_seed43022",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/cohmargin4M_lambda0_scale1p75_seed43022'),
        "endpoint": "final",
        "out_root": _public_path('experiments/archive/frontier_consolidation/data/cohmargin4M_lambda0_eval'),
        "collate_root": _public_path('experiments/archive/frontier_consolidation/data/cohmargin4M_lambda0_collate'),
        "role": "same-charge same-row no-margin-gradient isolate",
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


def scores_from_payload(payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = payload.get("tasks", {})
    scores: dict[str, float | None] = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        scores[c] = float(rec["score"]) if isinstance(rec, dict) and rec.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if isinstance(rec, dict) and rec.get("score") is not None:
            gp.append(float(rec["score"]))
    scores["GlobalPIQA"] = float(mean(gp)) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd, dict) and isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        scores["Reading"] = float(rd["scores"]["Reading"])
    elif isinstance(rd, dict) and rd.get("score") is not None:
        scores["Reading"] = float(rd["score"])
    else:
        scores["Reading"] = None
    return scores


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def train_lambda0_command() -> str:
    return (
        "PYTHONDONTWRITEBYTECODE=1 python -B "
        f"{rel(TRAINER)} --output_dir {rel(ENDPOINTS['cohmargin4M_lambda0']['run_dir'])} "
        "--max_word_exposure 4000000 --view_charge_multiplier 2.0 --checkpoint_words 1000000 "
        "--micro_batch_size 128 --grad_accum_steps 2 --adapter_scale 1.75 "
        "--margin_lambda 0.0 --margin 0.20 --disrupt_span_tokens 8 --lr_total_steps 1265 "
        "--seed 43 --train_rng_seed 43022 --gpu <free_gpu>"
    )


def eval_command(cfg: dict[str, Any]) -> str:
    return (
        "PYTHONDONTWRITEBYTECODE=1 python -B "
        f"{rel(EVAL_SCRIPT)} --arm reinvest --run-dir {rel(cfg['run_dir'])} "
        f"--target {cfg['target']} --endpoint {cfg['endpoint']} "
        f"--out-root {rel(cfg['out_root'])} --collate-root {rel(cfg['collate_root'])} "
        f"--gpu <free_gpu> --columns {EVAL_COLUMNS}"
    )


def probe_command(name: str, holdout: bool = False) -> str:
    cfg = ENDPOINTS[name]
    suffix = "holdout64_after2M" if holdout else "train64"
    out_dir = WORKSPACE / f"data/cohmargin_nll_gap_{name}_{suffix}"
    skip = " --skip-coherent-words 1999862" if holdout else ""
    return (
        "PYTHONDONTWRITEBYTECODE=1 python -B "
        f"{rel(PROBE)} --label {name}_{suffix} --model-path {rel(cfg['run_dir'] / 'hf_model' / cfg['endpoint'])} "
        f"--out-dir {rel(out_dir)}{skip} --num-rows 64 --batch-size 8 --device cpu --probe-seed 15900"
    )


def endpoint_record(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    model_dir = cfg["run_dir"] / "hf_model" / cfg["endpoint"]
    payload = cfg["out_root"] / "per_target" / f"{cfg['target']}.json"
    rec: dict[str, Any] = {
        "role": cfg["role"],
        "run_dir": rel(cfg["run_dir"]),
        "model_dir": rel(model_dir),
        "model_exists": model_dir.exists(),
        "payload_path": rel(payload),
        "payload_exists": payload.exists(),
        "eval_command_not_run": eval_command(cfg),
    }
    if payload.exists():
        sc = scores_from_payload(read_json(payload))
        rec["scores"] = sc
        rec["cheap7"] = cheap7(sc)
        rec["cheap7_complete"] = rec["cheap7"] is not None
    else:
        rec["cheap7_complete"] = False
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {name: endpoint_record(name, cfg) for name, cfg in ENDPOINTS.items()}
    comparisons: dict[str, Any] = {}
    if rows["cohmargin4M_lambda0p10"].get("cheap7") is not None and rows["scale1p75_chck4M_ref"].get("cheap7") is not None:
        comparisons["pilot_minus_ordinary_chck4M"] = {
            "cheap7_delta": float(rows["cohmargin4M_lambda0p10"]["cheap7"] - rows["scale1p75_chck4M_ref"]["cheap7"]),
            "note": "This contrast mixes margin gradient with half as many coherent words and is not sufficient to attribute signal.",
        }
    if rows["cohmargin4M_lambda0p10"].get("cheap7") is not None and rows["cohmargin4M_lambda0"].get("cheap7") is not None:
        a = rows["cohmargin4M_lambda0p10"]["scores"]
        b = rows["cohmargin4M_lambda0"]["scores"]
        comparisons["pilot_minus_lambda0_isolate"] = {
            "cheap7_delta": float(rows["cohmargin4M_lambda0p10"]["cheap7"] - rows["cohmargin4M_lambda0"]["cheap7"]),
            "column_deltas": {c: float(a[c] - b[c]) for c in CHEAP},
            "note": "This is the attribution-relevant official-score contrast.",
        }
    commands = []
    if not rows["cohmargin4M_lambda0"]["model_exists"]:
        commands.append({"purpose": "train lambda-zero isolate if pilot score warrants it", "command": train_lambda0_command()})
    for name, rec in rows.items():
        if rec["model_exists"] and not rec["payload_exists"]:
            commands.append({"purpose": f"evaluate cheap7 for {name}", "command": rec["eval_command_not_run"]})
    for name, rec in rows.items():
        if rec["model_exists"] and not rec.get("cheap7_complete", False):
            commands.append({"purpose": f"evaluate/complete cheap7 for {name}", "command": rec["eval_command_not_run"]})
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "endpoints": rows,
        "comparisons": comparisons,
        "commands_not_run": commands,
        "scientific_rule": "An encouraging coherence-margin pilot score cannot be escalated to 20M until the same-charge same-row lambda-zero arm shows the official movement and NLL separation are caused by the margin gradient rather than exposure/RNG/disruption geometry.",
    }
    js = _public_path('experiments/archive/frontier_consolidation/data/cohmargin_isolation_readiness/cohmargin_isolation_readiness.json')
    md = _public_path('research/documents/frontier_consolidation/data/cohmargin_isolation_readiness/cohmargin_isolation_readiness.md')
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research coherence-margin isolation readiness",
        "",
        f"Status: **{out['status']}**",
        "",
        "| endpoint | role | model exists | payload exists | cheap7 complete | cheap7 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, rec in rows.items():
        lines.append(f"| {name} | {rec['role']} | {rec['model_exists']} | {rec['payload_exists']} | {rec.get('cheap7_complete', False)} | {rec.get('cheap7','NA')} |")
    if comparisons:
        lines += ["", "## Existing comparisons", ""]
        for name, comp in comparisons.items():
            lines += [f"### {name}", "", "```json", json.dumps(comp, indent=2, ensure_ascii=False), "```", ""]
    lines += ["", "## Commands not run", ""]
    for c in commands:
        lines += [f"### {c['purpose']}", "", "```bash", c["command"], "```", ""]
    lines += ["", out["scientific_rule"], "", f"JSON: `{rel(js)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md), "n_commands_not_run": len(commands), "comparison_keys": list(comparisons)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
