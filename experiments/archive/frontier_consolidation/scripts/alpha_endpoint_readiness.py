#!/usr/bin/env python3
"""research helper: integrate alpha0.5/0.75 SuperGLUE results when present.

This CPU-only helper reads existing cheap7 summaries and optional SuperGLUE-only
summaries for private-scale endpoints, computes exact Overall(AoA0), and emits the
safe carrier-materialization commands. It does not run materialization, upload, or
submit anything.
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
OUT = _public_path('experiments/archive/frontier_consolidation/data/alpha_endpoint_readiness')
CHCK82 = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
CHEAP_COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ENDPOINTS = {
    "alpha0p5": {
        "label": "coherent86_alpha0p5",
        "panel_arm": "coherent86_private_alpha0p5",
        "model_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/coherent86_private_scale_0p5/hf_model/final'),
        "cheap_summary": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/coherent86_private_scale_0p5_summary.json'),
        "cheap_payload": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json'),
        "superglue_summary_candidates": [
            _public_path('experiments/archive/frontier_consolidation/data/private_scale_superglue_summary/coherent86_private_scale_0p5_sg_superglue_summary.json'),
        ],
        "superglue_payload_candidates": [
            _public_path('experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5_sg.json'),
        ],
    },
    "alpha0p75": {
        "label": "coherent86_alpha0p75",
        "panel_arm": "coherent86_private_alpha0p75",
        "model_dir": _public_path('models/frontier'),
        "cheap_summary": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/coherent86_private_scale_0p75_summary.json'),
        "cheap_payload": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json'),
        "superglue_summary_candidates": [
            _public_path('experiments/archive/frontier_consolidation/data/private_scale_superglue_summary_alpha0p75/coherent86_private_scale_0p75_sg_retry_superglue_summary.json'),
            _public_path('experiments/archive/frontier_consolidation/data/private_scale_superglue_summary/coherent86_private_scale_0p75_sg_superglue_summary.json'),
        ],
        "superglue_payload_candidates": [
            _public_path('experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json'),
            _public_path('experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg.json'),
        ],
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


def first_existing(paths: list[pathlib.Path]) -> pathlib.Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = read_json(CHCK82)
    protected_scores = {k: float(v) for k, v in protected["score_arithmetic"]["scores"].items() if v is not None}
    protected_overall = float(protected["score_arithmetic"]["overall_reported"])
    protected_cheap7 = float(mean(protected_scores[c] for c in CHEAP_COLS))
    rows = {}
    commands = {}
    for name, cfg in ENDPOINTS.items():
        cheap = read_json(cfg["cheap_summary"])
        cheap_scores = {k: float(v) for k, v in cheap["scores"].items() if v is not None}
        cheap7 = float(mean(cheap_scores[c] for c in CHEAP_COLS))
        sg_summary_path = first_existing(cfg["superglue_summary_candidates"])
        sg_payload_path = first_existing(cfg["superglue_payload_candidates"])
        row = {
            "status": "PENDING_SUPERGLUE" if sg_summary_path is None or sg_payload_path is None else "SUPERGLUE_AVAILABLE",
            "label": cfg["label"],
            "panel_arm": cfg["panel_arm"],
            "model_dir": rel(cfg["model_dir"]),
            "cheap_summary": rel(cfg["cheap_summary"]),
            "cheap_payload": rel(cfg["cheap_payload"]),
            "cheap7": cheap7,
            "cheap7_delta_vs_chck82": cheap7 - protected_cheap7,
            "scores_cheap": cheap_scores,
            "superglue_summary": None if sg_summary_path is None else rel(sg_summary_path),
            "superglue_payload": None if sg_payload_path is None else rel(sg_payload_path),
        }
        if sg_summary_path is not None and sg_payload_path is not None:
            sg = float(read_json(sg_summary_path)["superglue"])
            overall = float(mean([*(cheap_scores[c] for c in CHEAP_COLS), sg, 0.0]))
            row.update({
                "superglue": sg,
                "superglue_delta_vs_chck82": sg - protected_scores["SuperGLUE"],
                "overall_with_aoa0": overall,
                "overall_delta_vs_chck82": overall - protected_overall,
            })
            cmd = (
                "PYTHONDONTWRITEBYTECODE=1 python -B "
                "experiments/archive/frontier_consolidation/scripts/materialize_truthful_private_scale_carrier.py "
                f"--label {cfg['label']} "
                f"--panel-arm {cfg['panel_arm']} "
                f"--model-dir {rel(cfg['model_dir'])} "
                f"--cheap-payload {rel(cfg['cheap_payload'])} "
                f"--cheap-summary {rel(cfg['cheap_summary'])} "
                f"--superglue-payload {rel(sg_payload_path)} "
                f"--superglue-summary {rel(sg_summary_path)}"
            )
            commands[name] = cmd
        rows[name] = row
    out = {
        "status": "COMPLETE",
        "created_utc": now(),
        "protected_chck82": {"overall": protected_overall, "cheap7": protected_cheap7, "superglue": protected_scores["SuperGLUE"], "source": rel(CHCK82)},
        "rows": rows,
        "carrier_materialization_commands_not_run": commands,
        "policy": "This helper never uploads or submits. It only computes endpoint arithmetic and prints local carrier materialization commands when SuperGLUE summaries exist.",
    }
    js = _public_path('experiments/archive/frontier_consolidation/data/alpha_endpoint_readiness/alpha_endpoint_readiness.json')
    md = _public_path('research/documents/frontier_consolidation/data/alpha_endpoint_readiness/alpha_endpoint_readiness.md')
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research alpha endpoint readiness",
        "",
        f"Status: **{out['status']}**",
        f"Protected chck82 Overall `{protected_overall}`; cheap7 `{protected_cheap7}`; SuperGLUE `{protected_scores['SuperGLUE']}`.",
        "",
        "| endpoint | status | cheap7 | SG | Overall(AoA0) | delta vs chck82 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['status']} | {r['cheap7']:.12g} | {r.get('superglue','NA')} | {r.get('overall_with_aoa0','NA')} | {r.get('overall_delta_vs_chck82','NA')} |")
    lines += ["", "## Materialization commands (not run)", ""]
    if commands:
        for name, cmd in commands.items():
            lines += [f"### {name}", "", "```bash", cmd, "```", ""]
    else:
        lines.append("No endpoint has both cheap and SuperGLUE payloads available yet.")
    lines += ["", out["policy"], "", f"JSON: `{rel(js)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md), "ready_commands": list(commands)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
