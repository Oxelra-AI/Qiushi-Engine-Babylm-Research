#!/usr/bin/env python3
"""Prepare frozen contextual cap-120 full-eval targets from the no-AoA ranking.

This helper should be run only after `rank_contextual_cap120_trajectory.py` has
created the ranking JSON.  It does not use AoA or SuperGLUE.  It selects a compact
candidate set for corrected full nine-column evaluation and pre-fills zero-shot/Reading
records from the no-AoA trajectory screen.

Selection policy:
- always include the best treatment checkpoint by equal7;
- include the matched official control at the same checkpoint;
- include the control's own best equal7 checkpoint if different, for recipe-baseline context;
- optionally include the best treatment checkpoint by recovery_R if different.
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
from typing import Any, Dict, List

WORKSPACE = _public_path('experiments/archive/compact_experience')
RANKING = _public_path('experiments/archive/compact_experience/data/contextual_cap120_trajectory_ranking.json')
PREFILL = _public_path('experiments/archive/compact_experience/scripts/prefill_contextual_cap120_eval_from_trajectory.py')
OUT = _public_path('experiments/archive/compact_experience/data/contextual_cap120_full_eval_targets.json')


def ckpt_to_m(ckpt: str) -> str:
    if not ckpt.startswith("chck_") or not ckpt.endswith("M"):
        raise ValueError(f"unexpected checkpoint name {ckpt}")
    return ckpt.replace("chck_", "")


def add_unique(xs: List[str], x: str) -> None:
    if x not in xs:
        xs.append(x)


def target_name(kind: str, ckpt: str) -> str:
    m = ckpt_to_m(ckpt)
    if kind == "treat":
        return f"context_cap120_treat_seed43022_{m}"
    if kind == "control":
        return f"context_cap120_control_seed43022_{m}"
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default=str(RANKING))
    ap.add_argument("--no_prefill", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    ranking_path = pathlib.Path(args.ranking)
    if not ranking_path.exists():
        raise SystemExit(f"missing ranking JSON: {ranking_path}")
    d: Dict[str, Any] = json.loads(ranking_path.read_text(encoding="utf-8"))
    targets: List[str] = []
    reasons: Dict[str, str] = {}

    bt = d.get("best_treatment_equal7") or {}
    if bt.get("checkpoint"):
        t = target_name("treat", str(bt["checkpoint"]))
        c = target_name("control", str(bt["checkpoint"]))
        add_unique(targets, t); reasons[t] = "best_treatment_equal7"
        add_unique(targets, c); reasons[c] = "matched_control_same_checkpoint_as_best_treatment"

    bc = d.get("best_control_equal7") or {}
    if bc.get("checkpoint"):
        c = target_name("control", str(bc["checkpoint"]))
        add_unique(targets, c); reasons.setdefault(c, "best_control_equal7")

    br = d.get("best_treatment_recovery_R") or {}
    if br.get("checkpoint"):
        t = target_name("treat", str(br["checkpoint"]))
        c = target_name("control", str(br["checkpoint"]))
        add_unique(targets, t); reasons.setdefault(t, "best_treatment_recovery_R")
        add_unique(targets, c); reasons.setdefault(c, "matched_control_same_checkpoint_as_best_recovery_R")

    payload = {
        "status": "CONTEXTUAL_CAP120_FULL_EVAL_TARGETS",
        "non_leakage_statement": "Targets selected from no-AoA BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading ranking only; SuperGLUE and AoA are measured later in corrected full eval.",
        "ranking": str(ranking_path),
        "targets": targets,
        "reasons": reasons,
        "best_treatment_equal7": bt,
        "best_control_equal7": bc,
        "best_treatment_recovery_R": br,
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.no_prefill:
        return
    if not targets:
        raise SystemExit("no contextual targets selected")
    cmd = [sys.executable, str(PREFILL), *targets, "--gpu", str(args.gpu)]
    rc = subprocess.run(cmd).returncode
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
