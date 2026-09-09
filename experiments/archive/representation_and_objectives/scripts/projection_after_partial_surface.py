#!/usr/bin/env python3
"""research: projection helper after a partial corrected-tokenizer evaluation surface.

No model loading. Given either a pristine-collate summary, a JSON containing score
fields, or command-line scores, compute whether the already-known columns leave a
credible path to clear the visible 41.8 Strict-Small leader before spending
SuperGLUE/AoA evaluation time.

BabyLM Strict-Small Overall here is the arithmetic mean of nine columns:
BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading, AoA.

Usage examples after a cheap surface:
  python projection_after_partial_surface.py \
    --seed 43022 --BLiMP 66 --Supplement 63 --EWoK 53 --Entity 28 \
    --COMPS 52 --GlobalPIQA 36 --Reading 8

If a pristine summary exists it can be read directly:
  python ... --summary path/to/pristine_collate_..._summary.json
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/projection_after_partial_surface.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/projection_after_partial_surface')

COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]
VISIBLE_LEADER = 41.8
REFERENCE = {
    "old_inherited_tokenizer_seed43022_official": {
        "BLiMP": 66.87232315173485,
        "Supplement": 63.27576417952158,
        "EWoK": 53.536575594886855,
        "Entity": 27.745741097952372,
        "COMPS": 51.968828052457084,
        "SuperGLUE": 71.03604952825312,
        "GlobalPIQA": 35.62135922330097,
        "Reading": 8.241572282566393,
        "AoA": 0.0,
        "Overall": 42.0331347900748,
    },
    "old_inherited_tokenizer_seed43122_official": {
        "BLiMP": 65.66440014112493,
        "Supplement": 61.95235293529443,
        "EWoK": 51.89171801728321,
        "Entity": 26.285338959270966,
        "COMPS": 51.5391941620175,
        "SuperGLUE": 69.90021312600398,
        "GlobalPIQA": 35.13592233009709,
        "Reading": 8.86501663100661,
        "AoA": 0.0,
        "Overall": 41.24823958912208,
    },
}


def extract_scores_from_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = []
    # research style pristine collate summaries
    try:
        candidates.append(data["score_summary"]["official_overall"]["scores"])
    except Exception:
        pass
    # research/48 style pristine collate summaries
    try:
        candidates.append(data["score_summary"]["scores"])
    except Exception:
        pass
    # posttrain controller may carry either score_summary shape directly
    try:
        candidates.append(data["score_summary"]["official_overall"]["scores"])
    except Exception:
        pass
    # plain top-level scores
    candidates.append(data.get("scores", data))
    for c in candidates:
        if isinstance(c, dict):
            out = {}
            for k in COLUMNS:
                v = c.get(k)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    out[k] = float(v)
            if out:
                return out
    return {}


def compute_projection(scores: dict[str, float]) -> dict[str, Any]:
    known = {k: float(v) for k, v in scores.items() if k in COLUMNS and isinstance(v, (int, float)) and math.isfinite(float(v))}
    missing = [k for k in COLUMNS if k not in known]
    known_sum = sum(known.values())
    required_missing_sum = 9 * VISIBLE_LEADER - known_sum
    required_missing_avg = required_missing_sum / len(missing) if missing else None
    hard_upper_bound = known_sum + 100.0 * len(missing)
    hard_upper_bound_can_still_reach = hard_upper_bound >= 9 * VISIBLE_LEADER
    if not missing:
        overall = known_sum / 9.0
    else:
        overall = None
    seven_keys = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
    seven_known = [known[k] for k in seven_keys if k in known]
    seven_sum = sum(seven_known)
    sg_aoa_required_if_seven_known = None
    sg_required_if_aoa0 = None
    first_priority = None
    if all(k in known for k in seven_keys):
        sg_aoa_required_if_seven_known = 9 * VISIBLE_LEADER - seven_sum
        sg_required_if_aoa0 = sg_aoa_required_if_seven_known  # AoA=0 is a scheduling comparison, not a missing-value substitute.
        first_priority = sg_required_if_aoa0 <= 72.0
    return {
        "known_scores": known,
        "missing_columns": missing,
        "known_sum": known_sum,
        "overall_if_complete": overall,
        "margin_over_visible_leader_if_complete": (overall - VISIBLE_LEADER) if overall is not None else None,
        "required_missing_sum_to_reach_41p8": required_missing_sum,
        "required_missing_average_to_reach_41p8": required_missing_avg,
        "hard_upper_bound_if_missing_columns_each_100": hard_upper_bound,
        "hard_upper_bound_can_still_reach_41p8": hard_upper_bound_can_still_reach,
        "seven_non_sg_non_aoa_sum": seven_sum if len(seven_known) == 7 else None,
        "required_SuperGLUE_plus_AoA_if_seven_surface_known": sg_aoa_required_if_seven_known,
        "required_SuperGLUE_if_AoA_zero_and_seven_surface_known": sg_required_if_aoa0,
        "priority_for_first_full_eval_slot": first_priority,
        "full_eval_required_for_two_seed_measurement": True,
        "reference_SuperGLUE_values": {
            "old_seed43022": REFERENCE["old_inherited_tokenizer_seed43022_official"]["SuperGLUE"],
            "old_seed43122": REFERENCE["old_inherited_tokenizer_seed43122_official"]["SuperGLUE"],
        },
        "interpretation": (
            "Use this projection to order evaluation and estimate urgency, not to omit full official "
            "measurement of a completed corrected-tokenizer seed. Missing AoA is not counted as zero; "
            "AoA=0 is only a comparison scenario. A hard stop would require the upper bound with every "
            "missing column at 100 to remain below the decision target, but the research two-seed policy still "
            "requires the same pristine full coordinate for both completed retrains to measure tokenizer-by-seed behavior."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="unknown")
    ap.add_argument("--summary", type=Path, default=None)
    for col in COLUMNS:
        ap.add_argument(f"--{col}", type=float, default=None)
    args = ap.parse_args()
    scores = {}
    if args.summary is not None:
        scores.update(extract_scores_from_json(args.summary if args.summary.is_absolute() else USER_ROOT / args.summary))
    for col in COLUMNS:
        v = getattr(args, col)
        if v is not None:
            scores[col] = float(v)
    projection = compute_projection(scores)
    payload = {
        "status": "PROJECTION_AFTER_PARTIAL_SURFACE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "summary_input": str(args.summary) if args.summary else None,
        "visible_leader_overall": VISIBLE_LEADER,
        "columns": COLUMNS,
        "projection": projection,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.seed.replace("/", "_")
    out_json = OUT_DIR / f"projection_{suffix}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "seed": args.seed, "out_json": str(out_json.relative_to(USER_ROOT)), **projection}, indent=2), flush=True)


if __name__ == "__main__":
    main()
