#!/usr/bin/env python3
"""research: compare dose endpoint ex-Entity displacement with 1x seed separation.

File-only supplement.  It quantifies independent review's nuance: ex-Entity V-R radius is nearly
constant with dose, but endpoint-to-endpoint direction changes; that change is
smaller than the observed same-coordinate two-seed separation at 1x.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path("experiments/archive/frontier_consolidation")
DATA = ROOT / "data"
OUT = DATA / "axis_noise_floor_readout"
DOSE = DATA / "conservation_vector_readout" / "dose_common10_80_net_norm.csv"
SEED = OUT / "seed_1x_vector_reproducibility.csv"
COLS = ["BLiMP", "Supplement", "EWoK", "COMPS", "Reading"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def vec_from_json(s: str) -> Dict[str, float]:
    obj = json.loads(s)
    return {c: float(obj[c]) for c in COLS if c in obj}


def l2_diff(a: Dict[str, float], b: Dict[str, float]) -> float:
    return float(np.linalg.norm([a[c] - b[c] for c in COLS]))


def l2(v: Dict[str, float]) -> float:
    return float(np.linalg.norm([v[c] for c in COLS]))


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    av = np.array([a[c] for c in COLS], dtype=float)
    bv = np.array([b[c] for c in COLS], dtype=float)
    return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv)))


def main() -> None:
    dose_rows = read_csv(DOSE)
    vecs: Dict[str, Dict[str, float]] = {}
    for r in dose_rows:
        if r.get("contrast") == "V_minus_R" and r.get("family_set") == "stable5_exEntity":
            vecs[r["dose_name"]] = vec_from_json(r["vector_json"])
    d1 = vecs["dose1"]
    d264 = vecs["dose2p64"]
    dose_endpoint_delta = l2_diff(d264, d1)

    seed_rows = read_csv(SEED)
    seed_common = next(r for r in seed_rows if r["window"] == "common10_80" and r["family_set"] == "stable5_exEntity")
    seed_distance = float(seed_common["difference_l2"])
    seed_ratio = float(seed_common["difference_over_mean_seed_l2"])
    seed_cos = float(seed_common["cosine"])

    entity_abs_change = 2.062499999999999 - 0.5587499999999999
    squared_total = dose_endpoint_delta**2 + entity_abs_change**2
    ex_entity_sq_share = dose_endpoint_delta**2 / squared_total if squared_total > 0 else None

    out = {
        "status": "ENDPOINT_VS_SEED_DISPLACEMENT_DONE",
        "cols_exEntity": COLS,
        "dose1_exEntity_vector": d1,
        "dose2p64_exEntity_vector": d264,
        "dose_endpoint_exEntity_l2_difference": dose_endpoint_delta,
        "dose_endpoint_entity_abs_difference": entity_abs_change,
        "dose_endpoint_exEntity_squared_share_of_stable6_endpoint_displacement": ex_entity_sq_share,
        "dose1_to_dose2p64_exEntity_cosine": cosine(d1, d264),
        "seed_common10_80_exEntity_l2_difference": seed_distance,
        "seed_common10_80_exEntity_difference_over_mean_seed_l2": seed_ratio,
        "seed_common10_80_exEntity_cosine": seed_cos,
        "dose_endpoint_exEntity_l2_over_seed_l2_difference": dose_endpoint_delta / seed_distance,
        "scientific_reading": "Ex-Entity V-R does reorient across dose, but the 1x-to-MAX displacement is smaller than the same-coordinate two-seed 1x separation and the seed directions are anti-aligned; therefore it is not yet a reproduced broad trade-off direction.",
        "no_model_loading_training_evaluation_gpu_globalpiqa_superglue_aoa_upload_or_leaderboard": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "endpoint_vs_seed_displacement.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
