#!/usr/bin/env python3
"""Merge research six-column evaluations with repaired Reading-only results."""
from __future__ import annotations
import json
from pathlib import Path

pairs = [
    (
        Path("experiments/archive/functional_learning/data/eval/standard_eval.json"),
        Path("experiments/archive/functional_learning/data/reading_only/standard_alpha075/standard_alpha075_reading_eval.json"),
        Path("experiments/archive/functional_learning/data/common_eval/merged_standard_alpha075/standard_alpha075_eval.json"),
        "standard_alpha075",
    ),
    (
        Path("experiments/archive/functional_learning/data/eval/carrier_residual_eval.json"),
        Path("experiments/archive/functional_learning/data/reading_only/stoch_carrier_residual_alpha075/stoch_carrier_residual_alpha075_reading_eval.json"),
        Path("experiments/archive/functional_learning/data/common_eval/merged_stoch_carrier_residual_alpha075/stoch_carrier_residual_alpha075_eval.json"),
        "stoch_carrier_residual_alpha075",
    ),
]
EQ7 = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]
for base_p, read_p, out_p, tag in pairs:
    base = json.loads(base_p.read_text())
    read = json.loads(read_p.read_text())
    scores = dict(base.get("scores", {}))
    for k in ["Reading", "Reading_eye", "Reading_self_paced"]:
        scores[k] = read.get("scores", {}).get(k)
    vals = [scores.get(k) for k in EQ7]
    good = [float(v) for v in vals if v is not None]
    scores["equal_valid_mean"] = sum(good) / len(good)
    scores["n_valid_equal_columns"] = len(good)
    scores["equal7"] = scores["equal_valid_mean"] if len(good) == 7 else None
    merged = dict(base)
    merged["tag"] = tag
    merged["scores"] = scores
    merged["Reading"] = read.get("Reading", {})
    merged["merge_note"] = {
        "six_column_eval": str(base_p),
        "reading_only_eval": str(read_p),
        "reason": "research original Reading omitted required data_path; research reran Reading with fast_eval reading_data.csv."
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(tag, scores["equal_valid_mean"], out_p)
