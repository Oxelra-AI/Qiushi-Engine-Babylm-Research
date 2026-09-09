#!/usr/bin/env python3
"""Summarize research mixed-objective no-AoA trajectories."""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
from pathlib import Path
ROOT = _public_path('experiments/archive/compact_experience')
OUT = _public_path('experiments/archive/compact_experience/data/mixed_objective_eval')
REF = 43.112857142857145
COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
rows = []
for target, arm in [("mixed_causal50", "causal50"), ("mixed_causal15", "causal15")]:
    path = OUT / f"{target}_trajectory_summary.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    for ckpt, row in data.get("table", {}).items():
        if not all(row.get(c) is not None for c in COLS):
            continue
        eq7 = sum(float(row[c]) for c in COLS) / 7
        rows.append({"arm": arm, "checkpoint": ckpt, "equal7": eq7,
                     "delta_vs_clean_qwen": eq7 - REF,
                     **{c: float(row[c]) for c in COLS}})
rows.sort(key=lambda x: x["equal7"], reverse=True)
payload = {"status": "MIXED_OBJECTIVE_NOAOA_SUMMARY",
           "clean_qwen_reference_equal7": REF,
           "n_rows": len(rows), "best": rows[0] if rows else None, "rows": rows}
path = _public_path('experiments/archive/compact_experience/data/mixed_objective_eval/mixed_objective_noaoa_summary.json')
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"out": str(path), "n_rows": len(rows), "best": payload["best"]}, indent=2))
