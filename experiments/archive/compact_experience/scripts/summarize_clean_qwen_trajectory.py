#!/usr/bin/env python3
"""Summarize clean-Qwen zero-shot/Reading checkpoint trajectories."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import pathlib

ROOT = _public_path('experiments/archive/compact_experience')
TRAJ = _public_path('experiments/archive/compact_experience/data/clean_qwen_checkpoint_trajectory')
OUT = _public_path('experiments/archive/compact_experience/data/clean_qwen_checkpoint_trajectory/clean_qwen_trajectory_comparison.json')
NOTE = _public_path('research/notes/compact_experience/clean_qwen_checkpoint_trajectory.md')
TARGETS = [
    "official_lengthmatched_seed43022",
    "qwen_clean_aligned_seed43022",
    "official_lengthmatched_seed43122",
    "qwen_clean_aligned_seed43122",
]
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading", "equal7_full_eval"]


def r(x, nd=4):
    return None if x is None else round(float(x), nd)


def load(name):
    p = TRAJ / f"{name}_trajectory_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    payload = {"status": "CLEAN_QWEN_TRAJECTORY_COMPARISON", "targets": {}, "paired_deltas_by_checkpoint": {}}
    for t in TARGETS:
        d = load(t)
        payload["targets"][t] = d
    for seed in ["seed43022", "seed43122"]:
        off = payload["targets"].get(f"official_lengthmatched_{seed}")
        qw = payload["targets"].get(f"qwen_clean_aligned_{seed}")
        if not off or not qw:
            continue
        rows = {}
        for ckpt, qrow in (qw.get("table") or {}).items():
            orow = (off.get("table") or {}).get(ckpt)
            if not orow:
                continue
            rows[ckpt] = {k: (None if qrow.get(k) is None or orow.get(k) is None else round(float(qrow[k])-float(orow[k]), 6)) for k in KEYS}
        valid = {k: v for k, v in rows.items() if v.get("equal7_full_eval") is not None}
        best = max(valid.items(), key=lambda kv: kv[1]["equal7_full_eval"]) if valid else None
        payload["paired_deltas_by_checkpoint"][seed] = {"table": rows, "best_delta_by_equal7": {"checkpoint": best[0], "delta": best[1]} if best else None}
    _public_path('experiments/archive/compact_experience/data/clean_qwen_checkpoint_trajectory').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research clean-Qwen checkpoint trajectory", "", f"Summary JSON: `{OUT}`", ""]
    for t in TARGETS:
        d = payload["targets"].get(t)
        if not d:
            lines.append(f"- `{t}` pending")
            continue
        best = d.get("best_by_equal7_full_eval")
        lines.append(f"- `{t}` best: {best.get('checkpoint') if best else None}; equal7={r(best.get('row',{}).get('equal7_full_eval')) if best else None}")
    lines.append("\n## Matched deltas by checkpoint")
    for seed, rec in payload["paired_deltas_by_checkpoint"].items():
        best = rec.get("best_delta_by_equal7")
        lines.append(f"- `{seed}` best qwen-official delta: {best.get('checkpoint') if best else None}; Δequal7={r(best.get('delta',{}).get('equal7_full_eval')) if best else None}")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(OUT), "note": str(NOTE), "complete_targets": [k for k,v in payload['targets'].items() if v], "delta_seeds": payload["paired_deltas_by_checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
