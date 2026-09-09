#!/usr/bin/env python3
"""Lightweight unit test for summarize_highlr_interaction.py.
Creates synthetic per-target payloads in the same official_overall.scores shape used
by custom_endpoint_full_eval.py and checks interaction arithmetic.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = _public_path('experiments/archive/compact_experience')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/summarizer_unit_test')
PER = _public_path('experiments/archive/compact_experience/data/summarizer_unit_test/per_target')
LR = "0.007"
CKPTS = ["chck_3M", "chck_5M", "chck_10M"]
COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
BASE_NAMES = {
    "adamw_official": f"adamw_lr{LR}_official_10M_seed43022",
    "lamb_official": f"lamb_lr{LR}_official_10M_seed43022",
    "adamw_qwen": f"adamw_lr{LR}_qwen_10M_seed43022",
    "lamb_qwen": f"lamb_lr{LR}_qwen_10M_seed43022",
}


def payload(scores):
    return {"target": "synthetic", "tasks": {}, "official_overall": {"scores": scores}}


def main() -> None:
    PER.mkdir(parents=True, exist_ok=True)
    # Clean old synthetic payloads.
    for old in PER.glob("*.json"):
        old.unlink()

    # Construct known effects: AdamW data effect +1 on all columns; LAMB data effect
    # +2 on all columns except Reading +0.5. Interaction therefore +1 for six
    # columns and -0.5 for Reading; equal7 interaction = (6*1 - 0.5)/7.
    for ckpt in CKPTS:
        for key, base in BASE_NAMES.items():
            scores = {c: 50.0 for c in COLS}
            if key == "adamw_qwen":
                scores = {c: 51.0 for c in COLS}
            elif key == "lamb_official":
                scores = {c: 50.0 for c in COLS}
            elif key == "lamb_qwen":
                scores = {c: 52.0 for c in COLS}
                scores["Reading"] = 50.5
            target = f"{base}_{ckpt}"
            (PER / f"{target}.json").write_text(json.dumps(payload(scores)), encoding="utf-8")

    cmd = [sys.executable, "-B", str(_public_path('experiments/archive/compact_experience/scripts/summarize_highlr_interaction.py')), str(OUT_ROOT), LR]
    subprocess.run(cmd, cwd=str(WORKSPACE), check=True)
    summary = json.loads((_public_path('experiments/archive/compact_experience/data/summarizer_unit_test/highlr_10M_interaction_summary.json')).read_text(encoding="utf-8"))
    expected = round((6.0 - 0.5) / 7.0, 4)
    got = summary["interactions_by_checkpoint"]["chck_3M"]["interaction_equal7"]
    if abs(got - expected) > 1e-9:
        raise SystemExit(f"interaction_equal7 mismatch: got {got}, expected {expected}")
    ss = summary["interactions_by_checkpoint"]["chck_3M"]["interaction_sign_structure"]
    if ss["n_positive"] != 6 or ss["n_negative"] != 1:
        raise SystemExit(f"sign structure mismatch: {ss}")
    print(json.dumps({"status": "SUMMARIZER_UNIT_TEST_PASS", "interaction_equal7": got, "expected": expected, "sign_structure": ss}, indent=2))


if __name__ == "__main__":
    main()
