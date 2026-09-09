#!/usr/bin/env python3
"""Summarize a compact 2x2 tokenizer × data screen.

Cells:
  O16: tok16_official_20M_b128
  Q16: tok16_qwen_20M_b128
  O40: tok40_official_20M_b128
  Q40: tok40_qwen_20M_b128

All are intended to hold architecture/optimizer/masking/fixed seq/batch/exposure/seed
constant. The summary estimates data effects, tokenizer effects, and the interaction
on route-screen columns only (zero-shot + Reading; no SuperGLUE or AoA).
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = _public_path('experiments/archive/compact_experience')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
CELL_FILES = {
    "O16": _public_path('experiments/archive/compact_experience/data/tok16_b128_noaoa_eval/per_target/tok16_official_20M_b128.json'),
    "Q16": _public_path('experiments/archive/compact_experience/data/tok16_b128_noaoa_eval/per_target/tok16_qwen_20M_b128.json'),
    "O40": _public_path('experiments/archive/compact_experience/data/tok40_isolation_noaoa_eval/per_target/tok40_official_20M_b128.json'),
    "Q40": _public_path('experiments/archive/compact_experience/data/tok40_isolation_noaoa_eval/per_target/tok40_qwen_20M_b128.json'),
}
RUNS = {
    "O16": _public_path('experiments/archive/compact_experience/training/runs/tok16_official_20M_b128_seed43022/scientific_metrics.json'),
    "Q16": _public_path('experiments/archive/compact_experience/training/runs/tok16_qwen_20M_b128_seed43022/scientific_metrics.json'),
    "O40": _public_path('experiments/archive/compact_experience/training/runs/tok40_official_20M_b128_seed43022/scientific_metrics.json'),
    "Q40": _public_path('experiments/archive/compact_experience/training/runs/tok40_qwen_20M_b128_seed43022/scientific_metrics.json'),
}
OUT = _public_path('experiments/archive/compact_experience/data/tok16_tok40_2x2_summary.json')

def f(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def extract_payload(path: Path) -> Dict[str, Any]:
    p = json.loads(path.read_text())
    tasks = p.get("tasks", {})
    scores: Dict[str, Optional[float]] = {k: None for k in KEYS}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        scores[col] = f(tasks.get(col, {}).get("score"))
    gp_vals = [f(tasks.get(c, {}).get("score")) for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]]
    gp_vals = [v for v in gp_vals if v is not None]
    if gp_vals:
        scores["GlobalPIQA"] = sum(gp_vals) / len(gp_vals)
    scores["Reading"] = f(tasks.get("Reading", {}).get("scores", {}).get("Reading"))
    complete = all(scores[k] is not None for k in KEYS)
    equal7 = sum(float(scores[k]) for k in KEYS) / len(KEYS) if complete else None
    return {
        "target": p.get("target"),
        "endpoint": p.get("endpoint"),
        "model_path": p.get("model_path"),
        "scores": scores,
        "subcolumns": {
            "GlobalPIQA_parallel": f(tasks.get("GlobalPIQA_parallel", {}).get("score")),
            "GlobalPIQA_nonparallel": f(tasks.get("GlobalPIQA_nonparallel", {}).get("score")),
            "Reading_eye": f(tasks.get("Reading", {}).get("scores", {}).get("Reading_eye")),
            "Reading_self_paced": f(tasks.get("Reading", {}).get("scores", {}).get("Reading_self_paced")),
        },
        "equal7_noaoa_nosuperglue": equal7,
        "complete": complete,
    }

def load_run(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    m = json.loads(path.read_text())
    return {
        "path": str(path),
        "word_exposure": m.get("word_exposure", m.get("actual_word_exposure")),
        "actual_training_steps": m.get("actual_training_steps", m.get("actual_steps")),
        "loss_first": m.get("loss_first", m.get("mlm_loss_first")),
        "loss_last": m.get("loss_last", m.get("mlm_loss_last")),
        "parameter_count": m.get("parameter_count"),
        "vocab_size": m.get("vocab_size"),
        "tokenizer_label": m.get("tokenizer_label"),
        "batch_size": m.get("batch_size"),
        "seq_length": m.get("seq_length"),
        "max_seq_length": m.get("max_seq_length"),
        "saved_checkpoints": len(m.get("saved_checkpoints", []) or []),
    }

def diff(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    out = {}
    for k in ["equal7_noaoa_nosuperglue"] + KEYS:
        av = a.get(k) if k.startswith("equal") else a["scores"].get(k)
        bv = b.get(k) if k.startswith("equal") else b["scores"].get(k)
        out[k] = None if av is None or bv is None else av - bv
    return out

def main() -> None:
    missing = {k: str(p) for k,p in CELL_FILES.items() if not p.exists()}
    cells = {}
    if not missing:
        for k,p in CELL_FILES.items():
            cells[k] = extract_payload(p)
    runs = {k: load_run(p) for k,p in RUNS.items()}
    effects: Dict[str, Any] = {}
    if cells:
        data_effect_16 = diff(cells["Q16"], cells["O16"])
        data_effect_40 = diff(cells["Q40"], cells["O40"])
        tokenizer_effect_official = diff(cells["O40"], cells["O16"])
        tokenizer_effect_qwen = diff(cells["Q40"], cells["Q16"])
        interaction = {}
        for k in data_effect_16:
            a = data_effect_40[k]; b = data_effect_16[k]
            interaction[k] = None if a is None or b is None else a - b
        effects = {
            "data_effect_16k_Q_minus_O": data_effect_16,
            "data_effect_40k_Q_minus_O": data_effect_40,
            "tokenizer_effect_official_40_minus_16": tokenizer_effect_official,
            "tokenizer_effect_qwen_40_minus_16": tokenizer_effect_qwen,
            "data_x_tokenizer_interaction": interaction,
        }
    report = {
        "status": "TOK16_TOK40_2X2_SUMMARY" if cells else "INCOMPLETE_2X2_SUMMARY",
        "purpose": "Compact factor screen after MNTP failure: separate data effect, tokenizer effect, and data×tokenizer interaction at matched 20M/b128/current-recipe conditions. No AoA or SuperGLUE included.",
        "missing_payloads": missing,
        "cells": cells,
        "runs": runs,
        "effects": effects,
        "interpretation_rules": {
            "100M_extension_signal": "A positive Q40 vs O40 data effect alone is insufficient. Prefer extension only if interaction is positive in EWoK/Entity/GlobalPIQA and no strong Supplement/Reading loss appears, or if Q40 clearly improves over Q16 on target columns.",
            "negative_signal": "If 40k helps official as much as or more than Qwen, or hurts target columns, tokenization is not the right next full route; pivot to a pure MLM cross-view masking or view-identity representation intervention.",
            "limits": "This no-AoA screen cannot establish final Overall or AoA safety; any 100M candidate still needs complete nine-column measurement."
        }
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
