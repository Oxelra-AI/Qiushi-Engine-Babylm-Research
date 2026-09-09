#!/usr/bin/env python3
"""Summarize one compact tokenizer × data no-AoA endpoint screen.

This is a generic companion to launch_tok2x2_endpoint_noaoa_eval.sh.
It reads four per-target payloads for one checkpoint endpoint and computes:
  O16, Q16, O40, Q40 cell scores
  data effects under each tokenizer
  tokenizer effects under each data source
  data×tokenizer interaction

The screen covers zero-shot columns plus Reading only; it does not include
SuperGLUE or AoA and must not be used as final SOTA evidence.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = _public_path('experiments/archive/compact_experience')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


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


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in ["equal7_noaoa_nosuperglue"] + KEYS:
        av = a.get(k) if k.startswith("equal") else a["scores"].get(k)
        bv = b.get(k) if k.startswith("equal") else b["scores"].get(k)
        out[k] = None if av is None or bv is None else float(av) - float(bv)
    return out


def summarize(out_root: Path, endpoint: str, suffix: str) -> Dict[str, Any]:
    per = out_root / "per_target"
    target_names = {
        "O16": f"tok16_official_{suffix}_b128",
        "Q16": f"tok16_qwen_{suffix}_b128",
        "O40": f"tok40_official_{suffix}_b128",
        "Q40": f"tok40_qwen_{suffix}_b128",
    }
    cell_files = {k: per / f"{v}.json" for k, v in target_names.items()}
    missing = {k: str(p) for k, p in cell_files.items() if not p.exists()}
    cells: Dict[str, Any] = {}
    if not missing:
        cells = {k: extract_payload(p) for k, p in cell_files.items()}
    run_files = {
        "O16": _public_path('experiments/archive/compact_experience/training/runs/tok16_official_20M_b128_seed43022/scientific_metrics.json'),
        "Q16": _public_path('experiments/archive/compact_experience/training/runs/tok16_qwen_20M_b128_seed43022/scientific_metrics.json'),
        "O40": _public_path('experiments/archive/compact_experience/training/runs/tok40_official_20M_b128_seed43022/scientific_metrics.json'),
        "Q40": _public_path('experiments/archive/compact_experience/training/runs/tok40_qwen_20M_b128_seed43022/scientific_metrics.json'),
    }
    runs = {k: load_run(p) for k, p in run_files.items()}
    effects: Dict[str, Any] = {}
    if cells:
        data_effect_16 = diff(cells["Q16"], cells["O16"])
        data_effect_40 = diff(cells["Q40"], cells["O40"])
        tokenizer_effect_official = diff(cells["O40"], cells["O16"])
        tokenizer_effect_qwen = diff(cells["Q40"], cells["Q16"])
        interaction = {}
        for k in data_effect_16:
            a = data_effect_40[k]
            b = data_effect_16[k]
            interaction[k] = None if a is None or b is None else a - b
        effects = {
            "data_effect_16k_Q_minus_O": data_effect_16,
            "data_effect_40k_Q_minus_O": data_effect_40,
            "tokenizer_effect_official_40_minus_16": tokenizer_effect_official,
            "tokenizer_effect_qwen_40_minus_16": tokenizer_effect_qwen,
            "data_x_tokenizer_interaction": interaction,
        }
    report = {
        "status": "TOK2X2_ENDPOINT_SUMMARY" if cells else "INCOMPLETE_TOK2X2_ENDPOINT_SUMMARY",
        "endpoint": endpoint,
        "suffix": suffix,
        "purpose": "Route-screen endpoint for the matched 20M/b128 data×tokenizer interface experiment. Zero-shot columns plus Reading only; no AoA or SuperGLUE.",
        "important_interpretation": "40k changes segmentation, visible subword targets, and about 11.36M embedding/output parameters; effects are a whole data-tokenizer interface signal, not reduced fragmentation alone.",
        "missing_payloads": missing,
        "cells": cells,
        "runs": runs,
        "effects": effects,
        "promotion_reading": {
            "positive": "A full 100M candidate would need a broad, substantial Qwen×40k interface signal: Q40 clearly above Q16 and O16, gains not dominated by one task, and no strong Supplement/Reading damage. This screen still cannot establish AoA or SuperGLUE safety.",
            "negative": "A narrow or tradeoff-dominated endpoint should not be extrapolated to 100M, because these experiments have shown many small early/proxy gains reverse under full scale and full scoring."
        }
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_root = ROOT / args.out_root if not Path(args.out_root).is_absolute() else Path(args.out_root)
    report = summarize(out_root, args.endpoint, args.suffix)
    out = Path(args.out) if args.out else out_root / f"tok16_tok40_2x2_{args.suffix}_summary.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
