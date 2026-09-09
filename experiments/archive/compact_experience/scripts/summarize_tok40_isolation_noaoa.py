#!/usr/bin/env python3
"""Summarize the research compact tok40 isolation no-AoA screen."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = _public_path('experiments/archive/compact_experience')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/tok40_isolation_noaoa_eval')
OUT = _public_path('experiments/archive/compact_experience/data/tok40_isolation_noaoa_eval/tok40_isolation_noaoa_summary.json')
KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]

# Existing 16k 20M references should be extracted from prior payloads when available.
# They are not hard-coded here because full official-style eval at intermediate endpoints
# was not always preserved in the same format.

def f(x):
    try:
        return float(x)
    except Exception:
        return None


def extract(path: Path) -> Dict[str, Any]:
    p = json.loads(path.read_text())
    tasks = p.get("tasks", {})
    scores: Dict[str, Optional[float]] = {k: None for k in KEYS}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        scores[col] = f(tasks.get(col, {}).get("score"))
    gp = []
    for col in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        v = f(tasks.get(col, {}).get("score"))
        if v is not None:
            gp.append(v)
    if gp:
        scores["GlobalPIQA"] = sum(gp)/len(gp)
    scores["Reading"] = f(tasks.get("Reading", {}).get("scores", {}).get("Reading"))
    complete = all(scores[k] is not None for k in KEYS)
    equal7 = sum(float(scores[k]) for k in KEYS)/len(KEYS) if complete else None
    return {
        "target": p.get("target"),
        "model_path": p.get("model_path"),
        "endpoint": p.get("endpoint"),
        "scores": scores,
        "equal7_noaoa_nosuperglue": equal7,
        "complete_noaoa_screen": complete,
        "run_summary": p.get("run_summary", {}),
    }


def main():
    per = _public_path('experiments/archive/compact_experience/data/tok40_isolation_noaoa_eval/per_target')
    official_p = per / "tok40_official_20M_b128.json"
    qwen_p = per / "tok40_qwen_20M_b128.json"
    missing = [str(p) for p in [official_p, qwen_p] if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    official = extract(official_p)
    qwen = extract(qwen_p)
    delta = {}
    for k in ["equal7_noaoa_nosuperglue"] + KEYS:
        qv = qwen.get(k) if k.startswith("equal") else qwen["scores"].get(k)
        ov = official.get(k) if k.startswith("equal") else official["scores"].get(k)
        delta[k] = None if qv is None or ov is None else qv - ov
    report = {
        "status": "TOK40_ISOLATION_NOAOA_SUMMARY",
        "purpose": "Compact isolated 40k-tokenizer data-interaction screen under the current 8x480 recipe; not final SOTA evidence.",
        "targets": {"official": official, "qwen": qwen},
        "qwen_minus_official": delta,
        "interpretation": {
            "isolated_factors": "Shared official-only 40k tokenizer with current 8x480/AdamW/fixed-seq256/WWM recipe; data arm differs official-only vs clean-Qwen; batch128 is used only to fit the larger vocabulary logits and is matched across arms.",
            "positive_signal": "A broad qwen-minus-official no-AoA gain, especially in EWoK/Entity/GlobalPIQA without Supplement collapse, would justify extending the tokenizer isolation to 100M and complete nine-column eval.",
            "negative_signal": "Flat or narrow tradeoff would argue against spending a full run on tokenization and favor a minimal representation intervention on the current 16k clean-Qwen backbone."
        }
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
