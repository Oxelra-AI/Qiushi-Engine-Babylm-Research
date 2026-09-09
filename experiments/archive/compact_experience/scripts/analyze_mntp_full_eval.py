#!/usr/bin/env python3
"""Analyze research MLM-primary MNTP full evaluation against trusted references.

This script is intentionally post-hoc descriptive: it reads official-style eval
payloads and reports column deltas. It does not select endpoints, does not touch
training data, and does not use AoA/CDI signals for training or scheduling.
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
TARGET = "mlm_mntp_aux015_100M"
PAYLOAD = _public_path('experiments/archive/compact_experience/data/mlm_mntp_full_eval/per_target') / f"{TARGET}.json"
OUT = _public_path('experiments/archive/compact_experience/data/mntp_eval_interpretation.json')

OFFICIAL_KEYS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]

# Trusted fully measured references from preserved experimental evidence.
REFERENCES: Dict[str, Dict[str, float]] = {
    "clean_qwen_seed43022_100M": {
        "Overall": 41.34429066479573,
        "BLiMP": 66.84,
        "Supplement": 62.84,
        "EWoK": 50.19,
        "Entity": 25.76,
        "COMPS": 51.78,
        "GlobalPIQA": 36.62,
        "SuperGLUE": 70.3086159833656,
        "Reading": 7.76,
        "AoA": 0.0,
    },
    "mixed_causal15_100M": {
        "Overall": 41.20913349834475,
        "BLiMP": 66.66,
        "Supplement": 59.90,
        "EWoK": 48.57,
        "Entity": 26.01,
        "COMPS": 51.57,
        "GlobalPIQA": 39.105,
        "SuperGLUE": 70.0322014851027,
        "Reading": 8.725,
        "AoA": 0.0,
    },
    "visible_leader_2026_wwm_curriculum_simplification_40k": {
        "Overall": 41.8,
        "BLiMP": 67.2,
        "Supplement": 56.01,
        "EWoK": 56.07,
        "Entity": 28.45,
        "COMPS": 53.57,
        "GlobalPIQA": 39.67,
        "SuperGLUE": 69.79,
        "Reading": 5.42,
        "AoA": 0.0,
    },
}


def f_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def extract_scores(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    oo = payload.get("official_overall") if isinstance(payload, dict) else None
    if isinstance(oo, dict) and isinstance(oo.get("scores"), dict):
        scores = {k: f_or_none(oo["scores"].get(k)) for k in OFFICIAL_KEYS}
        if all(scores[k] is not None for k in OFFICIAL_KEYS):
            scores["Overall"] = f_or_none(oo.get("overall", oo.get("Overall")))
            scores["equal7"] = f_or_none(oo.get("equal7", oo.get("NLP_average")))
            scores["nlp7_mean"] = f_or_none(oo.get("nlp7_mean", oo.get("NLP_average")))
            scores["human_mean"] = f_or_none(oo.get("human_mean", oo.get("Human_like_average")))
            return scores
    tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
    out: Dict[str, Optional[float]] = {k: None for k in OFFICIAL_KEYS}
    for key in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "Reading", "SuperGLUE", "GlobalPIQA"]:
        rec = tasks.get(key, {}) if isinstance(tasks, dict) else {}
        if isinstance(rec, dict):
            for cand in ["score", "overall", "accuracy", "mean", "leaderboard_score"]:
                if cand in rec:
                    out[key] = f_or_none(rec[cand])
                    break
    aoa = tasks.get("AoA", {}) if isinstance(tasks, dict) else {}
    if isinstance(aoa, dict):
        out["AoA"] = f_or_none(aoa.get("aoa_leaderboard_score", aoa.get("aoa_official", aoa.get("score"))))
    if all(out[k] is not None for k in OFFICIAL_KEYS):
        out["Overall"] = sum(float(out[k]) for k in OFFICIAL_KEYS) / 9.0
    return out


def delta_table(scores: Dict[str, Optional[float]], ref: Dict[str, float]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k in ["Overall"] + OFFICIAL_KEYS:
        v = scores.get(k)
        out[k] = None if v is None else float(v) - ref[k]
    return out


def summarize_route(scores: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Scientific interpretation rules for the existing measurements."""
    if scores.get("Overall") is None:
        return {"status": "incomplete_eval_payload", "next": "finish or repair full nine-column measurement before route choice"}
    d_clean = delta_table(scores, REFERENCES["clean_qwen_seed43022_100M"])
    d_leader = delta_table(scores, REFERENCES["visible_leader_2026_wwm_curriculum_simplification_40k"])
    losses = {k: v for k, v in d_clean.items() if k in OFFICIAL_KEYS and v is not None and v < -0.5}
    gains = {k: v for k, v in d_clean.items() if k in OFFICIAL_KEYS and v is not None and v > 0.5}

    if float(scores["Overall"]) > 41.8:
        interpretation = (
            "MNTP auxiliary produced a leader-crossing candidate in local official-style measurement; "
            "the next scientific work is endpoint integrity checks, second-seed replication, and official-compatible packaging, "
            "not further speculative route changes."
        )
    elif d_clean["Overall"] is not None and d_clean["Overall"] > 0:
        interpretation = (
            "MNTP auxiliary improved the trusted clean-Qwen coordinate but did not cross the visible leader. "
            "Use column deltas to decide whether a single matched strengthening is warranted; a bundled 12x384/40k/LAMB pivot remains unisolated."
        )
    else:
        interpretation = (
            "MNTP auxiliary did not improve the trusted clean-Qwen Overall. Close this exact same-stack auxiliary construction as a SOTA route, "
            "then choose the next run from the column failure mode: preserve the validated same-window Qwen data, and isolate one representation or data-by-recipe factor instead of adopting the old leader-inspired bundle wholesale."
        )

    # Mechanism-directed follow-up suggestions, conditional on the observed column profile.
    suggestions = []
    if d_clean.get("GlobalPIQA") is not None and d_clean["GlobalPIQA"] > 1.0 and any(k in losses for k in ["Supplement", "EWoK", "BLiMP"]):
        suggestions.append(
            "If MNTP repeats the causal15 pattern (GlobalPIQA/Reading gain paid by Supplement/EWoK/BLiMP loss), the issue is objective-level tradeoff, "
            "not lack of directional signal. Prefer a short matched gradient/representation probe or a lower-level architectural separation that keeps MLM reconstruction pressure clean, not another full objective-dose sweep."
        )
    if d_clean.get("EWoK") is not None and d_clean["EWoK"] > 1.0 and d_clean.get("Entity") is not None and d_clean["Entity"] > 1.0:
        suggestions.append(
            "If EWoK/Entity move together without destroying Supplement/Reading, the same-window correspondence signal may be better captured; replicate or test a minimal isolated variant before changing architecture/optimizer/tokenizer together."
        )
    if d_clean.get("AoA") is not None and d_clean["AoA"] < -1.0:
        suggestions.append(
            "If AoA becomes negative, do not endpoint-mine against AoA. Treat it as an aggregate-balance failure like the tail family and avoid continuation tricks that already destroyed AoA."
        )
    if not suggestions:
        suggestions.append(
            "For a negative or flat result, the clean next experiment should isolate one factor. Candidates: (i) matched clean-Qwen data with 40k tokenizer on the current 8x480/AdamW recipe; (ii) current 16k tokenizer with a minimal representation change such as GEGLU/attention-output gate/layer weighting while holding optimizer and schedule fixed; (iii) compact data-by-recipe screen trained to a fixed no-AoA exposure only if it separates data interaction from architecture/optimizer/vocab."
        )
    return {
        "status": "interpreted_complete_scores",
        "interpretation": interpretation,
        "losses_vs_clean_over_0p5": losses,
        "gains_vs_clean_over_0p5": gains,
        "mechanism_directed_suggestions": suggestions,
    }


def main() -> None:
    if not PAYLOAD.exists():
        raise FileNotFoundError(f"missing evaluation payload: {PAYLOAD}")
    payload = json.loads(PAYLOAD.read_text())
    scores = extract_scores(payload)
    report = {
        "status": "MNTP_EVAL_INTERPRETATION",
        "target": TARGET,
        "payload": str(PAYLOAD.relative_to(ROOT)),
        "scores": scores,
        "deltas": {name: delta_table(scores, ref) for name, ref in REFERENCES.items()},
        "route_interpretation": summarize_route(scores),
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
