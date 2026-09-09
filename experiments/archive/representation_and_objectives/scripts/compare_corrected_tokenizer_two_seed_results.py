#!/usr/bin/env python3
"""research: compare corrected-tokenizer two-seed official results.

CPU-only result synthesizer. It is meant to run after both research post-training
controllers have produced pristine collate summaries for the research retrains.
It can dry-run before those files exist, recording the missing evidence and the
old inherited-tokenizer reference vectors.

Outputs:
- corrected official vectors and margins over the visible 41.8 leader;
- per-column seed deltas under the corrected tokenizer;
- corrected two-seed mean and seed spread;
- corrected-vs-inherited tokenizer movement per seed and per column;
- a compact interpretation for whether the compliant representation preserves
  the compact_view_reinvest advance and how seed-sensitive it is.
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

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/compare_corrected_tokenizer_two_seed_results.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')

VISIBLE_LEADER = 41.8
COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]

OLD_REFERENCE = {
    "43022": {
        "summary_path": "experiments/archive/representation_and_objectives/data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json",
        "scores": {
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
        "status": "inherited Strict-100M tokenizer; mechanism evidence only, not compliant submission material",
    },
    "43122": {
        "summary_path": "experiments/archive/representation_and_objectives/data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json",
        "scores": {
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
        "status": "inherited Strict-100M tokenizer; mechanism evidence only, not compliant submission material",
    },
}

CORRECTED_SUMMARY_PATHS = {
    "43022": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_seed43022_pristine_collate/pristine_collate_strictsmalltok_seed43022_summary.json'),
    "43122": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_seed43122_pristine_collate/pristine_collate_strictsmalltok_seed43122_summary.json'),
}
POSTTRAIN_SUMMARY_PATHS = {
    "43022": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_seed43022_posttrain_eval_controller/posttrain_eval_seed43022_summary.json'),
    "43122": _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_seed43122_posttrain_eval_controller/posttrain_eval_seed43122_summary.json'),
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def extract_scores_from_obj(obj: Any) -> dict[str, float]:
    candidates = []
    if isinstance(obj, dict):
        # research style
        try:
            candidates.append(obj["score_summary"]["official_overall"]["scores"])
        except Exception:
            pass
        # research/48 style
        try:
            candidates.append(obj["score_summary"]["scores"])
        except Exception:
            pass
        # posttrain controller may carry score_summary either nested or flat
        ss = obj.get("score_summary")
        if isinstance(ss, dict):
            candidates.append(ss)
        candidates.append(obj.get("scores", obj))
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        out = {}
        for k in COLUMNS + ["Overall", "margin_over_visible_leader_41p8"]:
            v = cand.get(k)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[k] = float(v)
        if all(k in out for k in COLUMNS):
            if "Overall" not in out:
                out["Overall"] = sum(out[k] for k in COLUMNS) / 9.0
            out["margin_over_visible_leader_41p8"] = out["Overall"] - VISIBLE_LEADER
            return out
    return {}


def load_scores(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    meta = {"path": rel(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}
    if not path.exists():
        return {}, meta
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta["json_status"] = data.get("status") if isinstance(data, dict) else None
        meta["collated_sha256"] = data.get("collated_sha256") if isinstance(data, dict) else None
        scores = extract_scores_from_obj(data)
        meta["score_keys"] = sorted(scores)
        return scores, meta
    except Exception as exc:
        meta["error"] = repr(exc)
        return {}, meta


def vector_diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: a[k] - b[k] for k in COLUMNS + ["Overall"] if k in a and k in b}


def mean_vector(vectors: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for k in COLUMNS + ["Overall"]:
        vals = [v[k] for v in vectors if k in v]
        if vals:
            out[k] = sum(vals) / len(vals)
    if "Overall" in out:
        out["margin_over_visible_leader_41p8"] = out["Overall"] - VISIBLE_LEADER
    return out


def population_sd(vals: list[float]) -> float | None:
    return statistics.pstdev(vals) if len(vals) >= 2 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_public_path('experiments/archive/representation_and_objectives/data/corrected_tokenizer_two_seed_comparison/corrected_tokenizer_two_seed_comparison.json'))
    ap.add_argument("--allow-one", action="store_true", help="write analysis even if only one corrected seed is present")
    args = ap.parse_args()

    corrected: dict[str, Any] = {}
    complete_corrected = []
    missing = []
    for seed, path in CORRECTED_SUMMARY_PATHS.items():
        scores, meta = load_scores(path)
        if not scores and POSTTRAIN_SUMMARY_PATHS[seed].exists():
            scores, meta2 = load_scores(POSTTRAIN_SUMMARY_PATHS[seed])
            meta = {"preferred_pristine_summary": meta, "posttrain_summary_fallback": meta2}
        corrected[seed] = {"scores": scores, "summary": meta}
        if all(k in scores for k in COLUMNS):
            complete_corrected.append(seed)
        else:
            missing.append(seed)

    old_vectors = {seed: rec["scores"] for seed, rec in OLD_REFERENCE.items()}
    corrected_vectors = {seed: rec["scores"] for seed, rec in corrected.items() if all(k in rec["scores"] for k in COLUMNS)}
    comparisons: dict[str, Any] = {}
    if len(corrected_vectors) >= 1:
        for seed, scores in corrected_vectors.items():
            comparisons[f"corrected_minus_inherited_seed{seed}"] = vector_diff(scores, old_vectors[seed])
    if len(corrected_vectors) == 2:
        c430 = corrected_vectors["43022"]
        c431 = corrected_vectors["43122"]
        comparisons["corrected_seed43022_minus_seed43122"] = vector_diff(c430, c431)
        comparisons["corrected_two_seed_mean"] = mean_vector([c430, c431])
        comparisons["corrected_overall_spread_abs"] = abs(c430["Overall"] - c431["Overall"])
        comparisons["corrected_overall_population_sd"] = population_sd([c430["Overall"], c431["Overall"]])
        old_mean = mean_vector([old_vectors["43022"], old_vectors["43122"]])
        comparisons["inherited_two_seed_mean"] = old_mean
        comparisons["corrected_mean_minus_inherited_mean"] = vector_diff(comparisons["corrected_two_seed_mean"], old_mean)
        comparisons["both_clear_visible_41p8"] = all(v["Overall"] > VISIBLE_LEADER for v in [c430, c431])
        comparisons["any_clear_visible_41p8"] = any(v["Overall"] > VISIBLE_LEADER for v in [c430, c431])
        comparisons["two_seed_mean_clears_visible_41p8"] = comparisons["corrected_two_seed_mean"].get("Overall", -999) > VISIBLE_LEADER

    if missing and not args.allow_one:
        status = "CORRECTED_TOKENIZER_TWO_SEED_COMPARISON_WAITING_FOR_FULL_RESULTS"
    else:
        status = "CORRECTED_TOKENIZER_TWO_SEED_COMPARISON_READY"

    interpretation = []
    if len(corrected_vectors) == 0:
        interpretation.append("No corrected-tokenizer pristine full official vector exists yet; endpoint judgment remains unavailable.")
    elif len(corrected_vectors) == 1:
        seed = complete_corrected[0]
        ov = corrected_vectors[seed]["Overall"]
        interpretation.append(f"Only seed{seed} corrected-tokenizer full vector exists so far: Overall {ov:.6f}. This can guide scheduling but cannot measure the two-seed corrected-tokenizer effect.")
    else:
        mean_ov = comparisons["corrected_two_seed_mean"]["Overall"]
        spread = comparisons["corrected_overall_spread_abs"]
        if comparisons["both_clear_visible_41p8"]:
            interpretation.append("Both corrected-tokenizer seeds clear the visible 41.8 leader; this would be strong evidence that the compact_view_reinvest principle survives the compliant representation.")
        elif comparisons["any_clear_visible_41p8"]:
            interpretation.append("Exactly one corrected-tokenizer seed clears the visible 41.8 leader; this protects a potential endpoint but leaves seed stability as a central scientific issue.")
        else:
            interpretation.append("Neither corrected-tokenizer seed clears the visible 41.8 leader; the inherited-tokenizer 42.033 result remains mechanism evidence but not a compliant SOTA endpoint.")
        interpretation.append(f"Corrected two-seed mean Overall {mean_ov:.6f}; absolute seed spread {spread:.6f}.")

    payload = {
        "status": status,
        "created_utc": now_utc(),
        "visible_leader_overall": VISIBLE_LEADER,
        "policy": "Both research corrected-tokenizer retrains should ultimately be represented by the same pristine full official coordinate; surface-first outputs are scheduling aids only.",
        "columns": COLUMNS,
        "old_inherited_tokenizer_reference": OLD_REFERENCE,
        "corrected_tokenizer_results": corrected,
        "complete_corrected_seeds": complete_corrected,
        "missing_or_incomplete_corrected_seeds": missing,
        "comparisons": comparisons,
        "interpretation": interpretation,
    }
    out = args.out if args.out.is_absolute() else USER_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "out_json": rel(out),
        "complete_corrected_seeds": complete_corrected,
        "missing_or_incomplete_corrected_seeds": missing,
        "comparisons_available": sorted(comparisons),
        "interpretation": interpretation,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
