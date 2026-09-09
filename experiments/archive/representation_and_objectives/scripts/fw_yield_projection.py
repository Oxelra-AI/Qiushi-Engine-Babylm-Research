#!/usr/bin/env python3
"""research: estimate usable FW compact-view volume after preservation filtering.

Uses only CPU and already produced files.  It explains whether the low research
pilot yield reflects deliberately hard source enrichment or a likely full-corpus
shortfall, and what volume remains missing before any training is considered.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import json
import math
import pathlib
import statistics
import sys
import time
from typing import Any, Iterable

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
sys.path.insert(0, str(SCRIPT_DIR))
import fw_preservation_standard as std  # noqa: E402

PROMPTS = _public_path('experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_compact_prompts.jsonl')
PILOT_ROWS = _public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/new_qwen35_pilot_preservation_rows.jsonl')
EXISTING_ROWS = _public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/existing_a02_qwen35_preservation_rows.jsonl')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/fw_yield_projection')
NOTE = _public_path('research/notes/representation_and_objectives/fw_preservation_yield_projection.md')
TARGET_PAIR_WORDS = 1_494_110


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def qstats(xs: Iterable[float]) -> dict[str, Any]:
    vals = sorted(float(x) for x in xs)
    if not vals:
        return {"n": 0}
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = p * (len(vals) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return {"n": len(vals), "min": vals[0], "p05": q(0.05), "mean": statistics.fmean(vals), "median": statistics.median(vals), "p95": q(0.95), "max": vals[-1], "sum": sum(vals)}


def source_features_from_text(source: str) -> dict[str, Any]:
    sw = std.wc(source)
    if sw <= 18:
        length_bin = "short"
    elif sw <= 30:
        length_bin = "medium"
    else:
        length_bin = "long"
    neg = std.neg_signature(source)["has_negative_polarity"]
    mod = std.modality_signature(source)["has_modality"]
    cau = std.causal_signature(source)["has_causal"]
    comp = std.comparison_signature(source)["has_comparison"]
    ent = bool(std.real_entities(source))
    num = bool(std.nums(source))
    risk_count = sum([neg, mod, cau, comp])
    return {
        "length_bin": length_bin,
        "source_words": sw,
        "has_negative": neg,
        "has_modality": mod,
        "has_causal": cau,
        "has_comparison": comp,
        "has_entity": ent,
        "has_number": num,
        "risk_count": risk_count,
        "semantic_bucket": f"R{risk_count}_{length_bin}",
        "marker_mask": "".join(["N" if neg else "n", "M" if mod else "m", "C" if cau else "c", "P" if comp else "p"]),
    }


def row_features(r: dict[str, Any]) -> dict[str, Any]:
    return source_features_from_text(str(r.get("source_text") or r.get("text") or ""))


def summarize_distribution(items: list[dict[str, Any]]) -> dict[str, Any]:
    risk = collections.Counter(x["risk_count"] for x in items)
    length = collections.Counter(x["length_bin"] for x in items)
    mask = collections.Counter(x["marker_mask"] for x in items)
    return {
        "n": len(items),
        "source_words": sum(x["source_words"] for x in items),
        "source_word_stats": qstats(x["source_words"] for x in items),
        "risk_count": {str(k): v for k, v in sorted(risk.items())},
        "risk_count_fraction": {str(k): v / max(1, len(items)) for k, v in sorted(risk.items())},
        "length_bin": dict(length),
        "top_marker_masks": mask.most_common(20),
        "has_negative_frac": sum(x["has_negative"] for x in items) / max(1, len(items)),
        "has_modality_frac": sum(x["has_modality"] for x in items) / max(1, len(items)),
        "has_causal_frac": sum(x["has_causal"] for x in items) / max(1, len(items)),
        "has_comparison_frac": sum(x["has_comparison"] for x in items) / max(1, len(items)),
        "has_entity_frac": sum(x["has_entity"] for x in items) / max(1, len(items)),
        "has_number_frac": sum(x["has_number"] for x in items) / max(1, len(items)),
    }


def rate_table(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[str(key_fn(r))].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        ok = [r for r in rs if r.get("usable_for_compact_pair")]
        out[k] = {
            "n": len(rs),
            "usable": len(ok),
            "rate": len(ok) / max(1, len(rs)),
            "mean_rewrite_to_source_usable": (sum(r["rewrite_words"] for r in ok) / max(1, sum(r["source_words"] for r in ok))) if ok else None,
            "mean_pair_words_usable": statistics.fmean([r["pair_words"] for r in ok]) if ok else None,
        }
    return out


def estimate_from_rates(prompt_feats: list[dict[str, Any]], rates: dict[str, dict[str, Any]], fallback: dict[str, Any], key: str) -> dict[str, Any]:
    est_usable = 0.0
    est_src = 0.0
    est_rw = 0.0
    missing_groups = collections.Counter()
    used_groups = collections.Counter()
    for f in prompt_feats:
        group = f[key]
        rr = rates.get(str(group))
        if rr is None or rr.get("n", 0) < 5:
            rr = fallback
            missing_groups[str(group)] += 1
        else:
            used_groups[str(group)] += 1
        rate = float(rr.get("rate") or 0.0)
        ratio = rr.get("mean_rewrite_to_source_usable")
        if ratio is None:
            ratio = fallback.get("mean_rewrite_to_source_usable") or 0.63
        est_usable += rate
        est_src += rate * f["source_words"]
        est_rw += rate * f["source_words"] * float(ratio)
    return {
        "key": key,
        "estimated_usable_rows": est_usable,
        "estimated_usable_source_words": est_src,
        "estimated_usable_rewrite_words": est_rw,
        "estimated_usable_pair_words": est_src + est_rw,
        "fraction_of_target_pair_words": (est_src + est_rw) / TARGET_PAIR_WORDS,
        "estimated_pair_words_missing": max(0.0, TARGET_PAIR_WORDS - (est_src + est_rw)),
        "used_groups": used_groups.most_common(20),
        "fallback_groups": missing_groups.most_common(20),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(PROMPTS)
    pilot_rows = read_jsonl(PILOT_ROWS)
    existing_rows = read_jsonl(EXISTING_ROWS)
    prompt_feats = [source_features_from_text(str(p.get("source_text") or "")) for p in prompts]
    pilot_feats = [row_features(r) for r in pilot_rows]
    existing_feats = [row_features(r) for r in existing_rows]

    for r, f in zip(pilot_rows, pilot_feats):
        r.update({"_semantic_bucket": f["semantic_bucket"], "_risk_count": f["risk_count"], "_marker_mask": f["marker_mask"]})
    for r, f in zip(existing_rows, existing_feats):
        r.update({"_semantic_bucket": f["semantic_bucket"], "_risk_count": f["risk_count"], "_marker_mask": f["marker_mask"]})

    pilot_overall_ok = [r for r in pilot_rows if r.get("usable_for_compact_pair")]
    existing_overall_ok = [r for r in existing_rows if r.get("usable_for_compact_pair")]
    pilot_fallback = {
        "n": len(pilot_rows),
        "usable": len(pilot_overall_ok),
        "rate": len(pilot_overall_ok) / max(1, len(pilot_rows)),
        "mean_rewrite_to_source_usable": sum(r["rewrite_words"] for r in pilot_overall_ok) / max(1, sum(r["source_words"] for r in pilot_overall_ok)),
    }
    existing_fallback = {
        "n": len(existing_rows),
        "usable": len(existing_overall_ok),
        "rate": len(existing_overall_ok) / max(1, len(existing_rows)),
        "mean_rewrite_to_source_usable": sum(r["rewrite_words"] for r in existing_overall_ok) / max(1, sum(r["source_words"] for r in existing_overall_ok)),
    }

    pilot_rates_bucket = rate_table(pilot_rows, lambda r: r["_semantic_bucket"])
    pilot_rates_risk = rate_table(pilot_rows, lambda r: r["_risk_count"])
    existing_rates_bucket = rate_table(existing_rows, lambda r: r["_semantic_bucket"])
    existing_rates_risk = rate_table(existing_rows, lambda r: r["_risk_count"])

    for i, f in enumerate(prompt_feats):
        f["prompt_index"] = i
    by_key_prompt = [dict(f) for f in prompt_feats]
    for f in by_key_prompt:
        f["semantic_bucket"] = f["semantic_bucket"]

    # Existing accepted pair words are measured, not estimated.  Projection applies only to the 26,015 new sources.
    estimates = {
        "pilot_overall_rate": estimate_from_rates(prompt_feats, {}, pilot_fallback, "semantic_bucket"),
        "pilot_by_risk_count": estimate_from_rates(prompt_feats, pilot_rates_risk, pilot_fallback, "risk_count"),
        "pilot_by_risk_count_and_length": estimate_from_rates(prompt_feats, pilot_rates_bucket, pilot_fallback, "semantic_bucket"),
        "existing_by_risk_count": estimate_from_rates(prompt_feats, existing_rates_risk, existing_fallback, "risk_count"),
        "existing_by_risk_count_and_length": estimate_from_rates(prompt_feats, existing_rates_bucket, existing_fallback, "semantic_bucket"),
    }
    measured_existing_pair_words = sum(r["pair_words"] for r in existing_overall_ok)
    for k, est in estimates.items():
        total_pair = measured_existing_pair_words + est["estimated_usable_pair_words"]
        est["plus_measured_existing_pair_words"] = total_pair
        est["plus_existing_fraction_of_target_pair_words"] = total_pair / TARGET_PAIR_WORDS
        est["plus_existing_pair_words_missing"] = max(0.0, TARGET_PAIR_WORDS - total_pair)

    payload = {
        "status": "FW_YIELD_PROJECTION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_pair_words": TARGET_PAIR_WORDS,
        "files": {
            "prompts": str(PROMPTS),
            "pilot_rows": str(PILOT_ROWS),
            "existing_rows": str(EXISTING_ROWS),
            "summary": str(_public_path('experiments/archive/representation_and_objectives/data/fw_yield_projection/yield_projection.json')),
            "note": str(NOTE),
        },
        "full_new_prompt_distribution": summarize_distribution(prompt_feats),
        "pilot_prompt_distribution": summarize_distribution(pilot_feats),
        "existing_source_distribution": summarize_distribution(existing_feats),
        "measured_existing_usable": {
            "usable_rows": len(existing_overall_ok),
            "usable_pair_words": measured_existing_pair_words,
            "usable_rate": len(existing_overall_ok) / max(1, len(existing_rows)),
            "rewrite_to_source_ratio": existing_fallback["mean_rewrite_to_source_usable"],
        },
        "measured_pilot_usable": {
            "usable_rows": len(pilot_overall_ok),
            "usable_pair_words": sum(r["pair_words"] for r in pilot_overall_ok),
            "usable_rate": len(pilot_overall_ok) / max(1, len(pilot_rows)),
            "rewrite_to_source_ratio": pilot_fallback["mean_rewrite_to_source_usable"],
        },
        "pilot_rates_by_risk_count": pilot_rates_risk,
        "pilot_rates_by_risk_count_and_length": pilot_rates_bucket,
        "existing_rates_by_risk_count": existing_rates_risk,
        "existing_rates_by_risk_count_and_length": existing_rates_bucket,
        "estimates_for_new_26015": estimates,
    }
    (_public_path('experiments/archive/representation_and_objectives/data/fw_yield_projection/yield_projection.json')).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    full_dist = payload["full_new_prompt_distribution"]
    pilot_dist = payload["pilot_prompt_distribution"]
    lines = [
        "# research — projected preserved compact-view yield\n\n",
        "The research pilot was deliberately enriched for semantically fragile sources, so its low usable rate should not be read as the full 26,015-source yield without reweighting. This CPU-only projection compares feature distributions and estimates how much pair volume may survive the research preservation standard.\n\n",
        "## Distribution contrast\n\n",
        f"- Full new prompts: {full_dist['n']:,} rows, {full_dist['source_words']:,} source words; risk-count fractions {full_dist['risk_count_fraction']}\n",
        f"- Pilot prompts: {pilot_dist['n']:,} rows, {pilot_dist['source_words']:,} source words; risk-count fractions {pilot_dist['risk_count_fraction']}\n\n",
        "## Measured usable anchors\n\n",
        f"- Existing A02/Qwen3.5 under research standard: {len(existing_overall_ok):,}/{len(existing_rows):,} usable, {measured_existing_pair_words:,} pair words.\n",
        f"- New Qwen3.5 pilot under research standard: {len(pilot_overall_ok):,}/{len(pilot_rows):,} usable, {sum(r['pair_words'] for r in pilot_overall_ok):,} pair words.\n\n",
        "## Estimated full-scale preserved volume\n\n",
    ]
    for name, est in estimates.items():
        lines.append(
            f"- {name}: new usable pair words ≈ {est['estimated_usable_pair_words']:.0f}; with measured existing ≈ {est['plus_measured_existing_pair_words']:.0f} "
            f"({est['plus_existing_fraction_of_target_pair_words']:.3f} of target), missing ≈ {est['plus_existing_pair_words_missing']:.0f}.\n"
        )
    lines.append(f"\nSummary JSON: `{_public_path('experiments/archive/representation_and_objectives/data/fw_yield_projection/yield_projection.json')}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "full_rows": full_dist["n"],
        "full_risk_fraction": full_dist["risk_count_fraction"],
        "pilot_risk_fraction": pilot_dist["risk_count_fraction"],
        "existing_usable_pair_words": measured_existing_pair_words,
        "pilot_usable_rate": payload["measured_pilot_usable"]["usable_rate"],
        "estimates": {k: {
            "new_pair_words": round(v["estimated_usable_pair_words"], 1),
            "with_existing_pair_words": round(v["plus_measured_existing_pair_words"], 1),
            "fraction_target": round(v["plus_existing_fraction_of_target_pair_words"], 3),
            "missing": round(v["plus_existing_pair_words_missing"], 1),
        } for k, v in estimates.items()},
        "note": str(NOTE),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
