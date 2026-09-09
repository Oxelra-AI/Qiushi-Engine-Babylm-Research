#!/usr/bin/env python3
"""Strict post-hoc validator for research source-attested fluent bridge outputs.

Adds checks that the first automatic analyzer missed, especially signed/units
number preservation. It does not generate, train, evaluate BabyLM tasks, upload,
or submit anything.
"""
from __future__ import annotations
import collections, json, math, re, statistics
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/frontier_consolidation")
ROWS = ROOT / "data/source_attested_fluent_bridge_prototype/source_attested_fluent_bridge_generation_rows.jsonl"
OUT_DIR = ROOT / "data/source_attested_fluent_bridge_prototype"
OUT_JSON = OUT_DIR / "source_attested_fluent_bridge_strict_validation.json"
OUT_REVIEW = OUT_DIR / "source_attested_fluent_bridge_strict_review_cases.jsonl"
NOTE = (ROOT.parents[2] / 'research/notes/frontier_consolidation/source_attested_fluent_bridge_strict_validation.md')

# Keep signs and common unit suffixes rather than collapsing to unsigned digits.
NUMBER_SURFACE_RE = re.compile(
    r"[−-]?\d+(?:[.,:/\-]\d+)*(?:\s*(?:%|percent|°\s*[CF]|degrees?\s*[CF]|/\s*°?\s*[CF]|per\s+cent))?",
    re.I,
)
BAD_GRAMMAR_PATTERNS = [
    ("bare_advised", re.compile(r"\b(allergies|individuals|people|patients)\s+advised\b", re.I)),
    ("missing_article_from_tray", re.compile(r"\bfrom\s+tray\b", re.I)),
    ("says_plural_subject", re.compile(r"\b(University\s+and\s+BirdLife\s+International)\s+says\b", re.I)),
    ("happy_the_bird", re.compile(r"\bfeel\s+happy\s+the\s+bird\b", re.I)),
]

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_number_surface(s: str) -> str:
    x = s.lower().replace("−", "-")
    x = re.sub(r"\s+", "", x)
    x = x.replace("percent", "%").replace("per cent", "%")
    x = x.replace("degrees", "°").replace("degree", "°")
    return x.strip(".,;:!?()[]{}\"'")

def number_surfaces(text: str) -> list[str]:
    vals = []
    for m in NUMBER_SURFACE_RE.finditer(text or ""):
        raw = norm_number_surface(m.group(0))
        if raw:
            vals.append(raw)
    return vals

def multiset_missing(src: list[str], out: list[str]) -> list[str]:
    c = collections.Counter(out)
    missing = []
    for x in src:
        if c[x] > 0:
            c[x] -= 1
        else:
            missing.append(x)
    return missing

def stat(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return {"n": 0}
    vals.sort()
    def q(p):
        if len(vals) == 1:
            return vals[0]
        z = p * (len(vals)-1); lo = math.floor(z); hi = math.ceil(z)
        return vals[lo] if lo == hi else vals[lo]*(hi-z)+vals[hi]*(z-lo)
    return {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals), "p10": q(0.1), "p25": q(0.25), "p75": q(0.75), "p90": q(0.9), "min": vals[0], "max": vals[-1]}

def main() -> None:
    rows = read_jsonl(ROWS)
    strict_rows = []
    reason_counts = collections.Counter()
    for r in rows:
        rr = dict(r)
        source_nums = number_surfaces(rr.get("source_text") or "")
        gen_nums = number_surfaces(rr.get("generated_text") or "")
        missing_num_surface = multiset_missing(source_nums, gen_nums)
        extra_num_surface = multiset_missing(gen_nums, source_nums)
        strict_reasons = []
        strict_reasons.extend(rr.get("hard_reasons") or [])
        if missing_num_surface:
            strict_reasons.append("missing_number_surface")
        if extra_num_surface:
            strict_reasons.append("extra_number_surface")
        if abs(float(rr.get("generated_to_natural_words_ratio") or 99) - 1.0) > 0.30:
            strict_reasons.append("geometry_word_ratio_outside_0p70_1p30")
        if rr.get("active_token_ratio_to_natural") is not None and abs(float(rr.get("active_token_ratio_to_natural")) - 1.0) > 0.35:
            strict_reasons.append("geometry_token_ratio_outside_0p65_1p35")
        grammar_hits = [name for name, rx in BAD_GRAMMAR_PATTERNS if rx.search(rr.get("generated_text") or "")]
        if grammar_hits:
            strict_reasons.append("fluency_proxy_pattern")
        rr["strict_missing_number_surfaces"] = missing_num_surface
        rr["strict_extra_number_surfaces"] = extra_num_surface
        rr["strict_fluency_proxy_hits"] = grammar_hits
        rr["strict_reasons"] = strict_reasons
        rr["strict_accept"] = len(strict_reasons) == 0
        reason_counts.update(strict_reasons)
        strict_rows.append(rr)
    by_pair = collections.defaultdict(list)
    for r in strict_rows:
        by_pair[r["pair_id"]].append(r)
    selected = []
    for pid, rs in by_pair.items():
        acc = [r for r in rs if r["strict_accept"]]
        if acc:
            selected.append(min(acc, key=lambda r: (abs(float(r.get("generated_to_natural_words_ratio") or 99)-1.0),
                                                    abs(float(r.get("active_token_ratio_to_natural") or 99)-1.0),
                                                    -float(r.get("source_content_recall") or 0))))
    def small(rs):
        return {
            "n": len(rs),
            "accepted": sum(1 for r in rs if r["strict_accept"]),
            "accepted_rate": sum(1 for r in rs if r["strict_accept"]) / max(1, len(rs)),
            "word_ratio": stat([r.get("generated_to_natural_words_ratio") for r in rs]),
            "token_ratio": stat([r.get("active_token_ratio_to_natural") for r in rs]),
            "source_recall": stat([r.get("source_content_recall") for r in rs]),
            "natural_overlap": stat([r.get("natural_content_overlap_recall") for r in rs]),
        }
    by_regime = collections.defaultdict(list)
    by_bucket = collections.defaultdict(list)
    for r in strict_rows:
        by_regime[r.get("regime")].append(r)
        by_bucket[r.get("prototype_bucket")].append(r)
    payload = {
        "status": "SOURCE_ATTESTED_FLUENT_BRIDGE_STRICT_VALIDATION",
        "input_rows": str(ROWS),
        "total_rows": len(strict_rows),
        "strict_row_summary": small(strict_rows),
        "total_pairs": len(by_pair),
        "accepted_any_pair_count": len(selected),
        "accepted_any_pair_rate": len(selected) / max(1, len(by_pair)),
        "selected_best_per_pair_summary": small(selected),
        "by_regime": {k: small(v) for k,v in sorted(by_regime.items())},
        "by_bucket": {k: small(v) for k,v in sorted(by_bucket.items())},
        "reason_counts": dict(reason_counts),
        "number_surface_rule": "signed/unit-preserving regex surfaces; source surfaces must appear in generated text with signs and attached units preserved.",
        "fluency_proxy_is_incomplete": True,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Cases for semantic review: all strict accepted hard cases and the failures with special validator hits.
    review = []
    for r in strict_rows:
        if r["strict_accept"] and int(r.get("natural_compact_absent_content_lemma_count") or 0) > 0:
            review.append({"sample_kind": "strict_accepted_hard", **r})
    for r in strict_rows:
        if ("missing_number_surface" in r["strict_reasons"] or "extra_number_surface" in r["strict_reasons"] or "fluency_proxy_pattern" in r["strict_reasons"]):
            review.append({"sample_kind": "strict_special_failure", **r})
    write_jsonl(OUT_REVIEW, review[:80])
    lines = [
        "# research strict validation of the source-attested fluent bridge prototype\n\n",
        "This post-hoc pass adds signed/unit-preserving numeric checks and a few explicit fluency proxy patterns to the first automatic analyzer. It is still not semantic proof.\n\n",
        f"Strict row accept: {payload['strict_row_summary']['accepted']}/{payload['strict_row_summary']['n']} ({payload['strict_row_summary']['accepted_rate']:.3f}). Pairs with at least one strict-accepted output: {len(selected)}/{len(by_pair)} ({payload['accepted_any_pair_rate']:.3f}).\n\n",
        f"Best strict accepted per pair: word ratio mean {payload['selected_best_per_pair_summary']['word_ratio'].get('mean', float('nan')):.3f}, token ratio mean {payload['selected_best_per_pair_summary']['token_ratio'].get('mean', float('nan')):.3f}, source recall mean {payload['selected_best_per_pair_summary']['source_recall'].get('mean', float('nan')):.3f}, natural overlap mean {payload['selected_best_per_pair_summary']['natural_overlap'].get('mean', float('nan')):.3f}.\n\n",
        f"Reason counts: {payload['reason_counts']}\n\n",
        f"JSON: `{OUT_JSON}`\n\nReview cases: `{OUT_REVIEW}`\n",
    ]
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "strict_row_accept": payload["strict_row_summary"]["accepted"], "strict_row_rate": payload["strict_row_summary"]["accepted_rate"], "accepted_pairs": len(selected), "accepted_pair_rate": payload["accepted_any_pair_rate"], "json": str(OUT_JSON), "note": str(NOTE)}, indent=2))

if __name__ == "__main__":
    main()
