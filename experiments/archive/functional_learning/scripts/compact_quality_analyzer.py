#!/usr/bin/env python3
"""research: Comprehensive quality analyzer for compact rewrites.

Measures whether shortened Qwen-internal second views preserve the useful learning
signal. The key scientific distinction: redundancy in expressed meaning is not 
necessarily redundancy for learning. A shorter view may preserve propositions but
lose the alternative-expression evidence that made the pair useful.

Measures:
1. Word savings: actual vs. expected, realized compression ratio
2. Entity preservation: named entities from source present in compact
3. Number preservation: all numbers from source in compact
4. Structural markers: negation, modality, role/coreference, causal/temporal, comparison
5. Expression diversity: how different is compact from original vs. current rewrite from original
6. Cross-span correspondence: tokens shared between compact and original vs. current and original

Usage:
    python compact_quality_analyzer.py --input pilot_generation_joined.jsonl
"""
import argparse, json, sys, re, statistics
from pathlib import Path
from collections import Counter

STUDY = Path("experiments/archive/functional_learning")
DATA = STUDY / "data" / "compact_pilot"
FIGURES = STUDY / "figures"

# ── structural marker patterns ──
NEGATION_PATTERNS = re.compile(
    r"\b(not|n't|no|never|neither|nor|nobody|nothing|nowhere|none|cannot|can't|won't|wouldn't|shouldn't|couldn't|didn't|doesn't|don't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b",
    re.IGNORECASE
)
MODALITY_PATTERNS = re.compile(
    r"\b(can|could|may|might|must|shall|should|will|would|ought|need)\b",
    re.IGNORECASE
)
CAUSAL_TEMPORAL = re.compile(
    r"\b(because|since|therefore|thus|hence|so|consequently|although|though|however|but|yet|while|when|before|after|until|during|meanwhile|then|finally|first|next|later|already|still|now|once|unless|if|whether)\b",
    re.IGNORECASE
)
COMPARISON = re.compile(
    r"\b(more|less|most|least|better|worse|best|worst|rather|than|as\s+as|bigger|smaller|larger|higher|lower|faster|slower|older|newer|longer|shorter)\b",
    re.IGNORECASE
)
ROLE_MARKERS = re.compile(
    r"\b(he|she|they|him|her|them|his|their|hers|theirs|himself|herself|themselves|I|me|my|mine|we|us|our|you|your|who|whom|whose)\b",
    re.IGNORECASE
)


def tokenize_words(text):
    """Simple whitespace tokenization."""
    return text.strip().split()


def extract_markers(text, pattern):
    """Extract all marker occurrences."""
    return [m.group().lower() for m in pattern.finditer(text)]


def word_overlap(a_tokens, b_tokens):
    """Jaccard overlap between two token sets."""
    a_set = set(t.lower().strip(".,;:!?'\"()-") for t in a_tokens)
    b_set = set(t.lower().strip(".,;:!?'\"()-") for t in b_tokens)
    a_set.discard("")
    b_set.discard("")
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def entity_recall(entities, text):
    """Fraction of source entities found in text (case-insensitive)."""
    if not entities:
        return 1.0  # no entities to check
    text_lower = text.lower()
    found = sum(1 for e in entities if e.lower() in text_lower)
    return found / len(entities)


def number_recall(numbers, text):
    """Fraction of source numbers found in text."""
    if not numbers:
        return 1.0
    found = sum(1 for n in numbers if n in text)
    return found / len(numbers)


def marker_preservation(source_markers, compact_markers):
    """Fraction of source structural markers preserved in compact."""
    if not source_markers:
        return 1.0
    source_set = Counter(source_markers)
    compact_set = Counter(compact_markers)
    preserved = sum(min(source_set[m], compact_set[m]) for m in source_set)
    return preserved / sum(source_set.values())


def analyze_pair(rec):
    """Analyze a single pair's compact rewrite quality."""
    original = rec["original"]
    current = rec["current_rewrite"]
    compact = rec["compact_rewrite"]
    target_words = rec.get("target_compact_words", 0)

    orig_tokens = tokenize_words(original)
    curr_tokens = tokenize_words(current)
    comp_tokens = tokenize_words(compact)

    orig_words = len(orig_tokens)
    curr_words = len(curr_tokens)
    comp_words = len(comp_tokens)

    # Word savings
    actual_saved = curr_words - comp_words
    expected_saved = rec.get("expected_saved_words", 0)
    savings_ratio = comp_words / curr_words if curr_words > 0 else 1.0
    target_hit = comp_words <= target_words if target_words > 0 else True

    # Entity and number preservation
    ent_recall_compact = entity_recall(rec.get("entity_source", []), compact)
    ent_recall_current = entity_recall(rec.get("entity_source", []), current)
    num_recall_compact = number_recall(rec.get("num_source", []), compact)
    num_recall_current = number_recall(rec.get("num_source", []), current)

    # Structural markers
    orig_neg = extract_markers(original, NEGATION_PATTERNS)
    comp_neg = extract_markers(compact, NEGATION_PATTERNS)
    curr_neg = extract_markers(current, NEGATION_PATTERNS)

    orig_mod = extract_markers(original, MODALITY_PATTERNS)
    comp_mod = extract_markers(compact, MODALITY_PATTERNS)
    
    orig_causal = extract_markers(original, CAUSAL_TEMPORAL)
    comp_causal = extract_markers(compact, CAUSAL_TEMPORAL)

    orig_comp = extract_markers(original, COMPARISON)
    comp_comp = extract_markers(compact, COMPARISON)

    orig_role = extract_markers(original, ROLE_MARKERS)
    comp_role = extract_markers(compact, ROLE_MARKERS)

    neg_pres = marker_preservation(orig_neg, comp_neg)
    mod_pres = marker_preservation(orig_mod, comp_mod)
    causal_pres = marker_preservation(orig_causal, comp_causal)
    comp_pres = marker_preservation(orig_comp, comp_comp)
    role_pres = marker_preservation(orig_role, comp_role)

    # Expression diversity: how different is compact from original
    # vs how different current rewrite is from original
    compact_orig_overlap = word_overlap(comp_tokens, orig_tokens)
    current_orig_overlap = word_overlap(curr_tokens, orig_tokens)
    compact_current_overlap = word_overlap(comp_tokens, curr_tokens)

    # A useful second view should have moderate overlap with original (not too high = copying,
    # not too low = unrelated). The compact should be closer to the current rewrite
    # (it's a compression of it) but still provide alternative expression of the original.

    return {
        "pair_id": rec["pair_id"],
        "source": rec.get("source", ""),
        "orig_words": orig_words,
        "current_words": curr_words,
        "compact_words": comp_words,
        "target_words": target_words,
        "actual_saved": actual_saved,
        "expected_saved": expected_saved,
        "savings_ratio": round(savings_ratio, 4),
        "target_hit": target_hit,
        "entity_recall_compact": round(ent_recall_compact, 4),
        "entity_recall_current": round(ent_recall_current, 4),
        "num_recall_compact": round(num_recall_compact, 4),
        "num_recall_current": round(num_recall_current, 4),
        "n_entities": len(rec.get("entity_source", [])),
        "n_numbers": len(rec.get("num_source", [])),
        "negation_preservation": round(neg_pres, 4),
        "modality_preservation": round(mod_pres, 4),
        "causal_temporal_preservation": round(causal_pres, 4),
        "comparison_preservation": round(comp_pres, 4),
        "role_preservation": round(role_pres, 4),
        "n_orig_negations": len(orig_neg),
        "n_compact_negations": len(comp_neg),
        "n_orig_modals": len(orig_mod),
        "n_compact_modals": len(comp_mod),
        "n_orig_causal": len(orig_causal),
        "n_compact_causal": len(comp_causal),
        "compact_orig_overlap": round(compact_orig_overlap, 4),
        "current_orig_overlap": round(current_orig_overlap, 4),
        "compact_current_overlap": round(compact_current_overlap, 4),
    }


def summarize_stats(values, label=""):
    """Compute summary statistics for a list of numbers."""
    if not values:
        return {"n": 0, "label": label}
    return {
        "label": label,
        "n": len(values),
        "min": round(min(values), 4),
        "p05": round(sorted(values)[max(0, int(0.05 * len(values)))], 4),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(sorted(values)[min(len(values) - 1, int(0.95 * len(values)))], 4),
        "max": round(max(values), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to pilot_generation_joined.jsonl")
    ap.add_argument("--output-dir", default=str(DATA))
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load joined data
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records", flush=True)

    # Filter out empty compacts
    valid = [r for r in records if r.get("compact_rewrite", "").strip()]
    print(f"Valid (non-empty compact): {len(valid)}", flush=True)

    # Analyze each pair
    analyses = []
    for rec in valid:
        try:
            a = analyze_pair(rec)
            analyses.append(a)
        except Exception as e:
            print(f"ERROR analyzing {rec.get('pair_id','?')}: {e}", file=sys.stderr)

    # Save per-pair analysis
    with open(out_dir / "pilot_pair_analysis.jsonl", "w") as f:
        for a in analyses:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # Aggregate statistics
    n = len(analyses)
    if n == 0:
        print("ERROR: no valid analyses", file=sys.stderr)
        sys.exit(1)

    # Word savings
    savings_stats = summarize_stats([a["actual_saved"] for a in analyses], "actual_saved_words")
    ratio_stats = summarize_stats([a["savings_ratio"] for a in analyses], "savings_ratio")
    target_hit_rate = sum(1 for a in analyses if a["target_hit"]) / n
    total_saved = sum(a["actual_saved"] for a in analyses)
    total_current = sum(a["current_words"] for a in analyses)
    total_compact = sum(a["compact_words"] for a in analyses)

    # Entity/number preservation (only for pairs that have them)
    ent_pairs = [a for a in analyses if a["n_entities"] > 0]
    num_pairs = [a for a in analyses if a["n_numbers"] > 0]

    # Structural preservation
    neg_pairs = [a for a in analyses if a["n_orig_negations"] > 0]
    mod_pairs = [a for a in analyses if a["n_orig_modals"] > 0]
    causal_pairs = [a for a in analyses if a["n_orig_causal"] > 0]

    # Expression diversity
    co_overlap = summarize_stats([a["compact_orig_overlap"] for a in analyses], "compact_orig_jaccard")
    cr_overlap = summarize_stats([a["current_orig_overlap"] for a in analyses], "current_orig_jaccard")
    cc_overlap = summarize_stats([a["compact_current_overlap"] for a in analyses], "compact_current_jaccard")

    # By source
    by_source = {}
    for src in set(a["source"] for a in analyses):
        src_a = [a for a in analyses if a["source"] == src]
        by_source[src] = {
            "n": len(src_a),
            "mean_savings_ratio": round(statistics.mean([a["savings_ratio"] for a in src_a]), 4),
            "target_hit_rate": round(sum(1 for a in src_a if a["target_hit"]) / len(src_a), 4),
            "mean_entity_recall": round(statistics.mean([a["entity_recall_compact"] for a in src_a if a["n_entities"] > 0]) if any(a["n_entities"] > 0 for a in src_a) else 1.0, 4),
            "mean_negation_pres": round(statistics.mean([a["negation_preservation"] for a in src_a if a["n_orig_negations"] > 0]) if any(a["n_orig_negations"] > 0 for a in src_a) else 1.0, 4),
            "mean_causal_pres": round(statistics.mean([a["causal_temporal_preservation"] for a in src_a if a["n_orig_causal"] > 0]) if any(a["n_orig_causal"] > 0 for a in src_a) else 1.0, 4),
            "mean_compact_orig_overlap": round(statistics.mean([a["compact_orig_overlap"] for a in src_a]), 4),
            "total_saved_words": sum(a["actual_saved"] for a in src_a),
        }

    # Identify failure cases
    failures = {
        "entity_loss": [a["pair_id"] for a in analyses if a["n_entities"] > 0 and a["entity_recall_compact"] < 1.0],
        "number_loss": [a["pair_id"] for a in analyses if a["n_numbers"] > 0 and a["num_recall_compact"] < 1.0],
        "negation_loss": [a["pair_id"] for a in analyses if a["n_orig_negations"] > 0 and a["negation_preservation"] < 0.5],
        "over_target": [a["pair_id"] for a in analyses if not a["target_hit"]],
        "negative_savings": [a["pair_id"] for a in analyses if a["actual_saved"] < 0],
    }

    summary = {
        "status": "COMPACT_QUALITY_ANALYZED",
        "n_analyzed": n,
        "word_savings": {
            "stats": savings_stats,
            "ratio_stats": ratio_stats,
            "target_hit_rate": round(target_hit_rate, 4),
            "total_current_words": total_current,
            "total_compact_words": total_compact,
            "total_saved_words": total_saved,
            "effective_compression": round(total_compact / total_current, 4) if total_current > 0 else None,
        },
        "entity_preservation": {
            "n_pairs_with_entities": len(ent_pairs),
            "mean_recall_compact": round(statistics.mean([a["entity_recall_compact"] for a in ent_pairs]), 4) if ent_pairs else None,
            "mean_recall_current": round(statistics.mean([a["entity_recall_current"] for a in ent_pairs]), 4) if ent_pairs else None,
            "perfect_recall_rate": round(sum(1 for a in ent_pairs if a["entity_recall_compact"] >= 1.0) / len(ent_pairs), 4) if ent_pairs else None,
            "n_entity_loss_pairs": len(failures["entity_loss"]),
        },
        "number_preservation": {
            "n_pairs_with_numbers": len(num_pairs),
            "mean_recall_compact": round(statistics.mean([a["num_recall_compact"] for a in num_pairs]), 4) if num_pairs else None,
            "perfect_recall_rate": round(sum(1 for a in num_pairs if a["num_recall_compact"] >= 1.0) / len(num_pairs), 4) if num_pairs else None,
            "n_number_loss_pairs": len(failures["number_loss"]),
        },
        "structural_preservation": {
            "negation": {
                "n_pairs_with_negation": len(neg_pairs),
                "mean_preservation": round(statistics.mean([a["negation_preservation"] for a in neg_pairs]), 4) if neg_pairs else None,
                "stats": summarize_stats([a["negation_preservation"] for a in neg_pairs], "negation_pres") if neg_pairs else {},
            },
            "modality": {
                "n_pairs_with_modals": len(mod_pairs),
                "mean_preservation": round(statistics.mean([a["modality_preservation"] for a in mod_pairs]), 4) if mod_pairs else None,
            },
            "causal_temporal": {
                "n_pairs_with_causal": len(causal_pairs),
                "mean_preservation": round(statistics.mean([a["causal_temporal_preservation"] for a in causal_pairs]), 4) if causal_pairs else None,
            },
        },
        "expression_diversity": {
            "compact_vs_original": co_overlap,
            "current_vs_original": cr_overlap,
            "compact_vs_current": cc_overlap,
            "interpretation": "Compact-original overlap should be moderate (different expression), compact-current overlap should be higher (compressed from current). If compact-original overlap approaches 1.0, the compact is just copying the original and provides no alternative expression.",
        },
        "by_source": by_source,
        "failures": {k: {"n": len(v), "pair_ids_sample": v[:10]} for k, v in failures.items()},
    }

    out_path = out_dir / "compact_quality_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    # Also save a compact table for quick reading
    table_path = out_dir / "compact_quality_table.md"
    with open(table_path, "w") as f:
        f.write("# Compact Rewrite Quality Pilot (n=%d)\n\n" % n)
        f.write("## Word Savings\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Total current words | {total_current} |\n")
        f.write(f"| Total compact words | {total_compact} |\n")
        f.write(f"| Total saved | {total_saved} |\n")
        f.write(f"| Effective compression | {summary['word_savings']['effective_compression']:.4f} |\n")
        f.write(f"| Target hit rate | {target_hit_rate:.4f} |\n")
        f.write(f"| Mean savings ratio | {ratio_stats['mean']:.4f} |\n\n")
        f.write("## Entity/Number Preservation\n")
        f.write("| Metric | Compact | Current |\n|---|---|---|\n")
        ep = summary["entity_preservation"]
        f.write(f"| Entity recall (n={ep['n_pairs_with_entities']}) | {ep['mean_recall_compact']} | {ep['mean_recall_current']} |\n")
        np_ = summary["number_preservation"]
        f.write(f"| Number recall (n={np_['n_pairs_with_numbers']}) | {np_['mean_recall_compact']} | {np_.get('mean_recall_current','N/A')} |\n\n")
        f.write("## Structural Preservation\n")
        f.write("| Marker | N pairs | Mean preservation |\n|---|---|---|\n")
        sp = summary["structural_preservation"]
        f.write(f"| Negation | {sp['negation']['n_pairs_with_negation']} | {sp['negation']['mean_preservation']} |\n")
        f.write(f"| Modality | {sp['modality']['n_pairs_with_modals']} | {sp['modality']['mean_preservation']} |\n")
        f.write(f"| Causal/temporal | {sp['causal_temporal']['n_pairs_with_causal']} | {sp['causal_temporal']['mean_preservation']} |\n\n")
        f.write("## Expression Diversity (Jaccard overlap)\n")
        ed = summary["expression_diversity"]
        f.write("| Comparison | Mean overlap |\n|---|---|\n")
        f.write(f"| Compact vs Original | {ed['compact_vs_original']['mean']:.4f} |\n")
        f.write(f"| Current vs Original | {ed['current_vs_original']['mean']:.4f} |\n")
        f.write(f"| Compact vs Current | {ed['compact_vs_current']['mean']:.4f} |\n\n")
        f.write("## By Source\n")
        f.write("| Source | N | Compression | Target hit | Entity recall | Negation pres |\n|---|---|---|---|---|---|\n")
        for src, sv in sorted(by_source.items()):
            f.write(f"| {src} | {sv['n']} | {sv['mean_savings_ratio']:.3f} | {sv['target_hit_rate']:.3f} | {sv['mean_entity_recall']:.3f} | {sv['mean_negation_pres']:.3f} |\n")

    print(f"\nTable: {table_path}", flush=True)


if __name__ == "__main__":
    main()
