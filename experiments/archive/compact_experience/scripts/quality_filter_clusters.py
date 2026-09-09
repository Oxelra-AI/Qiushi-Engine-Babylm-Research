#!/usr/bin/env python3
"""research: Quality-filter natural entity-relation clusters for mechanism test.

Reads the full preflight clusters JSONL from research, applies:
  1. Anchor-type preference: cap > quote > num >> rep (heavy downweight)
  2. Source/topic cap: max 3 clusters per block_id
  3. Sensitive-content exclusion via keyword blocklist
  4. Preference for held-out third sentence (allows internal diagnostic)
  5. Complementarity filter: low pairwise content Jaccard + diverse predicates
  6. Word budget for 2% low-dose (target ~200K cluster words)

Outputs a filtered clusters JSONL and summary metadata.
Does NOT read any downstream evaluation outputs or official AoA/CDI material.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import json
import pathlib
import random
import re
from typing import Any

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_IN = _public_path('experiments/archive/compact_experience/data/natural_entity_relation_clusters/clusters_sample.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/cluster_mechanism_test')

# Sensitive content keywords (case-insensitive, substring)
SENSITIVE_KEYWORDS = [
    "rape", "molest", "sexual assault", "pedophil", "incest",
    "genocide", "massacre", "torture", "mutilat", "dismember",
    "nigger", "faggot", "kike", "spic", "wetback",
    "concentration camp", "ethnic cleansing", "child porn",
    "suicide bomb", "terrorist attack", "mass shooting",
    "serial killer", "serial murder",
]
SENSITIVE_RE = re.compile("|".join(re.escape(k) for k in SENSITIVE_KEYWORDS), re.IGNORECASE)

# Filler/discourse anchors that should not drive clusters
FILLER_ANCHORS = {
    "rep:right", "rep:thing", "rep:things", "rep:people", "rep:really",
    "rep:going", "rep:about", "rep:would", "rep:think", "rep:there",
    "rep:could", "rep:should", "rep:still", "rep:every", "rep:never",
    "rep:always", "rep:women", "rep:other", "rep:years", "rep:first",
    "rep:before", "rep:after", "rep:little", "rep:great", "rep:place",
    "rep:those", "rep:these", "rep:where", "rep:which", "rep:being",
    "rep:having", "rep:found", "rep:makes", "rep:makes", "rep:quite",
    "rep:often", "rep:might", "rep:shall", "rep:rather",
}


def anchor_quality_score(anchor: str) -> float:
    """Higher = better anchor type for entity-relation evidence."""
    if anchor.startswith("cap:"):
        # Multi-word named entities are best
        parts = anchor.split("_")
        return 3.0 + min(len(parts) - 1, 3) * 0.5
    elif anchor.startswith("quote:"):
        return 2.5
    elif anchor.startswith("num:") or re.match(r"\d", anchor):
        return 1.5
    elif anchor.startswith("rep:"):
        if anchor in FILLER_ANCHORS:
            return 0.0
        # Only keep rep: if it looks like a proper noun or specific term
        term = anchor.split(":", 1)[1] if ":" in anchor else anchor
        if term[0].isupper() or len(term) >= 7:
            return 1.0
        return 0.3
    return 0.5


def has_sensitive_content(cluster: dict) -> bool:
    """Check all texts and heldout for sensitive content."""
    all_text = " ".join(cluster.get("texts", []))
    if cluster.get("heldout_text"):
        all_text += " " + cluster["heldout_text"]
    return bool(SENSITIVE_RE.search(all_text))


def complementarity_score(cluster: dict) -> float:
    """Score complementary predicate/attribute evidence.
    
    Low Jaccard = different content words = different predicates/attributes.
    High unique content words = rich evidence.
    Combined into a single score favoring diverse, information-rich clusters.
    """
    jaccard = cluster.get("pairwise_content_jaccard_mean", 1.0)
    unique_cw = cluster.get("unique_content_words", 0)
    # Best: low jaccard (< 0.1) and many unique content words (> 50)
    jaccard_score = max(0, 1.0 - jaccard * 5)  # 0.0 → 1.0, 0.2 → 0.0
    richness_score = min(unique_cw / 60.0, 1.5)
    return jaccard_score * 0.6 + richness_score * 0.4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(DEFAULT_IN))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--target_words", type=int, default=200_000,
                    help="Target word budget for cluster dose (~2% of 10M)")
    ap.add_argument("--max_per_block", type=int, default=3)
    ap.add_argument("--min_anchor_quality", type=float, default=0.5)
    ap.add_argument("--require_heldout", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=43043)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all clusters
    in_path = pathlib.Path(args.input)
    clusters: list[dict] = []
    with in_path.open("r") as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))

    print(f"Loaded {len(clusters)} raw clusters")

    # === FILTER PASS 1: basic exclusions ===
    block_counts: dict[str, int] = collections.defaultdict(int)
    rejected = collections.Counter()
    passed: list[dict] = []

    for c in clusters:
        anchor = c.get("anchor", "")
        aq = anchor_quality_score(anchor)
        
        # Anchor quality gate
        if aq < args.min_anchor_quality:
            rejected["low_anchor_quality"] += 1
            continue
        
        # Sensitive content
        if has_sensitive_content(c):
            rejected["sensitive_content"] += 1
            continue
        
        # Word budget sanity: need at least 30 words for meaningful evidence
        words = c.get("words", 0)
        if words < 30 or words > 140:
            rejected["word_count_out_of_range"] += 1
            continue
        
        # Need at least 2 sentences
        texts = c.get("texts", [])
        if len(texts) < 2:
            rejected["too_few_sentences"] += 1
            continue
        
        # Source/block cap
        block_id = c.get("block_id", "")
        if block_counts[block_id] >= args.max_per_block:
            rejected["block_cap_exceeded"] += 1
            continue
        
        # Heldout preference (not a hard requirement unless --require_heldout)
        if args.require_heldout and not c.get("heldout_sentence_id"):
            rejected["no_heldout"] += 1
            continue
        
        block_counts[block_id] += 1
        c["_anchor_quality"] = aq
        c["_complementarity"] = complementarity_score(c)
        c["_has_heldout"] = c.get("heldout_sentence_id") is not None
        c["_composite_score"] = (
            aq * 0.3 +
            c["_complementarity"] * 0.5 +
            (0.3 if c["_has_heldout"] else 0.0)
        )
        passed.append(c)

    print(f"After filter pass 1: {len(passed)} clusters")
    print(f"Rejections: {dict(rejected)}")

    # === FILTER PASS 2: rank by composite score, select within word budget ===
    rng = random.Random(args.seed)
    # Slight randomization to avoid pure deterministic bias
    for c in passed:
        c["_composite_score"] += rng.uniform(0, 0.05)
    
    passed.sort(key=lambda c: c["_composite_score"], reverse=True)

    selected: list[dict] = []
    total_words = 0
    source_counts: dict[str, int] = collections.defaultdict(int)
    anchor_type_counts: dict[str, int] = collections.defaultdict(int)
    
    # Source cap: no single source > 50% of selected clusters
    SOURCE_CAP_FRAC = 0.50
    
    for c in passed:
        if total_words >= args.target_words:
            break
        
        source = c.get("source", "")
        # Soft source cap
        if source_counts[source] > len(selected) * SOURCE_CAP_FRAC + 10:
            continue
        
        words = c.get("words", 0)
        selected.append(c)
        total_words += words
        source_counts[source] += 1
        
        anchor = c.get("anchor", "")
        atype = anchor.split(":")[0] if ":" in anchor else "other"
        anchor_type_counts[atype] += 1

    print(f"Selected {len(selected)} clusters, {total_words} words")
    print(f"Source distribution: {dict(source_counts)}")
    print(f"Anchor type distribution: {dict(anchor_type_counts)}")

    # === OUTPUT ===
    # Filtered clusters JSONL (without internal scoring fields)
    out_jsonl = out_dir / "filtered_clusters.jsonl"
    heldout_clusters = 0
    with out_jsonl.open("w") as f:
        for c in selected:
            # Remove internal scoring keys
            out_c = {k: v for k, v in c.items() if not k.startswith("_")}
            f.write(json.dumps(out_c, ensure_ascii=False) + "\n")
            if c.get("heldout_sentence_id") is not None:
                heldout_clusters += 1

    # Held-out diagnostic file
    heldout_items: list[dict] = []
    for c in selected:
        if c.get("heldout_text") and c.get("heldout_sentence_id") is not None:
            heldout_items.append({
                "cluster_id": c["cluster_id"],
                "anchor": c["anchor"],
                "source": c.get("source", ""),
                "heldout_text": c["heldout_text"],
                "cluster_texts": c["texts"],
            })
    
    heldout_path = out_dir / "heldout_diagnostic.jsonl"
    with heldout_path.open("w") as f:
        for item in heldout_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Summary metadata
    summary = {
        "status": "QUALITY_FILTERED_CLUSTERS",
        "input": str(in_path),
        "raw_clusters": len(clusters),
        "after_filter_pass1": len(passed),
        "selected_clusters": len(selected),
        "selected_words": total_words,
        "target_words": args.target_words,
        "dose_fraction_of_10M": total_words / 10_000_000,
        "heldout_clusters": heldout_clusters,
        "heldout_diagnostic_items": len(heldout_items),
        "source_distribution": dict(source_counts),
        "anchor_type_distribution": dict(anchor_type_counts),
        "rejection_counts": dict(rejected),
        "filters_applied": {
            "min_anchor_quality": args.min_anchor_quality,
            "max_per_block": args.max_per_block,
            "sensitive_keywords": len(SENSITIVE_KEYWORDS),
            "require_heldout": args.require_heldout,
            "source_cap_fraction": SOURCE_CAP_FRAC,
            "target_words": args.target_words,
        },
        "output_files": {
            "filtered_clusters": str(out_jsonl),
            "heldout_diagnostic": str(heldout_path),
        },
        "non_leakage_statement": "No official AoA/CDI words, child curves, or downstream evaluation outputs used.",
    }

    summary_path = out_dir / "quality_filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps({
        "summary": str(summary_path),
        "filtered_clusters": str(out_jsonl),
        "heldout_diagnostic": str(heldout_path),
        "selected_clusters": len(selected),
        "selected_words": total_words,
        "heldout_clusters": heldout_clusters,
    }, indent=2))


if __name__ == "__main__":
    main()
