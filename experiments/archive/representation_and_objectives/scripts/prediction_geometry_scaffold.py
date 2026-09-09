#!/usr/bin/env python3
"""research: Build prediction-geometry scaffold for causal topology × data 2×2 experiment.

Corrected design:
  - Same aligned pair-token multiset across all arms
  - Topology axis: one-way (source always first) vs reciprocal (balanced random order)
  - Data axis: own compact rewrite vs extractive (contiguous source span, 100% source overlap)
  - Adjbreak: external correspondence perturbation (same-length cyclic derangement)
  - Copied-vs-noncopied target analysis: by trainer, not scaffold

Arms:
  1. own_compact_oneway        4. extractive_oneway       (optional) adjbreak_oneway
  2. own_compact_reciprocal    5. extractive_reciprocal   (optional) adjbreak_reciprocal

Extractive: contiguous source span of same word count as compact rewrite.
  100% source overlap — the literal replay baseline.
  Compare with compact's ~83% overlap to isolate semantic restructuring.

Output: per-arm pair JSONL + manifest. Trainer handles filler/packing/tokenization.
"""

import json, pathlib, hashlib, random, statistics, collections, sys

PAIR_FILE = pathlib.Path("experiments/archive/frontier_consolidation/data"
    "density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl")
FILLER_FILE = pathlib.Path("experiments/archive/frontier_consolidation/data"
    "causal_transfer_scaffold/filler_rows.jsonl")
TOKENIZER_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data"
    "causal_transfer_scaffold/neutral_tokenizer")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/prediction_geometry_scaffold")


def build_extractive(source_text, target_word_count, pair_seed):
    """Contiguous source span of target_word_count words. Random start (pair-specific seed).
    If source is shorter, return full source."""
    words = source_text.split()
    if len(words) <= target_word_count:
        return source_text
    rng = random.Random(pair_seed)
    start = rng.randint(0, len(words) - target_word_count)
    return " ".join(words[start:start + target_word_count])


def build_derangement(pairs):
    """Same-length cyclic derangement: each pair gets the next pair's rewrite within
    its rewrite-length bucket. Excludes singleton-length pairs."""
    by_len = collections.defaultdict(list)
    for p in pairs:
        by_len[p["rewrite_words"]].append(p)
    deranged = {}
    excluded = []
    for length, group in sorted(by_len.items()):
        if len(group) < 2:
            excluded.extend(g["pair_id"] for g in group)
            continue
        for i, p in enumerate(group):
            donor = group[(i + 1) % len(group)]
            deranged[p["pair_id"]] = donor["rewrite_text"]
    return deranged, excluded


def word_overlap_frac(source, sideb):
    """Fraction of sideb words (lowered) found in source word set."""
    src_set = set(source.lower().split())
    sb_words = sideb.lower().split()
    return sum(1 for w in sb_words if w in src_set) / max(len(sb_words), 1)


def percentile(vals, p):
    s = sorted(vals)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load pairs ──
    pairs = [json.loads(l) for l in PAIR_FILE.open()]
    print(f"Loaded {len(pairs)} pairs")

    # ── Build extractive rewrites ──
    extractives = {}
    edge_cases = 0
    for p in pairs:
        seed = int(hashlib.sha256(p["pair_id"].encode()).hexdigest()[:8], 16)
        ext = build_extractive(p["source_text"], p["rewrite_words"], seed)
        extractives[p["pair_id"]] = ext
        if len(p["source_text"].split()) <= p["rewrite_words"]:
            edge_cases += 1
    print(f"Extractive rewrites: {len(extractives)} built, {edge_cases} edge cases")

    # ── Build adjbreak derangement ──
    deranged, excl_ids = build_derangement(pairs)
    print(f"Adjbreak derangement: {len(deranged)} pairs, {len(excl_ids)} excluded singletons")

    # ── Overlap statistics ──
    ov_compact = [word_overlap_frac(p["source_text"], p["rewrite_text"]) for p in pairs]
    ov_extract = [word_overlap_frac(p["source_text"], extractives[p["pair_id"]]) for p in pairs]
    ov_adjbrk  = [word_overlap_frac(p["source_text"], deranged[p["pair_id"]])
                  for p in pairs if p["pair_id"] in deranged]

    for label, ov in [("compact", ov_compact), ("extractive", ov_extract), ("adjbreak", ov_adjbrk)]:
        print(f"  {label:12s}: mean={statistics.mean(ov):.4f}  med={statistics.median(ov):.4f}  "
              f"p10={percentile(ov,0.1):.4f}  p90={percentile(ov,0.9):.4f}")

    # ── Extractive length-match audit ──
    len_mismatch = sum(1 for p in pairs
                       if len(extractives[p["pair_id"]].split()) != p["rewrite_words"])
    print(f"Extractive-compact word-count mismatches: {len_mismatch}")

    # ── Build arm JSONLs ──
    RECIP_SEED = 213042
    arm_configs = [
        ("own_compact_oneway",      "own_compact", False),
        ("own_compact_reciprocal",  "own_compact", True),
        ("extractive_oneway",       "extractive",  False),
        ("extractive_reciprocal",   "extractive",  True),
        ("adjbreak_oneway",         "adjbreak",    False),
        ("adjbreak_reciprocal",     "adjbreak",    True),
    ]

    rng_recip = random.Random(RECIP_SEED)
    arm_stats = {}

    for arm_name, dtype, recip in arm_configs:
        fp = OUT_DIR / f"{arm_name}.jsonl"
        rows = []
        n_sf = n_bf = tot_src = tot_sb = skip = 0

        for p in pairs:
            src = p["source_text"]
            if dtype == "own_compact":
                sb = p["rewrite_text"]
            elif dtype == "extractive":
                sb = extractives[p["pair_id"]]
            elif dtype == "adjbreak":
                if p["pair_id"] not in deranged:
                    skip += 1
                    continue
                sb = deranged[p["pair_id"]]

            source_first = (not recip) or (rng_recip.random() < 0.5)
            if source_first:
                seg_a, seg_b = src, sb
                n_sf += 1
            else:
                seg_a, seg_b = sb, src
                n_bf += 1

            tot_src += len(src.split())
            tot_sb  += len(sb.split())

            rows.append(json.dumps({
                "pair_id": p["pair_id"],
                "segment_a": seg_a,
                "segment_b": seg_b,
                "a_is_source": source_first,
                "data_type": dtype,
                "source_wc": len(src.split()),
                "sideb_wc": len(sb.split()),
            }, ensure_ascii=False))

        fp.write_text("\n".join(rows) + "\n", encoding="utf-8")
        arm_stats[arm_name] = dict(
            rows=len(rows), skipped=skip, source_first=n_sf, sideb_first=n_bf,
            total_source_words=tot_src, total_sideb_words=tot_sb,
            total_pair_words=tot_src + tot_sb, file=str(fp))
        print(f"  {arm_name}: {len(rows)} rows  sf={n_sf} bf={n_bf}  "
              f"pair_words={tot_src+tot_sb}  skip={skip}")

    # ── Manifest ──
    def ov_stats(vals):
        return dict(mean=statistics.mean(vals), median=statistics.median(vals),
                    p10=percentile(vals, 0.1), p90=percentile(vals, 0.9),
                    min=min(vals), max=max(vals))

    manifest = {
        "status": "PREDICTION_GEOMETRY_SCAFFOLD_BUILT",
        "design": ("2×2 topology (oneway/reciprocal) × data (own_compact/extractive) "
                   "+ adjbreak perturbation; revised prediction geometry"),
        "pair_source": str(PAIR_FILE),
        "total_pairs": len(pairs),
        "filler_reference": str(FILLER_FILE),
        "filler_words": 9576489,
        "tokenizer_reference": str(TOKENIZER_DIR),
        "extractive_method": ("Contiguous source span of same word count as compact rewrite; "
                              "random start (pair-specific SHA256 seed); 100% source overlap"),
        "adjbreak_method": "Same-length cyclic derangement; singletons excluded",
        "adjbreak_excluded_pair_ids": excl_ids,
        "extractive_edge_cases": edge_cases,
        "extractive_length_mismatches": len_mismatch,
        "reciprocal_seed": RECIP_SEED,
        "overlap": {
            "compact_to_source": ov_stats(ov_compact),
            "extractive_to_source": ov_stats(ov_extract),
            "adjbreak_to_source": ov_stats(ov_adjbrk),
        },
        "arms": arm_stats,
    }
    mf = OUT_DIR / "manifest.json"
    mf.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest: {mf}")
    print(f"Status: {manifest['status']}")


if __name__ == "__main__":
    main()
