#!/usr/bin/env python3
"""research: Census of directional pair evidence in the compact-view 10M corpus.

The question: does the 10M BabyLM Strict-Small corpus contain enough naturally
occurring paired directional evidence to train a four-cell interaction loss?

We need sentence pairs where:
1. High lexical/syntactic overlap
2. A localized relational pivot differs (e.g., "above"/"below", "before"/"after")
3. The consequence changes accordingly
4. Both directions of the relation appear

This script:
1. Loads the compact-view 10M corpus
2. Extracts relational pivot tokens and their dependents per sentence
3. Counts directional pair candidates (sentences sharing structure but differing in pivot)
4. Estimates the usable pair inventory for a four-cell interaction loss
5. Also counts pivot-visible-dependent-masked training opportunities (for refined PGDC)
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path("experiments/archive/representation_and_objectives/data/corpus_directional_census")
NOTE = Path("research/notes/representation_and_objectives/corpus_directional_pair_census.md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Directional pivot pairs: words that reverse relational direction
DIRECTIONAL_PIVOT_PAIRS = [
    # spatial
    ("above", "below"), ("over", "under"), ("up", "down"), ("top", "bottom"),
    ("left", "right"), ("inside", "outside"), ("in", "out"), ("into", "out of"),
    ("before", "after"), ("behind", "in front of"), ("on", "off"),
    ("north", "south"), ("east", "west"), ("near", "far"),
    # temporal
    ("before", "after"), ("earlier", "later"), ("past", "future"),
    ("first", "last"), ("begin", "end"), ("start", "stop"),
    # causal/state
    ("hot", "cold"), ("warm", "cool"), ("open", "close"), ("opened", "closed"),
    ("rise", "fall"), ("rose", "fell"), ("rising", "falling"),
    ("increase", "decrease"), ("increased", "decreased"),
    ("grow", "shrink"), ("grew", "shrank"), ("expand", "contract"),
    ("melt", "freeze"), ("melted", "froze"), ("melting", "freezing"),
    ("wet", "dry"), ("light", "dark"), ("bright", "dim"),
    ("big", "small"), ("large", "small"), ("tall", "short"), ("long", "short"),
    ("fast", "slow"), ("heavy", "light"), ("strong", "weak"),
    ("hard", "soft"), ("loud", "quiet"), ("high", "low"),
    # comparative
    ("more", "less"), ("greater", "smaller"), ("better", "worse"),
    ("higher", "lower"), ("longer", "shorter"), ("faster", "slower"),
    # verb direction
    ("push", "pull"), ("pushed", "pulled"),
    ("give", "take"), ("gave", "took"), ("giving", "taking"),
    ("add", "remove"), ("added", "removed"),
    ("build", "destroy"), ("built", "destroyed"),
    ("create", "destroy"), ("created", "destroyed"),
    ("buy", "sell"), ("bought", "sold"),
    ("win", "lose"), ("won", "lost"),
    ("love", "hate"), ("loved", "hated"),
    ("agree", "disagree"), ("agreed", "disagreed"),
    ("accept", "reject"), ("accepted", "rejected"),
    ("connect", "disconnect"), ("connected", "disconnected"),
    # negation-adjacent
    ("can", "cannot"), ("will", "won't"), ("is", "isn't"),
    ("true", "false"), ("yes", "no"), ("positive", "negative"),
]

# Relational structure patterns (for PGDC pivot-visible dependent masking count)
CAUSAL_MARKERS = {
    "because", "since", "therefore", "so", "thus", "hence", "consequently",
    "causes", "caused", "causing", "leads", "led", "results", "resulted",
    "makes", "made", "forces", "forced", "prevents", "prevented",
    "enables", "enabled", "allows", "allowed",
    "if", "unless", "although", "despite", "whether",
}
SPATIAL_PREPS = {
    "above", "below", "over", "under", "behind", "beside", "between",
    "inside", "outside", "near", "far", "left", "right", "north", "south",
    "east", "west", "on", "off", "in", "out", "into", "onto", "upon",
    "through", "across", "along", "around", "toward", "towards", "away",
}
TEMPORAL_MARKERS = {
    "before", "after", "during", "when", "while", "until", "since",
    "then", "next", "first", "last", "finally", "meanwhile", "already",
    "soon", "later", "earlier", "previously", "subsequently",
}
COMPARATIVE_MARKERS = {
    "more", "less", "most", "least", "better", "worse", "best", "worst",
    "greater", "smaller", "larger", "higher", "lower", "faster", "slower",
    "longer", "shorter", "bigger", "taller", "heavier", "lighter",
    "than", "compared", "versus", "rather",
}


def load_corpus(path: Path) -> list[dict]:
    """Load JSONL corpus, return list of {text, words, ...}."""
    rows = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def tokenize_simple(text: str) -> list[str]:
    """Whitespace + basic punct split."""
    return text.lower().split()


def find_pivot_tokens(tokens: list[str]) -> dict:
    """Find all relational pivot tokens in a token list.
    Returns {pivot_word: [positions]}."""
    pivot_vocab = set()
    for a, b in DIRECTIONAL_PIVOT_PAIRS:
        for w in a.split():
            pivot_vocab.add(w.lower())
        for w in b.split():
            pivot_vocab.add(w.lower())
    
    result = defaultdict(list)
    for i, t in enumerate(tokens):
        clean = t.strip(".,;:!?\"'()[]{}").lower()
        if clean in pivot_vocab:
            result[clean].append(i)
    return dict(result)


def sentence_signature(tokens: list[str], pivot_positions: set) -> str:
    """Create a structural signature: replace pivot tokens with PIVOT, keep others.
    Two sentences sharing a signature but differing at PIVOT positions are candidates."""
    sig_tokens = []
    for i, t in enumerate(tokens):
        clean = t.strip(".,;:!?\"'()[]{}").lower()
        if i in pivot_positions:
            sig_tokens.append("__PIVOT__")
        else:
            sig_tokens.append(clean)
    return " ".join(sig_tokens)


def count_relational_structure(tokens: list[str]) -> dict:
    """Count relational tokens in the sentence."""
    tset = set(t.strip(".,;:!?\"'()[]{}").lower() for t in tokens)
    return {
        "causal": len(tset & CAUSAL_MARKERS),
        "spatial": len(tset & SPATIAL_PREPS),
        "temporal": len(tset & TEMPORAL_MARKERS),
        "comparative": len(tset & COMPARATIVE_MARKERS),
        "any_relational": int(bool(tset & (CAUSAL_MARKERS | SPATIAL_PREPS | TEMPORAL_MARKERS | COMPARATIVE_MARKERS))),
    }


def main():
    start = time.time()
    
    # Load the 10M compact corpus
    corpus_path = Path("experiments/archive/representation_and_objectives/data/fw_full_arms/fw_preserved_compact_view_10M.jsonl")
    if not corpus_path.exists():
        print(json.dumps({"error": f"corpus not found: {corpus_path}"}), flush=True)
        sys.exit(1)
    
    rows = load_corpus(corpus_path)
    print(json.dumps({"event": "corpus_loaded", "rows": len(rows),
                       "total_words": sum(r.get("words", 0) for r in rows)}), flush=True)
    
    # Split each row into sentences (simple split on . ! ?)
    sentences = []
    for i, row in enumerate(rows):
        text = row["text"]
        # Split on sentence boundaries
        for sent in re.split(r'(?<=[.!?])\s+', text):
            words = sent.split()
            if len(words) >= 5:  # at least 5 words
                sentences.append({"text": sent, "row_idx": i, "words": len(words),
                                  "source": row.get("source", "")})
    
    print(json.dumps({"event": "sentences_extracted", "n_sentences": len(sentences),
                       "n_rows": len(rows)}), flush=True)
    
    # === Part 1: Count relational structure density ===
    rel_counts = Counter()
    any_relational = 0
    pivot_sentences = 0
    total_pivots = 0
    
    for s in sentences:
        tokens = tokenize_simple(s["text"])
        rel = count_relational_structure(tokens)
        for k, v in rel.items():
            if v > 0:
                rel_counts[k] += 1
        if rel["any_relational"]:
            any_relational += 1
        
        pivots = find_pivot_tokens(tokens)
        if pivots:
            pivot_sentences += 1
            total_pivots += sum(len(v) for v in pivots.values())
    
    relational_density = {
        "total_sentences": len(sentences),
        "with_any_relational_marker": any_relational,
        "frac_any_relational": round(any_relational / max(1, len(sentences)), 4),
        "with_directional_pivot": pivot_sentences,
        "frac_directional_pivot": round(pivot_sentences / max(1, len(sentences)), 4),
        "total_pivot_tokens": total_pivots,
        "by_category": {k: v for k, v in sorted(rel_counts.items(), key=lambda x: -x[1])},
    }
    print(json.dumps({"event": "relational_density", **relational_density}), flush=True)
    
    # === Part 2: Find directional pair candidates ===
    # Build a pivot-pair lookup
    pivot_pair_map = {}  # word -> set of antonym words
    for a, b in DIRECTIONAL_PIVOT_PAIRS:
        for wa in a.lower().split():
            pivot_pair_map.setdefault(wa, set()).add(b.lower().split()[0])
        for wb in b.lower().split():
            pivot_pair_map.setdefault(wb, set()).add(a.lower().split()[0])
    
    # For each sentence, compute a "structural template" by replacing pivots with __PIVOT__
    # Then group sentences by template
    template_groups = defaultdict(list)
    
    for si, s in enumerate(sentences):
        tokens = tokenize_simple(s["text"])
        pivots = find_pivot_tokens(tokens)
        if not pivots:
            continue
        
        for pivot_word, positions in pivots.items():
            for pos in positions:
                # Create template by replacing this pivot position
                template_tokens = []
                for i, t in enumerate(tokens):
                    clean = t.strip(".,;:!?\"'()[]{}").lower()
                    if i == pos:
                        template_tokens.append("__PIVOT__")
                    else:
                        template_tokens.append(clean)
                
                # Use a context window around the pivot as key (±5 words)
                lo = max(0, pos - 5)
                hi = min(len(template_tokens), pos + 6)
                local_key = " ".join(template_tokens[lo:hi])
                
                template_groups[local_key].append({
                    "sent_idx": si, "pivot_word": pivot_word,
                    "pivot_pos": pos, "n_tokens": len(tokens),
                })
    
    # Find groups where different pivot words appear in the same template
    directional_pairs = []
    for key, members in template_groups.items():
        pivot_words = set(m["pivot_word"] for m in members)
        if len(pivot_words) < 2:
            continue
        # Check if any pair of pivot words are in our directional pair list
        for m1 in members:
            for m2 in members:
                if m1["sent_idx"] >= m2["sent_idx"]:
                    continue
                w1, w2 = m1["pivot_word"], m2["pivot_word"]
                if w1 != w2 and w2 in pivot_pair_map.get(w1, set()):
                    directional_pairs.append({
                        "template_key": key,
                        "sent1_idx": m1["sent_idx"],
                        "sent2_idx": m2["sent_idx"],
                        "pivot1": w1,
                        "pivot2": w2,
                        "sent1_preview": sentences[m1["sent_idx"]]["text"][:120],
                        "sent2_preview": sentences[m2["sent_idx"]]["text"][:120],
                    })
    
    # Deduplicate by sentence pair
    seen_pairs = set()
    unique_pairs = []
    for p in directional_pairs:
        pair_key = (min(p["sent1_idx"], p["sent2_idx"]), max(p["sent1_idx"], p["sent2_idx"]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_pairs.append(p)
    
    # Pivot pair type distribution
    pair_type_counts = Counter()
    for p in unique_pairs:
        pair_type_counts[f"{p['pivot1']}/{p['pivot2']}"] += 1
    
    pair_census = {
        "n_template_groups": len(template_groups),
        "n_multi_pivot_groups": sum(1 for k, v in template_groups.items()
                                     if len(set(m["pivot_word"] for m in v)) >= 2),
        "n_directional_pairs_raw": len(directional_pairs),
        "n_directional_pairs_unique": len(unique_pairs),
        "pair_type_distribution": dict(pair_type_counts.most_common(30)),
        "sample_pairs": unique_pairs[:20],
    }
    print(json.dumps({"event": "directional_pair_census",
                       "n_unique_pairs": len(unique_pairs),
                       "n_pair_types": len(pair_type_counts)}), flush=True)
    
    # === Part 3: Pivot-visible dependent masking opportunities ===
    # Count sentences where a relational pivot can be kept visible
    # while a dependent/consequence token can be masked
    pvdm_candidates = 0
    pvdm_total_maskable = 0
    
    for s in sentences:
        tokens = tokenize_simple(s["text"])
        tset = set(t.strip(".,;:!?\"'()[]{}").lower() for t in tokens)
        
        rel_positions = set()
        non_rel_content = set()
        
        for i, t in enumerate(tokens):
            clean = t.strip(".,;:!?\"'()[]{}").lower()
            if clean in (CAUSAL_MARKERS | SPATIAL_PREPS | TEMPORAL_MARKERS | COMPARATIVE_MARKERS):
                rel_positions.add(i)
            elif len(clean) > 2 and clean.isalpha():
                non_rel_content.add(i)
        
        if rel_positions and non_rel_content:
            # Count non-relational content tokens near (within 5 positions of) a pivot
            near_pivot = set()
            for rp in rel_positions:
                for np in non_rel_content:
                    if abs(rp - np) <= 5:
                        near_pivot.add(np)
            if near_pivot:
                pvdm_candidates += 1
                pvdm_total_maskable += len(near_pivot)
    
    pvdm_stats = {
        "sentences_with_pivot_visible_dependent_masking": pvdm_candidates,
        "frac_sentences": round(pvdm_candidates / max(1, len(sentences)), 4),
        "total_maskable_dependents": pvdm_total_maskable,
        "mean_maskable_per_eligible_sentence": round(pvdm_total_maskable / max(1, pvdm_candidates), 2),
    }
    print(json.dumps({"event": "pvdm_stats", **pvdm_stats}), flush=True)
    
    # === Save results ===
    census = {
        "status": "CORPUS_DIRECTIONAL_PAIR_CENSUS",
        "corpus_path": str(corpus_path),
        "elapsed_sec": round(time.time() - start, 1),
        "relational_density": relational_density,
        "directional_pair_census": pair_census,
        "pivot_visible_dependent_masking": pvdm_stats,
    }
    
    out_json = OUT_DIR / "corpus_directional_pair_census.json"
    out_json.write_text(json.dumps(census, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Save sample pairs
    pairs_jsonl = OUT_DIR / "directional_pairs.jsonl"
    with pairs_jsonl.open("w") as f:
        for p in unique_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    # Write note
    note_lines = [
        "# research — Corpus directional pair census",
        "",
        "## Purpose",
        "Before any mechanism training, quantify the density of usable paired directional",
        "evidence in the 10M compact-view corpus. This determines whether a four-cell",
        "interaction loss can find training signal, or whether we must rely on weaker mechanisms.",
        "",
        "## Key results",
        f"- Total sentences (≥5 words): {len(sentences):,}",
        f"- With any relational marker: {any_relational:,} ({relational_density['frac_any_relational']:.1%})",
        f"- With directional pivot token: {pivot_sentences:,} ({relational_density['frac_directional_pivot']:.1%})",
        f"- Unique directional pairs found: {len(unique_pairs):,}",
        f"- Pair types: {len(pair_type_counts)}",
        f"- Top pair types: {dict(pair_type_counts.most_common(10))}",
        "",
        f"## Pivot-visible dependent masking",
        f"- Eligible sentences: {pvdm_candidates:,} ({pvdm_stats['frac_sentences']:.1%})",
        f"- Maskable dependents: {pvdm_total_maskable:,}",
        f"- Mean per eligible sentence: {pvdm_stats['mean_maskable_per_eligible_sentence']}",
        "",
        "## Files",
        f"- Census JSON: `{out_json}`",
        f"- Directional pairs JSONL: `{pairs_jsonl}`",
    ]
    NOTE.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    
    print(json.dumps({
        "status": "CORPUS_DIRECTIONAL_PAIR_CENSUS",
        "n_sentences": len(sentences),
        "n_directional_pairs": len(unique_pairs),
        "pvdm_eligible": pvdm_candidates,
        "out_json": str(out_json),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
