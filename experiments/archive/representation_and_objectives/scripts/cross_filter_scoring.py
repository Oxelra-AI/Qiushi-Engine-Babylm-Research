#!/usr/bin/env python3
"""Score previously accepted rewrites under the stricter preservation standard.

Purpose: Measure how much acceptance disagreement between filters comes from surface
marker matching. The initial filter accepted 18,682 rewrites; the preservation standard
applies stricter checks on polarity, modality, causal direction, comparison
direction, entity preservation, and number preservation.

The measurement answers: if we applied the stricter filter to the initial accepted set,
how many would pass? This determines whether the preservation standard is
over-rejecting (hurting yield) or catching real semantic damage.
"""
import json, re, pathlib, sys, collections

A02_ACCEPTED = pathlib.Path("experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_accepted_rewrites.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/cross_filter")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import the research filter logic inline (simplified to avoid import issues) ──
# These are the key hard-check functions from fw_preservation_standard.py

NEGATION_WORDS = frozenset(["not","no","never","neither","nor","none","nothing",
    "nobody","nowhere","cannot","can't","won't","wouldn't","shouldn't",
    "couldn't","doesn't","didn't","don't","isn't","aren't","wasn't",
    "weren't","hasn't","haven't","hadn't","without","lack","lacks",
    "lacking","absent","impossible","unable","fail","fails","failed",
    "prevent","prevents","prevented","exclude","excludes","excluded",
    "refuse","refuses","refused","deny","denies","denied","reject",
    "rejects","rejected","despite","regardless","rarely","seldom","hardly",
    "scarcely","barely"])

MODALITY_WORDS = frozenset(["must","should","could","would","might","may",
    "shall","ought","need","dare","can","will","able","unable","possible",
    "impossible","necessary","required","mandatory","optional","likely",
    "unlikely","probable","improbable","certain","uncertain","definite",
    "perhaps","maybe","possibly","probably","certainly","necessarily",
    "inevitably","potentially","conceivably","supposedly","allegedly",
    "apparently","presumably","seemingly","ostensibly"])

CAUSAL_WORDS = frozenset(["cause","causes","caused","causing","because",
    "since","therefore","thus","hence","consequently","accordingly",
    "result","results","resulted","resulting","led","lead","leads",
    "leading","produce","produces","produced","producing","trigger",
    "triggers","triggered","triggering","induce","induces","induced",
    "inducing","generate","generates","generated","generating",
    "create","creates","created","creating","enable","enables","enabled",
    "due","owing","thanks"])

COMPARISON_UP = frozenset(["more","most","greater","better","higher","larger",
    "bigger","faster","stronger","longer","wider","deeper","heavier",
    "taller","richer","warmer","hotter","louder","brighter","superior",
    "increase","increased","improve","improved","exceed","exceeded",
    "surpass","surpassed","outperform","outperformed"])

COMPARISON_DOWN = frozenset(["less","least","fewer","worse","lower","smaller",
    "slower","weaker","shorter","narrower","shallower","lighter",
    "poorer","cooler","colder","quieter","darker","inferior",
    "decrease","decreased","reduce","reduced","decline","declined",
    "diminish","diminished"])

def extract_words(text):
    return set(re.findall(r'[a-z]+', text.lower()))

def extract_entities_simple(text):
    """Simple entity extraction: capitalized word sequences, excluding sentence-initial."""
    ents = set()
    words = text.split()
    for i, w in enumerate(words):
        # Skip sentence-initial words
        if i == 0:
            continue
        if i > 0 and words[i-1][-1] in '.!?':
            continue
        if re.match(r'^[A-Z][a-z]+$', w) and w not in ("The","This","That","These","Those",
            "However","Although","While","When","Where","Because","Since","After","Before",
            "Also","Furthermore","Moreover","Nevertheless","Therefore","Meanwhile",
            "Additionally","Similarly","But","And","Or","Yet","So","If","Then","There",
            "It","He","She","They","We","Our","His","Her","Its","Many","Some","Most",
            "Several","Few","All","Each","Every","Other","Another","Such","Much"):
            ents.add(w)
    # Also extract multi-word proper nouns
    for m in re.finditer(r'(?<!\. )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text):
        ents.add(m.group(1))
    return ents

def extract_numbers(text):
    return set(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))

def check_preservation(source_text, rewrite_text):
    """Apply research-like preservation checks. Returns (usable, hard_failures, soft_flags)."""
    hard_failures = []
    soft_flags = []
    
    rw = rewrite_text.strip()
    if not rw or len(rw) < 5:
        hard_failures.append("empty_or_malformed")
        return False, hard_failures, soft_flags
    
    src_words = source_text.split()
    rw_words = rw.split()
    
    # Too short
    if len(rw_words) < 4 or (len(src_words) > 0 and len(rw_words) / len(src_words) < 0.2):
        hard_failures.append("too_short")
    
    # Copy-like
    src_set = set(w.lower() for w in src_words)
    rw_set = set(w.lower() for w in rw_words)
    if src_set and rw_set:
        jaccard = len(src_set & rw_set) / len(src_set | rw_set)
        if jaccard > 0.95:
            hard_failures.append("copy_like")
    
    src_lower = extract_words(source_text)
    rw_lower = extract_words(rewrite_text)
    
    # Entity preservation
    src_ents = extract_entities_simple(source_text)
    rw_ents = extract_entities_simple(rewrite_text)
    if src_ents:
        missing = src_ents - rw_ents
        if missing and len(missing) / len(src_ents) > 0.5:
            hard_failures.append("entity_loss")
    
    # Number preservation
    src_nums = extract_numbers(source_text)
    rw_nums = extract_numbers(rewrite_text)
    if src_nums:
        missing_nums = src_nums - rw_nums
        if missing_nums:
            hard_failures.append("number_loss")
    
    # Polarity
    src_neg = NEGATION_WORDS & src_lower
    rw_neg = NEGATION_WORDS & rw_lower
    if src_neg and not rw_neg:
        hard_failures.append("polarity_lost")
    elif not src_neg and rw_neg:
        soft_flags.append("polarity_added")
    
    # Modality
    src_mod = MODALITY_WORDS & src_lower
    rw_mod = MODALITY_WORDS & rw_lower
    if src_mod and not rw_mod:
        soft_flags.append("modality_lost")  # soft, not hard
    
    # Causal
    src_causal = CAUSAL_WORDS & src_lower
    rw_causal = CAUSAL_WORDS & rw_lower
    if src_causal and not rw_causal:
        hard_failures.append("causal_direction_lost")
    
    # Comparison direction
    src_up = COMPARISON_UP & src_lower
    src_down = COMPARISON_DOWN & src_lower
    rw_up = COMPARISON_UP & rw_lower
    rw_down = COMPARISON_DOWN & rw_lower
    if (src_up and not src_down) and (rw_down and not rw_up):
        hard_failures.append("comparison_direction_reversed")
    elif (src_down and not src_up) and (rw_up and not rw_down):
        hard_failures.append("comparison_direction_reversed")
    elif (src_up or src_down) and not (rw_up or rw_down):
        soft_flags.append("comparison_lost")
    
    usable = len(hard_failures) == 0
    return usable, hard_failures, soft_flags

def main():
    print("Loading previously accepted rewrites...")
    a02_rows = []
    with open(A02_ACCEPTED) as f:
        for line in f:
            a02_rows.append(json.loads(line))
    print(f"  Previously accepted rows: {len(a02_rows)}")
    
    # Apply research preservation filter
    print("Applying research preservation filter...")
    usable = 0
    failure_counts = collections.Counter()
    soft_counts = collections.Counter()
    rejected_examples = []
    
    for row in a02_rows:
        src = row.get("source_text", "")
        rw = row.get("rewrite_text", "")
        ok, hard, soft = check_preservation(src, rw)
        if ok:
            usable += 1
        else:
            for h in hard:
                failure_counts[h] += 1
            if len(rejected_examples) < 30:
                rejected_examples.append({
                    "prompt_id": row.get("prompt_id", ""),
                    "source": src[:120],
                    "rewrite": rw[:120],
                    "hard_failures": hard,
                    "soft_flags": soft,
                })
        for s in soft:
            soft_counts[s] += 1
    
    rate = usable / len(a02_rows) if a02_rows else 0
    print(f"\n  Accepted rows passing preservation filter: {usable}/{len(a02_rows)} ({rate:.3f})")
    print(f"  Hard failure distribution:")
    for k, v in failure_counts.most_common():
        print(f"    {k}: {v} ({v/len(a02_rows):.3f})")
    print(f"  Soft flag distribution:")
    for k, v in soft_counts.most_common():
        print(f"    {k}: {v} ({v/len(a02_rows):.3f})")
    
    # Compare with research existing filter results
    # research found: old_usable 8386/12152 = 0.690, pilot 89/256 = 0.348
    print(f"\n  Reference: existing-rewrite usable rate: 0.690 (8386/12152)")
    print(f"  Reference: research hard-pilot usable rate: 0.348 (89/256)")
    print(f"  Previously accepted set usable rate: {rate:.3f} ({usable}/{len(a02_rows)})")
    
    # Save results
    result = {
        "status": "A02_CROSS_FILTER",
        "a02_total": len(a02_rows),
        "a01_step100_usable": usable,
        "usable_rate": rate,
        "hard_failure_counts": dict(failure_counts.most_common()),
        "soft_flag_counts": dict(soft_counts.most_common()),
        "rejected_examples": rejected_examples[:20],
        "reference_step100_existing_rate": 0.690,
        "reference_step100_pilot_rate": 0.348,
    }
    
    out_json = OUT_DIR / "cross_filter.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    note_path = pathlib.Path("research/notes/representation_and_objectives/compact_experience_filter.md")
    note_path.parent.mkdir(parents=True, exist_ok=True)
    with open(note_path, "w") as f:
        f.write(f"# Compact-view source-filter scoring\n\n")
        f.write(f"Applied research preservation standard to A02's 18,682 accepted rewrites.\n\n")
        f.write(f"- A02 accepted: {len(a02_rows)}\n")
        f.write(f"- Pass research filter: {usable} ({rate:.3f})\n")
        f.write(f"- Top hard failures:\n")
        for k, v in failure_counts.most_common(5):
            f.write(f"  - {k}: {v}\n")
        f.write(f"\nReference: research existing-A02 rate 0.690, hard-pilot rate 0.348\n")
        f.write(f"Artifacts: `{out_json}`\n")
    
    print(f"\nSaved: {out_json}")
    return result

if __name__ == "__main__":
    main()
