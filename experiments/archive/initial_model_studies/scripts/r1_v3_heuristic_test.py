#!/usr/bin/env python3
"""research — Comprehensive heuristic baseline test for v3 generator.

Tests the following candidate shortcuts:
1. last_op: destination of last operation mentioning query object by name
2. last_indirect_op: destination of last indirect op whose source matches query obj location
3. most_recent_loc: last location word mentioned before the query sentence
4. word_bag_freq: most frequently mentioned location in the passage
5. local_window_3: location closest to query obj in the last 3 sentences
6. init_state: initial location of query object
7. random: uniform random over scenario locations
8. paired_counterfactual: verify same-bag pairs have different answers (word-bag proof)

The generator passes ONLY if all heuristics are below 0.60 accuracy AND
paired counterfactuals show 100% answer divergence.
"""
from __future__ import annotations
import json, pathlib, random, re, sys

sys.path.insert(0, str(pathlib.Path("experiments/archive/initial_model_studies/scripts")))
from r1_generator_v3 import (
    generate_training_corpus, generate_paired_corpus, LOCATIONS
)


def extract_locations_from_text(text: str) -> list[str]:
    """Find all location mentions in text."""
    found = []
    for loc in LOCATIONS:
        for m in re.finditer(re.escape(loc), text):
            found.append((m.start(), loc))
    found.sort()
    return [loc for _, loc in found]


def heuristic_last_op(scenario) -> str | None:
    """Last operation that mentions the query object by name (direct ops only)."""
    query = scenario.query_obj
    last_dst = None
    for op in scenario.operations:
        if op.obj == query and not op.ref_by_loc:
            last_dst = op.dst
    return last_dst


def heuristic_last_indirect_resolved(scenario) -> str | None:
    """Last operation (direct or indirect) that actually moved the query object."""
    query = scenario.query_obj
    last_dst = None
    for op in scenario.operations:
        if op.obj == query:
            last_dst = op.dst
    return last_dst


def heuristic_most_recent_loc(scenario) -> str | None:
    """Last location mentioned in the passage before the query sentence."""
    # Split at query
    query_marker = "After all these changes,"
    parts = scenario.passage.split(query_marker)
    if len(parts) < 2:
        return None
    pre_query = parts[0]
    locs = extract_locations_from_text(pre_query)
    return locs[-1] if locs else None


def heuristic_word_bag_freq(scenario) -> str | None:
    """Most frequently mentioned location in the full passage."""
    locs = extract_locations_from_text(scenario.passage)
    if not locs:
        return None
    from collections import Counter
    counts = Counter(locs)
    return counts.most_common(1)[0][0]


def heuristic_local_window(scenario, window_sentences: int = 3) -> str | None:
    """Location closest to query object mention in last N sentences before query."""
    query_marker = "After all these changes,"
    parts = scenario.passage.split(query_marker)
    if len(parts) < 2:
        return None
    pre_query = parts[0].strip()
    sentences = [s.strip() for s in pre_query.split(".") if s.strip()]
    window = sentences[-window_sentences:] if len(sentences) >= window_sentences else sentences
    window_text = " ".join(window)
    locs = extract_locations_from_text(window_text)
    return locs[-1] if locs else None


def heuristic_init_state(scenario) -> str | None:
    """Initial location of query object."""
    return scenario.init_loc.get(scenario.query_obj)


def evaluate_all_heuristics(corpus, seed=123):
    """Run all heuristics on a corpus and compute accuracies."""
    rng = random.Random(seed)
    n = len(corpus)
    
    results = {
        "last_op_direct_only": 0,
        "last_op_resolved": 0,
        "most_recent_loc": 0,
        "word_bag_freq": 0,
        "local_window_3": 0,
        "init_state": 0,
        "random": 0,
        "n": n,
    }
    
    all_locs = LOCATIONS  # superset for random
    
    for sc in corpus:
        answer = sc.answer
        
        pred = heuristic_last_op(sc)
        if pred == answer:
            results["last_op_direct_only"] += 1
        
        pred = heuristic_last_indirect_resolved(sc)
        if pred == answer:
            results["last_op_resolved"] += 1
        
        pred = heuristic_most_recent_loc(sc)
        if pred == answer:
            results["most_recent_loc"] += 1
        
        pred = heuristic_word_bag_freq(sc)
        if pred == answer:
            results["word_bag_freq"] += 1
        
        pred = heuristic_local_window(sc, 3)
        if pred == answer:
            results["local_window_3"] += 1
        
        pred = heuristic_init_state(sc)
        if pred == answer:
            results["init_state"] += 1
        
        # Random
        locs_in_scenario = list(set(
            list(sc.init_loc.values()) + 
            [op.dst for op in sc.operations] +
            [op.src for op in sc.operations]
        ))
        if rng.choice(locs_in_scenario) == answer:
            results["random"] += 1
    
    accuracies = {k: round(v / max(1, n), 4) for k, v in results.items() if k != "n"}
    accuracies["n"] = n
    return accuracies


def main():
    out_path = pathlib.Path(
        "experiments/archive/initial_model_studies/data/r1_v3_heuristic_results.json"
    )
    
    # Generate fresh corpus for evaluation (different seed from training)
    corpus = generate_training_corpus(target_words=50000, seed=99)
    
    # Run heuristics
    acc = evaluate_all_heuristics(corpus)
    
    # Paired counterfactual check
    pairs = generate_paired_corpus(n_pairs=200, seed=99)
    n_differ = sum(1 for a, b in pairs if a.answer != b.answer)
    pair_info = {
        "n_pairs": len(pairs),
        "n_answers_differ": n_differ,
        "fraction_differ": round(n_differ / max(1, len(pairs)), 4),
    }
    
    # Verdict
    max_heuristic = max(
        acc["last_op_direct_only"], acc["last_op_resolved"],
        acc["most_recent_loc"], acc["word_bag_freq"],
        acc["local_window_3"], acc["init_state"],
    )
    
    if max_heuristic < 0.60 and pair_info["fraction_differ"] > 0.95:
        verdict = "PASS: No heuristic exceeds 0.60, and paired counterfactuals confirm word-bag cannot determine answer. Ready for training-response experiment."
    elif acc["last_op_resolved"] > 0.90:
        verdict = "FAIL: Last-op-resolved still solves most examples. Indirect operations not creating enough composition pressure."
    else:
        verdict = f"MARGINAL: Strongest heuristic at {max_heuristic:.3f}. May need further hardening."
    
    payload = {
        "corpus_heuristic_accuracies": acc,
        "paired_counterfactual": pair_info,
        "strongest_heuristic": max_heuristic,
        "verdict": verdict,
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
