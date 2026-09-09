#!/usr/bin/env python3
"""research — Heuristic baseline test for R1 generated data.

Tests whether simple baselines can predict the query answer without multi-step
composition. If they can, the generator doesn't create binding pressure.

Heuristics tested:
1. last_op: destination of the last operation mentioning the query object
2. init_state: initial location of the query object  
3. random: uniformly random container from the scenario
4. most_recent_any: destination of the most recent operation (any object)
5. first_op: destination of the first operation mentioning the query object

If last_op accuracy is high in 'ordered' mode, the generator is TOO EASY and
the model can solve it without true multi-step composition.
"""
from __future__ import annotations
import json, pathlib, random, sys
sys.path.insert(0, str(pathlib.Path("experiments/archive/initial_model_studies/scripts")))
from r1_generator import generate_corpus, DOMAINS


def evaluate_baselines(corpus, seed=123):
    rng = random.Random(seed)
    results = {"last_op": 0, "init_state": 0, "random": 0,
               "most_recent_any": 0, "first_op": 0, "total": 0}
    
    for sc in corpus:
        query_obj = sc.query_obj
        answer = sc.answer
        ops = sc.operations
        containers = list(set(
            [sc.init_location[o] for o in sc.init_location] +
            [op["dst"] for op in ops] + [op["src"] for op in ops]
        ))
        results["total"] += 1
        
        # 1. last_op: last operation mentioning query_obj as the moved object
        last_dst = None
        for op in ops:
            if op["obj"] == query_obj:
                last_dst = op["dst"]
        if last_dst == answer:
            results["last_op"] += 1
        
        # 2. init_state: initial location
        if sc.init_location[query_obj] == answer:
            results["init_state"] += 1
        
        # 3. random: pick random container
        if rng.choice(containers) == answer:
            results["random"] += 1
        
        # 4. most_recent_any: destination of the very last operation
        if ops and ops[-1]["dst"] == answer:
            results["most_recent_any"] += 1
        
        # 5. first_op: first operation mentioning query_obj
        first_dst = None
        for op in ops:
            if op["obj"] == query_obj:
                first_dst = op["dst"]
                break
        if first_dst == answer:
            results["first_op"] += 1
    
    n = max(1, results["total"])
    accuracies = {k: round(v / n, 4) for k, v in results.items() if k != "total"}
    accuracies["n"] = n
    return accuracies


def main():
    out_path = pathlib.Path("experiments/archive/initial_model_studies/data/r1_heuristic_baselines.json")
    
    # Generate fresh corpora for evaluation
    ordered = generate_corpus(target_words=50000, mode="ordered", seed=99)
    independent = generate_corpus(target_words=50000, mode="independent", seed=99)
    
    ordered_acc = evaluate_baselines(ordered)
    independent_acc = evaluate_baselines(independent)
    
    payload = {
        "ordered_mode_baselines": ordered_acc,
        "independent_mode_baselines": independent_acc,
        "interpretation": {
            "last_op_ordered": "If high (>0.9), the answer is trivially predictable from the last operation mentioning the query object. Generator is TOO EASY for composition learning.",
            "last_op_independent": "Should be ~1.0 by design since each object is moved at most once.",
            "init_state_ordered": "Should be low if objects are actually moved.",
            "random_baseline": "Should be ~1/n_containers, roughly 0.15-0.25.",
        },
        "verdict": None,
    }
    
    # Verdict
    if ordered_acc["last_op"] > 0.90:
        payload["verdict"] = "FAIL: ordered mode is solvable by last-operation heuristic. Generator must be redesigned to require true multi-step composition."
    elif ordered_acc["last_op"] < 0.50:
        payload["verdict"] = "PASS: last-operation heuristic fails significantly on ordered mode. Multi-step composition may be needed."
    else:
        payload["verdict"] = "MARGINAL: last-operation heuristic has moderate success. Consider strengthening non-commutative or indirect operations."
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
