#!/usr/bin/env python3
"""research — R1 state-dynamics generator v3: counterfactual-paired, indirect ops.

Core mechanism: operations reference entities by CURRENT LOCATION, not by name.
  "The item in the kitchen was moved to the garage."
Which item this refers to depends on prior state. Swapping operation order
changes which entity is affected, producing different final states from the
same word bag.

Two output modes:
  'paired': generates counterfactual pairs (same token bag, different order,
            different answer) for heuristic verification and binding probes.
  'corpus': generates training passages (ordered, with indirect ops).

Vocabulary is independent of official BabyLM Entity Tracking (no boxes 1-7,
no BNC-100 nouns).
"""
from __future__ import annotations
import argparse, json, pathlib, random
from dataclasses import dataclass, field
from collections import Counter

# ── Vocabulary pools ───────────────────────────────────────────────────────────

LOCATIONS = [
    "the kitchen", "the garage", "the attic", "the cellar", "the study",
    "the porch", "the closet", "the shed", "the vault", "the loft",
    "the hallway", "the pantry",
]

ITEMS = [
    "the lamp", "the clock", "the vase", "the mirror", "the rug",
    "the painting", "the cushion", "the blanket", "the candle", "the plant",
    "the photo", "the trophy", "the statue", "the basket", "the fan",
    "the radio", "the stool", "the hammer", "the broom", "the kettle",
]


@dataclass
class Operation:
    """One state-changing operation."""
    kind: str           # 'direct' or 'indirect'
    obj: str            # the actual object affected (resolved)
    src: str            # source location
    dst: str            # destination location
    text: str           # natural language sentence
    ref_by_loc: bool    # True if text uses location-based reference


@dataclass
class Scenario:
    init_loc: dict[str, str]       # object -> initial location
    operations: list[Operation]
    final_loc: dict[str, str]      # object -> final location
    query_obj: str
    answer: str
    passage: str
    word_count: int
    n_indirect: int


def resolve_indirect(state: dict[str, str], loc: str) -> str | None:
    """Find which object is currently at `loc`. Returns None if empty/ambiguous."""
    objs_at = [o for o, l in state.items() if l == loc]
    if len(objs_at) == 1:
        return objs_at[0]
    return None


def make_indirect_text(rng, src_loc, dst_loc):
    """Generate indirect-reference operation text."""
    templates = [
        f"The item in {src_loc} was moved to {dst_loc}.",
        f"Whatever was in {src_loc} got transferred to {dst_loc}.",
        f"Someone took what was in {src_loc} and placed it in {dst_loc}.",
    ]
    return rng.choice(templates)


def make_direct_text(rng, obj, src, dst):
    """Generate direct-reference operation text."""
    templates = [
        f"{obj} was moved from {src} to {dst}.",
        f"Someone took {obj} from {src} and put it in {dst}.",
        f"{obj} ended up in {dst} after being in {src}.",
    ]
    return rng.choice(templates)


def generate_operations(
    rng: random.Random,
    items: list[str],
    locations: list[str],
    init_loc: dict[str, str],
    n_ops: int = 5,
    indirect_fraction: float = 0.6,
) -> tuple[list[Operation], dict[str, str]]:
    """Generate a sequence of operations, some indirect (location-referenced)."""
    state = dict(init_loc)
    ops = []
    
    for _ in range(n_ops):
        use_indirect = rng.random() < indirect_fraction
        
        if use_indirect:
            # Pick a location with exactly one item (so reference is unambiguous)
            occupied = {}
            for obj, loc in state.items():
                occupied.setdefault(loc, []).append(obj)
            single_locs = [loc for loc, objs in occupied.items() if len(objs) == 1]
            
            if single_locs:
                src_loc = rng.choice(single_locs)
                obj = occupied[src_loc][0]
                dst_candidates = [l for l in locations if l != src_loc]
                dst_loc = rng.choice(dst_candidates)
                text = make_indirect_text(rng, src_loc, dst_loc)
                ops.append(Operation(
                    kind="indirect", obj=obj, src=src_loc, dst=dst_loc,
                    text=text, ref_by_loc=True,
                ))
                state[obj] = dst_loc
                continue
        
        # Direct operation
        obj = rng.choice(items)
        src = state[obj]
        dst_candidates = [l for l in locations if l != src]
        dst = rng.choice(dst_candidates)
        text = make_direct_text(rng, obj, src, dst)
        ops.append(Operation(
            kind="direct", obj=obj, src=src, dst=dst,
            text=text, ref_by_loc=False,
        ))
        state[obj] = dst
    
    return ops, state


def build_passage(init_loc, ops, query_obj, final_loc) -> str:
    """Assemble a passage from init, operations, and query."""
    lines = []
    for obj, loc in init_loc.items():
        lines.append(f"Initially, {obj} is in {loc}.")
    for op in ops:
        lines.append(op.text)
    answer = final_loc[query_obj]
    lines.append(f"After all these changes, {query_obj} is in {answer}.")
    return " ".join(lines)


def generate_scenario(
    rng: random.Random,
    n_locs: int = 5,
    n_items: int = 4,
    n_ops: int = 5,
    indirect_fraction: float = 0.6,
) -> Scenario:
    """Generate one ordered state-dynamics passage with indirect operations."""
    locs = rng.sample(LOCATIONS, min(n_locs, len(LOCATIONS)))
    items = rng.sample(ITEMS, min(n_items, len(ITEMS)))
    init_loc = {item: rng.choice(locs) for item in items}
    
    ops, final_loc = generate_operations(
        rng, items, locs, init_loc, n_ops, indirect_fraction
    )
    
    # Choose query: prefer objects affected by indirect ops
    indirect_objs = [op.obj for op in ops if op.ref_by_loc]
    if indirect_objs:
        query_obj = rng.choice(indirect_objs)
    else:
        moved = [op.obj for op in ops]
        query_obj = rng.choice(moved) if moved else rng.choice(items)
    
    # Add 2-3 DISTRACTOR operations after the last query-relevant op.
    # These move OTHER objects to OTHER locations, defeating most-recent-loc
    # and local-window heuristics.
    other_items = [it for it in items if it != query_obj]
    n_distract = rng.randint(2, 3)
    for _ in range(min(n_distract, len(other_items))):
        d_obj = rng.choice(other_items)
        d_src = final_loc[d_obj]
        d_candidates = [l for l in locs if l != d_src and l != final_loc[query_obj]]
        if not d_candidates:
            d_candidates = [l for l in locs if l != d_src]
        if not d_candidates:
            continue
        d_dst = rng.choice(d_candidates)
        d_text = make_direct_text(rng, d_obj, d_src, d_dst)
        ops.append(Operation(
            kind="direct", obj=d_obj, src=d_src, dst=d_dst,
            text=d_text, ref_by_loc=False,
        ))
        final_loc[d_obj] = d_dst
    
    passage = build_passage(init_loc, ops, query_obj, final_loc)
    n_indirect = sum(1 for op in ops if op.ref_by_loc)
    
    return Scenario(
        init_loc=init_loc, operations=ops, final_loc=final_loc,
        query_obj=query_obj, answer=final_loc[query_obj],
        passage=passage, word_count=len(passage.split()),
        n_indirect=n_indirect,
    )


def try_generate_counterfactual_pair(
    rng: random.Random,
    n_locs: int = 5,
    n_items: int = 4,
    n_ops: int = 5,
    indirect_fraction: float = 0.7,
    max_attempts: int = 50,
) -> tuple[Scenario, Scenario] | None:
    """Try to generate a counterfactual pair: same init + same op texts in
    different order producing different final states for the query object.
    
    Returns (scenario_A, scenario_B) or None if no valid pair found.
    """
    locs = rng.sample(LOCATIONS, min(n_locs, len(LOCATIONS)))
    items = rng.sample(ITEMS, min(n_items, len(ITEMS)))
    init_loc = {item: rng.choice(locs) for item in items}
    
    # Generate base operations
    ops_a, final_a = generate_operations(
        rng, items, locs, init_loc, n_ops, indirect_fraction
    )
    
    if len(ops_a) < 2:
        return None
    
    # Try swapping adjacent pairs to find non-commutative pair
    for attempt in range(max_attempts):
        i = rng.randint(0, len(ops_a) - 2)
        j = i + 1
        
        # Replay with swapped operations i and j
        ops_b_order = list(range(len(ops_a)))
        ops_b_order[i], ops_b_order[j] = ops_b_order[j], ops_b_order[i]
        
        # Replay operations in new order from init_loc
        state_b = dict(init_loc)
        ops_b = []
        valid = True
        
        for idx in ops_b_order:
            orig_op = ops_a[idx]
            if orig_op.ref_by_loc:
                # Re-resolve indirect reference in current state_b
                occupied = {}
                for obj, loc in state_b.items():
                    occupied.setdefault(loc, []).append(obj)
                src_objs = occupied.get(orig_op.src, [])
                if len(src_objs) != 1:
                    valid = False
                    break
                actual_obj = src_objs[0]
                ops_b.append(Operation(
                    kind="indirect", obj=actual_obj,
                    src=orig_op.src, dst=orig_op.dst,
                    text=orig_op.text, ref_by_loc=True,
                ))
                state_b[actual_obj] = orig_op.dst
            else:
                # Direct op: same object
                obj = orig_op.obj
                actual_src = state_b[obj]
                # Keep same dst even if src changed (preserves word bag)
                ops_b.append(Operation(
                    kind="direct", obj=obj,
                    src=actual_src, dst=orig_op.dst,
                    text=orig_op.text, ref_by_loc=False,
                ))
                state_b[obj] = orig_op.dst
        
        if not valid:
            continue
        
        final_b = dict(state_b)
        
        # Check if any object ended up in different place
        diff_objs = [o for o in items if final_a.get(o) != final_b.get(o)]
        if not diff_objs:
            continue
        
        # Choose query from objects with different final states
        query_obj = rng.choice(diff_objs)
        
        passage_a = build_passage(init_loc, ops_a, query_obj, final_a)
        passage_b = build_passage(init_loc, ops_b, query_obj, final_b)
        
        # Verify word bags are similar (ops texts are reused)
        bag_a = Counter(passage_a.split())
        bag_b = Counter(passage_b.split())
        # They may differ slightly due to direct-op src text changes
        overlap = sum((bag_a & bag_b).values()) / max(1, sum(bag_a.values()))
        
        if overlap < 0.85:
            continue  # Too different, skip
        
        sc_a = Scenario(
            init_loc=init_loc, operations=ops_a, final_loc=final_a,
            query_obj=query_obj, answer=final_a[query_obj],
            passage=passage_a, word_count=len(passage_a.split()),
            n_indirect=sum(1 for op in ops_a if op.ref_by_loc),
        )
        sc_b = Scenario(
            init_loc=init_loc, operations=ops_b, final_loc=final_b,
            query_obj=query_obj, answer=final_b[query_obj],
            passage=passage_b, word_count=len(passage_b.split()),
            n_indirect=sum(1 for op in ops_b if op.ref_by_loc),
        )
        return (sc_a, sc_b)
    
    return None


def generate_paired_corpus(n_pairs: int, seed: int = 42):
    """Generate counterfactual pairs for binding probes."""
    rng = random.Random(seed)
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        result = try_generate_counterfactual_pair(
            rng, n_locs=5, n_items=4, n_ops=5, indirect_fraction=0.7
        )
        if result:
            pairs.append(result)
    return pairs


def generate_training_corpus(target_words: int, seed: int = 42):
    """Generate training passages (ordered, indirect)."""
    rng = random.Random(seed)
    corpus = []
    total = 0
    while total < target_words:
        n_ops = rng.randint(4, 7)
        n_items = rng.randint(3, 5)
        n_locs = rng.randint(4, 7)
        sc = generate_scenario(rng, n_locs, n_items, n_ops, indirect_fraction=0.6)
        corpus.append(sc)
        total += sc.word_count
    return corpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["pairs", "corpus"], default="pairs")
    p.add_argument("--n_pairs", type=int, default=500)
    p.add_argument("--target_words", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "pairs":
        pairs = generate_paired_corpus(args.n_pairs, args.seed)
        records = []
        for sc_a, sc_b in pairs:
            records.append({
                "passage_a": sc_a.passage,
                "passage_b": sc_b.passage,
                "query_obj": sc_a.query_obj,
                "answer_a": sc_a.answer,
                "answer_b": sc_b.answer,
                "answers_differ": sc_a.answer != sc_b.answer,
                "word_count_a": sc_a.word_count,
                "word_count_b": sc_b.word_count,
                "n_indirect_a": sc_a.n_indirect,
                "n_indirect_b": sc_b.n_indirect,
            })
        (out / "counterfactual_pairs.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        valid = sum(1 for r in records if r["answers_differ"])
        meta = {
            "mode": "pairs", "seed": args.seed,
            "n_pairs_requested": args.n_pairs,
            "n_pairs_generated": len(pairs),
            "n_answers_differ": valid,
            "fraction_differ": round(valid / max(1, len(pairs)), 4),
        }
        (out / "pairs_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))
    
    elif args.mode == "corpus":
        corpus = generate_training_corpus(args.target_words, args.seed)
        text_path = out / "state_dynamics_v3.txt"
        with text_path.open("w", encoding="utf-8") as f:
            for sc in corpus:
                f.write(sc.passage + "\n")
        total_words = sum(sc.word_count for sc in corpus)
        n_indirect = sum(1 for sc in corpus if sc.n_indirect > 0)
        meta = {
            "mode": "corpus", "seed": args.seed,
            "n_passages": len(corpus), "total_words": total_words,
            "target_words": args.target_words,
            "indirect_passages": n_indirect,
            "indirect_fraction": round(n_indirect / max(1, len(corpus)), 4),
            "mean_ops": round(sum(len(sc.operations) for sc in corpus) / max(1, len(corpus)), 2),
            "text_path": str(text_path),
        }
        (out / "corpus_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
