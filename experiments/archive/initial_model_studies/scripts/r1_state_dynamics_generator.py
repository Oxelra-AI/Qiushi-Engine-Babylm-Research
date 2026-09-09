#!/usr/bin/env python3
"""research — R1 state-dynamics passage generator.

Generates legal natural-language passages describing ordered entity-state operations.
Two modes:
  1. 'ordered': multi-step compositions where final state requires tracking all ops.
  2. 'independent': same surface form but each entity touched at most once (no composition).

Design principles:
- Independent vocabulary: NO official Entity Tracking words (boxes 1-7, BNC 100 nouns).
- Multiple surface domains: rooms/objects, bags/items, shelves/books, people/keys.
- Relexicalized: entity names drawn from an independent pool, near-uniform frequency.
- Final-state query embedded in the passage so WWM masked targets can depend on it.
- Exact whitespace word counts for BabyLM accounting.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, random
from dataclasses import dataclass, field
from typing import Optional

# ── Independent vocabulary pools (disjoint from official Entity Tracking) ──────

CONTAINERS = {
    "room": ["the kitchen", "the garage", "the attic", "the cellar", "the study",
             "the porch", "the closet", "the shed", "the vault", "the loft"],
    "shelf": ["the top shelf", "the middle shelf", "the bottom shelf",
              "the left shelf", "the right shelf", "the corner shelf",
              "the glass shelf", "the metal shelf", "the wooden shelf", "the back shelf"],
    "bag": ["the red bag", "the blue bag", "the green bag", "the black bag",
            "the white bag", "the canvas bag", "the leather bag", "the small bag",
            "the large bag", "the cloth bag"],
    "person": ["Alice", "Ben", "Carol", "David", "Elena", "Felix", "Grace",
               "Hugo", "Iris", "James"],
}

OBJECTS = {
    "room": ["the lamp", "the clock", "the vase", "the mirror", "the rug",
             "the painting", "the cushion", "the blanket", "the candle", "the plant",
             "the photo", "the trophy", "the statue", "the basket", "the fan"],
    "shelf": ["the red cup", "the blue mug", "the green plate", "the white bowl",
              "the tin can", "the glass jar", "the small box", "the wooden toy",
              "the metal ring", "the stone figure", "the paper roll", "the clay pot",
              "the brass coin", "the silver spoon", "the gold pin"],
    "bag": ["a notebook", "a pencil", "a phone", "a wallet", "a compass",
            "a flashlight", "a bottle", "a scarf", "a glove", "a ticket",
            "a map", "a charger", "a snack", "a towel", "a key"],
    "person": ["the letter", "the badge", "the card", "the coin", "the ticket",
               "the folder", "the receipt", "the permit", "the token", "the pass",
               "the seal", "the stamp", "the note", "the voucher", "the tag"],
}

MOVE_TEMPLATES = {
    "room": [
        "Someone moved {obj} from {src} to {dst}.",
        "{obj} was taken from {src} and placed in {dst}.",
        "After a moment, {obj} ended up in {dst} instead of {src}.",
    ],
    "shelf": [
        "{obj} was moved from {src} to {dst}.",
        "Someone rearranged {obj}, putting it on {dst} instead of {src}.",
        "{obj} was shifted from {src} over to {dst}.",
    ],
    "bag": [
        "{obj} was transferred from {src} to {dst}.",
        "Someone took {obj} out of {src} and put it into {dst}.",
        "{obj} went from {src} into {dst}.",
    ],
    "person": [
        "{src} gave {obj} to {dst}.",
        "{obj} was handed from {src} to {dst}.",
        "{dst} received {obj} from {src}.",
    ],
}

INIT_TEMPLATES = {
    "room": "Initially, {obj} is in {loc}.",
    "shelf": "At first, {obj} sits on {loc}.",
    "bag": "At the start, {obj} is inside {loc}.",
    "person": "Originally, {person} has {obj}.",
}

QUERY_TEMPLATES = {
    "room": "After all changes, {obj} is in {answer}.",
    "shelf": "Now, {obj} is on {answer}.",
    "bag": "In the end, {obj} is inside {answer}.",
    "person": "Finally, {person_or_answer} has {obj}.",
}


@dataclass
class WorldState:
    domain: str
    containers: list[str]
    objects: list[str]
    # object -> container mapping
    location: dict[str, str] = field(default_factory=dict)


def generate_scenario(
    rng: random.Random,
    domain: str,
    n_containers: int = 5,
    n_objects: int = 4,
    n_ops: int = 4,
    mode: str = "ordered",
) -> dict:
    """Generate one state-dynamics passage.
    
    mode='ordered': objects can be moved multiple times, requiring composition.
    mode='independent': each object is moved at most once (no composition needed).
    """
    containers = rng.sample(CONTAINERS[domain], min(n_containers, len(CONTAINERS[domain])))
    objects = rng.sample(OBJECTS[domain], min(n_objects, len(OBJECTS[domain])))
    
    # Initial placement
    state = WorldState(domain=domain, containers=containers, objects=objects)
    for obj in objects:
        state.location[obj] = rng.choice(containers)
    
    # Generate operations
    operations = []
    if mode == "ordered":
        # Objects CAN be moved multiple times; final state requires full composition
        for _ in range(n_ops):
            obj = rng.choice(objects)
            src = state.location[obj]
            dst = rng.choice([c for c in containers if c != src])
            operations.append({"obj": obj, "src": src, "dst": dst})
            state.location[obj] = dst
    elif mode == "independent":
        # Each object moved AT MOST once; final state = single operation lookup
        available = list(objects)
        rng.shuffle(available)
        for i in range(min(n_ops, len(available))):
            obj = available[i]
            src = state.location[obj]
            dst = rng.choice([c for c in containers if c != src])
            operations.append({"obj": obj, "src": src, "dst": dst})
            state.location[obj] = dst
    
    # Build text
    sentences = []
    # Initial state description
    for obj in objects:
        init_loc = None
        # Find initial location (before any ops)
        init_state = {o: rng.choice(containers) for o in objects}  # Will be overwritten
        break
    
    # Recompute: need to track initial state separately
    init_location = {}
    for obj in objects:
        init_location[obj] = rng.choice(containers)
    
    # Replay operations from init_location
    current = dict(init_location)
    ops_text = []
    actual_ops = []
    for op_spec in operations:
        obj = op_spec["obj"]
        if mode == "ordered":
            src = current[obj]
            dst = rng.choice([c for c in containers if c != src])
        else:
            src = current[obj]
            dst = rng.choice([c for c in containers if c != src])
        current[obj] = dst
        actual_ops.append({"obj": obj, "src": src, "dst": dst})
    
    # Build passage text
    lines = []
    for obj in objects:
        if domain == "person":
            lines.append(INIT_TEMPLATES[domain].format(person=init_location[obj], obj=obj))
        else:
            lines.append(INIT_TEMPLATES[domain].format(obj=obj, loc=init_location[obj]))
    
    for op in actual_ops:
        templates = MOVE_TEMPLATES[domain]
        tmpl = rng.choice(templates)
        if domain == "person":
            lines.append(tmpl.format(obj=op["obj"], src=op["src"], dst=op["dst"]))
        else:
            lines.append(tmpl.format(obj=op["obj"], src=op["src"], dst=op["dst"]))
    
    # Query: ask about an object that was moved (interesting case)
    moved_objs = [op["obj"] for op in actual_ops]
    if moved_objs:
        query_obj = rng.choice(moved_objs)
    else:
        query_obj = rng.choice(objects)
    
    answer = current[query_obj]
    if domain == "person":
        lines.append(QUERY_TEMPLATES[domain].format(person_or_answer=answer, obj=query_obj))
    else:
        lines.append(QUERY_TEMPLATES[domain].format(obj=query_obj, answer=answer))
    
    passage = " ".join(lines)
    word_count = len(passage.split())
    
    return {
        "passage": passage,
        "word_count": word_count,
        "domain": domain,
        "mode": mode,
        "n_ops": len(actual_ops),
        "query_obj": query_obj,
        "answer": answer,
        "init_location": init_location,
        "final_location": current,
        "operations": actual_ops,
        "n_moved_more_than_once": sum(1 for o in objects if sum(1 for op in actual_ops if op["obj"] == o) > 1),
    }


def generate_corpus(
    n_passages: int,
    target_words: int,
    mode: str = "ordered",
    seed: int = 42,
    domains: Optional[list[str]] = None,
) -> list[dict]:
    """Generate a corpus of state-dynamics passages."""
    rng = random.Random(seed)
    if domains is None:
        domains = list(CONTAINERS.keys())
    
    corpus = []
    total_words = 0
    for i in range(n_passages * 3):  # overshoot, then trim
        if total_words >= target_words:
            break
        domain = rng.choice(domains)
        n_ops = rng.randint(3, 6) if mode == "ordered" else rng.randint(3, 5)
        n_objects = rng.randint(3, 5)
        n_containers = rng.randint(4, 6)
        scenario = generate_scenario(
            rng, domain, n_containers=n_containers,
            n_objects=n_objects, n_ops=n_ops, mode=mode,
        )
        corpus.append(scenario)
        total_words += scenario["word_count"]
    
    return corpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["ordered", "independent"], default="ordered")
    p.add_argument("--target_words", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    corpus = generate_corpus(
        n_passages=args.target_words // 40,  # ~40 words per passage avg
        target_words=args.target_words,
        mode=args.mode,
        seed=args.seed,
    )
    
    # Save passages as plain text (one per line)
    text_path = out / f"state_dynamics_{args.mode}.txt"
    with text_path.open("w", encoding="utf-8") as f:
        for item in corpus:
            f.write(item["passage"] + "\n")
    
    # Save metadata
    total_words = sum(item["word_count"] for item in corpus)
    multi_move = sum(1 for item in corpus if item["n_moved_more_than_once"] > 0)
    meta = {
        "mode": args.mode,
        "seed": args.seed,
        "n_passages": len(corpus),
        "total_words": total_words,
        "target_words": args.target_words,
        "domains_used": list(set(item["domain"] for item in corpus)),
        "multi_move_passages": multi_move,
        "multi_move_fraction": multi_move / max(1, len(corpus)),
        "mean_ops": sum(item["n_ops"] for item in corpus) / max(1, len(corpus)),
        "text_path": str(text_path),
    }
    (out / f"state_dynamics_{args.mode}_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
