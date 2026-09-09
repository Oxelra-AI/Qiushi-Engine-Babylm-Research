#!/usr/bin/env python3
"""research — R1 state-dynamics passage generator (v2 clean rewrite).

Two modes:
  'ordered': objects moved multiple times; final state needs full composition.
  'independent': each object moved at most once; final state = single-op lookup.

CRITICAL: vocabulary is independent of official BabyLM Entity Tracking task.
Official task uses: "Box 1"–"Box 7", 100 BNC nouns. We use: rooms/shelves/bags/people
with completely different entity/container names.
"""
from __future__ import annotations
import argparse, json, pathlib, random
from dataclasses import dataclass, field

# ── Independent vocabulary ─────────────────────────────────────────────────────

DOMAINS = {
    "room": {
        "containers": ["the kitchen", "the garage", "the attic", "the cellar",
                       "the study", "the porch", "the closet", "the shed",
                       "the vault", "the loft"],
        "objects": ["the lamp", "the clock", "the vase", "the mirror", "the rug",
                    "the painting", "the cushion", "the blanket", "the candle",
                    "the plant", "the photo", "the trophy", "the statue",
                    "the basket", "the fan"],
        "init": "Initially, {obj} is in {loc}.",
        "move": [
            "{obj} was moved from {src} to {dst}.",
            "Someone took {obj} from {src} and placed it in {dst}.",
            "{obj} ended up in {dst} after being in {src}.",
        ],
        "query": "After all these changes, {obj} is in",
        "query_full": "After all these changes, {obj} is in {answer}.",
    },
    "shelf": {
        "containers": ["the top shelf", "the middle shelf", "the bottom shelf",
                       "the left shelf", "the right shelf", "the back shelf",
                       "the glass shelf", "the metal shelf", "the front shelf",
                       "the corner shelf"],
        "objects": ["the red cup", "the blue mug", "the green plate",
                    "the white bowl", "the tin can", "the glass jar",
                    "the small box", "the wooden toy", "the metal ring",
                    "the stone figure", "the paper roll", "the clay pot",
                    "the brass coin", "the silver spoon", "the gold pin"],
        "init": "At first, {obj} sits on {loc}.",
        "move": [
            "{obj} was moved from {src} to {dst}.",
            "Someone shifted {obj} from {src} to {dst}.",
            "{obj} was rearranged from {src} onto {dst}.",
        ],
        "query": "Now, {obj} is on",
        "query_full": "Now, {obj} is on {answer}.",
    },
    "bag": {
        "containers": ["the red bag", "the blue bag", "the green bag",
                       "the black bag", "the white bag", "the canvas bag",
                       "the leather bag", "the small bag", "the large bag",
                       "the cloth bag"],
        "objects": ["a notebook", "a pencil", "a phone", "a wallet", "a compass",
                    "a flashlight", "a bottle", "a scarf", "a glove", "a ticket",
                    "a map", "a charger", "a snack", "a towel", "a keychain"],
        "init": "At the start, {obj} is inside {loc}.",
        "move": [
            "{obj} was transferred from {src} to {dst}.",
            "Someone took {obj} out of {src} and put it into {dst}.",
            "{obj} went from {src} into {dst}.",
        ],
        "query": "In the end, {obj} is inside",
        "query_full": "In the end, {obj} is inside {answer}.",
    },
    "person": {
        "containers": ["Alice", "Ben", "Carol", "David", "Elena",
                       "Felix", "Grace", "Hugo", "Iris", "James"],
        "objects": ["the letter", "the badge", "the card", "the coin",
                    "the folder", "the receipt", "the permit", "the token",
                    "the pass", "the seal", "the stamp", "the note",
                    "the voucher", "the tag", "the ribbon"],
        "init": "Originally, {loc} has {obj}.",
        "move": [
            "{src} gave {obj} to {dst}.",
            "{obj} was handed from {src} to {dst}.",
            "{dst} received {obj} from {src}.",
        ],
        "query": "Finally, {obj} is held by",
        "query_full": "Finally, {obj} is held by {answer}.",
    },
}


@dataclass
class Scenario:
    passage: str
    word_count: int
    domain: str
    mode: str
    n_ops: int
    query_obj: str
    answer: str
    init_location: dict
    final_location: dict
    operations: list
    multi_move_count: int  # objects moved >1 time


def generate_scenario(
    rng: random.Random,
    domain: str,
    n_containers: int = 5,
    n_objects: int = 4,
    n_ops: int = 4,
    mode: str = "ordered",
) -> Scenario:
    """Generate one state-dynamics passage."""
    dom = DOMAINS[domain]
    containers = rng.sample(dom["containers"], min(n_containers, len(dom["containers"])))
    objects = rng.sample(dom["objects"], min(n_objects, len(dom["objects"])))

    # Initial random placement
    init_loc = {obj: rng.choice(containers) for obj in objects}
    current = dict(init_loc)

    # Generate operations
    ops = []
    if mode == "ordered":
        # Allow repeated moves of same object
        for _ in range(n_ops):
            obj = rng.choice(objects)
            src = current[obj]
            candidates = [c for c in containers if c != src]
            if not candidates:
                continue
            dst = rng.choice(candidates)
            ops.append({"obj": obj, "src": src, "dst": dst})
            current[obj] = dst
    elif mode == "independent":
        # Each object moved at most once
        pool = list(objects)
        rng.shuffle(pool)
        for obj in pool[:n_ops]:
            src = current[obj]
            candidates = [c for c in containers if c != src]
            if not candidates:
                continue
            dst = rng.choice(candidates)
            ops.append({"obj": obj, "src": src, "dst": dst})
            current[obj] = dst

    # Choose query object: prefer one that was moved (and ideally moved >1 time)
    moved_counts = {}
    for op in ops:
        moved_counts[op["obj"]] = moved_counts.get(op["obj"], 0) + 1
    multi_moved = [o for o, c in moved_counts.items() if c > 1]
    if multi_moved and mode == "ordered":
        query_obj = rng.choice(multi_moved)
    elif moved_counts:
        query_obj = rng.choice(list(moved_counts.keys()))
    else:
        query_obj = rng.choice(objects)

    answer = current[query_obj]

    # Build passage text
    lines = []
    for obj in objects:
        lines.append(dom["init"].format(obj=obj, loc=init_loc[obj]))
    for op in ops:
        tmpl = rng.choice(dom["move"])
        lines.append(tmpl.format(**op))
    lines.append(dom["query_full"].format(obj=query_obj, answer=answer))

    passage = " ".join(lines)
    word_count = len(passage.split())
    multi_move_count = len(multi_moved)

    return Scenario(
        passage=passage, word_count=word_count, domain=domain, mode=mode,
        n_ops=len(ops), query_obj=query_obj, answer=answer,
        init_location=init_loc, final_location=current, operations=ops,
        multi_move_count=multi_move_count,
    )


def generate_corpus(target_words: int, mode: str, seed: int, domains=None):
    rng = random.Random(seed)
    if domains is None:
        domains = list(DOMAINS.keys())
    corpus = []
    total = 0
    while total < target_words:
        domain = rng.choice(domains)
        n_ops = rng.randint(3, 6) if mode == "ordered" else rng.randint(3, 5)
        n_obj = rng.randint(3, 5)
        n_cont = rng.randint(4, 7)
        sc = generate_scenario(rng, domain, n_cont, n_obj, n_ops, mode)
        corpus.append(sc)
        total += sc.word_count
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

    corpus = generate_corpus(args.target_words, args.mode, args.seed)

    text_path = out / f"state_dynamics_{args.mode}.txt"
    with text_path.open("w", encoding="utf-8") as f:
        for sc in corpus:
            f.write(sc.passage + "\n")

    total_words = sum(sc.word_count for sc in corpus)
    multi = sum(1 for sc in corpus if sc.multi_move_count > 0)
    meta = {
        "mode": args.mode, "seed": args.seed, "n_passages": len(corpus),
        "total_words": total_words, "target_words": args.target_words,
        "domains": list(set(sc.domain for sc in corpus)),
        "multi_move_passages": multi,
        "multi_move_fraction": round(multi / max(1, len(corpus)), 4),
        "mean_ops": round(sum(sc.n_ops for sc in corpus) / max(1, len(corpus)), 2),
        "text_path": str(text_path),
    }
    meta_path = out / f"state_dynamics_{args.mode}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
