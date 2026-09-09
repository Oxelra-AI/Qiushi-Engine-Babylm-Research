#!/usr/bin/env python3
"""research: Build crossed-sign interaction probe for BabyLM structural binding.

2×2 factorial design:
  Two contexts (C1, C2) × Two alternatives (alt_A, alt_B).
  C1 makes alt_A correct; C2 makes alt_B correct.
  Interaction Δ = [PLL(A|C1) - PLL(B|C1)] - [PLL(A|C2) - PLL(B|C2)]
  Expected Δ > 0 for every item.
  Crossed-sign accuracy = fraction where margin(C1) > 0 AND margin(C2) < 0.

Families:
  entity_state  — multi-step location tracking (depth 2/3/4)
  spatial_put   — simple placement binding
  property_bind — property attribution via inference ("the big one is...")
  agent_action  — agent-to-role binding ("the reader is...")
  temporal_order— temporal precedence ("the first to ... was...")

Designed for potential conversion to training signal: L = -log(σ(Δ)).
All vocabulary is basic English present in the BabyLM legal corpus.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json, hashlib, random, pathlib, time
from collections import Counter

SEED = 42


def main():
    random.seed(SEED)
    t0 = time.time()

    ws = _public_path('experiments/archive/frontier_consolidation')
    out_dir = ws / "data" / "crossed_sign_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── vocabulary pools ────────────────────────────────────────────
    NAMES = ["Sam", "Alice", "Bob", "Emma", "Tom", "Kate", "Jack", "Max",
             "Lily", "Ben", "Anna", "Noah", "Ella", "Leo", "Mia", "Dan",
             "Zoe", "Ryan", "Ivy", "Luke"]
    OBJECTS = ["ball", "cup", "book", "toy", "hat", "key", "ring", "coin",
               "pen", "stone", "doll", "sock", "shoe", "card", "bottle", "lamp"]
    ON_LOCS = ["table", "shelf", "bench", "desk", "bed", "chair", "floor", "counter"]
    IN_LOCS = ["box", "basket", "bag", "drawer", "bucket", "jar", "bowl", "pocket"]
    MOVE_VERBS = ["moves", "takes", "carries", "brings"]
    PUT_VERBS = ["puts", "places", "sets", "drops"]

    PROP_PAIRS = [("big", "small"), ("tall", "short"), ("heavy", "light"),
                  ("old", "new"), ("fast", "slow"), ("happy", "sad"),
                  ("hot", "cold"), ("clean", "dirty"), ("loud", "quiet"),
                  ("strong", "weak")]
    ENTITIES = ["cat", "dog", "bird", "fish", "horse", "cow", "rabbit",
                "mouse", "lion", "bear", "frog", "duck", "sheep", "goat",
                "fox", "deer"]

    ACTION_PAIRS = [                        # (v1_3p, v2_3p, nominal_for_v1)
        ("reads", "writes", "reader"),
        ("sings", "dances", "singer"),
        ("teaches", "learns", "teacher"),
        ("runs", "walks", "runner"),
        ("cooks", "cleans", "cook"),
        ("speaks", "listens", "speaker"),
        ("draws", "paints", "drawer"),
        ("drives", "rides", "driver"),
    ]
    TEMPORAL = [                            # (past, infinitive)
        ("left", "leave"), ("arrived", "arrive"), ("woke up", "wake up"),
        ("started", "start"), ("finished", "finish"), ("spoke", "speak"),
        ("ate", "eat"), ("sat down", "sit down"),
    ]

    items = []
    used = set()

    # ── entity_state depth 2  (target 35) ───────────────────────────
    cnt = 0
    for _ in range(500):
        if cnt >= 35:
            break
        obj = random.choice(OBJECTS)
        prep = random.choice(["on", "in"])
        locs = ON_LOCS if prep == "on" else IN_LOCS
        la, lb = random.sample(locs, 2)
        agent = random.choice(NAMES)
        verb = random.choice(MOVE_VERBS)
        key = ("es2", obj, la, lb)
        if key in used:
            continue
        used.add(key)
        items.append(dict(
            id=f"entity_state_d2_{cnt:03d}", family="entity_state", depth=2,
            context_1=f"The {obj} is {prep} the {la}. {agent} {verb} the {obj} to the {lb}.",
            context_2=f"The {obj} is {prep} the {lb}. {agent} {verb} the {obj} to the {la}.",
            alt_A=f"The {obj} is now {prep} the {lb}.",
            alt_B=f"The {obj} is now {prep} the {la}.",
            expected=1))
        cnt += 1
    print(json.dumps({"event": "entity_state_d2", "count": cnt}))

    # ── entity_state depth 3  (target 25) ───────────────────────────
    cnt = 0
    for _ in range(500):
        if cnt >= 25:
            break
        obj = random.choice(OBJECTS)
        prep = random.choice(["on", "in"])
        locs = ON_LOCS if prep == "on" else IN_LOCS
        if len(locs) < 3:
            continue
        la, lb, lc = random.sample(locs, 3)
        a1, a2 = random.sample(NAMES, 2)
        v1, v2 = random.sample(MOVE_VERBS, 2)
        key = ("es3", obj, la, lb, lc)
        if key in used:
            continue
        used.add(key)
        items.append(dict(
            id=f"entity_state_d3_{cnt:03d}", family="entity_state", depth=3,
            context_1=f"The {obj} is {prep} the {la}. {a1} {v1} the {obj} to the {lb}. {a2} {v2} the {obj} to the {lc}.",
            context_2=f"The {obj} is {prep} the {lc}. {a1} {v1} the {obj} to the {lb}. {a2} {v2} the {obj} to the {la}.",
            alt_A=f"The {obj} is now {prep} the {lc}.",
            alt_B=f"The {obj} is now {prep} the {la}.",
            expected=1))
        cnt += 1
    print(json.dumps({"event": "entity_state_d3", "count": cnt}))

    # ── entity_state depth 4  (target 10) ───────────────────────────
    cnt = 0
    for _ in range(500):
        if cnt >= 10:
            break
        obj = random.choice(OBJECTS)
        prep = random.choice(["on", "in"])
        locs = ON_LOCS if prep == "on" else IN_LOCS
        if len(locs) < 4:
            continue
        la, lb, lc, ld = random.sample(locs, 4)
        a1, a2, a3 = random.sample(NAMES, 3)
        v1, v2, v3 = random.sample(MOVE_VERBS + ["puts"], 3)
        key = ("es4", obj, la, lb, lc, ld)
        if key in used:
            continue
        used.add(key)
        items.append(dict(
            id=f"entity_state_d4_{cnt:03d}", family="entity_state", depth=4,
            context_1=(f"The {obj} is {prep} the {la}. {a1} {v1} the {obj} to the {lb}. "
                       f"{a2} {v2} the {obj} to the {lc}. {a3} {v3} the {obj} to the {ld}."),
            context_2=(f"The {obj} is {prep} the {ld}. {a3} {v3} the {obj} to the {lc}. "
                       f"{a2} {v2} the {obj} to the {lb}. {a1} {v1} the {obj} to the {la}."),
            alt_A=f"The {obj} is now {prep} the {ld}.",
            alt_B=f"The {obj} is now {prep} the {la}.",
            expected=1))
        cnt += 1
    print(json.dumps({"event": "entity_state_d4", "count": cnt}))

    # ── spatial_put  (target 45) ────────────────────────────────────
    cnt = 0
    for _ in range(500):
        if cnt >= 45:
            break
        obj = random.choice(OBJECTS)
        prep = random.choice(["on", "in"])
        locs = ON_LOCS if prep == "on" else IN_LOCS
        la, lb = random.sample(locs, 2)
        agent = random.choice(NAMES)
        verb = random.choice(PUT_VERBS)
        key = ("sp", obj, la, lb)
        if key in used:
            continue
        used.add(key)
        items.append(dict(
            id=f"spatial_put_{cnt:03d}", family="spatial_put", depth=1,
            context_1=f"{agent} {verb} the {obj} {prep} the {la}.",
            context_2=f"{agent} {verb} the {obj} {prep} the {lb}.",
            alt_A=f"The {obj} is {prep} the {la}.",
            alt_B=f"The {obj} is {prep} the {lb}.",
            expected=1))
        cnt += 1
    print(json.dumps({"event": "spatial_put", "count": cnt}))

    # ── property_bind  (target 45) ──────────────────────────────────
    cnt = 0
    for _ in range(500):
        if cnt >= 45:
            break
        adj1, adj2 = random.choice(PROP_PAIRS)
        e1, e2 = random.sample(ENTITIES, 2)
        key = ("pb", e1, e2, adj1, adj2)
        if key in used:
            continue
        used.add(key)
        items.append(dict(
            id=f"property_bind_{cnt:03d}", family="property_bind", depth=1,
            context_1=f"The {e1} is {adj1} and the {e2} is {adj2}.",
            context_2=f"The {e1} is {adj2} and the {e2} is {adj1}.",
            alt_A=f"The {adj1} one is the {e1}.",
            alt_B=f"The {adj1} one is the {e2}.",
            expected=1))
        cnt += 1
    print(json.dumps({"event": "property_bind", "count": cnt}))

    # ── agent_action  (target 45) ───────────────────────────────────
    cnt = 0
    for _ in range(500):
        if cnt >= 45:
            break
        v1, v2, nom = random.choice(ACTION_PAIRS)
        n1, n2 = random.sample(NAMES, 2)
        key = ("aa", n1, n2, v1, v2)
        if key in used:
            continue
        used.add(key)
        items.append(dict(
            id=f"agent_action_{cnt:03d}", family="agent_action", depth=1,
            context_1=f"{n1} {v1} and {n2} {v2}.",
            context_2=f"{n2} {v1} and {n1} {v2}.",
            alt_A=f"The {nom} is {n1}.",
            alt_B=f"The {nom} is {n2}.",
            expected=1))
        cnt += 1
    print(json.dumps({"event": "agent_action", "count": cnt}))

    # ── temporal_order  (target 45) ─────────────────────────────────
    cnt = 0
    for _ in range(500):
        if cnt >= 45:
            break
        past, inf = random.choice(TEMPORAL)
        n1, n2 = random.sample(NAMES, 2)
        key = ("to", n1, n2, past)
        if key in used:
            continue
        used.add(key)
        items.append(dict(
            id=f"temporal_order_{cnt:03d}", family="temporal_order", depth=1,
            context_1=f"{n1} {past} before {n2}.",
            context_2=f"{n2} {past} before {n1}.",
            alt_A=f"The first to {inf} was {n1}.",
            alt_B=f"The first to {inf} was {n2}.",
            expected=1))
        cnt += 1
    print(json.dumps({"event": "temporal_order", "count": cnt}))

    # ── finalize ────────────────────────────────────────────────────
    random.shuffle(items)

    # assign sequential idx
    for i, it in enumerate(items):
        it["idx"] = i

    fam_counts = Counter(it["family"] for it in items)
    content_str = json.dumps(
        [{k: v for k, v in it.items()} for it in items], sort_keys=True)
    sha = hashlib.sha256(content_str.encode()).hexdigest()

    result = {
        "probe_name": "crossed_sign_interaction_v1",
        "seed": SEED,
        "n_items": len(items),
        "families": dict(fam_counts),
        "content_sha256": sha,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "items": items,
    }

    out_json = out_dir / "crossed_sign_probe.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    # markdown summary
    lines = [
        "# research crossed-sign interaction probe",
        "",
        f"Created: {result['created']}",
        f"Seed: {SEED}",
        f"Content SHA256: `{sha}`",
        f"Total items: {len(items)}",
        "",
        "## Families",
        "",
    ]
    for fam in sorted(fam_counts):
        lines.append(f"- **{fam}**: {fam_counts[fam]}")
    # depth breakdown for entity_state
    es_items = [it for it in items if it["family"] == "entity_state"]
    depth_counts = Counter(it.get("depth", 0) for it in es_items)
    if depth_counts:
        lines.append("")
        lines.append("### entity_state depth breakdown")
        for d in sorted(depth_counts):
            lines.append(f"- depth {d}: {depth_counts[d]}")

    lines.extend(["", "## Design", "",
        "2×2 factorial: C1 makes alt_A correct, C2 makes alt_B correct.",
        "Interaction Δ = [PLL(A|C1) - PLL(B|C1)] - [PLL(A|C2) - PLL(B|C2)]",
        "Expected Δ > 0. Crossed-sign = margin(C1) > 0 AND margin(C2) < 0.",
        "",
        "Alternatives differ in exactly one content word/phrase.",
        "Entity_state alternatives vary in location (same preposition).",
        "Property_bind alternatives vary in entity name (\"the adj one is...\").",
        "Agent_action alternatives vary in person name (\"the role is...\").",
        "Temporal_order alternatives vary in person name (\"the first to... was...\").",
        "Spatial_put alternatives vary in location (same preposition).",
        "",
        "## Sample items", ""])

    # show one example per family
    for fam in sorted(fam_counts):
        ex = next(it for it in items if it["family"] == fam)
        lines.append(f"### {fam} (id: {ex['id']})")
        lines.append(f"- C1: {ex['context_1']}")
        lines.append(f"- C2: {ex['context_2']}")
        lines.append(f"- alt_A: {ex['alt_A']}")
        lines.append(f"- alt_B: {ex['alt_B']}")
        lines.append("")

    out_md = out_dir / "crossed_sign_probe.md"
    out_md.write_text("\n".join(lines))

    elapsed = time.time() - t0
    print(json.dumps({
        "status": "CROSSED_SIGN_PROBE_BUILT",
        "n_items": len(items),
        "families": dict(fam_counts),
        "content_sha256": sha,
        "out_json": str(out_json),
        "out_md": str(out_md),
        "elapsed_sec": round(elapsed, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
