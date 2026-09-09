#!/usr/bin/env python3
"""research: Counterbalanced paired-quartet corpus for entity-keyed memory.

The research corpus was fatally flawed: entity B started in A's result state,
so both affected and unaffected queries had the SAME answer (100% confirmed).
The 0.97-0.99 attention finding was an artifact.

This rebuild ensures:
1. Both entities start in the SAME initial state within a family
2. The action changes ONLY the affected entity to the result state
3. Within each quartet, the answer differs between affected/unaffected queries
   SOLELY through event-entity and query correspondence
4. Interleaved reversals test selective slot updating
5. Proper counterbalancing: both initial-state directions (X start, Y start)

Quartet structure (fixed entities A,B; family; direction):
  actor=A, query=A → result_state  (affected: state was updated)
  actor=A, query=B → initial_state (unaffected: state preserved)
  actor=B, query=A → initial_state (unaffected: state preserved)
  actor=B, query=B → result_state  (affected: state was updated)

Held-out: entity x transition recombination, held templates, multi-event order.
All lexical components occur in acquisition examples.
"""
from __future__ import annotations
import argparse, hashlib, json, random, re
from collections import defaultdict
from pathlib import Path

ENTITIES = ["cup", "bowl", "box", "jar", "door", "window",
            "cloth", "rope", "lamp", "balloon", "drawer", "gate"]

# (family, state_x, state_y, verb_x_to_y, verb_y_to_x)
TRANSITIONS = [
    ("empty_full",       "empty",    "full",     "filled",       "emptied"),
    ("open_closed",      "closed",   "open",     "opened",       "closed"),
    ("clean_dirty",      "dirty",    "clean",    "cleaned",      "dirtied"),
    ("dry_wet",          "dry",      "wet",      "wetted",       "dried"),
    ("cold_hot",         "cold",     "hot",      "heated",       "cooled"),
    ("dark_lit",         "dark",     "lit",      "lit",          "extinguished"),
    ("flat_inflated",    "flat",     "inflated", "inflated",     "deflated"),
    ("unlocked_locked",  "unlocked", "locked",   "locked",       "unlocked"),
]

# Templates: {a}=entityA, {b}=entityB, {s}=shared_initial_state,
# {verb}=action verb, {actor}=acted entity, {query}=queried entity
TRAIN_TEMPLATES = [
    "The {a} was {s}. The {b} was {s}. Mira {verb} the {actor}. The {query} is now <mask>.",
    "At first, the {a} was {s}, and the {b} was {s}. Then Mira {verb} the {actor}. Now the {query} is <mask>.",
    "The {a} started {s}. The {b} started {s}. Mira then {verb} the {actor}. Afterwards, the {query} is <mask>.",
]

HELD_TEMPLATES = [
    "Both the {a} and the {b} were {s}. After Mira {verb} the {actor}, the {query} became <mask>.",
    "The {a} and the {b}, both {s}, sat nearby. Mira {verb} the {actor}; afterward the {query} was <mask>.",
]

# Component acquisition templates
SINGLE_TRANSITION_TPLS = [
    "The {a} was {s}. Mira {verb} the {a}. The {a} is now <mask>.",
    "Initially the {a} was {s}. After Mira {verb} it, the {a} became <mask>.",
]

STATE_ACQ_TPLS = [
    "The {a} is {s}. The state of the {a} is <mask>.",
    "A {s} {a} is here. This {a} is <mask>.",
]

IDENTITY_TPL = "The {a} was {sa}. The {b} was {sb}. Without any change, the {query} remained <mask>."

# Multi-event interleaved reversal template (3 events)
# Both start initial. Fill A, then fill B, then reverse A → final: A=initial, B=result
MULTI_INTERLEAVE_TPL = (
    "The {a} was {s}. The {b} was {s}. "
    "Mira {fwd} the {a}. Next Mira {fwd} the {b}. "
    "Finally Mira {rev} the {a}. The {query} is now <mask>."
)
# Counterbalanced: Fill B, then fill A, then reverse B → final: A=result, B=initial
MULTI_INTERLEAVE_COUNTER_TPL = (
    "The {a} was {s}. The {b} was {s}. "
    "Mira {fwd} the {b}. Next Mira {fwd} the {a}. "
    "Finally Mira {rev} the {b}. The {query} is now <mask>."
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def wc(text: str) -> int:
    return len(re.findall(r"\S+", text))


def held_pair(entity_i: int, trans_i: int) -> bool:
    return (entity_i + 3 * trans_i) % 4 == 0


def rec(kind, split, text, answer, foil, entities, actor_entity, query_entity,
        family, template_id, quartet_id=None, is_affected=None, event_order="single"):
    return {
        "kind": kind, "split": split, "text": text,
        "answer": answer, "foil": foil,
        "entities": entities, "actor_entity": actor_entity,
        "query_entity": query_entity,
        "is_affected_query": is_affected,
        "family": family, "template_id": template_id,
        "quartet_id": quartet_id, "event_order": event_order,
        "word_count": wc(text),
    }


def make_quartet(a, b, fam, initial, result, verb, template, template_id, 
                 split, held, quartet_base_id):
    """Generate a counterbalanced quartet: 4 items with same surface but different answers."""
    items = []
    for actor, query in [(a, a), (a, b), (b, a), (b, b)]:
        is_affected = (actor == query)
        ans = result if is_affected else initial
        foil = initial if is_affected else result
        text = template.format(a=a, b=b, s=initial, verb=verb, actor=actor, query=query)
        qid = f"{quartet_base_id}_{actor}_{query}"
        items.append(rec("binding", split, text, ans, foil, [a, b], actor, query,
                         fam, template_id, qid, is_affected, "single"))
    return items


def make_multi_event_pair(a, b, fam, initial, result, fwd_verb, rev_verb,
                          template, template_id, split, quartet_base_id,
                          reversed_entity=None):
    """Interleaved reversal with explicit reversed-entity tracking.
    reversed_entity ends at initial; the other entity ends at result."""
    if reversed_entity is None:
        reversed_entity = a  # default: standard template reverses A
    items = []
    for query in [a, b]:
        is_affected_by_final = (query == reversed_entity)
        # reversed_entity → initial (reversed back), other → result (forward only)
        ans = initial if query == reversed_entity else result
        foil = result if query == reversed_entity else initial
        text = template.format(a=a, b=b, s=initial, fwd=fwd_verb, rev=rev_verb, query=query)
        qid = f"{quartet_base_id}_multi_{query}"
        items.append(rec("multi_event", split, text, ans, foil, [a, b], "both", query,
                         fam, template_id, qid, is_affected_by_final, "interleaved"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=29199)
    ap.add_argument("--train-records", type=int, default=12000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train = []
    eval_rows = []

    # ---- Component acquisition (all entities × all transitions, both directions) ----
    for ei, a in enumerate(ENTITIES):
        for ti, (fam, sx, sy, fwd, rev) in enumerate(TRANSITIONS):
            # State acquisition: expose each state
            for state, foil in [(sx, sy), (sy, sx)]:
                for k, tpl in enumerate(STATE_ACQ_TPLS):
                    text = tpl.format(a=a, s=state)
                    train.append(rec("state_acq", "train", text, state, foil,
                                     [a], None, a, fam, f"sacq{k}"))
            # Transition acquisition: single-entity transition both directions
            for initial, result, verb in [(sx, sy, fwd), (sy, sx, rev)]:
                for k, tpl in enumerate(SINGLE_TRANSITION_TPLS):
                    text = tpl.format(a=a, s=initial, verb=verb)
                    train.append(rec("transition_acq", "train", text, result, initial,
                                     [a], a, a, fam, f"tacq{k}"))

    # ---- Identity selection: exposes entity-query without transitions ----
    for ei, a in enumerate(ENTITIES):
        for ti, (fam, sx, sy, _, _) in enumerate(TRANSITIONS):
            b = ENTITIES[(ei + 1 + ti) % len(ENTITIES)]
            for query, ans, foil, sa, sb in [(a, sx, sy, sx, sy), (b, sy, sx, sx, sy)]:
                text = IDENTITY_TPL.format(a=a, b=b, sa=sa, sb=sb, query=query)
                train.append(rec("identity_acq", "train", text, ans, foil,
                                 [a, b], None, query, fam, "ident0"))

    # ---- Binding quartets (counterbalanced) ----
    binding_pool = []
    held_pool = []

    for ei in range(len(ENTITIES)):
        a = ENTITIES[ei]
        for ti, (fam, sx, sy, fwd, rev) in enumerate(TRANSITIONS):
            is_held = held_pair(ei, ti)
            # Two directions: both start sx (action sx→sy), both start sy (action sy→sx)
            for initial, result, verb in [(sx, sy, fwd), (sy, sx, rev)]:
                if is_held:
                    templates = HELD_TEMPLATES
                    tpl_prefix = "held"
                    split = "eval_held_recomb"
                else:
                    templates = TRAIN_TEMPLATES
                    tpl_prefix = "train"
                    split = "train_binding"

                for k, tpl in enumerate(templates):
                    # Sample partner entity
                    b = ENTITIES[(ei + 1 + k) % len(ENTITIES)]
                    if b == a:
                        b = ENTITIES[(ei + 2 + k) % len(ENTITIES)]
                    qbase = f"Q_{a}_{b}_{fam}_{initial}_{tpl_prefix}{k}"
                    quartet = make_quartet(a, b, fam, initial, result, verb,
                                           tpl, f"{tpl_prefix}{k}", split, is_held, qbase)
                    if is_held:
                        held_pool.extend(quartet)
                    else:
                        binding_pool.extend(quartet)

    # Fill training to target count
    acq_count = len(train)
    while len(train) < args.train_records:
        train.append(dict(rng.choice(binding_pool)))
    rng.shuffle(train)

    # ---- Eval: held recombination quartets ----
    eval_rows.extend(held_pool)

    # ---- Eval: in-distribution binding sample ----
    eval_rows.extend(rng.sample(binding_pool, min(512, len(binding_pool))))

    # ---- Eval: multi-event interleaved reversals (held pairs) ----
    held_pairs = [(ei, ti) for ei in range(len(ENTITIES))
                  for ti in range(len(TRANSITIONS)) if held_pair(ei, ti)]
    for ei, ti in held_pairs:
        a = ENTITIES[ei]
        fam, sx, sy, fwd, rev = TRANSITIONS[ti]
        b = ENTITIES[(ei + 3) % len(ENTITIES)]
        if b == a:
            b = ENTITIES[(ei + 4) % len(ENTITIES)]

        for initial, result, fwd_v, rev_v in [(sx, sy, fwd, rev), (sy, sx, rev, fwd)]:
            qbase = f"MULTI_{a}_{b}_{fam}_{initial}"
            # Standard order: fill A, fill B, reverse A → A=initial, B=result
            eval_rows.extend(make_multi_event_pair(
                a, b, fam, initial, result, fwd_v, rev_v,
                MULTI_INTERLEAVE_TPL, "multi_std", "eval_held_order", qbase + "_std",
                reversed_entity=a))
            # Counter order: fill B, fill A, reverse B → A=result, B=initial
            eval_rows.extend(make_multi_event_pair(
                a, b, fam, initial, result, fwd_v, rev_v,
                MULTI_INTERLEAVE_COUNTER_TPL, "multi_ctr", "eval_held_order", qbase + "_ctr",
                reversed_entity=b))

    # ---- Write files ----
    files = {}
    for name, rows in [("train", train), ("eval", eval_rows)]:
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                rr = dict(r)
                rr["id"] = f"S199_{name}_{i:06d}"
                f.write(json.dumps(rr, ensure_ascii=False) + "\n")
        files[name] = {
            "path": str(p), "sha256": sha(p),
            "records": len(rows),
            "words": sum(r["word_count"] for r in rows),
        }

    # ---- Verify critical properties ----
    # 1. All lexical components in training
    lexical = defaultdict(set)
    for r in train:
        lexical["entity"].update(r["entities"])
        lexical["state"].update([r["answer"], r["foil"]])
        if r.get("actor_entity") and r["actor_entity"] not in ("both", None):
            lexical["actor"].add(r["actor_entity"])
    # 2. No held binding pairs in training binding
    train_bind = set()
    for r in train:
        if r["kind"] == "binding" and r["actor_entity"]:
            train_bind.add((r["entities"][0], r["family"]))
    eval_held_set = set()
    for r in eval_rows:
        if r["split"] == "eval_held_recomb":
            eval_held_set.add((r["entities"][0], r["family"]))
    overlap = sorted(train_bind & eval_held_set)

    # 3. Within-quartet answer diversity: every quartet MUST have different answers
    quartet_answers = defaultdict(set)
    for r in eval_rows:
        if r["quartet_id"]:
            qbase = "_".join(r["quartet_id"].rsplit("_", 2)[:1])  # strip actor_query
            quartet_answers[r["quartet_id"]].add(r["answer"])
    # Group by quartet base (strip the last actor_query part)
    quartet_groups = defaultdict(list)
    for r in eval_rows:
        if r.get("quartet_id") and r["kind"] == "binding":
            parts = r["quartet_id"].rsplit("_", 2)
            base = parts[0] if len(parts) >= 3 else r["quartet_id"]
            quartet_groups[base].append(r)
    n_mixed = sum(1 for grp in quartet_groups.values()
                  if len(set(r["answer"] for r in grp)) > 1)
    n_uniform = sum(1 for grp in quartet_groups.values()
                    if len(set(r["answer"] for r in grp)) == 1)

    manifest = {
        "status": "COUNTERBALANCED_CORPUS_FROZEN",
        "seed": args.seed,
        "files": files,
        "acquisition_count": acq_count,
        "binding_pool_size": len(binding_pool),
        "held_pool_size": len(held_pool),
        "held_pair_rule": "(entity_index + 3*transition_index) % 4 == 0",
        "held_pair_overlap": overlap,
        "lexical_coverage": {
            "entities": sorted(lexical["entity"]),
            "states": sorted(lexical["state"]),
            "actors": sorted(lexical["actor"]),
        },
        "quartet_answer_diversity": {
            "eval_quartets_with_mixed_answers": n_mixed,
            "eval_quartets_with_uniform_answers": n_uniform,
        },
        "design": (
            "Counterbalanced quartets: both entities start in the SAME state. "
            "Affected query → result_state, unaffected query → initial_state. "
            "Answer differs between affected/unaffected solely through entity-event-query correspondence. "
            "Multi-event interleaved reversals test selective slot updating."
        ),
    }
    mp = out / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(mp),
        "train_records": len(train),
        "train_words": files["train"]["words"],
        "eval_records": len(eval_rows),
        "held_overlap": len(overlap),
        "quartets_mixed_answers": n_mixed,
        "quartets_uniform_answers": n_uniform,
    }, indent=2))


if __name__ == "__main__":
    main()
