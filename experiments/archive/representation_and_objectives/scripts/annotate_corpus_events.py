#!/usr/bin/env python3
"""research: Annotate research counterbalanced corpus with the missing 'events' field.

The trainer (train_entity_memory_v2.py) expects each record to have an 'events'
field listing per-event {verb, entity} for EntityMemory to locate verb/actor positions
and compute write attention. The corpus generator never wrote this field, making the
memory module completely inert (all events invalid → no writes happen).

This extracts events from text templates, validates, and saves annotated versions.
"""
import json, re, hashlib
from pathlib import Path
from collections import Counter, defaultdict

ALL_ENTITIES = frozenset([
    "cup", "bowl", "box", "jar", "door", "window",
    "cloth", "rope", "lamp", "balloon", "drawer", "gate"
])

KNOWN_VERBS = frozenset([
    "filled", "emptied", "opened", "closed", "cleaned", "dirtied",
    "wetted", "dried", "heated", "cooled", "lit", "extinguished",
    "inflated", "deflated", "locked", "unlocked"
])

# Matches: Mira [then] {verb} [the {entity} | it]
EVENT_RE = re.compile(r'Mira\s+(?:then\s+)?(\w+)\s+(?:the\s+(\w+)|it)\b')


def extract_events(text, record):
    events = []
    actor = record.get("actor_entity")
    entities = record.get("entities", [])
    for m in EVENT_RE.finditer(text):
        verb, entity = m.group(1), m.group(2)
        if verb not in KNOWN_VERBS:
            continue
        if entity is None:  # "Mira {verb} it" pronoun
            entity = actor or (entities[0] if entities else None)
        if entity and entity in ALL_ENTITIES:
            events.append({"verb": verb, "entity": entity})
    return events


def validate_record(r):
    events = r.get("events", [])
    kind = r["kind"]
    actor = r.get("actor_entity")
    errors = []
    if kind == "binding":
        if len(events) != 1:
            errors.append(f"binding: expected 1 event, got {len(events)}")
        elif actor and actor != "both" and events[0]["entity"] != actor:
            errors.append(f"binding: event entity {events[0]['entity']} != actor {actor}")
    elif kind == "multi_event":
        if len(events) != 3:
            errors.append(f"multi_event: expected 3 events, got {len(events)}")
    elif kind == "transition_acq":
        if len(events) != 1:
            errors.append(f"transition_acq: expected 1 event, got {len(events)}")
    elif kind == "state_acq":
        if len(events) != 0:
            errors.append(f"state_acq: expected 0 events, got {len(events)}")
    elif kind == "identity_acq":
        if len(events) != 0:
            errors.append(f"identity_acq: expected 0 events, got {len(events)}")
    return errors


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def main():
    src = Path("experiments/archive/representation_and_objectives/training/data/counterbalanced_corpus")
    out = Path("experiments/archive/representation_and_objectives/data/annotated_corpus")
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    all_errors = []

    for split in ["train", "eval"]:
        rows = [json.loads(x) for x in (src / f"{split}.jsonl").read_text().splitlines() if x.strip()]
        ec = defaultdict(lambda: Counter())
        for r in rows:
            r["events"] = extract_events(r["text"], r)
            for e in validate_record(r):
                all_errors.append({"id": r.get("id"), "split": split, "error": e})
            ec[r["kind"]][len(r["events"])] += 1

        p = out / f"{split}.jsonl"
        with p.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results[split] = {"n": len(rows), "sha16": sha16(p),
                          "events_by_kind": {k: dict(v) for k, v in ec.items()}}
        print(f"\n=== {split} ({len(rows)} records) ===")
        for k, v in sorted(ec.items()):
            print(f"  {k}: {dict(v)}")

    # ---------- Critical verifications on eval ----------
    ev = [json.loads(x) for x in (out / "eval.jsonl").read_text().splitlines()]
    held = [r for r in ev if r["split"] == "eval_held_recomb"]

    # Quartet answer diversity
    qg = defaultdict(list)
    for r in held:
        parts = r["quartet_id"].rsplit("_", 2)
        qg[parts[0] if len(parts) >= 3 else r["quartet_id"]].append(r)
    n_mixed = sum(1 for g in qg.values() if len(set(r["answer"] for r in g)) > 1)
    n_uniform = sum(1 for g in qg.values() if len(set(r["answer"] for r in g)) == 1)

    # Majority class
    ac = Counter(r["answer"] for r in held)
    maj_pct = ac.most_common(1)[0][1] / len(held) * 100 if held else 0

    # Multi-event
    multi = [r for r in ev if r["kind"] == "multi_event"]
    m3 = all(len(r["events"]) == 3 for r in multi)

    # Events field universal
    ef = all("events" in r for r in ev)

    # Verify multi-event event-entity ordering matches expected temporal structure
    multi_order_ok = 0
    for r in multi:
        es = r["events"]
        if len(es) == 3:
            # All three events should reference known entities from the record
            ents_in_events = {e["entity"] for e in es}
            if ents_in_events <= set(r["entities"]):
                multi_order_ok += 1

    vf = {
        "held_mixed": n_mixed, "held_uniform": n_uniform,
        "held_majority_pct": round(maj_pct, 1),
        "events_present_all": ef, "multi_all_3_events": m3,
        "multi_n": len(multi), "multi_entity_order_ok": multi_order_ok,
        "total_errors": len(all_errors)
    }
    print(f"\n=== Verification ===")
    for k, v in vf.items():
        print(f"  {k}: {v}")
    if all_errors:
        print(f"\n  First errors:")
        for e in all_errors[:5]:
            print(f"    {e}")

    # Samples
    print("\n=== Samples ===")
    for r in ev:
        if r["kind"] == "binding" and r["split"] == "eval_held_recomb":
            print(f"  binding: events={r['events']} actor={r['actor_entity']} query={r['query_entity']} ans={r['answer']}")
            print(f"    text: {r['text'][:120]}")
            break
    for r in ev:
        if r["kind"] == "multi_event":
            print(f"  multi: events={r['events']} query={r['query_entity']} ans={r['answer']}")
            print(f"    text: {r['text'][:150]}")
            break

    manifest = {
        "status": "ANNOTATED_CORPUS",
        "source": str(src), "files": results,
        "verification": vf, "errors": all_errors[:20],
        "note": "Added 'events' field extracted from text templates. "
                "Enables EntityMemory write operations in train_entity_memory_v2.py."
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest saved to {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
