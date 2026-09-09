#!/usr/bin/env python3
"""Quick quality audit of the graph packet transformations."""
import json

rows = []
with open("experiments/archive/frontier_consolidation/data/graph_packet_v0/graph_packet_v0.jsonl") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

print(f"Total rows: {len(rows)}")

# 1. Entity rename quality
rename_good = 0
rename_partial = 0
rename_failed = 0
for r in rows:
    rm = r.get("entity_rename_map", {})
    if not rm:
        rename_failed += 1
        continue
    er_text = r.get("entity_renamed", "").lower()
    new_found = sum(1 for new in rm.values() if new.lower() in er_text)
    old_found = sum(1 for old in rm.keys() if old.lower() in er_text)
    new_frac = new_found / max(1, len(rm))
    old_frac = old_found / max(1, len(rm))
    if new_frac >= 0.5 and old_frac <= 0.3:
        rename_good += 1
    elif new_frac >= 0.3:
        rename_partial += 1
    else:
        rename_failed += 1

print(f"\nEntity rename: good={rename_good}, partial={rename_partial}, failed={rename_failed}")

# 2. Edge changed quality
import difflib
edge_different = 0
edge_same = 0
edge_similar = 0
for r in rows:
    src = r.get("source_text", "").strip()
    ec = r.get("edge_changed", "").strip()
    if not ec:
        edge_same += 1
        continue
    ratio = difflib.SequenceMatcher(None, src, ec).ratio()
    if ratio > 0.95:
        edge_same += 1
    elif ratio > 0.7:
        edge_similar += 1
    else:
        edge_different += 1

print(f"Edge changed: different={edge_different}, similar={edge_similar}, nearly same={edge_same}")

# 3. Graph compact quality
gc_short = 0
gc_preserved_ents = 0
for r in rows:
    gc = r.get("graph_compact", "")
    if not gc:
        continue
    src_words = r.get("source_words", 160)
    gc_words = len(gc.split())
    if gc_words < src_words * 0.6:
        gc_short += 1
    ents = [e.get("name", "") for e in r.get("graph", {}).get("entities", [])]
    found = sum(1 for e in ents if e.lower() in gc.lower())
    if found >= len(ents) * 0.5:
        gc_preserved_ents += 1

print(f"Graph compact: shorter than 60% src={gc_short}, preserved >=50% entities={gc_preserved_ents}")

# 4. Ordinary compact quality
oc_short = 0
for r in rows:
    oc = r.get("ordinary_compact", "")
    if oc and len(oc.split()) < r.get("source_words", 160) * 0.6:
        oc_short += 1

print(f"Ordinary compact: shorter than 60% src={oc_short}")

# 5. Overall usability
usable = 0
for r in rows:
    issues = r.get("verification_issues", [])
    gc = r.get("graph_compact", "")
    oc = r.get("ordinary_compact", "")
    er = r.get("entity_renamed", "")
    ec = r.get("edge_changed", "")
    if gc and oc and er and ec:
        usable += 1

print(f"\nAll 4 transforms present: {usable}/{len(rows)}")
