#!/usr/bin/env python3
"""Repair truncated JSON graphs by closing open brackets/braces."""
import json, re

def repair_truncated_json(text: str) -> dict | None:
    """Try to repair truncated JSON by closing open structures."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    
    # Direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to close truncated JSON
    # Remove trailing incomplete string (no closing quote)
    repaired = text
    # Remove last incomplete key-value pair
    repaired = re.sub(r',\s*"[^"]*"?\s*:?\s*"?[^"]*$', '', repaired)
    repaired = re.sub(r',\s*\{[^}]*$', '', repaired)  # Remove last incomplete object in array
    
    # Close open brackets
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    
    # Remove trailing comma before closing
    repaired = re.sub(r',\s*$', '', repaired)
    
    repaired += ']' * max(0, open_brackets)
    repaired += '}' * max(0, open_braces)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    
    # More aggressive: find the largest valid JSON prefix
    for end in range(len(text), 50, -1):
        candidate = text[:end]
        open_b = candidate.count('{') - candidate.count('}')
        open_k = candidate.count('[') - candidate.count(']')
        candidate = re.sub(r',\s*$', '', candidate)
        candidate += ']' * max(0, open_k)
        candidate += '}' * max(0, open_b)
        try:
            return json.loads(candidate)
        except:
            continue
    return None

# Process all outputs
outputs = []
with open("experiments/archive/frontier_consolidation/data/graph_packet_v0/stage1b_compact_graph_outputs.jsonl") as f:
    for line in f:
        if line.strip():
            outputs.append(json.loads(line))

valid = 0
repaired = 0
still_failed = 0
for o in outputs:
    text = o.get("output", "").strip()
    g = repair_truncated_json(text)
    if g and isinstance(g, dict):
        ents = g.get("ents", g.get("entities", []))
        rels = g.get("rels", g.get("relations", []))
        if isinstance(ents, list) and len(ents) >= 2 and isinstance(rels, list) and len(rels) >= 1:
            # Check if it was already valid without repair
            try:
                json.loads(text)
                valid += 1
            except:
                repaired += 1
                valid += 1
        else:
            still_failed += 1
    else:
        still_failed += 1

print(f"Originally valid: {valid - repaired}")
print(f"Repaired: {repaired}")
print(f"Total valid after repair: {valid}")
print(f"Still failed: {still_failed}")
