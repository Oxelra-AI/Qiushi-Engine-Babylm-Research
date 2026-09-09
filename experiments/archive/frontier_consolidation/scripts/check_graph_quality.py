#!/usr/bin/env python3
"""Quick check: how many compact graph extractions produced valid JSON."""
import json, re, collections

outputs = []
with open("experiments/archive/frontier_consolidation/data/graph_packet_v0/stage1b_compact_graph_outputs.jsonl") as f:
    for line in f:
        if line.strip():
            outputs.append(json.loads(line))

print(f"Total outputs: {len(outputs)}")
tokens = [o.get("generated_tokens", 0) for o in outputs]
print(f"Token counts: min={min(tokens)} max={max(tokens)} mean={sum(tokens)/len(tokens):.0f}")
print(f"At max (400): {sum(1 for t in tokens if t >= 399)}")

valid = 0
failed_reasons = []
for o in outputs:
    text = o.get("output", "").strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                parsed = json.loads(m.group())
            except:
                pass
    if parsed and isinstance(parsed, dict):
        ents = parsed.get("ents", parsed.get("entities", []))
        rels = parsed.get("rels", parsed.get("relations", []))
        if isinstance(ents, list) and len(ents) >= 2 and isinstance(rels, list) and len(rels) >= 1:
            valid += 1
        else:
            failed_reasons.append(f"structure")
    else:
        failed_reasons.append("no-json")

print(f"\nValid graphs: {valid}/{len(outputs)}")
if failed_reasons:
    print("Failures:", collections.Counter(failed_reasons).most_common(5))

# Show first valid
for o in outputs[:10]:
    text = o.get("output", "").strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        g = json.loads(text)
        ents = g.get("ents", g.get("entities", []))
        rels = g.get("rels", g.get("relations", []))
        if isinstance(ents, list) and len(ents) >= 2:
            print(f"\nSample: {o.get('prompt_id','?')}, tokens={o.get('generated_tokens')}")
            print(f"  entities: {len(ents)}, relations: {len(rels)}")
            print(f"  first ent: {ents[0]}")
            print(f"  first rel: {rels[0] if rels else 'none'}")
            break
    except:
        pass
