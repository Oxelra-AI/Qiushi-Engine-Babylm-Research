#!/usr/bin/env python3
"""research Stage 2: Parse extracted graphs and prepare transformation prompts.

Reads Stage 1 graph extraction outputs, validates JSON graphs, then creates
matched transformation prompts: graph-preserving compact, entity-renamed,
one-edge-changed, and ordinary compact (baseline).

All transformations use the same model and similar prompt style so that
surface form, fluency, and generation process are tightly matched. The only
variable is the structural constraint.
"""
from __future__ import annotations
import json, pathlib, random, re, collections, hashlib
from typing import Any, Optional

ROOT = pathlib.Path(".")
DATA_DIR = ROOT / "experiments/archive/frontier_consolidation/data/graph_packet_v0"
GRAPH_OUT = DATA_DIR / "stage1b_compact_graph_outputs.jsonl"
SELECTED_PATH = DATA_DIR / "selected_source_rows.jsonl"
TRANSFORM_PROMPTS = DATA_DIR / "stage2_transform_prompts.jsonl"
PARSED_GRAPHS = DATA_DIR / "parsed_graphs.jsonl"
STAGE2_META = DATA_DIR / "stage2_metadata.json"

# ---- Replacement name pools (deterministic, balanced) ----
PERSON_NAMES = [
    "Alex", "Morgan", "Jordan", "Sam", "Taylor", "Casey", "Robin", "Pat",
    "Quinn", "Drew", "Riley", "Blake", "Avery", "Dakota", "Skyler", "Reese",
    "Finley", "Emery", "Sage", "Rowan", "Charlie", "Harper", "Devon", "Ellis",
    "Frankie", "Lee", "Marley", "Phoenix", "Sterling", "Wren",
]
PLACE_NAMES = [
    "Millbrook", "Thornfield", "Ashbury", "Redstone", "Whitehaven",
    "Clearwater", "Oakridge", "Silverdale", "Greenhill", "Stonewall",
    "Fairview", "Meadowdale", "Riverside", "Sunhill", "Pinewood",
]
GROUP_NAMES = [
    "the council", "the committee", "the assembly", "the society", "the guild",
    "the alliance", "the company", "the brigade", "the association", "the order",
]

TRANSFORM_SYSTEM = (
    "You rewrite English text passages precisely as instructed. Follow the "
    "constraints exactly. Output ONLY the rewritten passage, no commentary, "
    "no labels, no quotation marks around the whole output."
)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def try_parse_json(text: str) -> Optional[dict]:
    """Try to extract JSON from LLM output, handling markdown fences and truncation."""
    text = text.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to repair truncated JSON by closing open structures
    repaired = text
    repaired = re.sub(r',\s*"[^"]*"?\s*:?\s*"?[^"]*$', '', repaired)
    repaired = re.sub(r',\s*\{[^}]*$', '', repaired)
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')
    repaired = re.sub(r',\s*$', '', repaired)
    repaired += ']' * max(0, open_brackets)
    repaired += '}' * max(0, open_braces)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Aggressive: find largest valid JSON prefix
    for end in range(len(text), 50, -1):
        candidate = text[:end]
        ob = candidate.count('{') - candidate.count('}')
        ok = candidate.count('[') - candidate.count(']')
        candidate = re.sub(r',\s*$', '', candidate)
        candidate += ']' * max(0, ok) + '}' * max(0, ob)
        try:
            return json.loads(candidate)
        except:
            continue
    return None


def normalize_graph(g: dict) -> dict:
    """Normalize compact graph keys to standard form."""
    out = {}
    # Entities: compact key 'ents' or standard 'entities'
    raw_ents = g.get("ents", g.get("entities", []))
    out["entities"] = []
    for e in (raw_ents if isinstance(raw_ents, list) else []):
        if isinstance(e, dict):
            out["entities"].append({
                "name": e.get("n", e.get("name", "")),
                "type": e.get("t", e.get("type", "")),
            })
    # Relations: compact key 'rels' or standard 'relations'
    raw_rels = g.get("rels", g.get("relations", []))
    out["relations"] = []
    for r in (raw_rels if isinstance(raw_rels, list) else []):
        if isinstance(r, dict):
            out["relations"].append({
                "type": r.get("t", r.get("type", "")),
                "arg1": r.get("a", r.get("arg1", "")),
                "arg2": r.get("b", r.get("arg2", "")),
                "detail": r.get("d", r.get("detail", "")),
            })
    # States
    raw_states = g.get("states", [])
    out["states"] = raw_states if isinstance(raw_states, list) else []
    # Events: compact key 'events' or 'key_events'
    raw_events = g.get("events", g.get("key_events", []))
    out["key_events"] = []
    for ev in (raw_events if isinstance(raw_events, list) else []):
        if isinstance(ev, dict):
            out["key_events"].append({
                "event": ev.get("e", ev.get("event", ev.get("description", ""))),
                "participants": ev.get("who", ev.get("participants", [])),
            })
    out["polarity"] = g.get("pol", g.get("polarity", "affirmed"))
    out["modality"] = g.get("mod", g.get("modality", "factual"))
    return out


def validate_graph(g: dict) -> tuple[bool, str]:
    """Validate extracted graph has minimum required structure."""
    if not isinstance(g, dict):
        return False, "not a dict"
    ng = normalize_graph(g)
    entities = ng.get("entities", [])
    relations = ng.get("relations", [])
    if len(entities) < 2:
        return False, f"need >=2 entities, got {len(entities)}"
    if len(relations) < 1:
        return False, f"need >=1 relation, got {len(relations)}"
    for e in entities:
        if not e.get("name"):
            return False, "entity missing name"
    for r in relations:
        if not r.get("arg1") and not r.get("arg2"):
            return False, "relation missing args"
    return True, "ok"


def make_entity_rename_map(graph: dict, rng: random.Random) -> dict[str, str]:
    """Create deterministic entity name replacements."""
    entities = graph.get("entities", [])
    rename_map = {}
    person_pool = list(PERSON_NAMES)
    place_pool = list(PLACE_NAMES)
    group_pool = list(GROUP_NAMES)
    rng.shuffle(person_pool)
    rng.shuffle(place_pool)
    rng.shuffle(group_pool)
    
    pi, pli, gi = 0, 0, 0
    for e in entities:
        name = e.get("name", "")
        etype = str(e.get("type", "")).lower()
        if name in rename_map:
            continue
        if "place" in etype or "location" in etype:
            if pli < len(place_pool):
                rename_map[name] = place_pool[pli]
                pli += 1
            else:
                rename_map[name] = f"Place_{pli}"
                pli += 1
        elif "group" in etype:
            if gi < len(group_pool):
                rename_map[name] = group_pool[gi]
                gi += 1
            else:
                rename_map[name] = f"Group_{gi}"
                gi += 1
        else:  # person, object, concept, default
            if pi < len(person_pool):
                rename_map[name] = person_pool[pi]
                pi += 1
            else:
                rename_map[name] = f"Entity_{pi}"
                pi += 1
    return rename_map


def select_edge_to_change(graph: dict) -> Optional[dict]:
    """Select the most interesting relation to change for the one-edge counterpart."""
    relations = graph.get("relations", [])
    states = graph.get("states", [])
    
    # Prefer state changes, then causal/temporal, then others
    priority = {"state_change": 3, "causal": 2, "temporal": 2, "social": 1,
                "quantitative": 1, "possessive": 1, "spatial": 1, "action": 2}
    
    candidates = []
    for r in relations:
        rtype = str(r.get("type", "")).lower()
        matches = [priority.get(t, 0) for t in priority if t in rtype]
        score = max(matches) if matches else 0
        candidates.append((score, r))
    
    # Also consider states
    for s in states:
        if s.get("value_after") and s.get("value_before"):
            candidates.append((3, {
                "type": "state_change",
                "arg1": s.get("entity", ""),
                "arg2": f"{s.get('property', '')}: {s.get('value_before')} -> {s.get('value_after')}",
                "detail": f"{s.get('entity')} changed {s.get('property')} from {s.get('value_before')} to {s.get('value_after')}",
            }))
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def format_relations_for_prompt(graph: dict) -> str:
    """Format graph relations as a concise constraint list for prompts."""
    lines = []
    for r in graph.get("relations", []):
        detail = r.get("detail", f"{r.get('arg1', '?')} {r.get('type', '?')} {r.get('arg2', '?')}")
        lines.append(f"- {detail}")
    for s in graph.get("states", []):
        if s.get("value_after"):
            lines.append(f"- {s.get('entity', '?')}'s {s.get('property', '?')}: {s.get('value_before', '?')} → {s.get('value_after', '?')}")
    for ev in graph.get("key_events", [])[:3]:
        lines.append(f"- Event: {ev.get('event', '?')}")
    return "\n".join(lines[:10])  # Cap at 10 lines


def make_transform_prompts(row: dict, graph: dict, idx: int, rng: random.Random) -> list[dict]:
    """Create all four transformation prompts for one source row."""
    text = row["text"]
    row_idx = row["row_index"]
    base_id = f"t{idx:03d}_row{row_idx}"
    relations_text = format_relations_for_prompt(graph)
    
    prompts = []
    
    # 1. Graph-preserving compact view
    prompts.append({
        "prompt_id": f"{base_id}_graph_compact",
        "transform_type": "graph_compact",
        "row_index": row_idx,
        "system": TRANSFORM_SYSTEM,
        "prompt": (
            f"Rewrite the following passage more concisely (roughly half the length) "
            f"while preserving ALL of these specific facts and relationships:\n"
            f"{relations_text}\n\n"
            f"Keep every named entity and every relationship listed above. "
            f"Use fewer words but do not drop any fact.\n\n"
            f"PASSAGE:\n{text}"
        ),
    })
    
    # 2. Entity-renamed view (same structure, different names)
    rename_map = make_entity_rename_map(graph, rng)
    rename_instructions = "\n".join(f'  "{old}" → "{new}"' for old, new in rename_map.items())
    prompts.append({
        "prompt_id": f"{base_id}_entity_renamed",
        "transform_type": "entity_renamed",
        "row_index": row_idx,
        "rename_map": rename_map,
        "system": TRANSFORM_SYSTEM,
        "prompt": (
            f"Rewrite the following passage, replacing entity names as specified below. "
            f"Keep ALL events, actions, relationships, states, temporal order, and structure "
            f"exactly the same. Change ONLY the names.\n\n"
            f"Name replacements:\n{rename_instructions}\n\n"
            f"PASSAGE:\n{text}"
        ),
    })
    
    # 3. One-edge-changed coherent counterpart
    edge = select_edge_to_change(graph)
    if edge:
        edge_desc = edge.get("detail", f"{edge.get('arg1', '?')} {edge.get('type', '?')} {edge.get('arg2', '?')}")
        prompts.append({
            "prompt_id": f"{base_id}_edge_changed",
            "transform_type": "edge_changed",
            "row_index": row_idx,
            "changed_edge": edge,
            "system": TRANSFORM_SYSTEM,
            "prompt": (
                f"Rewrite the following passage, changing ONLY this one fact:\n"
                f"  Original: {edge_desc}\n"
                f"  Change it to a different but coherent alternative.\n\n"
                f"Keep everything else exactly the same: all other events, entities, "
                f"relationships, and temporal order. The rewrite must be internally "
                f"coherent and grammatical.\n\n"
                f"PASSAGE:\n{text}"
            ),
        })
    
    # 4. Ordinary compact (no structural constraint — baseline)
    prompts.append({
        "prompt_id": f"{base_id}_ordinary_compact",
        "transform_type": "ordinary_compact",
        "row_index": row_idx,
        "system": TRANSFORM_SYSTEM,
        "prompt": (
            f"Summarize the following passage more concisely (roughly half the length). "
            f"Capture the main idea but do not worry about preserving every specific "
            f"name, relationship, or detail.\n\n"
            f"PASSAGE:\n{text}"
        ),
    })
    
    return prompts


def main():
    # Load selected source rows and graph extraction outputs
    selected = read_jsonl(SELECTED_PATH)
    graph_outputs = read_jsonl(GRAPH_OUT)
    # Load input prompts to recover metadata (batch tool strips custom fields)
    input_prompts = read_jsonl(DATA_DIR / "stage1b_compact_graph_prompts.jsonl")
    
    # Index by row_index for matching
    selected_by_row = {r["row_index"]: r for r in selected}
    # Build index→prompt_metadata map
    prompt_by_index = {i: p for i, p in enumerate(input_prompts)}
    
    # Parse graphs and match to source rows
    parsed = []
    failed = []
    for go in graph_outputs:
        # Match by output index to input prompt
        idx = go.get("index")
        if idx is not None and idx in prompt_by_index:
            meta = prompt_by_index[idx]
            row_idx = meta.get("row_index")
        else:
            # Fallback: try prompt_id from output
            prompt_id = go.get("prompt_id", "")
            m = re.search(r"row(\d+)", prompt_id)
            row_idx = int(m.group(1)) if m else None
        
        output_text = go.get("output", "")
        graph = try_parse_json(output_text)
        
        if graph is None:
            failed.append({"row_index": row_idx, "reason": "JSON parse failed", "output_preview": output_text[:200]})
            continue
        
        valid, reason = validate_graph(graph)
        if not valid:
            failed.append({"row_index": row_idx, "reason": reason, "output_preview": output_text[:200]})
            continue
        
        if row_idx not in selected_by_row:
            failed.append({"row_index": row_idx, "reason": "row_index not in selected"})
            continue
        
        ng = normalize_graph(graph)
        parsed.append({
            "row_index": row_idx,
            "source_row": selected_by_row[row_idx],
            "graph": ng,
            "n_entities": len(ng.get("entities", [])),
            "n_relations": len(ng.get("relations", [])),
            "n_states": len(ng.get("states", [])),
            "n_events": len(ng.get("key_events", [])),
        })
    
    # Save parsed graphs
    with PARSED_GRAPHS.open("w", encoding="utf-8") as f:
        for p in parsed:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    # Create transformation prompts
    rng = random.Random(165829)
    all_prompts = []
    for idx, p in enumerate(parsed):
        row = p["source_row"]
        graph = p["graph"]
        transforms = make_transform_prompts(row, graph, idx, rng)
        all_prompts.extend(transforms)
    
    with TRANSFORM_PROMPTS.open("w", encoding="utf-8") as f:
        for pr in all_prompts:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    
    # Stats
    type_counts = collections.Counter(p["transform_type"] for p in all_prompts)
    
    meta = {
        "status": "STAGE2_TRANSFORM_PROMPTS_READY",
        "graph_extraction_total": len(graph_outputs),
        "parsed_valid": len(parsed),
        "failed": len(failed),
        "failed_details": failed[:10],  # First 10 failures for inspection
        "transform_prompts_total": len(all_prompts),
        "by_transform_type": dict(sorted(type_counts.items())),
        "parsed_graphs_file": str(PARSED_GRAPHS),
        "transform_prompts_file": str(TRANSFORM_PROMPTS),
        "generation_command": (
            f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} "
            f"--model qwen3.5-9b "
            f"--prompts-jsonl {TRANSFORM_PROMPTS} "
            f"--output-jsonl {DATA_DIR / 'stage2_transform_outputs.jsonl'} "
            f"--batch-size 8 --max-new-tokens 300 --temperature 0.3 --device cuda"
        ),
    }
    STAGE2_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
