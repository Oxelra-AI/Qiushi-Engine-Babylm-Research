#!/usr/bin/env python3
"""research: prepare prompts for lexically matched graph-transfer probes.

The prompt generator consumes the research graph packet but does not train or score
models.  Its purpose is to ask a generator for a much stricter diagnostic set than
research: relation target/distractor words must be absent from both context views,
context length and lexical overlap must be matched, and a reversed context must
alter only the edge direction/state/polarity/consequence.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import random
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
IN_PACKET = _public_path('experiments/archive/frontier_consolidation/data/graph_packet_v0/graph_packet_v0.jsonl')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe')
PROMPTS = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_prompts.jsonl')
META = _public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/prompt_manifest.json')

PAIR_BY_FAMILY = {
    "event_temporal_causal": ["temporal", "causal"],
    "entity_state_update": ["state", "causal", "temporal"],
    "quantity_change_compare": ["quantity", "temporal"],
    "social_belief_report": ["belief", "polarity", "social"],
    "polarity_contrast_event": ["polarity", "causal", "contrast"],
}

ALLOWED_PAIRS = [
    ["after", "before"],
    ["before", "after"],
    ["because", "despite"],
    ["despite", "because"],
    ["more", "less"],
    ["less", "more"],
    ["inside", "outside"],
    ["outside", "inside"],
    ["accepted", "rejected"],
    ["rejected", "accepted"],
    ["gained", "lost"],
    ["lost", "gained"],
    ["entered", "left"],
    ["left", "entered"],
    ["true", "false"],
    ["false", "true"],
]


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with IN_PACKET.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # Prefer rows with clean verification, enough relation material, and no graph_compact coverage warning.
            issues = obj.get("verification_issues") or []
            graph = obj.get("graph") or {}
            rels = graph.get("relations") or []
            ents = graph.get("entities") or []
            score = 0
            score += 4 if not issues else -2 * len(issues)
            score += min(5, len(rels))
            score += min(4, len(ents))
            score += 2 if obj.get("source_bucket") == "SimpleWiki" else 0
            score += 1 if obj.get("edge_changed") and obj.get("edge_changed") != obj.get("source_text") else 0
            obj["_selection_score"] = score
            rows.append(obj)
    return rows


def compact_graph_text(row: dict[str, Any], max_chars: int = 2200) -> str:
    graph = row.get("graph") or {}
    parts = []
    ents = graph.get("entities") or []
    rels = graph.get("relations") or []
    events = graph.get("key_events") or []
    if ents:
        parts.append("Entities: " + "; ".join(f"{e.get('name')} ({e.get('type')})" for e in ents[:8]))
    if rels:
        parts.append("Relations: " + "; ".join(f"{r.get('type')}: {r.get('arg1')} -> {r.get('arg2')} :: {r.get('detail')}" for r in rels[:8]))
    if events:
        parts.append("Events: " + "; ".join(str(e.get("event")) for e in events[:6]))
    txt = "\n".join(parts)
    return txt[:max_chars]


def make_prompt(row: dict[str, Any], k: int) -> dict[str, Any]:
    family = row.get("family")
    suggested = PAIR_BY_FAMILY.get(family, [])
    source = str(row.get("source_text") or "")[:1500]
    graph_txt = compact_graph_text(row)
    prompt = f"""You are constructing a STRICT diagnostic example for a masked-language-model research probe. Use ONLY the factual relation structure present in the source and graph below. Return one JSON object only; no markdown, no explanation.

Source family: {family}
Suggested relation families: {', '.join(suggested) if suggested else 'any temporal/causal/state/quantity/belief/polarity relation'}
Source text (may be truncated):
{source}

Extracted graph:
{graph_txt}

Required JSON schema:
{{
  "ok": true,
  "relation_family": "temporal|causal|state|quantity|belief|polarity|social",
  "new_names": ["2-5 new names or neutral entity labels, none copied from source"],
  "target": "one word from the allowed target/distractor pairs",
  "distractor": "the opposite word",
  "view_structured": "18-34 words, one natural sentence, same style as view_neutral, encodes the correct edge WITHOUT using target or distractor",
  "view_neutral": "18-34 words, one natural sentence, similar length and lexical overlap, uses the same names/events/objects but does not encode the decisive edge",
  "view_reversed": "18-34 words, minimally different from view_structured, changes ONLY the one edge direction/state/polarity/consequence, WITHOUT using target or distractor",
  "query_prefix": "prefix before the masked relation word; uses the same new names and a different lexicalization from the views",
  "query_suffix": "suffix after the masked relation word",
  "correct_full_query": "query_prefix + target + query_suffix",
  "distractor_full_query": "query_prefix + distractor + query_suffix",
  "edge_correct": "plain description of the correct edge",
  "edge_reversed": "plain description of the reversed edge"
}}

Allowed target/distractor pairs: {ALLOWED_PAIRS}
Hard constraints:
1. target and distractor must be exactly one word each, selected as one of the allowed pairs.
2. Neither target nor distractor may appear anywhere in view_structured, view_neutral, or view_reversed, even as a substring or inflection.
3. view_structured and view_neutral must have the same generation style and differ by at most 5 words in length.
4. view_neutral must contain the same entity names and most of the same content words as view_structured, but omit/neutralize the decisive direction, state, polarity, quantity comparison, or consequence.
5. view_reversed must keep the same entity names and content words but flip only the chosen edge.
6. The query must be answerable from view_structured, contradicted by view_reversed, and not answerable from lexical copying because target/distractor are absent from all views.
7. Use new names/labels in the views and query, not the source names. Avoid exotic words. Avoid lists, tables, arrows, labels, parentheses, and quotation marks inside the views.
8. If no clean edge can be made under these constraints, output {{"ok": false, "reason": "..."}}.
"""
    return {
        "prompt_id": f"match_{k:03d}_row{row.get('row_index')}",
        "index_hint": k,
        "row_index": row.get("row_index"),
        "family": family,
        "source_bucket": row.get("source_bucket"),
        "selection_score": row.get("_selection_score"),
        "prompt": prompt,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    # Balance roughly by family while preferring clean rows.
    by_family: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_family.setdefault(str(r.get("family")), []).append(r)
    for xs in by_family.values():
        xs.sort(key=lambda r: (-int(r.get("_selection_score", 0)), int(r.get("row_index", 10**9))))
    selected = []
    for fam in sorted(by_family):
        selected.extend(by_family[fam][:6])
    # If fewer than 30, fill with next best remaining.
    seen = {id(r) for r in selected}
    remaining = [r for r in sorted(rows, key=lambda r: (-int(r.get("_selection_score", 0)), int(r.get("row_index", 10**9)))) if id(r) not in seen]
    selected.extend(remaining[:max(0, 30 - len(selected))])
    selected = selected[:30]
    prompts = [make_prompt(r, i) for i, r in enumerate(selected)]
    with PROMPTS.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {
        "status": "MATCHED_PROBE_PROMPTS_READY",
        "input_packet": str(IN_PACKET.relative_to(ROOT)),
        "selected": len(selected),
        "by_family": {fam: sum(1 for r in selected if r.get("family") == fam) for fam in sorted(by_family)},
        "prompt_file": str(PROMPTS.relative_to(ROOT)),
        "recommended_command": "CUDA_VISIBLE_DEVICES=0 \"${BABYLM_GENERATOR:?configure-an-external-generator}\" --model qwen3.5-9b --prompts-jsonl %s --output-jsonl %s --batch-size 8 --max-new-tokens 650 --temperature 0.2 --device cuda" % (str(PROMPTS.relative_to(ROOT)), str((_public_path('experiments/archive/frontier_consolidation/data/matched_graph_transfer_probe/matched_probe_generation_outputs.jsonl')).relative_to(ROOT))),
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
