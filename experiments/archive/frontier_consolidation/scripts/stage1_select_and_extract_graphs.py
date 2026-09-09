#!/usr/bin/env python3
"""research Stage 1: Select high-quality source rows and prepare graph extraction prompts.

Reads the research structural packet candidates, selects ~50 high-quality
Gutenberg/SimpleWiki rows, and writes a JSONL prompt file for graph extraction
using Qwen3.5-9B.
"""
from __future__ import annotations
import json, hashlib, pathlib, random, collections
from typing import Any

ROOT = pathlib.Path(".")
CAND_PATH = ROOT / "experiments/archive/frontier_consolidation/data/structural_packet_candidates/structural_packet_candidates.jsonl"
OUT_DIR = ROOT / "experiments/archive/frontier_consolidation/data/graph_packet_v0"
PROMPT_PATH = OUT_DIR / "stage1_graph_extraction_prompts.jsonl"
SELECTED_PATH = OUT_DIR / "selected_source_rows.jsonl"
META_PATH = OUT_DIR / "stage1_metadata.json"

TARGET_TOTAL = 50
MIN_PER_FAMILY = 8

GRAPH_SYSTEM = (
    "You are a precise information extractor. Given a text passage, you extract the "
    "relational structure as a JSON object. Output ONLY valid JSON, no commentary."
)

GRAPH_PROMPT_TEMPLATE = """Extract the relational structure of this text as JSON.

TEXT:
{text}

Return ONLY valid JSON with this exact structure (no other text before or after):
{{
  "entities": [
    {{"name": "exact name from text", "type": "person|object|place|concept|group", "mentions": 2}}
  ],
  "relations": [
    {{"type": "causal|temporal|spatial|possessive|social|quantitative|state_change|action", "arg1": "entity name", "arg2": "entity name or value", "detail": "one-sentence description"}}
  ],
  "states": [
    {{"entity": "entity name", "property": "property name", "value_before": "value or null", "value_after": "value or null"}}
  ],
  "key_events": [
    {{"event": "brief description", "participants": ["entity names"], "order": 1}}
  ],
  "polarity": "affirmed|negated|mixed",
  "modality": "factual|reported|hypothetical|mixed"
}}"""


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def quality_score(c: dict[str, Any]) -> float:
    """Score candidate by relation diversity and entity richness."""
    hits = c.get("hits", {})
    # Number of distinct relation families with hits
    active_families = sum(1 for k in ["temporal", "causal", "state", "quantity", "social", "modality"]
                          if hits.get(k, 0) > 0)
    # Named entity count (deduplicated)
    named = len(set(c.get("named_spans", [])))
    # Sentence count (more sentences = more structure)
    sents = max(1, c.get("sentence_count", 1))
    # Prefer mid-range sentence counts (not too fragmented, not too few)
    sent_bonus = 1.0 if 4 <= sents <= 12 else 0.7
    return active_families * named * sent_bonus / sents


def select_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select ~TARGET_TOTAL high-quality rows balanced across families."""
    # Filter: Gutenberg/SimpleWiki, quality criteria
    eligible = [c for c in candidates
                if c.get("source_bucket") in ("Gutenberg", "SimpleWiki")
                and c.get("score", 0) > 15
                and c.get("words", 0) > 80
                and len(c.get("named_spans", [])) >= 3]
    
    # Score and sort within each family
    by_family: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in eligible:
        c["_quality"] = quality_score(c)
        by_family[c["family"]].append(c)
    
    for fam in by_family:
        by_family[fam].sort(key=lambda x: -x["_quality"])
    
    # Allocate: MIN_PER_FAMILY per family, rest by global rank
    selected = []
    selected_ids = set()
    
    for fam, rows in sorted(by_family.items()):
        for r in rows[:MIN_PER_FAMILY]:
            if r["row_index"] not in selected_ids:
                selected.append(r)
                selected_ids.add(r["row_index"])
    
    # Fill remaining slots by global quality rank
    remaining = [c for c in eligible if c["row_index"] not in selected_ids]
    remaining.sort(key=lambda x: -x["_quality"])
    for r in remaining:
        if len(selected) >= TARGET_TOTAL:
            break
        selected.append(r)
        selected_ids.add(r["row_index"])
    
    # Deterministic order
    selected.sort(key=lambda x: x["row_index"])
    return selected


def make_graph_prompt(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Create a graph extraction prompt for one source row."""
    return {
        "prompt_id": f"graph_{idx:03d}_row{row['row_index']}",
        "system": GRAPH_SYSTEM,
        "prompt": GRAPH_PROMPT_TEMPLATE.format(text=row["text"]),
        "row_index": row["row_index"],
        "example_id": row.get("example_id"),
        "family": row["family"],
        "source_bucket": row["source_bucket"],
        "words": row["words"],
        "quality_score": row.get("_quality", 0),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = read_jsonl(CAND_PATH)
    selected = select_rows(candidates)
    
    # Save selected source rows (with full text for later use)
    with SELECTED_PATH.open("w", encoding="utf-8") as f:
        for r in selected:
            # Remove internal scoring field
            row_out = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
    
    # Create graph extraction prompts
    prompts = []
    for idx, row in enumerate(selected):
        prompts.append(make_graph_prompt(row, idx))
    
    with PROMPT_PATH.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    # Summary
    family_counts = collections.Counter(r["family"] for r in selected)
    source_counts = collections.Counter(r["source_bucket"] for r in selected)
    
    meta = {
        "status": "STAGE1_PROMPTS_READY",
        "selected_rows": len(selected),
        "by_family": dict(sorted(family_counts.items())),
        "by_source": dict(sorted(source_counts.items())),
        "prompt_file": str(PROMPT_PATH),
        "selected_file": str(SELECTED_PATH),
        "quality_score_range": [
            min(r.get("_quality", 0) for r in selected),
            max(r.get("_quality", 0) for r in selected),
        ],
        "generation_command": (
            f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} "
            f"--model qwen3.5-9b "
            f"--prompts-jsonl {PROMPT_PATH} "
            f"--output-jsonl {OUT_DIR / 'stage1_graph_extraction_outputs.jsonl'} "
            f"--batch-size 8 --max-new-tokens 600 --temperature 0.1 --device cuda"
        ),
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
