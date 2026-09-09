#!/usr/bin/env python3
"""research Stage 1b: Compact graph extraction prompts.

Simplified graph format that fits within reasonable token limits.
Uses abbreviated keys and requests only essential structure.
"""
from __future__ import annotations
import json, pathlib, collections
from typing import Any

ROOT = pathlib.Path(".")
SELECTED_PATH = ROOT / "experiments/archive/frontier_consolidation/data/graph_packet_v0/selected_source_rows.jsonl"
OUT_DIR = ROOT / "experiments/archive/frontier_consolidation/data/graph_packet_v0"
PROMPT_PATH = OUT_DIR / "stage1b_compact_graph_prompts.jsonl"

GRAPH_SYSTEM = (
    "You extract relational structure from text as compact JSON. "
    "Use SHORT keys and brief values. Output ONLY the JSON object."
)

GRAPH_PROMPT = """Extract the relational structure of this text as compact JSON.

TEXT:
{text}

Output ONLY valid JSON (no markdown, no commentary):
{{"ents":[{{"n":"Name","t":"person/place/object/group"}}],"rels":[{{"t":"causal/temporal/state/social/spatial/possessive","a":"arg1","b":"arg2","d":"brief fact"}}],"events":[{{"e":"what happened","who":["names"]}}],"pol":"affirmed/negated","mod":"factual/reported/hypothetical"}}

Keep entity names exact from text. Keep descriptions under 10 words each. List at most 6 entities, 5 relations, 4 events."""


def read_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    selected = read_jsonl(SELECTED_PATH)
    prompts = []
    for idx, row in enumerate(selected):
        prompts.append({
            "prompt_id": f"g{idx:03d}_row{row['row_index']}",
            "system": GRAPH_SYSTEM,
            "prompt": GRAPH_PROMPT.format(text=row["text"]),
            "row_index": row["row_index"],
            "family": row["family"],
            "source_bucket": row["source_bucket"],
        })
    
    with PROMPT_PATH.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(json.dumps({
        "status": "STAGE1B_PROMPTS_READY",
        "rows": len(prompts),
        "prompt_file": str(PROMPT_PATH),
        "cmd": (
            f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} "
            f"--model qwen3.5-9b "
            f"--prompts-jsonl {PROMPT_PATH} "
            f"--output-jsonl {OUT_DIR / 'stage1b_compact_graph_outputs.jsonl'} "
            f"--batch-size 16 --max-new-tokens 400 --temperature 0.1 --device cuda"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
