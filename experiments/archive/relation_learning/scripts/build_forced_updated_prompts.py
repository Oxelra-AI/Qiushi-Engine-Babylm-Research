#!/usr/bin/env python3
"""research: build high-precision UPDATED_USE prompts for Qwen3.5-9B.

The first natural prompt generated many usable distractor-retention packets but
few updated-use packets after adding the necessary state-content check.  This
prompt keeps the natural corpus source sentences but constrains the output so the
use sentence must lexically carry the newly chosen state.  It is intended to
increase the deep-update half without touching the Entity benchmark register.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

ROOT = pathlib.Path.cwd()
DATA = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
BASE_PROMPTS = DATA / "state_use_prompts_9b.jsonl"
OUT = DATA / "state_use_prompts_9b_updated_forced_step044.jsonl"
META = DATA / "forced_updated_prompt_metadata_revision_044.json"

PROMPT_TEMPLATE = """You must produce ONE LINE of strict JSON. No explanation, no markdown, no extra text.

Task: Create an UPDATED current-state training packet from this natural source sentence.

Source sentence:
{sentence}

Entities found in the source: {entities}

Hard requirements:
1. Pick exactly one target_entity from the entity list. updated_entity must be the same text.
2. source_state must be a short phrase that is actually expressed in the source sentence about target_entity. Prefer a location, role, affiliation, property, condition, activity, or relation; do not invent the source_state.
3. Invent a plausible new_state that clearly differs from source_state and uses different content words.
4. update_sentence must mention the target_entity and must include the exact new_state phrase you wrote.
5. use_sentence must mention the target_entity and must include the exact new_state phrase you wrote. This sentence should require the current state after the update.
6. use_sentence must not repeat the old source_state content.
7. Avoid generic placeholders and avoid the words box, basket, container, marble. Do not use the form "Now X is in Y".
8. Keep update_sentence 8-24 words and use_sentence 6-18 words.

Output JSON keys exactly: target_entity, source_state, updated_entity, new_state, update_sentence, use_sentence

Examples:
{{"target_entity":"Sarah","source_state":"lives in Portland","updated_entity":"Sarah","new_state":"lives in Denver","update_sentence":"After accepting a new position, Sarah now lives in Denver.","use_sentence":"Sarah now lives in Denver and commutes by train."}}
{{"target_entity":"the old workshop","source_state":"painted dark green","updated_entity":"the old workshop","new_state":"painted bright yellow","update_sentence":"Volunteers repainted the old workshop painted bright yellow over the weekend.","use_sentence":"Visitors recognize the old workshop by its painted bright yellow exterior."}}
{{"target_entity":"Dr. Patel","source_state":"head of the biology department","updated_entity":"Dr. Patel","new_state":"dean of the science faculty","update_sentence":"The university promoted Dr. Patel to dean of the science faculty.","use_sentence":"Dr. Patel serves as dean of the science faculty this year."}}

JSON:"""


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    prompts = []
    with BASE_PROMPTS.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("packet_type") != "UPDATED_USE":
                continue
            out = dict(rec)
            out["base_prompt_id"] = rec.get("id")
            out["prompt_family"] = "forced_updated_use"
            out["prompt"] = PROMPT_TEMPLATE.format(
                sentence=rec["source_sentence"],
                entities=json.dumps(rec.get("candidate_entities", [])[:7], ensure_ascii=False),
            )
            prompts.append(out)
    with OUT.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {
        "status": "FORCED_UPDATED_PROMPTS_READY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_prompt_path": str(BASE_PROMPTS),
        "prompt_count": len(prompts),
        "prompt_path": str(OUT),
        "sha256_prompts": sha256_file(OUT),
        "purpose": "Increase high-precision UPDATED_USE yield while keeping natural corpus sources; output use_sentence must carry exact new_state phrase.",
    }
    META.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
