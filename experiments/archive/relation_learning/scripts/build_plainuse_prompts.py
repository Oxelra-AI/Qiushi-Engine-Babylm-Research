#!/usr/bin/env python3
"""research: build cue-controlled plain-use state/update prompts.

The research changed-form prompts solved the contiguous-copy problem, but manual
reading exposed a possible shortcut: use sentences often contained persistence or
change markers (still/remains/continues/now/new/current).  In a fixed
source/update/use packet those words can tell the learner whether to read the
state from sentence 1 or sentence 2 without entity-state binding.  This script
keeps the same natural sources and state/update/use design while asking Qwen3.5
for a plain present-tense final use sentence with no temporal/change cue.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import time
from typing import Any

ROOT = pathlib.Path.cwd()
DATA = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
BASE_PROMPTS = DATA / "state_use_prompts_9b.jsonl"
UPDATED_OUT = DATA / "state_use_prompts_9b_updated_plainuse_step047.jsonl"
DISTRACTOR_OUT = DATA / "state_use_prompts_9b_distractor_plainuse_step047.jsonl"
UPDATED_PILOT = DATA / "state_use_prompts_9b_updated_plainuse_step047_pilot256.jsonl"
DISTRACTOR_PILOT = DATA / "state_use_prompts_9b_distractor_plainuse_step047_pilot256.jsonl"
META = DATA / "plainuse_prompt_metadata_revision_047.json"

RNG_SEED = 47047
PILOT_N = 256

PLAIN_CUE_WORDS = "still, remains, remain, stayed, stays, continues, continue, today, now, currently, current, new, newly, former, formerly, no longer, anymore, again, later"

UPDATED_TEMPLATE = """You must produce ONE LINE of strict JSON. No explanation, no markdown, no extra text.

Task: Create an UPDATED current-state packet from this natural source sentence.

Source sentence:
{sentence}

Entities found in the source: {entities}

Rules:
1. Pick exactly one target_entity from the entity list. updated_entity must be the same entity.
2. source_state must be a short phrase that the source sentence really says about target_entity. Prefer a role, place, condition, affiliation, activity, possession, property, or relation.
3. Invent a plausible new_state with at least TWO meaningful content words and clearly different content from source_state.
4. update_sentence must mention target_entity and make the new_state true.
5. use_sentence must mention target_entity and use the entity in its updated state.
6. The use_sentence must be a different sentence from the update_sentence. Do not copy the update sentence, and do not reuse a long phrase from it.
7. Plain-use requirement: write use_sentence as an ordinary present-tense sentence. Do not explicitly mark change, persistence, time, or recency.
8. In use_sentence, avoid these cue words or phrases: {cue_words}.
9. Do not put the exact new_state phrase into both update_sentence and use_sentence. Reuse key content words if needed, but change the surrounding wording and word order.
10. Do not copy phrases from the source sentence. Do not use any six-word sequence from the source or update sentence.
11. Avoid generic placeholders and avoid the words box, basket, container, marble. Do not write "Now X is in Y".
12. Keep update_sentence 8-24 words and use_sentence 6-18 words.

Output JSON keys exactly: target_entity, source_state, updated_entity, new_state, update_sentence, use_sentence

Good examples:
{{"target_entity":"Sarah","source_state":"home in Portland","updated_entity":"Sarah","new_state":"Denver apartment","update_sentence":"After accepting the job, Sarah moved her home to a Denver apartment.","use_sentence":"Sarah commutes from her Denver apartment each morning."}}
{{"target_entity":"the old workshop","source_state":"painted dark green","updated_entity":"the old workshop","new_state":"yellow exterior","update_sentence":"Volunteers repainted the old workshop with a bright yellow exterior over the weekend.","use_sentence":"Visitors recognize the old workshop by its yellow exterior."}}
{{"target_entity":"Dr. Patel","source_state":"head of the biology department","updated_entity":"Dr. Patel","new_state":"science faculty dean","update_sentence":"The university promoted Dr. Patel to become science faculty dean.","use_sentence":"Dr. Patel oversees the science faculty as dean."}}

JSON:"""

DISTRACTOR_TEMPLATE = """You must produce ONE LINE of strict JSON. No explanation, no markdown, no extra text.

Task: Create a DISTRACTOR current-state packet from this natural source sentence.

Source sentence:
{sentence}

Entities found in the source: {entities}

Rules:
1. Pick one entity from the entity list as target_entity. Its original state will remain true, but do not say that it remains true in the use sentence.
2. Pick a different updated_entity from the entity list when possible; otherwise invent a different plausible entity related to the scene.
3. source_state must be a short phrase that the source sentence really says about target_entity. It must contain at least TWO meaningful content words.
4. Invent a plausible new_state for updated_entity that clearly differs from the target's source_state.
5. update_sentence must mention updated_entity and make new_state true. It must not change target_entity.
6. use_sentence must mention target_entity and express target_entity's original source_state after the distractor update.
7. The use_sentence must be changed-form, not a copy of the source. Keep at least two key content words from source_state, but rewrite the sentence with different wording or word order.
8. Plain-use requirement: write use_sentence as an ordinary present-tense sentence. Do not explicitly mark persistence, time, unchanged status, or recency.
9. In use_sentence, avoid these cue words or phrases: {cue_words}.
10. Do not repeat the exact source_state phrase. Do not copy any six-word sequence from the source sentence.
11. Avoid generic placeholders and avoid the words box, basket, container, marble. Do not write "Now X is in Y".
12. Keep update_sentence 8-24 words and use_sentence 6-18 words.

Output JSON keys exactly: target_entity, source_state, updated_entity, new_state, update_sentence, use_sentence

Good examples:
{{"target_entity":"the cathedral","source_state":"on the hill overlooking the river","updated_entity":"Marcus","new_state":"hospital job","update_sentence":"Marcus left the factory and took a hospital job downtown.","use_sentence":"Visitors climb the river hill to reach the cathedral."}}
{{"target_entity":"Elena","source_state":"teaches piano at the conservatory","updated_entity":"the corner shop","new_state":"organic produce market","update_sentence":"The corner shop reopened as an organic produce market.","use_sentence":"Elena gives piano lessons inside the conservatory."}}
{{"target_entity":"the lighthouse","source_state":"guides ships along the northern coast","updated_entity":"Captain Torres","new_state":"village retirement","update_sentence":"Captain Torres ended his naval service for village retirement.","use_sentence":"Sailors rely on the northern-coast lighthouse for guidance."}}

JSON:"""


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_base() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with BASE_PROMPTS.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                r.setdefault("original_prompt_index", i)
                rows.append(r)
    return rows


def make_prompt(rec: dict[str, Any], family: str) -> dict[str, Any]:
    out = dict(rec)
    out["base_prompt_id"] = rec.get("id")
    out["prompt_family"] = family
    tmpl = UPDATED_TEMPLATE if rec.get("packet_type") == "UPDATED_USE" else DISTRACTOR_TEMPLATE
    out["prompt"] = tmpl.format(
        sentence=rec["source_sentence"],
        entities=json.dumps(rec.get("candidate_entities", [])[:7], ensure_ascii=False),
        cue_words=PLAIN_CUE_WORDS,
    )
    return out


def main() -> None:
    rows = load_base()
    updated = [make_prompt(r, "updated_plainuse") for r in rows if r.get("packet_type") == "UPDATED_USE"]
    distractor = [make_prompt(r, "distractor_plainuse") for r in rows if r.get("packet_type") == "UNCHANGED_DISTRACTOR_USE"]
    rng = random.Random(RNG_SEED)
    upd_pilot = updated[:]
    dis_pilot = distractor[:]
    rng.shuffle(upd_pilot)
    rng.shuffle(dis_pilot)
    upd_pilot = upd_pilot[:min(PILOT_N, len(upd_pilot))]
    dis_pilot = dis_pilot[:min(PILOT_N, len(dis_pilot))]

    write_jsonl(UPDATED_OUT, updated)
    write_jsonl(DISTRACTOR_OUT, distractor)
    write_jsonl(UPDATED_PILOT, upd_pilot)
    write_jsonl(DISTRACTOR_PILOT, dis_pilot)

    meta = {
        "status": "PLAINUSE_PROMPTS_READY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_prompt_path": str(BASE_PROMPTS),
        "updated_count": len(updated),
        "distractor_count": len(distractor),
        "pilot_n_each": PILOT_N,
        "plain_use_cue_words": PLAIN_CUE_WORDS,
        "paths": {
            "updated_full": str(UPDATED_OUT),
            "distractor_full": str(DISTRACTOR_OUT),
            "updated_pilot": str(UPDATED_PILOT),
            "distractor_pilot": str(DISTRACTOR_PILOT),
        },
        "sha256": {
            "updated_full": sha256_file(UPDATED_OUT),
            "distractor_full": sha256_file(DISTRACTOR_OUT),
            "updated_pilot": sha256_file(UPDATED_PILOT),
            "distractor_pilot": sha256_file(DISTRACTOR_PILOT),
        },
        "purpose": "Cue-controlled natural source/update/use packets: final use sentence must carry state in changed form without temporal, persistence, or change marker shortcuts.",
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
