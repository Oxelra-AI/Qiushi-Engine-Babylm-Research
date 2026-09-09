#!/usr/bin/env python3
"""research: build same-source contrastive entity-binding generation prompts.

Purpose
-------
The state-update replacement arm showed that ordinary WWM on weak natural packets
mainly trains coarse in-source retention/recency responses rather than entity-
gated selection. The credit-allocation pilot showed that focused answer masking
can train the missing retain/update discrimination. This script prepares the
side of the next crossed experiment: same-source contrastive packets where the
only intended sufficient predictor of the answer slot is whether the update is
about the target entity or a different entity.

The generator is asked for one source sentence, two updates, and one shared use
frame. The two training rows derived from an accepted JSON are:
  UPDATE: source + target_update_sentence + use_frame(new_answer)
  RETAIN: source + distractor_update_sentence + use_frame(source_answer)
The two use sentences differ only in the masked answer phrase.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import random
import sys
import time
from collections import Counter
from typing import Any

ROOT = pathlib.Path.cwd()
SELECTED_PAIRS = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl"
research = ROOT / "experiments/archive/relation_learning/scripts/build_9b_state_use_prompts.py"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/contrastive_binding_generation"
PROMPTS_ALL = OUT_DIR / "contrastive_binding_prompts.jsonl"
PROMPTS_PILOT = OUT_DIR / "contrastive_binding_prompts_pilot512.jsonl"
META = OUT_DIR / "contrastive_binding_prompt_metadata.json"

RNG_SEED = 57057
PILOT_N = 512

PROMPT_TEMPLATE = """You must produce ONE LINE of strict JSON. No explanation, no markdown, no extra text.

Task: Create a CONTRASTIVE entity-state packet from this natural source sentence. The packet will become two training examples with the SAME source and SAME final use frame:
- TARGET-UPDATE example: the target entity is updated, so the final use frame should take the target's NEW answer phrase.
- DISTRACTOR-UPDATE example: a different entity is updated, so the final use frame should take the target's SOURCE answer phrase.

Source sentence:
{sentence}

Candidate entities from the source: {entities}

Rules:
1. Choose target_entity and distractor_entity from the candidate list. They must be different real entities in the source.
2. target_source_state must be a short state, role, relation, possession, location, activity, attribute, or condition that the source really says about target_entity.
3. target_source_answer is the answer phrase that will be inserted into the use frame when the distractor is updated. It should express the target_source_state, but should NOT be an exact copied phrase from the source if a natural changed wording is possible.
4. target_new_update_state is a plausible new state for target_entity, different from target_source_state.
5. target_new_answer is the answer phrase inserted into the use frame when target_entity is updated. It should express target_new_update_state, but should NOT be an exact copied phrase from target_update_sentence if a natural changed wording is possible.
6. distractor_source_state may be brief; distractor_new_update_state must be plausible for distractor_entity and different from the target states.
7. target_update_sentence must mention target_entity and make target_new_update_state true. It should NOT mention distractor_entity.
8. distractor_update_sentence must mention distractor_entity and make distractor_new_update_state true. It should NOT mention target_entity and should NOT change target_entity.
9. use_sentence_frame must mention target_entity once and contain exactly one literal placeholder {{STATE}}. It must be an ordinary sentence with no time, persistence, change, or recency cue.
10. Avoid cue words in the use frame and updates: still, remains, remain, stayed, stays, continues, today, now, currently, current, new, newly, former, formerly, no longer, anymore, again, later.
11. Avoid benchmark-like containers or toy states: box, basket, container, marble.
12. Keep each update sentence 8-22 words, and the completed use sentence 7-18 words.

Output JSON keys exactly:
target_entity, distractor_entity, target_source_state, target_source_answer, target_new_update_state, target_new_answer, distractor_source_state, distractor_new_update_state, target_update_sentence, distractor_update_sentence, use_sentence_frame

Good examples:
{{"target_entity":"Professor Chen","distractor_entity":"Dr. Reyes","target_source_state":"kept leather-bound journals on the shelf","target_source_answer":"bound research journals","target_new_update_state":"used digital tablets for cataloging","target_new_answer":"cataloging tablets","distractor_source_state":"stored handwritten field notes in her cabinet","distractor_new_update_state":"used printed spreadsheets for records","target_update_sentence":"Professor Chen replaced the shelf materials with digital tablets for cataloging.","distractor_update_sentence":"Dr. Reyes converted her cabinet records into printed spreadsheets after the audit.","use_sentence_frame":"A visiting scholar found {{STATE}} in Professor Chen's workspace."}}
{{"target_entity":"Chef Kim","distractor_entity":"Chef Park","target_source_state":"prepared spicy tofu dishes at the restaurant","target_source_answer":"spiced tofu plates","target_new_update_state":"served mushroom risotto as the signature dish","target_new_answer":"mushroom risotto","distractor_source_state":"specialized in grilled salmon plates","distractor_new_update_state":"served lamb stew as the signature dish","target_update_sentence":"Chef Kim redesigned the menu around mushroom risotto as the signature dish.","distractor_update_sentence":"Chef Park redesigned his menu around lamb stew for the evening service.","use_sentence_frame":"Food critics praised Chef Kim for the exceptional {{STATE}} served at the tasting."}}
{{"target_entity":"Maria","distractor_entity":"James","target_source_state":"lived in a small apartment near the university","target_source_answer":"campus apartment","target_new_update_state":"moved into a converted warehouse studio","target_new_answer":"warehouse studio","distractor_source_state":"occupied a large house by the lake","distractor_new_update_state":"moved into a garden cottage","target_update_sentence":"Maria moved into a converted warehouse studio for more room to paint.","distractor_update_sentence":"James left the lake house and settled into a garden cottage outside town.","use_sentence_frame":"Friends visiting Maria found her in a {{STATE}} with high ceilings."}}

JSON:"""


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def import_step043():
    spec = importlib.util.spec_from_file_location("prompts", research)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {research}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_pairs() -> list[dict[str, Any]]:
    rows = []
    with SELECTED_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prompt(pair: dict[str, Any], ents: list[str], order_index: int) -> dict[str, Any]:
    source = " ".join(str(pair["original"]).replace("\u00a0", " ").split())
    return {
        "id": f"cb57_{pair['pair_id']}",
        "prompt": PROMPT_TEMPLATE.format(sentence=source, entities=json.dumps(ents[:8], ensure_ascii=False)),
        "pair_id": pair["pair_id"],
        "source_sentence": source,
        "qwen_rewrite": " ".join(str(pair.get("rewrite", "")).split()),
        "candidate_entities": ents[:8],
        "source_words": len(source.split()),
        "source": pair.get("source_name") or pair.get("source", ""),
        "example_id": pair.get("source_example_id") or pair.get("example_id", -1),
        "source_sentence_idx": pair.get("source_sentence_idx"),
        "cohort": pair.get("cohort", ""),
        "order_index": order_index,
        "prompt_family": "contrastive_binding_same_source",
    }


def main() -> None:
    rng = random.Random(RNG_SEED)
    mod = import_step043()
    pairs = load_pairs()
    amenable = []
    ent_count = Counter()
    for p in pairs:
        ents = mod.entity_candidates(p)
        ent_count[len(ents)] += 1
        if len(ents) >= 2:
            amenable.append((p, ents))
    rng.shuffle(amenable)
    prompts = [build_prompt(p, ents, i) for i, (p, ents) in enumerate(amenable)]
    pilot = prompts[:min(PILOT_N, len(prompts))]
    write_jsonl(PROMPTS_ALL, prompts)
    write_jsonl(PROMPTS_PILOT, pilot)
    reg = Counter(r.get("source", "") for r in prompts)
    cohort = Counter(r.get("cohort", "") for r in prompts)
    meta = {
        "status": "CONTRASTIVE_BINDING_PROMPTS_READY",
        "created_utc": now_utc(),
        "selected_pairs": len(pairs),
        "entity_count_distribution_first10": {str(k): ent_count.get(k, 0) for k in range(10)},
        "amenable_ge2_entities": len(prompts),
        "pilot_n": len(pilot),
        "purpose": "same-source contrastive target-update vs distractor-update packets for crossed data/objective credit experiment",
        "outputs": {"all_prompts": str(PROMPTS_ALL), "pilot_prompts": str(PROMPTS_PILOT)},
        "sha256": {"all_prompts": sha256_file(PROMPTS_ALL), "pilot_prompts": sha256_file(PROMPTS_PILOT)},
        "register_distribution": dict(reg.most_common()),
        "cohort_distribution": dict(cohort.most_common()),
        "validation_plan": "Use validate_contrastive_binding_outputs.py after Qwen generation; accepted JSONs materialize UPDATE and RETAIN rows with the same source and same use frame, differing only in the answer phrase and whether the update targets target_entity or distractor_entity.",
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
