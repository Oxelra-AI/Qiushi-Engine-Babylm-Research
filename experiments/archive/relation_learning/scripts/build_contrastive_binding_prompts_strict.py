#!/usr/bin/env python3
"""research: build strict same-source contrastive binding generation prompts.

This supersedes the research prompt set for the contrastive x credit experiment.
research allowed Qwen to choose roles and separate target/distractor update states.
The strict design makes entity identity the intended discriminative variable:
  * A target and distractor entity are assigned by the script from source-order
    candidates; role/source-position direction alternates.
  * The generator must use those exact entities.
  * Both entities must have source states.
  * A single shared update sentence frame with one {ENTITY} placeholder is
    generated; UPDATE and RETAIN variants differ in the update entity mention
    only, while the new-state phrase is identical.
  * A single target use frame with one {STATE} placeholder is shared.

Rows derived from an accepted object:
  UPDATE = source + update_frame(target_entity) + use_frame(shared_new_answer)
  RETAIN = source + update_frame(distractor_entity) + use_frame(target_source_answer)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import random
import re
import sys
import time
from collections import Counter
from typing import Any

ROOT = pathlib.Path.cwd()
SELECTED_PAIRS = ROOT / "experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl"
research = ROOT / "experiments/archive/relation_learning/scripts/build_9b_state_use_prompts.py"
OUT_DIR = ROOT / "experiments/archive/relation_learning/data/contrastive_binding_generation_strict"
PROMPTS_ALL = OUT_DIR / "contrastive_binding_prompts_strict.jsonl"
PROMPTS_PILOT = OUT_DIR / "contrastive_binding_prompts_strict_pilot512.jsonl"
META = OUT_DIR / "contrastive_binding_prompt_metadata_strict.json"

RNG_SEED = 58058
PILOT_N = 512
MAX_CANDIDATES = 8

PROMPT_TEMPLATE = """You must produce ONE LINE of strict JSON. No explanation, no markdown, no extra text.

Task: Create a STRICT CONTRASTIVE entity-state packet from this natural source sentence. The experiment will make two examples with the SAME source and SAME target use frame. Entity identity should be the only informative difference between the examples.

Source sentence:
{sentence}

All candidate entities from the source: {entities}
Assigned target_entity: {target_entity}
Assigned distractor_entity: {distractor_entity}

The two examples will be constructed by the validator as:
- TARGET-UPDATE: source + shared_update_sentence_frame with {{ENTITY}} replaced by the target_entity + target use frame with {{STATE}} replaced by shared_new_answer.
- DISTRACTOR-UPDATE / RETAIN: source + the SAME shared_update_sentence_frame with {{ENTITY}} replaced by the distractor_entity + the SAME target use frame with {{STATE}} replaced by target_source_answer.

Rules:
1. Use exactly the assigned target_entity and assigned distractor_entity. Do not choose different entities.
2. The source must say a distinct state, role, relation, possession, location, activity, attribute, or condition about BOTH assigned entities.
3. target_source_state and target_source_answer describe the target's source state. target_source_answer should be a short answer phrase that is actually supported by the source.
4. distractor_source_state and distractor_source_answer describe the distractor's source state. They make the distractor a real state-bearing entity, although the final use frame asks about the target.
5. shared_new_update_state and shared_new_answer describe one plausible changed state that could apply to either assigned entity. This is the same recent competitor phrase in both examples.
6. shared_update_sentence_frame must contain exactly one literal placeholder {{ENTITY}}. Replacing it with either assigned entity must produce a grammatical update sentence, and the rest of the sentence must be identical. Do not use pronouns that refer back to the entity.
7. use_sentence_frame must mention the target_entity exactly once and contain exactly one literal placeholder {{STATE}}. It must not mention the distractor_entity.
8. Do not use time, persistence, change, or recency cue words in either frame: still, remains, remain, stayed, stays, continues, today, now, currently, current, new, newly, former, formerly, no longer, anymore, again, later.
9. Avoid benchmark-like containers or toy states: box, basket, container, marble.
10. Keep the materialized update sentence 8-24 words and the completed use sentence 7-20 words.
11. The answer phrases should be short enough to mask and score: usually 1-4 content words.

Output JSON keys exactly:
target_entity, distractor_entity, target_source_state, target_source_answer, distractor_source_state, distractor_source_answer, shared_new_update_state, shared_new_answer, shared_update_sentence_frame, use_sentence_frame

Good examples:
{{"target_entity":"Professor Chen","distractor_entity":"Dr. Reyes","target_source_state":"kept leather-bound journals on the shelf","target_source_answer":"leather-bound journals","distractor_source_state":"stored handwritten field notes in her cabinet","distractor_source_answer":"handwritten field notes","shared_new_update_state":"used cataloging tablets for research materials","shared_new_answer":"cataloging tablets","shared_update_sentence_frame":"During the inventory, {{ENTITY}} adopted cataloging tablets for all research materials.","use_sentence_frame":"A visiting scholar found {{STATE}} in Professor Chen's workspace."}}
{{"target_entity":"Chef Kim","distractor_entity":"Chef Park","target_source_state":"prepared spicy tofu dishes at the restaurant","target_source_answer":"spicy tofu dishes","distractor_source_state":"specialized in grilled salmon plates","distractor_source_answer":"grilled salmon plates","shared_new_update_state":"served mushroom risotto as the signature dish","shared_new_answer":"mushroom risotto","shared_update_sentence_frame":"For the tasting menu, {{ENTITY}} served mushroom risotto as the signature dish.","use_sentence_frame":"Food critics praised Chef Kim for the exceptional {{STATE}} served at the tasting."}}
{{"target_entity":"Maria","distractor_entity":"James","target_source_state":"lived in a small apartment near the university","target_source_answer":"campus apartment","distractor_source_state":"occupied a large house by the lake","distractor_source_answer":"lake house","shared_new_update_state":"moved into a warehouse studio","shared_new_answer":"warehouse studio","shared_update_sentence_frame":"Before the exhibition season, {{ENTITY}} moved into a warehouse studio with skylights.","use_sentence_frame":"Friends visiting Maria admired the {{STATE}} with high ceilings."}}

JSON:"""


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_ws(x: str) -> str:
    return " ".join(str(x or "").replace("\u00a0", " ").split())


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
    rows: list[dict[str, Any]] = []
    with SELECTED_PAIRS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def entity_char_pos(source: str, ent: str) -> int:
    # Entity candidates come from exact source spans or common nouns. Use a tolerant
    # word-boundary search first and fall back to simple lower-case find.
    src = source.lower()
    e = norm_ws(ent).lower()
    if not e:
        return 10**9
    m = re.search(r"(?<![a-z0-9])" + re.escape(e) + r"(?![a-z0-9])", src)
    if m:
        return m.start()
    i = src.find(e)
    return i if i >= 0 else 10**9


def choose_entity_pair(source: str, ents: list[str], order_index: int, rng: random.Random) -> dict[str, Any] | None:
    capped = [norm_ws(e) for e in ents[:MAX_CANDIDATES] if norm_ws(e)]
    if len(capped) < 2:
        return None
    # Prefer salient nearby pairs from the first candidates, but rotate so the
    # same first-two positions are not always used.
    max_start = max(0, len(capped) - 2)
    start = rng.randint(0, max_start) if max_start else 0
    a_i, b_i = start, start + 1
    if entity_char_pos(source, capped[a_i]) <= entity_char_pos(source, capped[b_i]):
        earlier_i, later_i = a_i, b_i
    else:
        earlier_i, later_i = b_i, a_i
    if order_index % 2 == 0:
        target_i, distractor_i = earlier_i, later_i
        relation = "target_earlier"
    else:
        target_i, distractor_i = later_i, earlier_i
        relation = "target_later"
    return {
        "target_entity_assigned": capped[target_i],
        "distractor_entity_assigned": capped[distractor_i],
        "target_candidate_rank": target_i,
        "distractor_candidate_rank": distractor_i,
        "target_char_pos": entity_char_pos(source, capped[target_i]),
        "distractor_char_pos": entity_char_pos(source, capped[distractor_i]),
        "role_position_relation": relation,
        "candidate_pair_rank_span": [min(a_i, b_i), max(a_i, b_i)],
    }


def build_prompt(pair: dict[str, Any], ents: list[str], order_index: int, rng: random.Random) -> dict[str, Any] | None:
    source = norm_ws(pair["original"])
    role = choose_entity_pair(source, ents, order_index, rng)
    if role is None:
        return None
    target = role["target_entity_assigned"]
    distractor = role["distractor_entity_assigned"]
    prompt = PROMPT_TEMPLATE.format(
        sentence=source,
        entities=json.dumps([norm_ws(e) for e in ents[:MAX_CANDIDATES]], ensure_ascii=False),
        target_entity=json.dumps(target, ensure_ascii=False),
        distractor_entity=json.dumps(distractor, ensure_ascii=False),
    )
    return {
        "id": f"cb58_{pair['pair_id']}",
        "prompt": prompt,
        "pair_id": pair["pair_id"],
        "source_sentence": source,
        "qwen_rewrite": norm_ws(str(pair.get("rewrite", ""))),
        "candidate_entities": [norm_ws(e) for e in ents[:MAX_CANDIDATES]],
        **role,
        "source_words": len(source.split()),
        "source": pair.get("source_name") or pair.get("source", ""),
        "example_id": pair.get("source_example_id") or pair.get("example_id", -1),
        "source_sentence_idx": pair.get("source_sentence_idx"),
        "cohort": pair.get("cohort", ""),
        "order_index": order_index,
        "prompt_family": "strict_contrastive_binding_same_source_shared_update_frame",
    }


def main() -> None:
    rng = random.Random(RNG_SEED)
    mod = import_step043()
    pairs = load_pairs()
    amenable: list[tuple[dict[str, Any], list[str]]] = []
    ent_count: Counter[int] = Counter()
    for p in pairs:
        ents = mod.entity_candidates(p)
        ent_count[len(ents)] += 1
        if len(ents) >= 2:
            amenable.append((p, ents))
    rng.shuffle(amenable)
    prompts: list[dict[str, Any]] = []
    for i, (p, ents) in enumerate(amenable):
        row = build_prompt(p, ents, i, rng)
        if row is not None:
            prompts.append(row)
    pilot = prompts[:min(PILOT_N, len(prompts))]
    write_jsonl(PROMPTS_ALL, prompts)
    write_jsonl(PROMPTS_PILOT, pilot)
    reg = Counter(r.get("source", "") for r in prompts)
    cohort = Counter(r.get("cohort", "") for r in prompts)
    role_rel = Counter(r.get("role_position_relation", "") for r in prompts)
    pair_span = Counter(str(r.get("candidate_pair_rank_span", [])) for r in prompts)
    meta = {
        "status": "STRICT_CONTRASTIVE_BINDING_PROMPTS_READY",
        "created_utc": now_utc(),
        "selected_pairs": len(pairs),
        "entity_count_distribution_first10": {str(k): ent_count.get(k, 0) for k in range(10)},
        "amenable_ge2_entities": len(prompts),
        "pilot_n": len(pilot),
        "purpose": "strict same-source contrastive target-update vs distractor-update packets for crossed data/objective credit experiment",
        "strict_design": {
            "assigned_entities_not_qwen_chosen": True,
            "role_source_position_alternates": True,
            "role_position_distribution": dict(role_rel),
            "single_shared_update_sentence_frame": True,
            "update_variants_differ_only_by_entity_placeholder_substitution": True,
            "same_use_sentence_frame": True,
            "both_entities_require_source_states_in_validation": True,
            "success_readout_contract": "held-out same-source discrimination contrast and Entity relevant-update strata; T-U/copy are side faces",
            "substrate_contract": "all training cells should use the same adapter-stage base and the same held-out strict contrastive packet set; the 100M replacement arm is motivating prior, not a cell",
        },
        "outputs": {"all_prompts": str(PROMPTS_ALL), "pilot_prompts": str(PROMPTS_PILOT)},
        "sha256": {"all_prompts": sha256_file(PROMPTS_ALL), "pilot_prompts": sha256_file(PROMPTS_PILOT)},
        "register_distribution": dict(reg.most_common()),
        "cohort_distribution": dict(cohort.most_common()),
        "candidate_pair_rank_span_distribution": dict(pair_span.most_common(20)),
        "validation_plan": "Use validate_contrastive_binding_outputs_strict.py after Qwen generation; accepted objects materialize UPDATE and RETAIN rows with identical source, identical target use frame, and update sentence variants obtained from one shared update frame by substituting target vs distractor entity.",
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
