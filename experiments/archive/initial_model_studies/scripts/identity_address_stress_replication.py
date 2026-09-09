#!/usr/bin/env python3
"""research: independent stress replication of identity-addressed state retrieval.

research unexpectedly showed that both the exact continuation and a displaced
window work when their address is derived from the correct repeated entity,
while a wrong-entity address fails.  This script does not reinterpret the
research preregistered decision.  It tests the newly generated hypothesis on
fresh episode families with variable entity counts, multiple overwrites,
longer distractors, and unseen surface templates.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, DebertaV2Config


STUDY = _public_path('experiments/archive/initial_model_studies')
SOURCE = _public_path('experiments/archive/initial_model_studies/scripts/induction_seeded_state_subspace_gate.py')
SHARD_DIR = _public_path('experiments/archive/initial_model_studies/data/identity_address_stress_replication')
OUT = _public_path('experiments/archive/initial_model_studies/data/identity_address_stress_replication.json')
NOTE = _public_path('research/notes/initial_model_studies/identity_address_stress_replication.md')

spec = importlib.util.spec_from_file_location("for_step312", SOURCE)
state_subspace_probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = state_subspace_probe
assert spec.loader is not None
spec.loader.exec_module(state_subspace_probe)
induction_probe = state_subspace_probe.induction_probe
binding_task = state_subspace_probe.binding_task


NAMES = "Abel Bianca Cyrus Delia Edgar Flora Gordon Helena Ingram Jasmine Kelvin Lucille Malcolm Nina Oscar Paula Roland Sylvia Trevor Wendy Xavier Yvonne".split()
OBJECTS = "anchor brush compass diary feather goblet helmet lantern medal needle parcel quilt racket scarf teapot umbrella vase whistle yarn zipper badge kettle".split()

INITIAL_TEMPLATES = [
    "Near midday, {name} deposited the {obj} inside the {loc}.",
    "At the outset, {name} carried the {obj} over to the {loc}.",
    "For the first arrangement, the {obj} went with {name} into the {loc}.",
]
UPDATE_TEMPLATES = [
    "During change {round}, {name} redirected the {obj} toward the {loc}.",
    "On revision {round}, the {obj} was taken by {name} into the {loc}.",
    "For adjustment {round}, {name} shifted the {obj} across to the {loc}.",
]
QUERY_TEMPLATES = [
    "Tracing only the most recent change, the destination of the {obj} is the {answer}.",
    "Ignoring every older placement, the current location of the {obj} is the {answer}.",
    "After resolving all revisions, the {obj} now belongs in the {answer}.",
]
DISTRACTORS = [
    "Meanwhile the observers compared their notes and waited for the next instruction.",
    "A separate discussion continued quietly without changing any stored object.",
    "Several witnesses checked the schedule, but nobody altered the arrangement.",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_round(
    names: list[str],
    objects: list[str],
    locations: list[str],
    order: list[int],
    templates: list[str],
    round_index: int,
) -> str:
    return " ".join(
        templates[(position + round_index) % len(templates)].format(
            name=names[index],
            obj=objects[index],
            loc=locations[index],
            round=round_index,
        )
        for position, index in enumerate(order)
    )


def build_stress_pairs(
    tokenizer,
    locations: list[str],
    number: int,
    seed: int,
    entity_range: tuple[int, int],
    update_range: tuple[int, int],
    distractor_count: int,
) -> list:
    rng = random.Random(seed)
    pairs = []
    attempts = 0
    while len(pairs) < number and attempts < number * 80:
        attempts += 1
        entity_count = rng.randint(*entity_range)
        update_count = rng.randint(*update_range)
        names = rng.sample(NAMES, entity_count)
        objects = rng.sample(OBJECTS, entity_count)
        initial = rng.sample(locations, entity_count)
        used = set(initial)
        rounds = []
        for _ in range(update_count):
            available = [location for location in locations if location not in used]
            if len(available) < entity_count:
                available = list(locations)
            current = rng.sample(available, entity_count)
            used.update(current)
            rounds.append(current)
        target = rng.randrange(entity_count)
        partner = rng.choice([index for index in range(entity_count) if index != target])
        orders = []
        for _ in range(update_count + 1):
            order = list(range(entity_count))
            rng.shuffle(order)
            orders.append(order)

        final_a = list(rounds[-1])
        final_b = binding_task.swap(final_a, target, partner)
        rounds_a = [list(values) for values in rounds]
        rounds_b = [list(values) for values in rounds]
        rounds_a[-1] = final_a
        rounds_b[-1] = final_b

        history_a = render_round(
            names, objects, initial, orders[0], INITIAL_TEMPLATES, 0
        )
        history_b = history_a
        for round_index in range(update_count):
            history_a += " " + render_round(
                names,
                objects,
                rounds_a[round_index],
                orders[round_index + 1],
                UPDATE_TEMPLATES,
                round_index + 1,
            )
            history_b += " " + render_round(
                names,
                objects,
                rounds_b[round_index],
                orders[round_index + 1],
                UPDATE_TEMPLATES,
                round_index + 1,
            )
        distractors = " ".join(
            DISTRACTORS[(attempts + index) % len(DISTRACTORS)]
            for index in range(distractor_count)
        )
        query = rng.choice(QUERY_TEMPLATES)
        answer_a = final_a[target]
        answer_b = final_b[target]
        text_a = (
            history_a
            + (" " + distractors if distractors else "")
            + " "
            + query.format(obj=objects[target], answer=answer_a)
        )
        text_b = (
            history_b
            + (" " + distractors if distractors else "")
            + " "
            + query.format(obj=objects[target], answer=answer_b)
        )
        encoded_a = binding_task.encode_text(tokenizer, text_a, answer_a, max_length=160)
        encoded_b = binding_task.encode_text(tokenizer, text_b, answer_b, max_length=160)
        if encoded_a is None or encoded_b is None:
            continue
        ids_a, position_a, label_a = encoded_a
        ids_b, position_b, label_b = encoded_b
        if label_a == label_b or len(ids_a) != len(ids_b):
            continue
        if sorted(ids_a) != sorted(ids_b):
            continue
        pairs.append(
            binding_task.EncodedPair(
                input_a=ids_a,
                input_b=ids_b,
                position_a=position_a,
                position_b=position_b,
                label_a=label_a,
                label_b=label_b,
                kind=f"entities_{entity_count}_updates_{update_count}_distractors_{distractor_count}",
            )
        )
    if len(pairs) != number:
        raise RuntimeError(f"constructed only {len(pairs)}/{number} stress pairs")
    return pairs


def build_stress_splits(tokenizer, locations: list[str]) -> dict:
    return {
        "variable_entities": build_stress_pairs(
            tokenizer, locations, 600, 31201, (2, 5), (1, 1), 0
        ),
        "multiple_overwrites": build_stress_pairs(
            tokenizer, locations, 600, 31202, (3, 4), (2, 3), 0
        ),
        "long_distractors": build_stress_pairs(
            tokenizer, locations, 600, 31203, (2, 4), (1, 2), 2
        ),
    }


def run_shard(arm: str, seed: int) -> dict:
    binding_task.setup_environment()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(binding_task.TOKENIZER_PATH, use_fast=True)
    locations, train_pairs, _ = induction_probe.build_data(tokenizer)
    stress = build_stress_splits(tokenizer, locations)
    allowed = state_subspace_probe.eligible_token_ids(tokenizer)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=160,
        position_buckets=64,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model, training = state_subspace_probe.train_arm(
        arm,
        seed,
        config,
        train_pairs,
        tokenizer.pad_token_id,
        allowed,
        tokenizer.mask_token_id,
        device,
    )
    evaluations = {}
    interventions = {}
    for split_name, pairs in stress.items():
        evaluations[split_name] = induction_probe.evaluate_detailed(
            model, pairs, tokenizer.pad_token_id, device
        )[0]
        interventions[split_name] = {}
        for mode in ["off", "wrong_position", "wrong_identity", "identity_continuation"]:
            interventions[split_name][mode] = induction_probe.evaluate_detailed(
                model,
                pairs,
                tokenizer.pad_token_id,
                device,
                edge_mode=mode,
            )[0]
    payload = {
        "status": "SHARD_COMPLETE",
        "arm": arm,
        "seed": seed,
        "training": training,
        "stress_evaluation": evaluations,
        "same_model_interventions": interventions,
        "stress_counts": {name: len(pairs) for name, pairs in stress.items()},
        "device": str(device),
        "elapsed_sec": time.time() - started,
    }
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    path = SHARD_DIR / f"{arm}_seed{seed}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "arm": arm,
                "seed": seed,
                "stress_pair_accuracy": {
                    name: value["pair_accuracy"] for name, value in evaluations.items()
                },
                "elapsed_sec": payload["elapsed_sec"],
            }
        ),
        flush=True,
    )
    return payload


def aggregate() -> dict:
    arms = ["wrong_position", "identity_continuation"]
    seeds = [42, 43]
    shards = {
        arm: {
            str(seed): json.loads(
                (SHARD_DIR / f"{arm}_seed{seed}.json").read_text(encoding="utf-8")
            )
            for seed in seeds
        }
        for arm in arms
    }
    splits = ["variable_entities", "multiple_overwrites", "long_distractors"]
    exact = {
        split: [
            shards["identity_continuation"][str(seed)]["stress_evaluation"][split][
                "pair_accuracy"
            ]
            for seed in seeds
        ]
        for split in splits
    }
    displaced = {
        split: [
            shards["wrong_position"][str(seed)]["stress_evaluation"][split][
                "pair_accuracy"
            ]
            for seed in seeds
        ]
        for split in splits
    }
    causal = {}
    for split in splits:
        causal[split] = {}
        for seed in seeds:
            modes = shards["identity_continuation"][str(seed)][
                "same_model_interventions"
            ][split]
            true_value = modes["identity_continuation"]["pair_accuracy"]
            causal[split][str(seed)] = {
                "true": true_value,
                "minus_off": true_value - modes["off"]["pair_accuracy"],
                "minus_wrong_identity": true_value
                - modes["wrong_identity"]["pair_accuracy"],
                "minus_wrong_position": true_value
                - modes["wrong_position"]["pair_accuracy"],
            }
    exact_pass = all(min(exact[split]) >= 0.75 for split in splits)
    displaced_pass = all(min(displaced[split]) >= 0.60 for split in splits)
    causal_pass = all(
        causal[split][str(seed)]["minus_off"] >= 0.50
        and causal[split][str(seed)]["minus_wrong_identity"] >= 0.50
        for split in splits
        for seed in seeds
    )
    if exact_pass and displaced_pass and causal_pass:
        decision = "IDENTITY_ADDRESS_STATE_RETRIEVAL_REPLICATED_ON_STRESS_TESTS"
        next_action = (
            "Promote the mechanism family to a natural-text gate, not to an SOTA claim: "
            "implement word-span addressing, measure activation coverage and MLM retention, "
            "and compare true-address, shuffled-address, and standard WWM from matched initialization."
        )
    else:
        decision = "REJECT_IDENTITY_ADDRESS_GENERALIZATION_AFTER_STRESS_TEST"
        next_action = (
            "Do not scale the mechanism. The research result does not survive independent "
            "composition, overwrite, and distractor stress."
        )
    payload = {
        "status": "IDENTITY_ADDRESS_STRESS_REPLICATION_COMPLETE",
        "hypothesis_generated_after_step311": (
            "correct identity addressing, rather than exact lexical continuation copying, "
            "creates a reusable state bottleneck"
        ),
        "pre_registered_gate": {
            "exact_each_seed_each_split": ">= 0.75",
            "displaced_correct_address_each_seed_each_split": ">= 0.60",
            "same_exact_model_minus_off_each_seed_each_split": ">= 0.50",
            "same_exact_model_minus_wrong_identity_each_seed_each_split": ">= 0.50",
        },
        "exact_pair_accuracy": exact,
        "displaced_correct_address_pair_accuracy": displaced,
        "same_model_causal_effects": causal,
        "gate_components": {
            "exact_pass": exact_pass,
            "displaced_pass": displaced_pass,
            "causal_pass": causal_pass,
        },
        "decision": decision,
        "next_action": next_action,
        "shards": shards,
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/identity_address_stress_replication.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/identity_address_stress_replication.py')),
            "script_sha256": sha256_file(SOURCE),
            "no_babylm_evaluation_data": True,
            "shard_sha256": {
                f"{arm}_seed{seed}": sha256_file(
                    SHARD_DIR / f"{arm}_seed{seed}.json"
                )
                for arm in arms
                for seed in seeds
            },
        },
    }
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)
    lines = [
        "# research identity-address stress replication",
        "",
        f"Decision: **{decision}**",
        "",
        "Fresh tests vary entity count, overwrite depth, distractors, names, objects, and templates.",
        "No BabyLM evaluation data is used.",
        "",
        "| split | exact seed42/43 | displaced correct-address seed42/43 |",
        "|---|---:|---:|",
    ]
    for split in splits:
        lines.append(
            f"| {split} | {exact[split][0]:.3f} / {exact[split][1]:.3f} | "
            f"{displaced[split][0]:.3f} / {displaced[split][1]:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Gate components: {payload['gate_components']}",
            "",
            f"Next action: {next_action}",
            "",
            f"Evidence JSON: `{OUT}`",
        ]
    )
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decision": decision, "exact": exact, "displaced": displaced}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["wrong_position", "identity_continuation"])
    parser.add_argument("--seed", type=int, choices=[42, 43])
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
        return
    if args.arm is None or args.seed is None:
        parser.error("--arm and --seed are required unless --aggregate is used")
    run_shard(args.arm, args.seed)


if __name__ == "__main__":
    main()
