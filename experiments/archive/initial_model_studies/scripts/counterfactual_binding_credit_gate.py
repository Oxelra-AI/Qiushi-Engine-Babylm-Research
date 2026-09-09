#!/usr/bin/env python3
"""Mechanism gate for Counterfactual Binding Credit Assignment (CBCA).

CBCA operates only on the submission model's ordinary MLM logits. For each
example it creates two coherent histories with an identical masked-input token
multiset but different entity-state bindings. The two histories must reverse
their preference between the two possible state tokens.

This is a bounded, evaluation-free mechanism test. It does not read BabyLM
evaluation data and it does not claim an official-score improvement.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM


STUDY = _public_path('experiments/archive/initial_model_studies')
TOKENIZER_PATH = (
    _public_path('experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M')
)
OUT = _public_path('experiments/archive/initial_model_studies/data/counterfactual_binding_credit_gate.json')
NOTE = _public_path('research/notes/initial_model_studies/counterfactual_binding_credit_gate.md')

TRAIN_NAMES = "Alice Ben Clara David Emma Frank Grace Henry Irene Jack Karen Leo Maria Noah Olivia Peter Quinn Rose Simon Tina".split()
HELD_NAMES = "Aaron Bella Colin Diana Ethan Fiona Gavin Holly Isaac Julia Kevin Laura".split()
TRAIN_OBJECTS = "apple book coin doll flower glass hammer jacket key letter map notebook orange pencil ring spoon ticket watch ball candle".split()
HELD_OBJECTS = "bottle camera drum envelope fork glove ladder mirror pillow rope suitcase wallet".split()
LOCATION_CANDIDATES = "box room house school garden kitchen office store park street town city farm field forest river lake beach hotel church station shop library hospital village bedroom garage basement attic closet cabinet drawer basket shelf bag pocket porch hallway studio pantry counter warehouse theater airport museum".split()

TRAIN_EVENT_TEMPLATES = [
    "{name} placed the {obj} in the {loc}.",
    "{name} stored the {obj} in the {loc}.",
    "{name} left the {obj} in the {loc}.",
    "{name} carried the {obj} into the {loc}.",
]
HELD_EVENT_TEMPLATES = [
    "The {obj} was taken by {name} to the {loc}.",
    "For safekeeping, {name} brought the {obj} to the {loc}.",
]
TRAIN_UPDATE_TEMPLATES = [
    "Later, {name} moved the {obj} to the {loc}.",
    "After that, {name} took the {obj} into the {loc}.",
    "Then {name} transferred the {obj} to the {loc}.",
]
HELD_UPDATE_TEMPLATES = [
    "Before evening, the {obj} was relocated by {name} to the {loc}.",
    "Eventually, {name} brought the {obj} over to the {loc}.",
]
TRAIN_QUERY_TEMPLATES = [
    "When everything was finished, the {obj} was in the {answer}.",
    "At the end, the {obj} could be found in the {answer}.",
    "After all these events, the {obj} remained in the {answer}.",
]
HELD_QUERY_TEMPLATES = [
    "Once the activity stopped, the final place for the {obj} was the {answer}.",
    "Looking back afterward, the {obj} ended up in the {answer}.",
]


@dataclass
class EncodedPair:
    input_a: list[int]
    input_b: list[int]
    position_a: int
    position_b: int
    label_a: int
    label_b: int
    kind: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_environment() -> None:
    hf_home = _public_path('experiments/archive/initial_model_studies/training/hf_home')
    os.environ.setdefault("HF_HOME", str(hf_home.resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((hf_home / "hub").resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((hf_home / "transformers").resolve()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.set_num_threads(4)


def one_token_locations(tokenizer) -> list[str]:
    accepted = []
    for word in LOCATION_CANDIDATES:
        text = f"The object was in the {word}."
        start = text.rfind(word)
        encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        positions = [
            index
            for index, (left, right) in enumerate(encoded["offset_mapping"])
            if right > start and left < start + len(word)
        ]
        if len(positions) == 1:
            accepted.append(word)
    if len(accepted) < 24:
        raise RuntimeError(f"only {len(accepted)} one-token locations")
    return accepted


def render_events(
    names: list[str],
    objects: list[str],
    locations: list[str],
    order: list[int],
    templates: list[str],
) -> str:
    return " ".join(
        templates[position % len(templates)].format(
            name=names[index], obj=objects[index], loc=locations[index]
        )
        for position, index in enumerate(order)
    )


def swap(values: list[str], left: int, right: int) -> list[str]:
    result = list(values)
    result[left], result[right] = result[right], result[left]
    return result


def encode_text(tokenizer, text: str, answer: str, max_length: int = 128):
    start = text.rfind(answer)
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < start + len(answer)
    ]
    if len(positions) != 1:
        return None
    ids = list(encoded["input_ids"])
    position = positions[0]
    label = ids[position]
    ids[position] = tokenizer.mask_token_id
    return ids, position, label


def build_pairs(
    tokenizer,
    number: int,
    seed: int,
    names_pool: list[str],
    objects_pool: list[str],
    locations_pool: list[str],
    event_templates: list[str],
    update_templates: list[str],
    query_templates: list[str],
    force_update: bool | None = None,
) -> list[EncodedPair]:
    rng = random.Random(seed)
    pairs = []
    seen = set()
    attempts = 0
    while len(pairs) < number and attempts < number * 30:
        attempts += 1
        names = rng.sample(names_pool, 3)
        objects = rng.sample(objects_pool, 3)
        initial = rng.sample(locations_pool, 3)
        remaining = [item for item in locations_pool if item not in initial]
        final = rng.sample(remaining, 3)
        target = rng.randrange(3)
        partner = rng.choice([index for index in range(3) if index != target])
        order = [0, 1, 2]
        rng.shuffle(order)
        is_update = force_update if force_update is not None else rng.random() < 0.5
        key = (tuple(names), tuple(objects), tuple(initial), tuple(final), target, partner, tuple(order), is_update)
        if key in seen:
            continue
        seen.add(key)

        answer_locations = final if is_update else initial
        swapped_locations = swap(answer_locations, target, partner)
        base = render_events(names, objects, initial, order, event_templates)
        if is_update:
            history_a = base + " " + render_events(names, objects, answer_locations, order, update_templates)
            history_b = base + " " + render_events(names, objects, swapped_locations, order, update_templates)
            kind = "latest_update"
        else:
            history_a = render_events(names, objects, answer_locations, order, event_templates)
            history_b = render_events(names, objects, swapped_locations, order, event_templates)
            kind = "initial_binding"

        answer_a = answer_locations[target]
        answer_b = swapped_locations[target]
        query_template = rng.choice(query_templates)
        suffix_a = " Everyone waited quietly, and nothing else was moved. " + query_template.format(
            obj=objects[target], answer=answer_a
        )
        suffix_b = " Everyone waited quietly, and nothing else was moved. " + query_template.format(
            obj=objects[target], answer=answer_b
        )
        encoded_a = encode_text(tokenizer, history_a + suffix_a, answer_a)
        encoded_b = encode_text(tokenizer, history_b + suffix_b, answer_b)
        if encoded_a is None or encoded_b is None:
            continue
        ids_a, pos_a, label_a = encoded_a
        ids_b, pos_b, label_b = encoded_b
        if label_a == label_b or len(ids_a) != len(ids_b):
            continue
        if sorted(ids_a) != sorted(ids_b):
            continue
        pairs.append(
            EncodedPair(
                input_a=ids_a,
                input_b=ids_b,
                position_a=pos_a,
                position_b=pos_b,
                label_a=label_a,
                label_b=label_b,
                kind=kind,
            )
        )
    if len(pairs) != number:
        raise RuntimeError(f"constructed only {len(pairs)}/{number} pairs")
    return pairs


def pad_inputs(sequences: list[list[int]], pad_id: int, device: torch.device):
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), length), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention[row, : len(sequence)] = 1
    return ids.to(device), attention.to(device)


def batch_logits(model, pairs: list[EncodedPair], pad_id: int, device: torch.device, duplicate_a: bool = False):
    first = [pair.input_a for pair in pairs]
    second = [pair.input_a if duplicate_a else pair.input_b for pair in pairs]
    ids, attention = pad_inputs(first + second, pad_id, device)
    logits = model(input_ids=ids, attention_mask=attention).logits
    size = len(pairs)
    rows = torch.arange(size, device=device)
    positions_a = torch.tensor([pair.position_a for pair in pairs], device=device)
    positions_b = torch.tensor(
        [pair.position_a if duplicate_a else pair.position_b for pair in pairs], device=device
    )
    return logits[rows, positions_a], logits[rows + size, positions_b]


@torch.no_grad()
def evaluate(model, pairs: list[EncodedPair], pad_id: int, device: torch.device, batch_size: int = 64):
    model.eval()
    margins = []
    condition_correct = []
    pair_correct = []
    nll = []
    kinds = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        logits_a, logits_b = batch_logits(model, chunk, pad_id, device)
        labels_a = torch.tensor([pair.label_a for pair in chunk], device=device)
        labels_b = torch.tensor([pair.label_b for pair in chunk], device=device)
        rows = torch.arange(len(chunk), device=device)
        a_good = logits_a[rows, labels_a] - logits_a[rows, labels_b]
        b_good = logits_b[rows, labels_b] - logits_b[rows, labels_a]
        margins.extend((0.5 * (a_good + b_good)).cpu().tolist())
        a_ok = a_good > 0
        b_ok = b_good > 0
        condition_correct.extend(torch.stack((a_ok, b_ok), dim=1).cpu().reshape(-1).tolist())
        pair_correct.extend((a_ok & b_ok).cpu().tolist())
        nll.extend(
            (0.5 * (F.cross_entropy(logits_a, labels_a, reduction="none") + F.cross_entropy(logits_b, labels_b, reduction="none"))).cpu().tolist()
        )
        kinds.extend(pair.kind for pair in chunk)
    result = {
        "n": len(pairs),
        "condition_accuracy": float(np.mean(condition_correct)),
        "pair_accuracy": float(np.mean(pair_correct)),
        "cross_margin_mean": float(np.mean(margins)),
        "cross_margin_median": float(np.median(margins)),
        "correct_nll": float(np.mean(nll)),
    }
    for kind in sorted(set(kinds)):
        indices = [index for index, value in enumerate(kinds) if value == kind]
        result[f"{kind}_pair_accuracy"] = float(np.mean([pair_correct[index] for index in indices]))
        result[f"{kind}_margin_mean"] = float(np.mean([margins[index] for index in indices]))
    return result


def train_arm(
    arm: str,
    seed: int,
    config: DebertaV2Config,
    train_pairs: list[EncodedPair],
    pad_id: int,
    device: torch.device,
    steps: int,
    batch_size: int,
):
    torch.manual_seed(seed)
    model = DebertaV2ForMaskedLM(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    rng = random.Random(seed * 1009 + 305)
    losses = []
    cross_margins = []
    for _ in range(steps):
        batch = [train_pairs[rng.randrange(len(train_pairs))] for _ in range(batch_size)]
        duplicate_a = arm == "single_world_ce"
        logits_a, logits_b = batch_logits(model, batch, pad_id, device, duplicate_a=duplicate_a)
        labels_a = torch.tensor([pair.label_a for pair in batch], device=device)
        labels_b = torch.tensor(
            [pair.label_a if duplicate_a else pair.label_b for pair in batch], device=device
        )
        ce = 0.5 * (
            F.cross_entropy(logits_a, labels_a) + F.cross_entropy(logits_b, labels_b)
        )
        loss = ce
        if arm in {"random_cross", "cbca"}:
            true_labels_b = torch.tensor([pair.label_b for pair in batch], device=device)
            rows = torch.arange(len(batch), device=device)
            margin = 0.5 * (
                logits_a[rows, labels_a]
                - logits_a[rows, true_labels_b]
                + logits_b[rows, true_labels_b]
                - logits_b[rows, labels_a]
            )
            if arm == "random_cross":
                signs = torch.tensor(
                    [1.0 if rng.random() < 0.5 else -1.0 for _ in batch], device=device
                )
                margin = margin * signs
            cross_loss = F.relu(1.0 - margin).mean()
            loss = ce + 0.5 * cross_loss
            cross_margins.append(float(margin.detach().mean().cpu()))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model.eval(), {
        "loss_first20": float(np.mean(losses[:20])),
        "loss_last20": float(np.mean(losses[-20:])),
        "training_cross_margin_last20": (
            float(np.mean(cross_margins[-20:])) if cross_margins else None
        ),
    }


def main() -> None:
    setup_environment()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("tokenizer has no mask token")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has no pad token")
    locations = one_token_locations(tokenizer)
    # All output-state tokens receive lexical training. The held-out tests
    # isolate new bindings, entities, objects, and templates, not unseen output
    # embeddings that a random-init model could not possibly predict.
    train_locations = locations

    train_pairs = build_pairs(
        tokenizer,
        4000,
        30501,
        TRAIN_NAMES,
        TRAIN_OBJECTS,
        train_locations,
        TRAIN_EVENT_TEMPLATES,
        TRAIN_UPDATE_TEMPLATES,
        TRAIN_QUERY_TEMPLATES,
    )
    heldout = {
        "new_entities": build_pairs(
            tokenizer,
            500,
            30502,
            HELD_NAMES,
            HELD_OBJECTS,
            train_locations,
            TRAIN_EVENT_TEMPLATES,
            TRAIN_UPDATE_TEMPLATES,
            TRAIN_QUERY_TEMPLATES,
        ),
        "new_templates": build_pairs(
            tokenizer,
            500,
            30503,
            TRAIN_NAMES,
            TRAIN_OBJECTS,
            train_locations,
            HELD_EVENT_TEMPLATES,
            HELD_UPDATE_TEMPLATES,
            HELD_QUERY_TEMPLATES,
        ),
        "new_all": build_pairs(
            tokenizer,
            600,
            30504,
            HELD_NAMES,
            HELD_OBJECTS,
            train_locations,
            HELD_EVENT_TEMPLATES,
            HELD_UPDATE_TEMPLATES,
            HELD_QUERY_TEMPLATES,
        ),
        "new_all_updates": build_pairs(
            tokenizer,
            400,
            30505,
            HELD_NAMES,
            HELD_OBJECTS,
            train_locations,
            HELD_EVENT_TEMPLATES,
            HELD_UPDATE_TEMPLATES,
            HELD_QUERY_TEMPLATES,
            force_update=True,
        ),
    }
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
    arms = ["single_world_ce", "paired_ce", "random_cross", "cbca"]
    seeds = [42, 43]
    results = {}
    for arm in arms:
        results[arm] = {}
        for seed in seeds:
            print(json.dumps({"event": "train_start", "arm": arm, "seed": seed}), flush=True)
            model, training = train_arm(
                arm, seed, config, train_pairs, pad_id, device, steps=180, batch_size=24
            )
            evaluations = {
                name: evaluate(model, pairs, pad_id, device) for name, pairs in heldout.items()
            }
            results[arm][str(seed)] = {"training": training, "evaluation": evaluations}
            print(
                json.dumps(
                    {
                        "event": "train_complete",
                        "arm": arm,
                        "seed": seed,
                        "new_all": evaluations["new_all"],
                    }
                ),
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def seed_values(arm: str, metric: str) -> list[float]:
        return [
            results[arm][str(seed)]["evaluation"]["new_all"][metric] for seed in seeds
        ]

    pair_accuracy = {arm: seed_values(arm, "pair_accuracy") for arm in arms}
    margins = {arm: seed_values(arm, "cross_margin_mean") for arm in arms}
    cbca_delta = [
        pair_accuracy["cbca"][index]
        - max(pair_accuracy["paired_ce"][index], pair_accuracy["random_cross"][index])
        for index in range(len(seeds))
    ]
    paired_delta = [
        pair_accuracy["paired_ce"][index] - pair_accuracy["single_world_ce"][index]
        for index in range(len(seeds))
    ]
    if min(cbca_delta) >= 0.08 and min(pair_accuracy["cbca"]) >= 0.65:
        decision = "PROMOTE_CBCA_TO_NATURAL_DATA_PILOT"
        next_action = "Materialize diverse natural event pairs and integrate CBCA into the full DeBERTa trainer with a random-cross control."
    elif min(paired_delta) >= 0.08 and min(pair_accuracy["paired_ce"]) >= 0.65:
        decision = "PROMOTE_SYMMETRIC_COUNTERFACTUAL_EXPERIENCE_WITHOUT_EXTRA_MARGIN"
        next_action = "Use symmetric coherent counterfactual target supervision; omit the unnecessary cross-margin term."
    else:
        decision = "REJECT_CURRENT_COUNTERFACTUAL_BINDING_MECHANISM"
        next_action = "Do not scale this objective; return to the causal gap and seek a non-template mechanism."

    payload = {
        "status": "COUNTERFACTUAL_BINDING_CREDIT_GATE_COMPLETE",
        "design": {
            "principle": "allocate ordinary MLM credit to predictions that must reverse under a minimal binding intervention",
            "input_invariant": "paired masked inputs have identical token multisets",
            "model_interface": "standard DebertaV2ForMaskedLM logits; no auxiliary head or inference change",
            "arms": arms,
            "seeds": seeds,
            "steps": 180,
            "batch_pairs": 24,
            "train_pairs": len(train_pairs),
            "heldout_pairs": {name: len(pairs) for name, pairs in heldout.items()},
            "state_vocabulary": train_locations,
            "device": str(device),
            "no_babylm_eval_data": True,
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/counterfactual_binding_credit_gate.py')),
            "script_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/counterfactual_binding_credit_gate.py')),
            "tokenizer": str(_public_path('experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M')),
            "tokenizer_json_sha256": sha256_file(_public_path('experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M/tokenizer.json')),
        },
        "results": results,
        "primary": {
            "new_all_pair_accuracy": pair_accuracy,
            "new_all_cross_margin": margins,
            "cbca_minus_best_active_control_pair_accuracy": cbca_delta,
            "paired_minus_single_pair_accuracy": paired_delta,
        },
        "decision": decision,
        "next_action": next_action,
        "elapsed_sec": time.time() - started,
    }
    _public_path('experiments/archive/initial_model_studies/data').mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)
    lines = [
        "# research counterfactual binding credit gate",
        "",
        f"Decision: **{decision}**",
        "",
        "Primary split is disjoint in names, objects, event templates, update templates, and query templates; state tokens remain lexically trained while their bindings are new.",
        "",
        "| arm | seed 42 pair acc | seed 43 pair acc | mean cross margin |",
        "|---|---:|---:|---:|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm} | {pair_accuracy[arm][0]:.3f} | {pair_accuracy[arm][1]:.3f} | {np.mean(margins[arm]):+.3f} |"
        )
    lines.extend(
        [
            "",
            f"CBCA minus best active control by seed: {cbca_delta}",
            "",
            f"Paired CE minus single-world CE by seed: {paired_delta}",
            "",
            f"Next action: {next_action}",
            "",
            f"Evidence JSON: `{OUT}`",
        ]
    )
    _public_path('research/notes/initial_model_studies').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": decision,
                "primary": payload["primary"],
                "out": str(OUT),
                "elapsed_sec": payload["elapsed_sec"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
