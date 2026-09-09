#!/usr/bin/env python3
"""Final H2 positive control with bag-of-words-matched event histories.

The probe is synthetic and does not read BabyLM evaluation data. Each condition
contains exactly the same names, objects, locations, and query. The causal arm
swaps the queried object's location with another object's location; the null arm
swaps two unrelated objects' locations. A model that binds event history to the
queried entity should assign the original answer a larger loss increase in the
causal arm than in the null arm.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


STUDY = _public_path('experiments/archive/initial_model_studies')
PATH = _public_path('experiments/archive/initial_model_studies/scripts/leader_causal_use_gap.py')
PROTECTED_DEFAULT = (
    _public_path('experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M')
)
LEADER_DEFAULT = _public_path('experiments/archive/initial_model_studies/data/leader_package_revision_124/model_side/local')
OUT_DEFAULT = _public_path('experiments/archive/initial_model_studies/data/revision_302b_event_binding_positive_control.json')
NOTE_DEFAULT = _public_path('research/notes/initial_model_studies/revision_302b_event_binding_positive_control.md')

NAMES = [
    "Alice", "Ben", "Clara", "David", "Emma", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Karen", "Leo", "Maria", "Noah", "Olivia", "Peter",
]
OBJECTS = [
    "apple", "book", "coin", "doll", "flower", "glass", "hammer", "jacket",
    "key", "letter", "map", "notebook", "orange", "pencil", "ring", "spoon",
]
LOCATIONS = [
    "basket", "box", "cabinet", "drawer", "garage", "garden", "kitchen",
    "office", "pantry", "room", "shelf", "studio", "table", "closet",
    "hallway", "bedroom",
]


@dataclass
class Condition:
    text: str
    content_spans: list[tuple[int, int]]


@dataclass
class Case:
    target_word: str
    source: str
    orig: Condition
    entity_cf: Condition
    unrelated_cf: Condition


def load_step302_module():
    spec = importlib.util.spec_from_file_location("probe", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def condition(text: str, target: str) -> Condition:
    start = text.rfind(target)
    if start < 0:
        raise RuntimeError(f"target {target!r} missing from query")
    return Condition(text=text, content_spans=[(start, start + len(target))])


def render_initial(names: list[str], objects: list[str], locations: list[str], order: list[int]) -> str:
    templates = [
        "{name} placed the {obj} in the {loc}.",
        "{name} stored the {obj} in the {loc}.",
        "{name} left the {obj} in the {loc}.",
    ]
    return " ".join(
        templates[position % len(templates)].format(
            name=names[index], obj=objects[index], loc=locations[index]
        )
        for position, index in enumerate(order)
    )


def render_updates(names: list[str], objects: list[str], locations: list[str], order: list[int]) -> str:
    templates = [
        "Later, {name} moved the {obj} to the {loc}.",
        "After that, {name} carried the {obj} to the {loc}.",
        "Then {name} took the {obj} to the {loc}.",
    ]
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


def build_cases(number: int, seed: int) -> list[Case]:
    rng = random.Random(seed)
    cases = []
    for case_index in range(number):
        names = rng.sample(NAMES, 3)
        objects = rng.sample(OBJECTS, 3)
        initial_locations = rng.sample(LOCATIONS, 3)
        remaining = [item for item in LOCATIONS if item not in initial_locations]
        final_locations = rng.sample(remaining, 3)
        target_index = rng.randrange(3)
        other = [index for index in range(3) if index != target_index]
        causal_partner = rng.choice(other)
        order = [0, 1, 2]
        rng.shuffle(order)

        latest_update = case_index % 2 == 1
        answer_locations = final_locations if latest_update else initial_locations
        causal_locations = swap(answer_locations, target_index, causal_partner)
        null_locations = swap(answer_locations, other[0], other[1])

        if latest_update:
            prefix = render_initial(names, objects, initial_locations, order)
            orig_history = prefix + " " + render_updates(names, objects, answer_locations, order)
            causal_history = prefix + " " + render_updates(names, objects, causal_locations, order)
            null_history = prefix + " " + render_updates(names, objects, null_locations, order)
            source = "synthetic_latest_update_binding"
        else:
            orig_history = render_initial(names, objects, answer_locations, order)
            causal_history = render_initial(names, objects, causal_locations, order)
            null_history = render_initial(names, objects, null_locations, order)
            source = "synthetic_initial_binding"

        target = answer_locations[target_index]
        suffix = (
            " Everyone then waited quietly, and no object was moved again. "
            f"When the work was finished, the {objects[target_index]} was in the {target}."
        )
        cases.append(
            Case(
                target_word=target,
                source=source,
                orig=condition(orig_history + suffix, target),
                entity_cf=condition(causal_history + suffix, target),
                unrelated_cf=condition(null_history + suffix, target),
            )
        )
    return cases


def summarize_by_source(rows: dict[int, dict], cases: list[Case]) -> dict:
    result = {}
    for source in sorted({case.source for case in cases}):
        values = np.asarray(
            [
                row["normalized_extra_entity_effect"]
                for index, row in rows.items()
                if cases[index].source == source
            ],
            dtype=np.float64,
        )
        result[source] = {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "fraction_positive": float((values > 0).mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=420)
    parser.add_argument("--case-seed", type=int, default=3021)
    parser.add_argument("--bootstrap-seed", type=int, default=3022)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--protected-model", default=str(PROTECTED_DEFAULT))
    parser.add_argument("--leader-model", default=str(LEADER_DEFAULT))
    parser.add_argument("--out-json", default=str(OUT_DEFAULT))
    parser.add_argument("--out-note", default=str(NOTE_DEFAULT))
    args = parser.parse_args()

    started = time.time()
    research = load_step302_module()
    research.setup_environment()
    device = research.torch.device(args.device)
    if device.type == "cuda" and not research.torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA is unavailable; use --allow-cpu for the bounded fallback")
        device = research.torch.device("cpu")

    cases = build_cases(args.cases, args.case_seed)
    paths = {
        "protected_wwm42_100M": Path(args.protected_model).resolve(),
        "public_leader": Path(args.leader_model).resolve(),
    }
    scored = {
        name: research.score_model(
            name, path, cases, args.max_length, args.batch_size, device
        )
        for name, path in paths.items()
    }
    maps = {
        name: {row["case_index"]: row for row in result["rows"]}
        for name, result in scored.items()
    }
    common = sorted(set(maps["protected_wwm42_100M"]) & set(maps["public_leader"]))
    if len(common) < int(0.9 * len(cases)):
        raise RuntimeError(f"only {len(common)}/{len(cases)} cases are common across tokenizers")

    protected = np.asarray(
        [maps["protected_wwm42_100M"][index]["normalized_extra_entity_effect"] for index in common]
    )
    leader = np.asarray(
        [maps["public_leader"][index]["normalized_extra_entity_effect"] for index in common]
    )
    gap = leader - protected
    rng = np.random.default_rng(args.bootstrap_seed)
    statistics = {
        "protected": research.bootstrap_mean(protected, rng, args.bootstrap_repetitions),
        "leader": research.bootstrap_mean(leader, rng, args.bootstrap_repetitions),
        "leader_minus_protected": research.bootstrap_mean(gap, rng, args.bootstrap_repetitions),
        "cross_model_spearman_rho": float(spearmanr(protected, leader).statistic),
    }
    leader_ci = statistics["leader"]["bootstrap_ci95"]
    gap_ci = statistics["leader_minus_protected"]["bootstrap_ci95"]
    linked = (
        leader_ci[0] > 0
        and gap_ci[0] > 0
        and statistics["leader"]["fraction_positive"] >= 0.60
    )
    decision = (
        "H2_LINKED_TO_LEADER_ADVANTAGE"
        if linked
        else "H2_NOT_LINKED_MOVE_MAIN_ROUTE_TO_H1"
    )
    next_action = (
        "Design a single unavoidable-history objective with matched controls."
        if linked
        else "Close H2 as the explanation of the leaderboard gap; prioritize coverage-preserving semantic compression."
    )

    output = {
        "status": "STEP302B_EVENT_BINDING_POSITIVE_CONTROL_COMPLETE",
        "design": {
            "data": "deterministic synthetic diagnostic; no BabyLM evaluation data",
            "cases": len(cases),
            "common_cases": len(common),
            "arms": {
                "causal": "swap target and distractor locations while preserving the complete word multiset",
                "null": "swap two unrelated locations while preserving the target binding and complete word multiset",
            },
            "case_families": ["initial binding", "latest state after overwrite"],
            "success_rule": "leader and leader-minus-protected CI95 lower bounds > 0 and leader positive fraction >= 0.60",
        },
        "provenance": {
            "script": str(_public_path('experiments/archive/initial_model_studies/scripts/revision_302b_event_binding_positive_control.py')),
            "script_sha256": research.sha256_file(_public_path('experiments/archive/initial_model_studies/scripts/revision_302b_event_binding_positive_control.py')),
            "scorer": str(PATH),
            "scorer_sha256": research.sha256_file(PATH),
            "case_seed": args.case_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "device": str(device),
        },
        "models": {name: {"identity": result["identity"]} for name, result in scored.items()},
        "statistics": statistics,
        "descriptive_by_family": {
            name: summarize_by_source(maps[name], cases) for name in maps
        },
        "decision": decision,
        "next_action": next_action,
        "elapsed_sec": time.time() - started,
    }
    out_json = Path(args.out_json).resolve()
    out_note = Path(args.out_note).resolve()
    research.write_json(out_json, output)
    out_note.parent.mkdir(parents=True, exist_ok=True)
    out_note.write_text(
        "\n".join(
            [
                "# Step 302b event-binding positive control",
                "",
                f"Decision: **{decision}**",
                "",
                "The causal and null conditions preserve the complete word multiset. Only the event bindings differ.",
                "",
                "| measure | mean | 95% bootstrap CI | fraction > 0 |",
                "|---|---:|---:|---:|",
                f"| protected | {statistics['protected']['mean']:+.6f} | [{statistics['protected']['bootstrap_ci95'][0]:+.6f}, {statistics['protected']['bootstrap_ci95'][1]:+.6f}] | {statistics['protected']['fraction_positive']:.3f} |",
                f"| public leader | {statistics['leader']['mean']:+.6f} | [{statistics['leader']['bootstrap_ci95'][0]:+.6f}, {statistics['leader']['bootstrap_ci95'][1]:+.6f}] | {statistics['leader']['fraction_positive']:.3f} |",
                f"| leader - protected | {statistics['leader_minus_protected']['mean']:+.6f} | [{statistics['leader_minus_protected']['bootstrap_ci95'][0]:+.6f}, {statistics['leader_minus_protected']['bootstrap_ci95'][1]:+.6f}] | {statistics['leader_minus_protected']['fraction_positive']:.3f} |",
                "",
                f"Next action: {next_action}",
                "",
                f"Evidence JSON: `{out_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "decision": decision,
                "statistics": statistics,
                "out_json": str(out_json),
                "elapsed_sec": output["elapsed_sec"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
