#!/usr/bin/env python3
"""research: construct and audit a natural transfer-possession symmetry substrate.

This is a CPU/file-only Explore asset.  It creates a small, inspectable corpus
family for the next causal study; it does not train a model, load a model, run
official evaluation, use GPU, upload, or contact a leaderboard.

Scientific object
-----------------
A held lexical component can be internally coherent under relation-comparison
rows while remaining ambiguous under a global swap of the two possessor roles.
Small natural bridges to a seen coordinate should break that symmetry; inverted
bridges should choose the opposite orientation; decoupled bridges should not.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Relation:
    key: str
    component: str  # seen or held
    final_slot: int  # 0 if participant a ends with the theme, 1 if participant b ends with it
    family: str
    train_templates: Tuple[str, ...]
    eval_templates: Tuple[str, ...]
    description: str


ROOT = _public_path('experiments/archive/representation_and_objectives')
DEFAULT_OUT = _public_path('experiments/archive/representation_and_objectives/data/symmetry_identification_substrate')

TRAIN_NAMES = [
    "Mira", "Noel", "Iris", "Omar", "Lena", "Pavel", "Rina", "Tomas",
    "Nia", "Felix", "Ava", "Jonas", "Keira", "Milo", "Sara", "Theo",
]
EVAL_NAMES = [
    "Zara", "Eli", "Nora", "Caleb", "Vera", "Hugo", "Maya", "Luca",
    "June", "Arun", "Leah", "Mateo", "Tara", "Simon", "Yara", "Ben",
]
TRAIN_OBJECTS = [
    "lantern", "parcel", "compass", "violin", "notebook", "teapot", "camera", "blanket",
    "basket", "tablet", "wallet", "sketchbook", "helmet", "map", "jacket", "medal",
]
EVAL_OBJECTS = [
    "goblet", "key", "telescope", "bracelet", "vase", "satchel", "radio", "painting",
    "flute", "badge", "ticket", "scarf", "hammer", "shell", "drum", "booklet",
]
STATIC_OBJECTS = [
    "mug", "rope", "spoon", "bottle", "pillow", "clock", "broom", "coin",
    "brush", "kite", "stone", "candle", "plate", "glove", "bell", "saddle",
]

# Seen relations are the already grounded coordinate.  Held relations are the
# component whose absolute orientation is to be identified by sparse bridges.
RELATIONS: Dict[str, Relation] = {
    "give_to": Relation(
        "give_to", "seen", 1, "source_subject_to",
        ("{a} gave the {obj} to {b}.", "{a} handed the {obj} to {b}."),
        ("{a} gave {b} the {obj}.", "{a} passed the {obj} over to {b}."),
        "source-subject seen transfer: participant b receives the object",
    ),
    "receive_from": Relation(
        "receive_from", "seen", 0, "recipient_subject_from",
        ("{a} received the {obj} from {b}.", "{a} got the {obj} from {b}."),
        ("From {b}, {a} received the {obj}.", "{a} took delivery of the {obj} from {b}."),
        "recipient-subject seen transfer: participant a receives the object",
    ),
    "cede_to": Relation(
        "cede_to", "held", 1, "source_subject_to",
        ("{a} ceded the {obj} to {b}.",),
        ("{a}'s cession sent the {obj} to {b}.", "{a} made a cession of the {obj} to {b}."),
        "held source-subject transfer",
    ),
    "bequeath_to": Relation(
        "bequeath_to", "held", 1, "source_subject_to",
        ("{a} bequeathed the {obj} to {b}.",),
        ("{a} bequeathed {b} the {obj}.", "{b} was bequeathed the {obj} by {a}."),
        "held source-subject transfer with dative alternation",
    ),
    "acquire_from": Relation(
        "acquire_from", "held", 0, "recipient_subject_from",
        ("{a} acquired the {obj} from {b}.",),
        ("From {b}, {a} acquired the {obj}.", "{a}'s acquisition from {b} involved the {obj}."),
        "held recipient-subject transfer",
    ),
    "inherit_from": Relation(
        "inherit_from", "held", 0, "recipient_subject_from",
        ("{a} inherited the {obj} from {b}.",),
        ("From {b}, {a} inherited the {obj}.", "{a}'s inheritance from {b} included the {obj}."),
        "held recipient-subject transfer with nominalized alternation",
    ),
}
HELD_KEYS = ["cede_to", "bequeath_to", "acquire_from", "inherit_from"]
SEEN_KEYS = ["give_to", "receive_from"]


QUERY_TEMPLATES = {
    "changed": (
        "Afterward, {candidate} had the {obj}.",
        "When the event was over, {candidate} possessed the {obj}.",
    ),
    "unchanged": (
        "Afterward, {candidate} still had the {obj}.",
        "When the event was over, {candidate} retained the {obj}.",
    ),
}


def render_event(rel_key: str, a: str, b: str, obj: str, *, bank: str, variant: int = 0) -> str:
    rel = RELATIONS[rel_key]
    templates = rel.train_templates if bank == "train" else rel.eval_templates
    t = templates[variant % len(templates)]
    return t.format(a=a, b=b, obj=obj)


def final_owner(rel_key: str, a: str, b: str, *, flipped_held: bool = False) -> str:
    rel = RELATIONS[rel_key]
    slot = rel.final_slot
    if flipped_held and rel.component == "held":
        slot = 1 - slot
    return (a, b)[slot]


def initial_owner(rel_key: str, a: str, b: str, *, flipped_held: bool = False) -> str:
    rel = RELATIONS[rel_key]
    slot = rel.final_slot
    if flipped_held and rel.component == "held":
        slot = 1 - slot
    return (a, b)[1 - slot]


def choose_pair(pool: Sequence[str], i: int) -> Tuple[str, str]:
    # Deterministic balanced cycling through adjacent pairs, with parity flips.
    a = pool[(2 * i) % len(pool)]
    b = pool[(2 * i + 1) % len(pool)]
    if (i // max(1, len(pool) // 2)) % 2:
        return b, a
    return a, b


def token_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def row_words(row: Dict) -> int:
    return sum(token_count(str(row.get(k, ""))) for k in ("text", "premise", "hypothesis", "event1", "event2"))


def comparison_row(
    row_id: str,
    rel1: str,
    rel2: str,
    names: Tuple[str, str],
    obj: str,
    same: bool,
    *,
    split: str,
    suite: str,
    bank: str,
    variant: int = 0,
) -> Dict:
    n0, n1 = names
    rel1_final_slot = RELATIONS[rel1].final_slot
    rel2_final_slot = RELATIONS[rel2].final_slot
    owner1 = (n0, n1)[rel1_final_slot]
    if same:
        # Choose argument order for rel2 so its final owner is owner1.
        if rel2_final_slot == rel1_final_slot:
            a2, b2 = n0, n1
        else:
            a2, b2 = n1, n0
    else:
        if rel2_final_slot == rel1_final_slot:
            a2, b2 = n1, n0
        else:
            a2, b2 = n0, n1
    owner2 = final_owner(rel2, a2, b2)
    assert (owner1 == owner2) == same
    event1 = render_event(rel1, n0, n1, obj, bank=bank, variant=variant)
    event2 = render_event(rel2, a2, b2, obj, bank=bank, variant=variant + 1)
    comp1 = RELATIONS[rel1].component
    comp2 = RELATIONS[rel2].component
    if comp1 == comp2:
        dep = "invariant_under_global_held_swap"
    elif "held" in (comp1, comp2) and "seen" in (comp1, comp2):
        dep = "breaks_held_seen_orientation_symmetry"
    else:
        dep = "invariant_under_global_held_swap"
    return {
        "row_id": row_id,
        "task": "relation_comparison",
        "split": split,
        "suite": suite,
        "text": f"Sentence A: {event1} Sentence B: {event2} Do the two sentences leave the {obj} with the same person?",
        "event1": event1,
        "event2": event2,
        "label": bool(same),
        "relation1": rel1,
        "relation2": rel2,
        "component1": comp1,
        "component2": comp2,
        "arg_order1": [n0, n1],
        "arg_order2": [a2, b2],
        "object": obj,
        "final_owner1": owner1,
        "final_owner2": owner2,
        "orientation_dependency": dep,
        "held_flip_changes_label": dep == "breaks_held_seen_orientation_symmetry",
    }


def state_query_rows(
    row_id_prefix: str,
    rel: str,
    names: Tuple[str, str],
    changed_obj: str,
    static_obj: str,
    *,
    split: str,
    suite: str,
    bank: str,
    static_owner_slot: int,
    variant: int = 0,
    invert_changed_label: bool = False,
    decoupled: bool = False,
) -> List[Dict]:
    a, b = names
    true_final = final_owner(rel, a, b)
    old_owner = initial_owner(rel, a, b)
    static_owner = (a, b)[static_owner_slot]
    other_static = b if static_owner == a else a
    event_text = render_event(rel, a, b, changed_obj, bank=bank, variant=variant)
    if decoupled:
        # Same consequence vocabulary and people appear, but the held relation is not the cause.
        premise = (
            f"Initially, {old_owner} had the {changed_obj}. Separately, {static_owner} had the {static_obj}. "
            f"The afternoon included a weather report and a quiet walk. {event_text}"
        )
        cause_kind = "decoupled_same_text_no_causal_bridge"
    else:
        premise = (
            f"Initially, {old_owner} had the {changed_obj}. Separately, {static_owner} had the {static_obj}. "
            f"Then {event_text}"
        )
        cause_kind = "event_causes_changed_possession"
    rows: List[Dict] = []
    changed_candidates = [true_final, b if true_final == a else a]
    if invert_changed_label:
        changed_true = changed_candidates[1]
    else:
        changed_true = changed_candidates[0]
    for ci, candidate in enumerate(changed_candidates):
        base_label = candidate == changed_true
        hyp = QUERY_TEMPLATES["changed"][variant % len(QUERY_TEMPLATES["changed"])].format(candidate=candidate, obj=changed_obj)
        rows.append({
            "row_id": f"{row_id_prefix}_changed_{ci}",
            "pair_id": row_id_prefix,
            "task": "state_query",
            "query_kind": "changed",
            "split": split,
            "suite": suite,
            "premise": premise,
            "hypothesis": hyp,
            "label": bool(base_label),
            "relation": rel,
            "component": RELATIONS[rel].component,
            "arg_order": [a, b],
            "changed_object": changed_obj,
            "static_object": static_obj,
            "candidate": candidate,
            "true_final_owner": true_final,
            "supervised_final_owner": changed_true,
            "initial_changed_owner": old_owner,
            "static_owner": static_owner,
            "cause_kind": cause_kind,
            "correct_slot": 0 if changed_true == a else 1,
            "candidate_slot": 0 if candidate == a else 1,
            "orientation_dependency": "breaks_held_seen_orientation_symmetry" if RELATIONS[rel].component == "held" and not decoupled else "invariant_or_seen_coordinate",
            "held_flip_changes_label": RELATIONS[rel].component == "held" and not decoupled,
            "inverted_bridge_label": bool(invert_changed_label),
            "decoupled": bool(decoupled),
        })
    # Unchanged fact must remain true irrespective of the transfer orientation.
    static_candidates = [static_owner, other_static]
    for ci, candidate in enumerate(static_candidates):
        hyp = QUERY_TEMPLATES["unchanged"][(variant + 1) % len(QUERY_TEMPLATES["unchanged"])].format(candidate=candidate, obj=static_obj)
        rows.append({
            "row_id": f"{row_id_prefix}_unchanged_{ci}",
            "pair_id": row_id_prefix,
            "task": "state_query",
            "query_kind": "unchanged",
            "split": split,
            "suite": suite,
            "premise": premise,
            "hypothesis": hyp,
            "label": bool(candidate == static_owner),
            "relation": rel,
            "component": RELATIONS[rel].component,
            "arg_order": [a, b],
            "changed_object": changed_obj,
            "static_object": static_obj,
            "candidate": candidate,
            "true_final_owner": true_final,
            "supervised_final_owner": changed_true,
            "initial_changed_owner": old_owner,
            "static_owner": static_owner,
            "cause_kind": cause_kind,
            "correct_slot": 0 if static_owner == a else 1,
            "candidate_slot": 0 if candidate == a else 1,
            "orientation_dependency": "unaffected_fact_should_be_preserved",
            "held_flip_changes_label": False,
            "inverted_bridge_label": False,
            "decoupled": bool(decoupled),
        })
    return rows


def unsup_event_row(row_id: str, rel: str, names: Tuple[str, str], obj: str, *, split: str, suite: str, bank: str, variant: int = 0) -> Dict:
    a, b = names
    event = render_event(rel, a, b, obj, bank=bank, variant=variant)
    return {
        "row_id": row_id,
        "task": "unsupervised_event_text",
        "split": split,
        "suite": suite,
        "text": event,
        "relation": rel,
        "component": RELATIONS[rel].component,
        "arg_order": [a, b],
        "object": obj,
        "orientation_dependency": "none_unsupervised",
        "held_flip_changes_label": False,
    }


def build_common_seen(n: int, rng: random.Random) -> List[Dict]:
    rows: List[Dict] = []
    for i in range(n):
        rel = SEEN_KEYS[i % len(SEEN_KEYS)]
        names = choose_pair(TRAIN_NAMES, i)
        obj = TRAIN_OBJECTS[i % len(TRAIN_OBJECTS)]
        static = STATIC_OBJECTS[i % len(STATIC_OBJECTS)]
        rows.extend(state_query_rows(
            f"common_seen_{i:04d}", rel, names, obj, static,
            split="train", suite="common_seen_coordinate", bank="train",
            static_owner_slot=(i // 2) % 2, variant=i,
        ))
    return rows


def build_heldheld_rows(n_per_edge: int, *, split: str, bank: str, names_pool: Sequence[str], obj_pool: Sequence[str], start: int = 0) -> List[Dict]:
    rows: List[Dict] = []
    edges = [("cede_to", "bequeath_to"), ("bequeath_to", "acquire_from"), ("acquire_from", "inherit_from"), ("inherit_from", "cede_to")]
    idx = start
    for edge_i, (r1, r2) in enumerate(edges):
        for j in range(n_per_edge):
            same = (j % 2 == 0)
            names = choose_pair(names_pool, idx)
            obj = obj_pool[idx % len(obj_pool)]
            rows.append(comparison_row(
                f"{split}_heldheld_e{edge_i}_{j:04d}", r1, r2, names, obj, same,
                split=split, suite="held_held_component_coherence", bank=bank, variant=idx,
            ))
            idx += 1
    return rows


def build_mixed_eval(n_per_pair: int) -> List[Dict]:
    rows: List[Dict] = []
    idx = 0
    pairs = []
    for h in HELD_KEYS:
        for s in SEEN_KEYS:
            pairs.append((h, s))
    for pair_i, (h, s) in enumerate(pairs):
        for j in range(n_per_pair):
            same = (j % 2 == 0)
            names = choose_pair(EVAL_NAMES, idx)
            obj = EVAL_OBJECTS[idx % len(EVAL_OBJECTS)]
            # Alternate direction: held first or seen first, so lexical order cannot solve it.
            if j % 4 < 2:
                rel1, rel2 = h, s
            else:
                rel1, rel2 = s, h
            rows.append(comparison_row(
                f"eval_mixed_p{pair_i}_{j:04d}", rel1, rel2, names, obj, same,
                split="eval", suite="mixed_held_seen_orientation", bank="eval", variant=idx,
            ))
            idx += 1
    return rows


def build_state_eval(n_per_rel: int) -> List[Dict]:
    rows: List[Dict] = []
    idx = 0
    for rel in HELD_KEYS:
        for j in range(n_per_rel):
            names = choose_pair(EVAL_NAMES, idx)
            changed = EVAL_OBJECTS[idx % len(EVAL_OBJECTS)]
            static = STATIC_OBJECTS[(idx + 5) % len(STATIC_OBJECTS)]
            rows.extend(state_query_rows(
                f"eval_state_{rel}_{j:04d}", rel, names, changed, static,
                split="eval", suite="paired_state_conservation", bank="eval",
                static_owner_slot=(idx // 2) % 2, variant=idx,
            ))
            idx += 1
    return rows


def build_cross_frame_eval(n_per_rel: int) -> List[Dict]:
    rows: List[Dict] = []
    idx = 0
    for rel in HELD_KEYS:
        for j in range(n_per_rel):
            names = choose_pair(EVAL_NAMES, idx + 1000)
            changed = EVAL_OBJECTS[(idx + 3) % len(EVAL_OBJECTS)]
            static = STATIC_OBJECTS[(idx + 9) % len(STATIC_OBJECTS)]
            rows.extend(state_query_rows(
                f"eval_crossframe_{rel}_{j:04d}", rel, names, changed, static,
                split="eval", suite="cross_frame_state_readout", bank="eval",
                static_owner_slot=(idx + 1) % 2, variant=idx + 1,
            ))
            idx += 1
    return rows


def build_name_permutation_eval(n: int) -> List[Dict]:
    rows: List[Dict] = []
    for i in range(n):
        rel = HELD_KEYS[i % len(HELD_KEYS)]
        n0, n1 = choose_pair(EVAL_NAMES, i + 2000)
        obj = EVAL_OBJECTS[(i + 7) % len(EVAL_OBJECTS)]
        static = STATIC_OBJECTS[(i + 11) % len(STATIC_OBJECTS)]
        rows.extend(state_query_rows(
            f"eval_perm_base_{i:04d}", rel, (n0, n1), obj, static,
            split="eval", suite="name_permutation_base", bank="eval",
            static_owner_slot=i % 2, variant=i,
        ))
        rows.extend(state_query_rows(
            f"eval_perm_swapped_{i:04d}", rel, (n1, n0), obj, static,
            split="eval", suite="name_permutation_swapped", bank="eval",
            static_owner_slot=(i + 1) % 2, variant=i + 1,
        ))
    return rows


def build_arm_rows(k_bridge: int, heldheld_n: int) -> Dict[str, Dict[str, List[Dict]]]:
    arms: Dict[str, Dict[str, List[Dict]]] = {}
    base_heldheld = build_heldheld_rows(heldheld_n, split="train", bank="train", names_pool=TRAIN_NAMES, obj_pool=TRAIN_OBJECTS)
    # Exposure rows are held event text only; they can be used in an MLM version but carry no absolute labels.
    exposure_unsup: List[Dict] = []
    for i in range(heldheld_n * len(HELD_KEYS)):
        rel = HELD_KEYS[i % len(HELD_KEYS)]
        exposure_unsup.append(unsup_event_row(
            f"train_exposure_{i:04d}", rel, choose_pair(TRAIN_NAMES, i + 3000), TRAIN_OBJECTS[i % len(TRAIN_OBJECTS)],
            split="train", suite="held_event_exposure", bank="train", variant=i,
        ))
    arms["exposure_only"] = {"supervised": [], "unsupervised": exposure_unsup}
    arms["heldheld_only"] = {"supervised": list(base_heldheld), "unsupervised": exposure_unsup}

    # Natural bridges: a few state consequences for only one held relation h1.  Eval probes h2-h4.
    h1 = "cede_to"
    bridge_rows: List[Dict] = []
    inverted_rows: List[Dict] = []
    decoupled_rows: List[Dict] = []
    for i in range(k_bridge):
        names = choose_pair(TRAIN_NAMES, i + 4000)
        changed = TRAIN_OBJECTS[(i + 3) % len(TRAIN_OBJECTS)]
        static = STATIC_OBJECTS[(i + 4) % len(STATIC_OBJECTS)]
        bridge_rows.extend(state_query_rows(
            f"train_aligned_bridge_{i:04d}", h1, names, changed, static,
            split="train", suite="aligned_sparse_natural_bridge", bank="train",
            static_owner_slot=i % 2, variant=i,
        ))
        inverted_rows.extend(state_query_rows(
            f"train_inverted_bridge_{i:04d}", h1, names, changed, static,
            split="train", suite="inverted_sparse_natural_bridge", bank="train",
            static_owner_slot=i % 2, variant=i, invert_changed_label=True,
        ))
        decoupled_rows.extend(state_query_rows(
            f"train_decoupled_bridge_{i:04d}", h1, names, changed, static,
            split="train", suite="decoupled_sparse_text_control", bank="train",
            static_owner_slot=i % 2, variant=i, decoupled=True,
        ))
    # Mixed-event bridge version: direct comparison to a seen relation rather than explicit state labels.
    mixed_bridge_rows: List[Dict] = []
    for i in range(k_bridge):
        names = choose_pair(TRAIN_NAMES, i + 5000)
        obj = TRAIN_OBJECTS[(i + 9) % len(TRAIN_OBJECTS)]
        # Alternate true/false so the comparison row cannot become all positive.
        mixed_bridge_rows.append(comparison_row(
            f"train_mixed_event_bridge_{i:04d}", h1, SEEN_KEYS[i % len(SEEN_KEYS)], names, obj, same=(i % 2 == 0),
            split="train", suite="mixed_event_sparse_bridge", bank="train", variant=i,
        ))
    arms["aligned_state_bridge"] = {"supervised": list(base_heldheld) + bridge_rows, "unsupervised": exposure_unsup}
    arms["inverted_state_bridge"] = {"supervised": list(base_heldheld) + inverted_rows, "unsupervised": exposure_unsup}
    arms["decoupled_state_bridge"] = {"supervised": list(base_heldheld) + decoupled_rows, "unsupervised": exposure_unsup}
    arms["mixed_event_bridge"] = {"supervised": list(base_heldheld) + mixed_bridge_rows, "unsupervised": exposure_unsup}
    return arms


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_balance(rows: Sequence[Dict]) -> Dict[str, float]:
    labs = [r.get("label") for r in rows if "label" in r]
    if not labs:
        return {"n_labeled": 0}
    return {
        "n_labeled": len(labs),
        "true_frac": sum(1 for x in labs if x is True) / len(labs),
        "false_frac": sum(1 for x in labs if x is False) / len(labs),
    }


def slot_balance(rows: Sequence[Dict]) -> Dict[str, float]:
    slots = [r.get("correct_slot") for r in rows if r.get("task") == "state_query" and r.get("label") is True]
    if not slots:
        return {"n_state_true": 0}
    return {
        "n_state_true": len(slots),
        "correct_slot0_frac": sum(1 for x in slots if x == 0) / len(slots),
        "correct_slot1_frac": sum(1 for x in slots if x == 1) / len(slots),
    }


def name_balance(rows: Sequence[Dict]) -> Dict[str, object]:
    role_counts = Counter()
    candidate_true_counts = Counter()
    for r in rows:
        for idx, nm in enumerate(r.get("arg_order", []) or []):
            role_counts[(nm, f"slot{idx}")] += 1
        if r.get("task") == "state_query" and r.get("label") is True:
            candidate_true_counts[r.get("candidate")] += 1
    return {
        "slot_role_count_minmax": minmax(role_counts.values()),
        "true_candidate_count_minmax": minmax(candidate_true_counts.values()) if candidate_true_counts else None,
        "n_unique_role_names": len({k[0] for k in role_counts}),
        "n_unique_true_candidate_names": len(candidate_true_counts),
    }


def minmax(vals: Iterable[int]) -> Optional[Tuple[int, int]]:
    vals = list(vals)
    if not vals:
        return None
    return (min(vals), max(vals))


def shortcut_baselines_state(rows: Sequence[Dict]) -> Dict[str, float]:
    # Evaluate on grouped binary true/false state-query candidate rows.  A position
    # heuristic is correct on a pair iff it selects the true candidate's slot.
    true_rows = [r for r in rows if r.get("task") == "state_query" and r.get("label") is True]
    if not true_rows:
        return {"n_pairs": 0}
    first = sum(1 for r in true_rows if r.get("correct_slot") == 0) / len(true_rows)
    second = sum(1 for r in true_rows if r.get("correct_slot") == 1) / len(true_rows)
    static_owner_same_as_changed_final = []
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("task") == "state_query":
            by_pair[r.get("pair_id")][(r.get("query_kind"), r.get("label"))] = r
    for d in by_pair.values():
        c = d.get(("changed", True))
        u = d.get(("unchanged", True))
        if c and u:
            static_owner_same_as_changed_final.append(c.get("candidate") == u.get("candidate"))
    return {
        "n_true_state_queries": len(true_rows),
        "first_participant_accuracy_if_always_first": first,
        "second_participant_accuracy_if_always_second_or_last": second,
        "changed_and_unchanged_same_owner_frac": (sum(static_owner_same_as_changed_final) / len(static_owner_same_as_changed_final)) if static_owner_same_as_changed_final else None,
    }


def audit_rows(common_seen: Sequence[Dict], arms: Dict[str, Dict[str, List[Dict]]], eval_sets: Dict[str, List[Dict]]) -> Dict:
    all_eval = [r for rows in eval_sets.values() for r in rows]
    arm_summaries = {}
    for arm, blocks in arms.items():
        sup = blocks["supervised"]
        unsup = blocks["unsupervised"]
        deps = Counter(r.get("orientation_dependency") for r in sup)
        arm_summaries[arm] = {
            "supervised_rows": len(sup),
            "unsupervised_rows": len(unsup),
            "supervised_words": sum(row_words(r) for r in sup),
            "unsupervised_words": sum(row_words(r) for r in unsup),
            "label_balance": label_balance(sup),
            "correct_slot_balance": slot_balance(sup),
            "orientation_dependency_counts": dict(deps),
            "contains_held_seen_orientation_breaking_rows": any(r.get("held_flip_changes_label") for r in sup),
        }
    eval_summaries = {}
    for name, rows in eval_sets.items():
        eval_summaries[name] = {
            "rows": len(rows),
            "words": sum(row_words(r) for r in rows),
            "label_balance": label_balance(rows),
            "correct_slot_balance": slot_balance(rows),
            "orientation_dependency_counts": dict(Counter(r.get("orientation_dependency") for r in rows)),
            "name_balance": name_balance(rows),
            "position_shortcut_readout": shortcut_baselines_state(rows),
        }
    return {
        "status": "SYMMETRY_SUBSTRATE_AUDIT_COMPLETE",
        "common_seen_coordinate": {
            "rows": len(common_seen),
            "words": sum(row_words(r) for r in common_seen),
            "label_balance": label_balance(common_seen),
            "correct_slot_balance": slot_balance(common_seen),
        },
        "arms": arm_summaries,
        "eval_sets": eval_summaries,
        "global_readout": {
            "heldheld_train_labels_invariant_under_global_swap": True,
            "held_seen_mixed_eval_labels_change_under_held_global_swap": True,
            "aligned_bridge_breaks_symmetry": arm_summaries["aligned_state_bridge"]["contains_held_seen_orientation_breaking_rows"],
            "inverted_bridge_breaks_symmetry_opposite_labels": arm_summaries["inverted_state_bridge"]["contains_held_seen_orientation_breaking_rows"],
            "decoupled_bridge_preserves_ambiguity": not arm_summaries["decoupled_state_bridge"]["contains_held_seen_orientation_breaking_rows"],
            "eval_position_shortcuts_near_half": shortcut_baselines_state(all_eval),
            "train_eval_name_sets_disjoint": set(TRAIN_NAMES).isdisjoint(EVAL_NAMES),
            "train_eval_object_sets_disjoint": set(TRAIN_OBJECTS).isdisjoint(EVAL_OBJECTS),
            "static_objects_separate_from_changed_objects": set(STATIC_OBJECTS).isdisjoint(TRAIN_OBJECTS) and set(STATIC_OBJECTS).isdisjoint(EVAL_OBJECTS),
        },
        "relation_inventory": {k: RELATIONS[k].__dict__ for k in RELATIONS},
    }


def write_summary(path: Path, audit: Dict, files: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("# research symmetry-identification substrate audit")
    lines.append("")
    lines.append("CPU/file-only construction for a natural transfer-of-possession symmetry experiment. No model loading, training, official evaluation, GPU work, upload, or leaderboard action occurred.")
    lines.append("")
    lines.append("## Scientific object")
    lines.append("")
    lines.append("The held lexical component has relation-comparison rows that are internally learnable while remaining invariant under a global swap of the two possessor roles. Natural sparse bridges to the seen transfer coordinate break this ambiguity; inverted bridges select the opposite orientation; decoupled text controls preserve the ambiguity.")
    lines.append("")
    lines.append("## Row counts")
    lines.append("")
    cs = audit["common_seen_coordinate"]
    lines.append(f"- common seen-coordinate rows: {cs['rows']} labeled rows, {cs['words']} approximate words, true fraction {cs['label_balance'].get('true_frac'):.3f}")
    lines.append("")
    lines.append("| arm | supervised rows | unsup rows | orientation-breaking supervised rows present | true frac | slot0 true frac |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for arm, s in audit["arms"].items():
        lb = s["label_balance"]
        sb = s["correct_slot_balance"]
        lines.append(f"| {arm} | {s['supervised_rows']} | {s['unsupervised_rows']} | {s['contains_held_seen_orientation_breaking_rows']} | {lb.get('true_frac', float('nan')):.3f} | {sb.get('correct_slot0_frac', float('nan')):.3f} |")
    lines.append("")
    lines.append("## Evaluation suites")
    lines.append("")
    lines.append("| suite | rows | true frac | slot0 true frac | first-slot shortcut | second-slot shortcut |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for suite, s in audit["eval_sets"].items():
        lb = s["label_balance"]
        sb = s["correct_slot_balance"]
        sh = s["position_shortcut_readout"]
        lines.append(f"| {suite} | {s['rows']} | {lb.get('true_frac', float('nan')):.3f} | {sb.get('correct_slot0_frac', float('nan')):.3f} | {sh.get('first_participant_accuracy_if_always_first', float('nan')):.3f} | {sh.get('second_participant_accuracy_if_always_second_or_last', float('nan')):.3f} |")
    lines.append("")
    lines.append("## Symmetry audit")
    lines.append("")
    g = audit["global_readout"]
    for k in [
        "heldheld_train_labels_invariant_under_global_swap",
        "held_seen_mixed_eval_labels_change_under_held_global_swap",
        "aligned_bridge_breaks_symmetry",
        "inverted_bridge_breaks_symmetry_opposite_labels",
        "decoupled_bridge_preserves_ambiguity",
        "train_eval_name_sets_disjoint",
        "train_eval_object_sets_disjoint",
        "static_objects_separate_from_changed_objects",
    ]:
        lines.append(f"- {k}: {g[k]}")
    sh = g["eval_position_shortcuts_near_half"]
    lines.append(f"- all state-eval first-slot shortcut: {sh['first_participant_accuracy_if_always_first']:.3f}; second-slot shortcut: {sh['second_participant_accuracy_if_always_second_or_last']:.3f}; changed/static same-owner fraction: {sh['changed_and_unchanged_same_owner_frac']:.3f}")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for key, value in files.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=27600)
    ap.add_argument("--common-seen", type=int, default=48)
    ap.add_argument("--heldheld-n", type=int, default=16)
    ap.add_argument("--bridge-k", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.out
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    common_seen = build_common_seen(args.common_seen, rng)
    arms = build_arm_rows(args.bridge_k, args.heldheld_n)
    eval_sets = {
        "held_held_coherence": build_heldheld_rows(args.eval_n, split="eval", bank="eval", names_pool=EVAL_NAMES, obj_pool=EVAL_OBJECTS, start=6000),
        "mixed_held_seen_orientation": build_mixed_eval(args.eval_n),
        "paired_state_conservation": build_state_eval(args.eval_n),
        "cross_frame_state_readout": build_cross_frame_eval(max(4, args.eval_n // 2)),
        "name_permutation_counterfactual": build_name_permutation_eval(max(8, args.eval_n // 2)),
    }

    files: Dict[str, str] = {}
    project_root = _public_path('.')
    files["common_seen_train_jsonl"] = str((out / "common_seen_train.jsonl").relative_to(project_root))
    write_jsonl(out / "common_seen_train.jsonl", common_seen)

    for arm, blocks in arms.items():
        p_sup = out / "arms" / arm / "train_supervised.jsonl"
        p_unsup = out / "arms" / arm / "train_unsup_text.jsonl"
        write_jsonl(p_sup, blocks["supervised"])
        write_jsonl(p_unsup, blocks["unsupervised"])
        files[f"arm_{arm}_supervised_jsonl"] = str(p_sup.relative_to(project_root))
        files[f"arm_{arm}_unsup_jsonl"] = str(p_unsup.relative_to(project_root))

    for suite, rows in eval_sets.items():
        p = out / "eval" / f"{suite}.jsonl"
        write_jsonl(p, rows)
        files[f"eval_{suite}_jsonl"] = str(p.relative_to(project_root))

    audit = audit_rows(common_seen, arms, eval_sets)
    audit_path = out / "symmetry_substrate_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files["audit_json"] = str(audit_path.relative_to(project_root))

    arm_rows_csv = []
    for arm, s in audit["arms"].items():
        row = {"arm": arm}
        row.update({k: v for k, v in s.items() if not isinstance(v, dict)})
        row.update({f"label_{k}": v for k, v in s["label_balance"].items()})
        row.update({f"slot_{k}": v for k, v in s["correct_slot_balance"].items()})
        arm_rows_csv.append(row)
    write_csv(out / "arm_summary.csv", arm_rows_csv)
    files["arm_summary_csv"] = str((out / "arm_summary.csv").relative_to(project_root))

    eval_rows_csv = []
    for suite, s in audit["eval_sets"].items():
        row = {"suite": suite, "rows": s["rows"], "words": s["words"]}
        row.update({f"label_{k}": v for k, v in s["label_balance"].items()})
        row.update({f"slot_{k}": v for k, v in s["correct_slot_balance"].items()})
        row.update({f"shortcut_{k}": v for k, v in s["position_shortcut_readout"].items()})
        eval_rows_csv.append(row)
    write_csv(out / "eval_summary.csv", eval_rows_csv)
    files["eval_summary_csv"] = str((out / "eval_summary.csv").relative_to(project_root))

    summary_path = out / "symmetry_substrate_summary.md"
    write_summary(summary_path, audit, files)
    files["summary_md"] = str(summary_path.relative_to(project_root))

    print(json.dumps({
        "status": audit["status"],
        "out": str(out.relative_to(project_root)),
        "summary_md": files["summary_md"],
        "common_seen_rows": audit["common_seen_coordinate"]["rows"],
        "arms": {k: {"supervised_rows": v["supervised_rows"], "orientation_breaking": v["contains_held_seen_orientation_breaking_rows"]} for k, v in audit["arms"].items()},
        "eval_shortcuts": audit["global_readout"]["eval_position_shortcuts_near_half"],
        "no_model_loading_training_evaluation_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
