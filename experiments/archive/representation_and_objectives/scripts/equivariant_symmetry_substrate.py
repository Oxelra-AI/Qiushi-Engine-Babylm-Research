#!/usr/bin/env python3
"""research: equivariant symmetry-identification substrate.

Purpose
-------
Repair the research comparison surface before any further learned conclusion.
The surface keeps a natural reusable argument interface (active/passive
syntactic roles) but balances the visible order of names, event sides, voices,
and candidate positions so simple order/name rules cannot reproduce the 0.750
shortcut seen in research.

No model loading, no training, no official BabyLM evaluation, no upload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Any

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate')


@dataclass(frozen=True)
class Rel:
    key: str
    component: str
    final_slot: int  # 0 = syntactic role A/agent-like, 1 = role B/patient-like
    lex: str
    family: str


HELD: Dict[str, Rel] = {
    "h0_dax": Rel("h0_dax", "held", 1, "dax", "nonce_A"),
    "h1_mep": Rel("h1_mep", "held", 1, "mep", "nonce_A"),
    "h2_norp": Rel("h2_norp", "held", 0, "norp", "nonce_B"),
    "h3_ziv": Rel("h3_ziv", "held", 0, "ziv", "nonce_B"),
}
SEEN: Dict[str, Rel] = {
    "s_give": Rel("s_give", "seen", 1, "give", "source_to_goal"),
    "s_receive": Rel("s_receive", "seen", 0, "receive", "goal_from_source"),
}
REL: Dict[str, Rel] = {**HELD, **SEEN}
HELD_KEYS = list(HELD.keys())
SEEN_KEYS = list(SEEN.keys())
TRUE_ASSIGNMENT = {k: HELD[k].final_slot for k in HELD_KEYS}
INVERTED_ASSIGNMENT = {k: 1 - v for k, v in TRUE_ASSIGNMENT.items()}

TRAIN_NAMES = ["Mira", "Noel", "Iris", "Omar", "Lena", "Pavel", "Rina", "Tomas", "Nia", "Felix", "Ava", "Jonas", "Keira", "Milo", "Sara", "Theo"]
EVAL_NAMES = ["Zara", "Eli", "Nora", "Caleb", "Vera", "Hugo", "Maya", "Luca", "June", "Arun", "Leah", "Mateo", "Tara", "Simon", "Yara", "Ben"]
TRAIN_CHANGED = ["lantern", "parcel", "compass", "violin", "notebook", "teapot", "camera", "blanket", "basket", "tablet", "wallet", "sketchbook", "helmet", "map", "jacket", "medal"]
EVAL_CHANGED = ["goblet", "key", "telescope", "bracelet", "vase", "satchel", "radio", "painting", "flute", "badge", "ticket", "scarf", "hammer", "shell", "drum", "booklet"]
TRAIN_STATIC = ["mug", "rope", "spoon", "bottle", "pillow", "clock", "broom", "coin", "brush", "kite", "stone", "candle", "plate", "glove", "bell", "saddle"]
EVAL_STATIC = ["ruler", "mirror", "bucket", "comb", "ladder", "needle", "pencil", "towel", "anchor", "fork", "whistle", "basketball", "bowl", "rug", "lamp", "button"]


def choose_pair(pool: Sequence[str], i: int) -> Tuple[str, str]:
    n = len(pool)
    a = pool[i % n]
    b = pool[(i * 5 + 3) % n]
    if a == b:
        b = pool[(i + 7) % n]
    if (i // n) % 2:
        a, b = b, a
    return a, b


def held_event(rel_key: str, a: str, b: str, obj: str, voice: int) -> Tuple[str, List[str], str, str]:
    """Return text, visible name order, template id, and voice.

    Latent role slot 0 is the active subject/by-phrase agent-like role; slot 1
    is the direct object/passive subject patient-like role.  Active/passive
    alternation reverses visible order without changing latent roles.
    """
    lex = HELD[rel_key].lex
    templates = [
        ("During the {obj} episode, {a} {lex}ed {b}.", ["a", "b"], "active_front", "active"),
        ("During the {obj} episode, {b} was {lex}ed by {a}.", ["b", "a"], "passive_front", "passive"),
    ]
    tmpl, order_keys, tid, v = templates[voice % 2]
    text = tmpl.format(a=a, b=b, obj=obj, lex=lex)
    order = [a if x == "a" else b for x in order_keys]
    return text, order, tid, v


def seen_event(rel_key: str, a: str, b: str, obj: str, voice: int) -> Tuple[str, List[str], str, str]:
    """Seen transfer wording with matched visible-order reversals."""
    templates = {
        "s_give": [
            ("During the {obj} episode, {a} gave the {obj} to {b}.", ["a", "b"], "give_to_active", "active"),
            ("During the {obj} episode, the {obj} was given to {b} by {a}.", ["b", "a"], "give_to_passive", "passive"),
        ],
        "s_receive": [
            ("During the {obj} episode, {a} received the {obj} from {b}.", ["a", "b"], "receive_from_active", "active"),
            ("During the {obj} episode, from {b}, the {obj} was received by {a}.", ["b", "a"], "receive_from_fronted", "passive_like"),
        ],
    }
    tmpl, order_keys, tid, v = templates[rel_key][voice % 2]
    text = tmpl.format(a=a, b=b, obj=obj)
    order = [a if x == "a" else b for x in order_keys]
    return text, order, tid, v


def event(rel_key: str, a: str, b: str, obj: str, voice: int) -> Tuple[str, List[str], str, str]:
    if REL[rel_key].component == "held":
        return held_event(rel_key, a, b, obj, voice)
    return seen_event(rel_key, a, b, obj, voice)


def slot(rel_key: str, assignment: Dict[str, int]) -> int:
    if rel_key in HELD:
        return int(assignment[rel_key])
    return REL[rel_key].final_slot


def owner_from_slot(slot_idx: int, a: str, b: str) -> str:
    return (a, b)[slot_idx]


def final_owner(rel_key: str, a: str, b: str, assignment: Dict[str, int] = TRUE_ASSIGNMENT) -> str:
    return owner_from_slot(slot(rel_key, assignment), a, b)


def old_owner(rel_key: str, a: str, b: str, assignment: Dict[str, int] = TRUE_ASSIGNMENT) -> str:
    return owner_from_slot(1 - slot(rel_key, assignment), a, b)


def words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def row_words(r: Dict[str, Any]) -> int:
    keys = ("text", "premise", "hypothesis", "event1", "event2", "held_distractor_event", "cause_event")
    return sum(words(str(r.get(k, ""))) for k in keys)


def comparison_row(row_id: str, rel1: str, rel2: str, a: str, b: str, obj: str,
                   same: bool, split: str, suite: str, voice1: int, voice2: int) -> Dict[str, Any]:
    s1 = slot(rel1, TRUE_ASSIGNMENT)
    s2 = slot(rel2, TRUE_ASSIGNMENT)
    if same:
        a2, b2 = (a, b) if s1 == s2 else (b, a)
    else:
        a2, b2 = (b, a) if s1 == s2 else (a, b)
    e1, o1, t1, v1 = event(rel1, a, b, obj, voice1)
    e2, o2, t2, v2 = event(rel2, a2, b2, obj, voice2)
    assert (final_owner(rel1, a, b) == final_owner(rel2, a2, b2)) == same
    comps = (REL[rel1].component, REL[rel2].component)
    if comps.count("held") == 1:
        dep = "held_seen_breaks_symmetry"
    elif comps == ("held", "held"):
        dep = "heldheld_invariant"
    else:
        dep = "seen_seen_fixed"
    return {
        "row_id": row_id,
        "task": "relation_comparison",
        "split": split,
        "suite": suite,
        "relation1": rel1,
        "relation2": rel2,
        "component1": REL[rel1].component,
        "component2": REL[rel2].component,
        "arg_order1": [a, b],
        "arg_order2": [a2, b2],
        "surface_order1": o1,
        "surface_order2": o2,
        "template1": t1,
        "template2": t2,
        "voice1": v1,
        "voice2": v2,
        "object": obj,
        "event1": e1,
        "event2": e2,
        "text": f"Event A: {e1} Event B: {e2} Did Event A and Event B leave the {obj} with the same person?",
        "label": bool(same),
        "orientation_dependency": dep,
        "global_swap_changes_label": dep == "held_seen_breaks_symmetry",
    }


def comparison_orbit(row_prefix: str, rel1: str, rel2: str, a: str, b: str, obj: str,
                     same: bool, split: str, suite: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for v1 in [0, 1]:
        for v2 in [0, 1]:
            rows.append(comparison_row(f"{row_prefix}_v{v1}{v2}", rel1, rel2, a, b, obj, same, split, suite, v1, v2))
    return rows


def state_rows(row_id: str, rel_key: str, a: str, b: str, changed_obj: str, static_obj: str,
               split: str, suite: str, voice: int, static_slot: int,
               assignment_for_labels: Dict[str, int] = TRUE_ASSIGNMENT) -> List[Dict[str, Any]]:
    new = final_owner(rel_key, a, b, assignment_for_labels)
    old = owner_from_slot(1 - slot(rel_key, assignment_for_labels), a, b)
    # Static ownership is counterbalanced independently of event voice.  research's
    # first audit showed unchanged rows were otherwise solved by choosing the
    # first visible participant, which would invalidate conservation readouts.
    static = (a, b)[static_slot % 2]
    ev, surface_order, tmpl, vv = event(rel_key, a, b, changed_obj, voice)
    premise = f"At first, {old} had the {changed_obj}, and {static} had the {static_obj}. The event was this: {ev}"
    rows: List[Dict[str, Any]] = []
    for query_kind, obj, true_owner, dep in [
        ("changed", changed_obj, new, "held_seen_breaks_symmetry" if rel_key in HELD else "seen_seen_fixed"),
        ("unchanged", static_obj, static, "unaffected_fact_should_be_preserved"),
    ]:
        for cand in [a, b]:
            rows.append({
                "row_id": f"{row_id}_{query_kind}_{0 if cand == a else 1}",
                "pair_id": row_id,
                "task": "state_query",
                "split": split,
                "suite": suite,
                "relation": rel_key,
                "component": REL[rel_key].component,
                "arg_order": [a, b],
                "surface_order": surface_order,
                "template": tmpl,
                "voice": vv,
                "static_slot": static_slot,
                "changed_object": changed_obj,
                "static_object": static_obj,
                "query_kind": query_kind,
                "candidate": cand,
                "candidate_slot": 0 if cand == a else 1,
                "correct_slot": 0 if true_owner == a else 1,
                "premise": premise,
                "hypothesis": f"After the event, {cand} had the {obj}.",
                "label": bool(cand == true_owner),
                "orientation_dependency": dep,
                "global_swap_changes_label": dep == "held_seen_breaks_symmetry",
                "cause_relation": rel_key,
                "cause_event": ev,
                "static_owner": static,
                "supervised_changed_owner": new,
                "inverted_bridge_label": assignment_for_labels != TRUE_ASSIGNMENT,
            })
    return rows


def state_orbit(row_prefix: str, rel_key: str, a: str, b: str, changed_obj: str, static_obj: str,
                split: str, suite: str, assignment_for_labels: Dict[str, int] = TRUE_ASSIGNMENT) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for voice in [0, 1]:
        for static_slot in [0, 1]:
            rows.extend(state_rows(f"{row_prefix}_v{voice}_st{static_slot}", rel_key, a, b, changed_obj, static_obj, split, suite, voice, static_slot, assignment_for_labels))
    return rows


def decoupled_orbit(row_prefix: str, held_rel: str, seen_rel: str, a: str, b: str,
                    held_obj: str, cause_obj: str, static_obj: str, split: str, suite: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for hv in [0, 1]:
        for sv in [0, 1]:
            for static_slot in [0, 1]:
                held_ev, held_order, held_t, held_voice = held_event(held_rel, a, b, held_obj, hv)
                cause_ev, cause_order, cause_t, cause_voice = seen_event(seen_rel, a, b, cause_obj, sv)
                new = final_owner(seen_rel, a, b)
                old = old_owner(seen_rel, a, b)
                static = (a, b)[static_slot % 2]
                rid = f"{row_prefix}_hv{hv}_sv{sv}_st{static_slot}"
                premise = (
                    f"A separate note described this unrelated episode: {held_ev} "
                    f"For the tested object, at first {old} had the {cause_obj}, and {static} had the {static_obj}. "
                    f"The tested event was this: {cause_ev}"
                )
                for query_kind, obj, true_owner, dep in [
                    ("changed", cause_obj, new, "seen_seen_fixed"),
                    ("unchanged", static_obj, static, "unaffected_fact_should_be_preserved"),
                ]:
                    for cand in [a, b]:
                        rows.append({
                        "row_id": f"{rid}_{query_kind}_{0 if cand == a else 1}",
                        "pair_id": rid,
                        "task": "state_query",
                        "split": split,
                        "suite": suite,
                        "relation": seen_rel,
                        "held_distractor_relation": held_rel,
                        "component": "seen_with_held_distractor",
                        "arg_order": [a, b],
                        "surface_order": cause_order,
                        "held_distractor_surface_order": held_order,
                        "template": cause_t,
                        "held_distractor_template": held_t,
                        "voice": cause_voice,
                        "held_distractor_voice": held_voice,
                        "static_slot": static_slot,
                        "changed_object": cause_obj,
                        "held_distractor_object": held_obj,
                        "static_object": static_obj,
                        "query_kind": query_kind,
                        "candidate": cand,
                        "candidate_slot": 0 if cand == a else 1,
                        "correct_slot": 0 if true_owner == a else 1,
                        "premise": premise,
                        "hypothesis": f"After the event, {cand} had the {obj}.",
                        "label": bool(cand == true_owner),
                        "orientation_dependency": dep,
                        "global_swap_changes_label": False,
                        "cause_relation": seen_rel,
                        "cause_event": cause_ev,
                        "held_distractor_event": held_ev,
                        "static_owner": static,
                        "supervised_changed_owner": new,
                        "inverted_bridge_label": False,
                    })
    return rows


def unsup_row(row_id: str, rel: str, a: str, b: str, obj: str, split: str, suite: str, voice: int) -> Dict[str, Any]:
    ev, so, tmpl, vv = event(rel, a, b, obj, voice)
    return {
        "row_id": row_id,
        "task": "unsupervised_event_text",
        "split": split,
        "suite": suite,
        "relation": rel,
        "component": REL[rel].component,
        "arg_order": [a, b],
        "surface_order": so,
        "template": tmpl,
        "voice": vv,
        "object": obj,
        "text": ev,
        "orientation_dependency": "none_unsupervised",
        "global_swap_changes_label": False,
    }


def common_seen_train(n_worlds: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n_worlds):
        rel = SEEN_KEYS[i % len(SEEN_KEYS)]
        a, b = choose_pair(TRAIN_NAMES, i)
        rows.extend(state_orbit(f"common_seen_{i:04d}", rel, a, b,
                                TRAIN_CHANGED[i % len(TRAIN_CHANGED)],
                                TRAIN_STATIC[(i * 3) % len(TRAIN_STATIC)],
                                "train", "common_seen_coordinate"))
    return rows


def heldheld_train(n_per_edge: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Balanced connected graph: two same-orientation edges and two opposite-orientation edges.
    # This preserves exactly the global Z2 ambiguity but removes the research 2:1
    # edge-type imbalance that let latent/name-order rules score above chance.
    edges = [("h0_dax", "h1_mep"), ("h2_norp", "h3_ziv"), ("h1_mep", "h2_norp"), ("h0_dax", "h2_norp")]
    k = 0
    for ei, (r1, r2) in enumerate(edges):
        for j in range(n_per_edge):
            a, b = choose_pair(TRAIN_NAMES, k)
            rows.extend(comparison_orbit(f"train_hh_path_e{ei}_{j:04d}", r1, r2, a, b,
                                         TRAIN_CHANGED[k % len(TRAIN_CHANGED)],
                                         same=(j % 2 == 0), split="train", suite="heldheld_path_train"))
            k += 1
    return rows


def held_event_exposure(n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        rel = HELD_KEYS[i % len(HELD_KEYS)]
        a, b = choose_pair(TRAIN_NAMES, i + 2000)
        for voice in [0, 1]:
            rows.append(unsup_row(f"train_held_exposure_{i:04d}_v{voice}", rel, a, b,
                                  TRAIN_CHANGED[(i + 5) % len(TRAIN_CHANGED)], "train", "held_event_exposure", voice))
    return rows


def bridge_state(rel_keys: Sequence[str], k_per_rel: int, assignment: Dict[str, int], suite: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    for rel in rel_keys:
        for j in range(k_per_rel):
            a, b = choose_pair(TRAIN_NAMES, idx + 4000)
            rows.extend(state_orbit(f"train_{suite}_{rel}_{j:04d}", rel, a, b,
                                    TRAIN_CHANGED[(idx + 9) % len(TRAIN_CHANGED)],
                                    TRAIN_STATIC[(idx + 11) % len(TRAIN_STATIC)],
                                    "train", suite, assignment))
            idx += 1
    return rows


def bridge_mixed(rel_keys: Sequence[str], k_per_rel: int, suite: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    for rel in rel_keys:
        for j in range(k_per_rel):
            for seen in SEEN_KEYS:
                a, b = choose_pair(TRAIN_NAMES, idx + 5000)
                rows.extend(comparison_orbit(f"train_{suite}_{rel}_{seen}_{j:04d}", rel, seen, a, b,
                                             TRAIN_CHANGED[(idx + 13) % len(TRAIN_CHANGED)],
                                             same=(j % 2 == 0), split="train", suite=suite))
                idx += 1
    return rows


def build_arms(n_hh: int, k_bridge: int) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    hh = heldheld_train(n_hh)
    unsup = held_event_exposure(n_hh * len(HELD_KEYS))
    anchor_rels = ["h0_dax", "h2_norp"]
    dec: List[Dict[str, Any]] = []
    idx = 0
    for rel in anchor_rels:
        for j in range(k_bridge):
            for seen in SEEN_KEYS:
                a, b = choose_pair(TRAIN_NAMES, idx + 6000)
                dec.extend(decoupled_orbit(f"train_decoupled_{rel}_{seen}_{j:04d}", rel, seen, a, b,
                                           TRAIN_CHANGED[(idx + 1) % len(TRAIN_CHANGED)],
                                           TRAIN_CHANGED[(idx + 7) % len(TRAIN_CHANGED)],
                                           TRAIN_STATIC[(idx + 2) % len(TRAIN_STATIC)],
                                           "train", "neutral_decoupled_seen_with_held_distractor"))
                idx += 1
    return {
        "exposure_only": {"supervised": [], "unsupervised": unsup},
        "heldheld_only": {"supervised": hh, "unsupervised": unsup},
        "aligned_state_bridge": {"supervised": hh + bridge_state(anchor_rels, k_bridge, TRUE_ASSIGNMENT, "aligned_sparse_state_bridge"), "unsupervised": unsup},
        "inverted_state_bridge": {"supervised": hh + bridge_state(anchor_rels, k_bridge, INVERTED_ASSIGNMENT, "inverted_sparse_state_bridge"), "unsupervised": unsup},
        "neutral_decoupled": {"supervised": hh + dec, "unsupervised": unsup},
        "mixed_event_bridge": {"supervised": hh + bridge_mixed(anchor_rels, k_bridge, "mixed_event_sparse_bridge"), "unsupervised": unsup},
    }


def eval_heldheld_closure(n_per_pair: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Balanced held-held readout: two same-orientation and two opposite-orientation
    # relation pairs under disjoint eval names/objects.  The filename remains
    # compatible with research scripts, but the scientific role is now held-held
    # consistency on a surface-balanced role orbit rather than only unseen-edge closure.
    pairs = [("h0_dax", "h1_mep"), ("h2_norp", "h3_ziv"), ("h3_ziv", "h0_dax"), ("h1_mep", "h3_ziv")]
    k = 0
    for pi, (r1, r2) in enumerate(pairs):
        for j in range(n_per_pair):
            a, b = choose_pair(EVAL_NAMES, k)
            rows.extend(comparison_orbit(f"eval_hh_closure_p{pi}_{j:04d}", r1, r2, a, b,
                                         EVAL_CHANGED[k % len(EVAL_CHANGED)], same=(j % 2 == 0),
                                         split="eval", suite="heldheld_unseen_edge_closure"))
            k += 1
    return rows


def eval_mixed(n_per_rel_seen: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    k = 0
    for h in HELD_KEYS:
        for s in SEEN_KEYS:
            for j in range(n_per_rel_seen):
                a, b = choose_pair(EVAL_NAMES, k)
                same = (j % 2 == 0)
                rows.extend(comparison_orbit(f"eval_mixed_hs_{h}_{s}_{j:04d}", h, s, a, b,
                                             EVAL_CHANGED[k % len(EVAL_CHANGED)], same=same,
                                             split="eval", suite="mixed_held_seen_orientation"))
                rows.extend(comparison_orbit(f"eval_mixed_sh_{s}_{h}_{j:04d}", s, h, a, b,
                                             EVAL_CHANGED[(k + 5) % len(EVAL_CHANGED)], same=same,
                                             split="eval", suite="mixed_held_seen_orientation"))
                k += 1
    return rows


def eval_state(n_per_rel: int, suite: str, start: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    k = start
    for h in HELD_KEYS:
        for j in range(n_per_rel):
            a, b = choose_pair(EVAL_NAMES, k)
            rows.extend(state_orbit(f"eval_{suite}_{h}_{j:04d}", h, a, b,
                                    EVAL_CHANGED[k % len(EVAL_CHANGED)],
                                    EVAL_STATIC[(k * 3) % len(EVAL_STATIC)],
                                    "eval", suite, TRUE_ASSIGNMENT))
            k += 1
    return rows


def eval_name_permutation(n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        h = HELD_KEYS[i % len(HELD_KEYS)]
        a, b = choose_pair(EVAL_NAMES, i + 3000)
        rows.extend(state_orbit(f"eval_perm_base_{i:04d}", h, a, b,
                                EVAL_CHANGED[(i + 4) % len(EVAL_CHANGED)],
                                EVAL_STATIC[(i + 5) % len(EVAL_STATIC)],
                                "eval", "name_permutation_base", TRUE_ASSIGNMENT))
        rows.extend(state_orbit(f"eval_perm_swap_{i:04d}", h, b, a,
                                EVAL_CHANGED[(i + 4) % len(EVAL_CHANGED)],
                                EVAL_STATIC[(i + 9) % len(EVAL_STATIC)],
                                "eval", "name_permutation_swapped", TRUE_ASSIGNMENT))
    return rows


def build_eval(n_eval: int) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "heldheld_unseen_edge_closure": eval_heldheld_closure(n_eval),
        "mixed_held_seen_orientation": eval_mixed(n_eval),
        "paired_state_conservation": eval_state(n_eval, "paired_state_conservation"),
        "cross_template_state_readout": eval_state(max(4, n_eval // 2), "cross_template_state_readout", 1000),
        "name_permutation_counterfactual": eval_name_permutation(max(8, n_eval // 2)),
    }


def predict_label(row: Dict[str, Any], assignment: Dict[str, int]) -> bool:
    if row["task"] == "relation_comparison":
        r1, r2 = row["relation1"], row["relation2"]
        a1, b1 = row["arg_order1"]
        a2, b2 = row["arg_order2"]
        return final_owner(r1, a1, b1, assignment) == final_owner(r2, a2, b2, assignment)
    if row["task"] == "state_query":
        if row["query_kind"] == "unchanged":
            return row["candidate"] == row["static_owner"]
        cause = row.get("cause_relation", row.get("relation"))
        if cause in SEEN:
            return row["candidate_slot"] == SEEN[cause].final_slot
        return row["candidate_slot"] == assignment[cause]
    raise ValueError(row["task"])


def satisfying_assignments(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    labeled = [r for r in rows if r.get("task") in {"relation_comparison", "state_query"}]
    sats: List[Dict[str, int]] = []
    for bits in itertools.product([0, 1], repeat=len(HELD_KEYS)):
        ass = dict(zip(HELD_KEYS, bits))
        if all(predict_label(r, ass) == bool(r["label"]) for r in labeled):
            sats.append(ass)
    return sats


def true_state_rows(rows: Sequence[Dict[str, Any]], query_kind: str | None = None) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("task") == "state_query" and r.get("label") is True and (query_kind is None or r.get("query_kind") == query_kind)]


def frac(xs: Iterable[bool]) -> float | None:
    vals = list(xs)
    if not vals:
        return None
    return sum(vals) / len(vals)


def label_balance(rows: Sequence[Dict[str, Any]]) -> Dict[str, float | int]:
    labs = [bool(r["label"]) for r in rows if "label" in r]
    if not labs:
        return {"n_labeled": 0}
    return {"n_labeled": len(labs), "true_frac": sum(labs) / len(labs), "false_frac": 1 - sum(labs) / len(labs)}


def state_balance(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    trs = true_state_rows(rows)
    changed = true_state_rows(rows, "changed")
    unchanged = true_state_rows(rows, "unchanged")

    def slot_frac(rs: Sequence[Dict[str, Any]]) -> float | None:
        return frac(r["correct_slot"] == 0 for r in rs)

    def surface_frac(rs: Sequence[Dict[str, Any]]) -> float | None:
        vals = []
        for r in rs:
            so = r.get("surface_order") or []
            vals.append(bool(so and so[0] == r["candidate"]))
        return frac(vals)

    by_name = Counter(r["candidate"] for r in trs)
    return {
        "n_true_state": len(trs),
        "slot0_true_all": slot_frac(trs),
        "slot0_true_changed": slot_frac(changed),
        "slot0_true_unchanged": slot_frac(unchanged),
        "surface_first_true_changed": surface_frac(changed),
        "surface_first_true_unchanged": surface_frac(unchanged),
        "true_candidate_name_minmax": (min(by_name.values()), max(by_name.values())) if by_name else None,
    }


def comparison_order_features(r: Dict[str, Any]) -> Dict[str, bool]:
    so1 = r.get("surface_order1") or []
    so2 = r.get("surface_order2") or []
    ao1 = r.get("arg_order1") or []
    ao2 = r.get("arg_order2") or []
    return {
        "surface_first_same": bool(so1 and so2 and so1[0] == so2[0]),
        "surface_second_same": bool(len(so1) > 1 and len(so2) > 1 and so1[1] == so2[1]),
        "surface_order_exact_same": so1 == so2,
        "surface_order_reversed": len(so1) == 2 and len(so2) == 2 and so1[0] == so2[1] and so1[1] == so2[0],
        "latent_arg_order_same": ao1 == ao2,
        "latent_arg_order_reversed": len(ao1) == 2 and len(ao2) == 2 and ao1[0] == ao2[1] and ao1[1] == ao2[0],
        "voice_same": r.get("voice1") == r.get("voice2"),
        "held_first": r.get("component1") == "held",
    }


def state_order_features(r: Dict[str, Any]) -> Dict[str, bool]:
    so = r.get("surface_order") or []
    cand = r.get("candidate")
    return {
        "candidate_surface_first": bool(so and so[0] == cand),
        "candidate_surface_second": bool(len(so) > 1 and so[1] == cand),
        "candidate_latent_slot0": r.get("candidate_slot") == 0,
        "candidate_latent_slot1": r.get("candidate_slot") == 1,
        "active_voice": r.get("voice") == "active",
        "passive_or_fronted_voice": r.get("voice") != "active",
    }


def best_rule_accuracy(rows: Sequence[Dict[str, Any]], task: str, features: Sequence[str] | None = None) -> Dict[str, Any]:
    labeled = [r for r in rows if r.get("task") == task and "label" in r]
    if not labeled:
        return {"n": 0}
    feat_fun = comparison_order_features if task == "relation_comparison" else state_order_features
    feat_rows = [feat_fun(r) for r in labeled]
    if features is None:
        features = list(feat_rows[0].keys())
    out = {"n": len(labeled)}
    best_name, best_acc, best_pol = None, -1.0, None
    for f in features:
        vals = [bool(fr.get(f)) for fr in feat_rows]
        labs = [bool(r["label"]) for r in labeled]
        acc_pos = sum(v == y for v, y in zip(vals, labs)) / len(labs)
        acc_neg = sum((not v) == y for v, y in zip(vals, labs)) / len(labs)
        acc = max(acc_pos, acc_neg)
        out[f"rule_{f}"] = acc
        if acc > best_acc:
            best_name, best_acc, best_pol = f, acc, ("as_is" if acc_pos >= acc_neg else "inverted")
    out["best_rule"] = best_name
    out["best_accuracy"] = best_acc
    out["best_polarity"] = best_pol
    return out


def order_rule_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    comp = [r for r in rows if r.get("task") == "relation_comparison"]
    state_chg = [r for r in rows if r.get("task") == "state_query" and r.get("query_kind") == "changed"]
    state_unchg = [r for r in rows if r.get("task") == "state_query" and r.get("query_kind") == "unchanged"]
    return {
        "comparison": best_rule_accuracy(comp, "relation_comparison"),
        "state_changed": best_rule_accuracy(state_chg, "state_query"),
        "state_unchanged": best_rule_accuracy(state_unchg, "state_query"),
    }


def held_surface_leak(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    texts: List[str] = []
    for r in rows:
        if r.get("task") == "relation_comparison":
            if r.get("relation1") in HELD and "event1" in r:
                texts.append(str(r["event1"]))
            if r.get("relation2") in HELD and "event2" in r:
                texts.append(str(r["event2"]))
        elif r.get("task") == "state_query":
            if r.get("cause_relation", r.get("relation")) in HELD and "cause_event" in r:
                texts.append(str(r["cause_event"]))
            if r.get("held_distractor_relation") in HELD and "held_distractor_event" in r:
                texts.append(str(r["held_distractor_event"]))
        elif r.get("task") == "unsupervised_event_text" and r.get("relation") in HELD:
            texts.append(str(r.get("text", "")))
    pat = re.compile(r"\b(to|from|gave|give|given|handed|received|got|bequeathed|ceded|acquired|inherited)\b", re.I)
    bad = [t for t in texts if pat.search(t)]
    return {"held_related_texts_scanned": len(texts), "known_transfer_or_to_from_hits": len(bad), "example_hits": bad[:5]}


def pair_state_shortcuts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("task") == "state_query" and r.get("label") is True:
            by_pair[r["pair_id"]][r["query_kind"]] = r
    same_owner = [d["changed"]["candidate"] == d["unchanged"]["candidate"] for d in by_pair.values() if "changed" in d and "unchanged" in d]
    true_rows = true_state_rows(rows)
    return {
        "n_true": len(true_rows),
        "always_slot0_accuracy": frac(r["correct_slot"] == 0 for r in true_rows),
        "always_slot1_accuracy": frac(r["correct_slot"] == 1 for r in true_rows),
        "changed_static_same_owner_frac": frac(same_owner),
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def audit_like(common_seen: Sequence[Dict[str, Any]], arms: Dict[str, Dict[str, List[Dict[str, Any]]]], evals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    arm_summaries: Dict[str, Any] = {}
    for name, blocks in arms.items():
        sup = blocks["supervised"]
        sats = satisfying_assignments(sup)
        arm_summaries[name] = {
            "supervised_rows": len(sup),
            "unsupervised_rows": len(blocks["unsupervised"]),
            "supervised_words": sum(row_words(r) for r in sup),
            "unsupervised_words": sum(row_words(r) for r in blocks["unsupervised"]),
            "label_balance": label_balance(sup),
            "state_balance": state_balance(sup),
            "order_rule_report": order_rule_report(sup),
            "orientation_dependency_counts": dict(Counter(r.get("orientation_dependency") for r in sup)),
            "formal_satisfying_assignment_count": len(sats),
            "formal_satisfying_assignments": sats,
            "true_assignment_satisfies": TRUE_ASSIGNMENT in sats,
            "inverted_assignment_satisfies": INVERTED_ASSIGNMENT in sats,
        }
    eval_summaries: Dict[str, Any] = {}
    for name, rows in evals.items():
        eval_summaries[name] = {
            "rows": len(rows),
            "words": sum(row_words(r) for r in rows),
            "label_balance": label_balance(rows),
            "state_balance": state_balance(rows),
            "order_rule_report": order_rule_report(rows),
            "orientation_dependency_counts": dict(Counter(r.get("orientation_dependency") for r in rows)),
            "pair_state_shortcuts": pair_state_shortcuts(rows),
        }
    all_rows = list(common_seen)
    for b in arms.values():
        all_rows += b["supervised"] + b["unsupervised"]
    for rs in evals.values():
        all_rows += rs
    mixed_best = eval_summaries["mixed_held_seen_orientation"]["order_rule_report"]["comparison"].get("best_accuracy")
    state_best = eval_summaries["paired_state_conservation"]["order_rule_report"]["state_changed"].get("best_accuracy")
    return {
        "status": "EQUIVARIANT_SUBSTRATE_COMPLETE",
        "repair_intent": "active_passive_reusable_argument_slots_with_name_order_and_voice_orbits_balanced",
        "true_assignment": TRUE_ASSIGNMENT,
        "inverted_assignment": INVERTED_ASSIGNMENT,
        "common_seen_coordinate": {
            "rows": len(common_seen),
            "words": sum(row_words(r) for r in common_seen),
            "label_balance": label_balance(common_seen),
            "state_balance": state_balance(common_seen),
            "order_rule_report": order_rule_report(common_seen),
        },
        "arms": arm_summaries,
        "eval_sets": eval_summaries,
        "global_checks": {
            "train_eval_names_disjoint": set(TRAIN_NAMES).isdisjoint(EVAL_NAMES),
            "train_eval_changed_objects_disjoint": set(TRAIN_CHANGED).isdisjoint(EVAL_CHANGED),
            "train_eval_static_objects_disjoint": set(TRAIN_STATIC).isdisjoint(EVAL_STATIC),
            "changed_static_objects_disjoint_within_train": set(TRAIN_CHANGED).isdisjoint(TRAIN_STATIC),
            "changed_static_objects_disjoint_within_eval": set(EVAL_CHANGED).isdisjoint(EVAL_STATIC),
            "held_surface_leak_scan": held_surface_leak(all_rows),
            "heldheld_only_has_exactly_global_Z2": arm_summaries["heldheld_only"]["formal_satisfying_assignment_count"] == 2,
            "aligned_state_identifies_true": arm_summaries["aligned_state_bridge"]["formal_satisfying_assignments"] == [TRUE_ASSIGNMENT],
            "inverted_state_identifies_inverted": arm_summaries["inverted_state_bridge"]["formal_satisfying_assignments"] == [INVERTED_ASSIGNMENT],
            "neutral_decoupled_preserves_Z2": arm_summaries["neutral_decoupled"]["formal_satisfying_assignment_count"] == 2,
            "mixed_event_identifies_true": arm_summaries["mixed_event_bridge"]["formal_satisfying_assignments"] == [TRUE_ASSIGNMENT],
            "mixed_eval_order_rules_at_chance": mixed_best is not None and mixed_best <= 0.505,
            "changed_state_eval_order_rules_at_chance": state_best is not None and state_best <= 0.505,
        },
    }


def fmt(x: Any) -> str:
    if x is None:
        return "nan"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_summary(path: Path, rep: Dict[str, Any], files: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("# research equivariant symmetry-identification substrate")
    lines.append("")
    lines.append("This CPU/file-only construction repairs the research 0.750 order solution by replacing order-free conjoined nonce events with active/passive syntactic role orbits, a balanced held-held relation graph, and static-owner counterbalancing for unaffected facts. The learner still sees a reusable argument interface, but visible name order, event side, voice, relation-pair orientation type, changed-owner position, static-owner position, and candidate position are balanced against labels.")
    lines.append("")
    lines.append("## Formal assignments")
    lines.append("")
    lines.append(f"- true assignment: `{rep['true_assignment']}`")
    lines.append(f"- inverted assignment: `{rep['inverted_assignment']}`")
    lines.append("")
    lines.append("| arm | supervised rows | satisfying assignments | true? | inverted? | best comparison order rule | best changed-state order rule | slot0 true changed | surface-first true changed |")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|---:|")
    for arm, s in rep["arms"].items():
        comp_best = s["order_rule_report"]["comparison"].get("best_accuracy")
        chg_best = s["order_rule_report"]["state_changed"].get("best_accuracy")
        sb = s["state_balance"]
        lines.append(f"| {arm} | {s['supervised_rows']} | {s['formal_satisfying_assignment_count']} | {s['true_assignment_satisfies']} | {s['inverted_assignment_satisfies']} | {fmt(comp_best)} | {fmt(chg_best)} | {fmt(sb.get('slot0_true_changed'))} | {fmt(sb.get('surface_first_true_changed'))} |")
    lines.append("")
    lines.append("## Evaluation surface readout")
    lines.append("")
    lines.append("| suite | rows | true frac | best comparison order rule | best changed-state order rule | slot0 true changed | surface-first true changed | same changed/static owner |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for suite, s in rep["eval_sets"].items():
        lb = s["label_balance"]
        sb = s["state_balance"]
        ps = s["pair_state_shortcuts"]
        comp_best = s["order_rule_report"]["comparison"].get("best_accuracy")
        chg_best = s["order_rule_report"]["state_changed"].get("best_accuracy")
        lines.append(f"| {suite} | {s['rows']} | {fmt(lb.get('true_frac'))} | {fmt(comp_best)} | {fmt(chg_best)} | {fmt(sb.get('slot0_true_changed'))} | {fmt(sb.get('surface_first_true_changed'))} | {fmt(ps.get('changed_static_same_owner_frac'))} |")
    lines.append("")
    lines.append("## Global checks")
    lines.append("")
    for k, v in rep["global_checks"].items():
        if k == "held_surface_leak_scan":
            lines.append(f"- held surface transfer-word scan: {v['known_transfer_or_to_from_hits']} hits across {v['held_related_texts_scanned']} held-related fields")
            if v.get("example_hits"):
                for e in v["example_hits"]:
                    lines.append(f"  - {e}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for k, v in files.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--common-seen-worlds", type=int, default=64)
    ap.add_argument("--heldheld-per-edge", type=int, default=12)
    ap.add_argument("--bridge-k", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=8)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    common = common_seen_train(args.common_seen_worlds)
    arms = build_arms(args.heldheld_per_edge, args.bridge_k)
    evals = build_eval(args.eval_n)
    files: Dict[str, str] = {}

    p = out / "common_seen_train.jsonl"
    write_jsonl(p, common)
    files["common_seen_train_jsonl"] = str(p.relative_to(PROJECT_ROOT))
    for arm, blocks in arms.items():
        ps = out / "arms" / arm / "train_supervised.jsonl"
        pu = out / "arms" / arm / "train_unsup_text.jsonl"
        write_jsonl(ps, blocks["supervised"])
        write_jsonl(pu, blocks["unsupervised"])
        files[f"arm_{arm}_supervised_jsonl"] = str(ps.relative_to(PROJECT_ROOT))
        files[f"arm_{arm}_unsup_jsonl"] = str(pu.relative_to(PROJECT_ROOT))
    for suite, rows in evals.items():
        pe = out / "eval" / f"{suite}.jsonl"
        write_jsonl(pe, rows)
        files[f"eval_{suite}_jsonl"] = str(pe.relative_to(PROJECT_ROOT))

    rep = audit_like(common, arms, evals)
    pj = out / "equivariant_substrate_report.json"
    pj.write_text(json.dumps(rep, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files["report_json"] = str(pj.relative_to(PROJECT_ROOT))

    arm_csv = []
    for arm, s in rep["arms"].items():
        row = {
            "arm": arm,
            "supervised_rows": s["supervised_rows"],
            "supervised_words": s["supervised_words"],
            "formal_satisfying_assignment_count": s["formal_satisfying_assignment_count"],
            "true_assignment_satisfies": s["true_assignment_satisfies"],
            "inverted_assignment_satisfies": s["inverted_assignment_satisfies"],
            "best_comparison_order_rule": s["order_rule_report"]["comparison"].get("best_accuracy"),
            "best_changed_state_order_rule": s["order_rule_report"]["state_changed"].get("best_accuracy"),
        }
        row.update({f"label_{k}": v for k, v in s["label_balance"].items()})
        row.update({f"state_{k}": v for k, v in s["state_balance"].items()})
        arm_csv.append(row)
    pc = out / "arm_summary.csv"
    write_csv(pc, arm_csv)
    files["arm_summary_csv"] = str(pc.relative_to(PROJECT_ROOT))

    eval_csv = []
    for suite, s in rep["eval_sets"].items():
        row = {
            "suite": suite,
            "rows": s["rows"],
            "words": s["words"],
            "best_comparison_order_rule": s["order_rule_report"]["comparison"].get("best_accuracy"),
            "best_changed_state_order_rule": s["order_rule_report"]["state_changed"].get("best_accuracy"),
        }
        row.update({f"label_{k}": v for k, v in s["label_balance"].items()})
        row.update({f"state_{k}": v for k, v in s["state_balance"].items()})
        row.update({f"shortcut_{k}": v for k, v in s["pair_state_shortcuts"].items()})
        eval_csv.append(row)
    pec = out / "eval_summary.csv"
    write_csv(pec, eval_csv)
    files["eval_summary_csv"] = str(pec.relative_to(PROJECT_ROOT))

    ps = out / "equivariant_substrate_summary.md"
    write_summary(ps, rep, files)
    files["summary_md"] = str(ps.relative_to(PROJECT_ROOT))

    print(json.dumps({
        "status": rep["status"],
        "out": str(out.relative_to(PROJECT_ROOT)),
        "summary_md": files["summary_md"],
        "formal_counts": {k: v["formal_satisfying_assignment_count"] for k, v in rep["arms"].items()},
        "global_checks": rep["global_checks"],
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
