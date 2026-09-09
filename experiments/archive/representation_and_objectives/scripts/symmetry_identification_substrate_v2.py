#!/usr/bin/env python3
"""research v2: repaired symmetry-identification substrate.

CPU/file-only Explore asset.  It rebuilds the research prototype after independent review
identified lexical, decoupling, slot-balance, and shortcut flaws.

No model loading, no training, no official evaluation, no GPU, no upload.
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
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/symmetry_identification_substrate_v2')


@dataclass(frozen=True)
class Rel:
    key: str
    component: str  # held or seen
    final_slot: int
    lex: str
    family: str


HELD: Dict[str, Rel] = {
    "h0_dax": Rel("h0_dax", "held", 1, "dax", "nonce_A"),
    "h1_mep": Rel("h1_mep", "held", 1, "mep", "nonce_A"),
    "h2_norp": Rel("h2_norp", "held", 0, "norp", "nonce_B"),
    "h3_ziv": Rel("h3_ziv", "held", 0, "ziv", "nonce_B"),
}
SEEN: Dict[str, Rel] = {
    "s_give": Rel("s_give", "seen", 1, "give", "source_subject_to"),
    "s_receive": Rel("s_receive", "seen", 0, "receive", "recipient_subject_from"),
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
    # Complete-ish balanced pair cycling.  Each name appears in both latent slots.
    n = len(pool)
    a = pool[i % n]
    b = pool[(i * 5 + 3) % n]
    if a == b:
        b = pool[(i + 7) % n]
    if (i // n) % 2:
        a, b = b, a
    return a, b


def held_event(rel_key: str, a: str, b: str, obj: str, variant: int) -> Tuple[str, List[str], str]:
    """Return text, surface order, template id.

    The held relation text deliberately avoids to/from/dative/known-transfer
    verbs.  The nonce lexeme is a lexical relation component, not an address or
    coordinate label.  Surface order is balanced across variants.
    """
    lex = HELD[rel_key].lex
    # Symmetric templates: both participants appear in conjoined or
    # equivalent positions so that surface order does not predict who
    # ends up with the object.  This removes the pretrained agent-patient
    # frame cue, forcing the model to rely on bridge evidence.
    templates = [
        ("During the {obj} episode, {a} and {b} had a {lex}ing event.", ["a", "b"], "conjoined_ab"),
        ("During the {obj} episode, {b} and {a} had a {lex}ing event.", ["b", "a"], "conjoined_ba"),
        ("{a} and {b} completed a {lex} event involving the {obj}.", ["a", "b"], "completed_ab"),
        ("{b} and {a} completed a {lex} event involving the {obj}.", ["b", "a"], "completed_ba"),
    ]
    tmpl, order_keys, tid = templates[variant % len(templates)]
    text = tmpl.format(a=a, b=b, obj=obj, lex=lex)
    order = [a if x == "a" else b for x in order_keys]
    return text, order, tid


def seen_event(rel_key: str, a: str, b: str, obj: str, variant: int) -> Tuple[str, List[str], str]:
    templates = {
        "s_give": [
            ("{a} gave the {obj} to {b}.", ["a", "b"], "give_to"),
            ("{a} handed {b} the {obj}.", ["a", "b"], "double_object"),
        ],
        "s_receive": [
            ("{a} received the {obj} from {b}.", ["a", "b"], "receive_from"),
            ("From {b}, {a} got the {obj}.", ["b", "a"], "fronted_from"),
        ],
    }
    tmpl, order_keys, tid = templates[rel_key][variant % len(templates[rel_key])]
    text = tmpl.format(a=a, b=b, obj=obj)
    order = [a if x == "a" else b for x in order_keys]
    return text, order, tid


def event(rel_key: str, a: str, b: str, obj: str, variant: int) -> Tuple[str, List[str], str]:
    if REL[rel_key].component == "held":
        return held_event(rel_key, a, b, obj, variant)
    return seen_event(rel_key, a, b, obj, variant)


def slot(rel_key: str, assignment: Dict[str, int]) -> int:
    if REL[rel_key].component == "held":
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


def row_words(r: Dict) -> int:
    return sum(words(str(r.get(k, ""))) for k in ("text", "premise", "hypothesis", "event1", "event2", "held_distractor_event", "cause_event"))


def comparison_row(row_id: str, rel1: str, rel2: str, a: str, b: str, obj: str, same: bool, split: str, suite: str, variant: int) -> Dict:
    s1 = slot(rel1, TRUE_ASSIGNMENT)
    s2 = slot(rel2, TRUE_ASSIGNMENT)
    # If same, choose rel2's argument order so final owner equals rel1's final owner.
    if same:
        a2, b2 = (a, b) if s1 == s2 else (b, a)
    else:
        a2, b2 = (b, a) if s1 == s2 else (a, b)
    e1, o1, t1 = event(rel1, a, b, obj, variant)
    e2, o2, t2 = event(rel2, a2, b2, obj, variant + 1)
    assert (final_owner(rel1, a, b) == final_owner(rel2, a2, b2)) == same
    dep = "held_seen_breaks_symmetry" if (REL[rel1].component, REL[rel2].component).count("held") == 1 else "heldheld_invariant" if REL[rel1].component == REL[rel2].component == "held" else "seen_seen_fixed"
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
        "object": obj,
        "event1": e1,
        "event2": e2,
        "text": f"Event A: {e1} Event B: {e2} Did Event A and Event B leave the {obj} with the same person?",
        "label": bool(same),
        "orientation_dependency": dep,
        "global_swap_changes_label": dep == "held_seen_breaks_symmetry",
    }


def state_rows(row_id: str, rel_key: str, a: str, b: str, changed_obj: str, static_obj: str, split: str, suite: str, variant: int, assignment_for_labels: Dict[str, int] = TRUE_ASSIGNMENT) -> List[Dict]:
    # Labels for changed object are determined by assignment_for_labels.  This is
    # how inverted bridges impose the opposite global orientation.
    new = final_owner(rel_key, a, b, assignment_for_labels)
    old = owner_from_slot(1 - slot(rel_key, assignment_for_labels), a, b)
    static = (a, b)[(variant // 2) % 2]
    ev, surface_order, tmpl = event(rel_key, a, b, changed_obj, variant)
    premise = f"At first, {old} had the {changed_obj}, and {static} had the {static_obj}. The event was this: {ev}"
    rows: List[Dict] = []
    for query_kind, obj, true_owner, dep in [
        ("changed", changed_obj, new, "held_seen_breaks_symmetry" if REL[rel_key].component == "held" else "seen_seen_fixed"),
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


def decoupled_rows(row_id: str, held_rel: str, seen_rel: str, a: str, b: str, held_obj: str, cause_obj: str, static_obj: str, split: str, suite: str, variant: int) -> List[Dict]:
    # Held text is present as a distractor, but all labels concern the separate
    # seen transfer.  The labeled function is invariant under any held-role swap.
    held_ev, held_order, held_t = held_event(held_rel, a, b, held_obj, variant)
    cause_ev, cause_order, cause_t = seen_event(seen_rel, a, b, cause_obj, variant)
    new = final_owner(seen_rel, a, b)
    old = old_owner(seen_rel, a, b)
    static = (a, b)[(variant + 1) % 2]
    premise = (
        f"A separate note described this unrelated episode: {held_ev} "
        f"For the tested object, at first {old} had the {cause_obj}, and {static} had the {static_obj}. "
        f"The tested event was this: {cause_ev}"
    )
    rows: List[Dict] = []
    for query_kind, obj, true_owner, dep in [
        ("changed", cause_obj, new, "seen_seen_fixed"),
        ("unchanged", static_obj, static, "unaffected_fact_should_be_preserved"),
    ]:
        for cand in [a, b]:
            rows.append({
                "row_id": f"{row_id}_{query_kind}_{0 if cand == a else 1}",
                "pair_id": row_id,
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


def unsup_row(row_id: str, rel: str, a: str, b: str, obj: str, split: str, suite: str, variant: int) -> Dict:
    ev, so, tmpl = event(rel, a, b, obj, variant)
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
        "object": obj,
        "text": ev,
        "orientation_dependency": "none_unsupervised",
        "global_swap_changes_label": False,
    }


def common_seen_train(n_worlds: int) -> List[Dict]:
    rows: List[Dict] = []
    for i in range(n_worlds):
        rel = SEEN_KEYS[i % len(SEEN_KEYS)]
        a, b = choose_pair(TRAIN_NAMES, i)
        rows.extend(state_rows(f"common_seen_{i:04d}", rel, a, b, TRAIN_CHANGED[i % len(TRAIN_CHANGED)], TRAIN_STATIC[(i * 3) % len(TRAIN_STATIC)], "train", "common_seen_coordinate", i))
    return rows


def heldheld_train(n_per_edge: int) -> List[Dict]:
    rows: List[Dict] = []
    edges = [("h0_dax", "h1_mep"), ("h1_mep", "h2_norp"), ("h2_norp", "h3_ziv")]
    k = 0
    for ei, (r1, r2) in enumerate(edges):
        for j in range(n_per_edge):
            a, b = choose_pair(TRAIN_NAMES, k)
            rows.append(comparison_row(f"train_hh_path_e{ei}_{j:04d}", r1, r2, a, b, TRAIN_CHANGED[k % len(TRAIN_CHANGED)], same=(j % 2 == 0), split="train", suite="heldheld_path_train", variant=k))
            k += 1
    return rows


def held_event_exposure(n: int) -> List[Dict]:
    rows = []
    for i in range(n):
        rel = HELD_KEYS[i % len(HELD_KEYS)]
        a, b = choose_pair(TRAIN_NAMES, i + 2000)
        rows.append(unsup_row(f"train_held_exposure_{i:04d}", rel, a, b, TRAIN_CHANGED[(i + 5) % len(TRAIN_CHANGED)], "train", "held_event_exposure", i))
    return rows


def bridge_state(rel_keys: Sequence[str], k_per_rel: int, assignment: Dict[str, int], suite: str) -> List[Dict]:
    rows: List[Dict] = []
    idx = 0
    for rel in rel_keys:
        for j in range(k_per_rel):
            a, b = choose_pair(TRAIN_NAMES, idx + 4000)
            rows.extend(state_rows(f"train_{suite}_{rel}_{j:04d}", rel, a, b, TRAIN_CHANGED[(idx + 9) % len(TRAIN_CHANGED)], TRAIN_STATIC[(idx + 11) % len(TRAIN_STATIC)], "train", suite, idx, assignment_for_labels=assignment))
            idx += 1
    return rows


def bridge_mixed(rel_keys: Sequence[str], k_per_rel: int, suite: str, assignment: Dict[str, int] = TRUE_ASSIGNMENT) -> List[Dict]:
    # Mixed held-seen comparison rows.  The assignment argument is used only to
    # choose same/false order; labels are still generated by TRUE_ASSIGNMENT in
    # comparison_row, so inverted mixed bridges are intentionally not used here.
    rows: List[Dict] = []
    idx = 0
    for rel in rel_keys:
        for j in range(k_per_rel):
            seen = SEEN_KEYS[idx % len(SEEN_KEYS)]
            a, b = choose_pair(TRAIN_NAMES, idx + 5000)
            rows.append(comparison_row(f"train_{suite}_{rel}_{j:04d}", rel, seen, a, b, TRAIN_CHANGED[(idx + 13) % len(TRAIN_CHANGED)], same=(j % 2 == 0), split="train", suite=suite, variant=idx))
            idx += 1
    return rows


def build_arms(n_hh: int, k_bridge: int) -> Dict[str, Dict[str, List[Dict]]]:
    hh = heldheld_train(n_hh)
    unsup = held_event_exposure(n_hh * len(HELD_KEYS))
    # Two anchors with opposite final_slot balance orientation rows while still
    # requiring propagation to h1_mep and h3_ziv.
    anchor_rels = ["h0_dax", "h2_norp"]
    dec: List[Dict] = []
    idx = 0
    for rel in anchor_rels:
        for j in range(k_bridge):
            seen = SEEN_KEYS[idx % len(SEEN_KEYS)]
            a, b = choose_pair(TRAIN_NAMES, idx + 6000)
            dec.extend(decoupled_rows(f"train_decoupled_{rel}_{j:04d}", rel, seen, a, b, TRAIN_CHANGED[(idx + 1) % len(TRAIN_CHANGED)], TRAIN_CHANGED[(idx + 7) % len(TRAIN_CHANGED)], TRAIN_STATIC[(idx + 2) % len(TRAIN_STATIC)], "train", "neutral_decoupled_seen_with_held_distractor", idx))
            idx += 1
    return {
        "exposure_only": {"supervised": [], "unsupervised": unsup},
        "heldheld_only": {"supervised": hh, "unsupervised": unsup},
        "aligned_state_bridge": {"supervised": hh + bridge_state(anchor_rels, k_bridge, TRUE_ASSIGNMENT, "aligned_sparse_state_bridge"), "unsupervised": unsup},
        "inverted_state_bridge": {"supervised": hh + bridge_state(anchor_rels, k_bridge, INVERTED_ASSIGNMENT, "inverted_sparse_state_bridge"), "unsupervised": unsup},
        "neutral_decoupled": {"supervised": hh + dec, "unsupervised": unsup},
        "mixed_event_bridge": {"supervised": hh + bridge_mixed(anchor_rels, k_bridge, "mixed_event_sparse_bridge"), "unsupervised": unsup},
    }


def eval_heldheld_closure(n_per_pair: int) -> List[Dict]:
    rows: List[Dict] = []
    pairs = [("h3_ziv", "h0_dax"), ("h0_dax", "h2_norp"), ("h1_mep", "h3_ziv")]
    k = 0
    for pi, (r1, r2) in enumerate(pairs):
        for j in range(n_per_pair):
            a, b = choose_pair(EVAL_NAMES, k)
            rows.append(comparison_row(f"eval_hh_closure_p{pi}_{j:04d}", r1, r2, a, b, EVAL_CHANGED[k % len(EVAL_CHANGED)], same=(j % 2 == 0), split="eval", suite="heldheld_unseen_edge_closure", variant=k))
            k += 1
    return rows


def eval_mixed(n_per_pair: int) -> List[Dict]:
    rows: List[Dict] = []
    k = 0
    for h in HELD_KEYS:
        for s in SEEN_KEYS:
            for j in range(n_per_pair):
                a, b = choose_pair(EVAL_NAMES, k)
                if j % 4 < 2:
                    r1, r2 = h, s
                else:
                    r1, r2 = s, h
                rows.append(comparison_row(f"eval_mixed_{h}_{s}_{j:04d}", r1, r2, a, b, EVAL_CHANGED[k % len(EVAL_CHANGED)], same=(j % 2 == 0), split="eval", suite="mixed_held_seen_orientation", variant=k))
                k += 1
    return rows


def eval_state(n_per_rel: int, suite: str, start: int = 0) -> List[Dict]:
    rows: List[Dict] = []
    k = start
    for h in HELD_KEYS:
        for j in range(n_per_rel):
            a, b = choose_pair(EVAL_NAMES, k)
            rows.extend(state_rows(f"eval_{suite}_{h}_{j:04d}", h, a, b, EVAL_CHANGED[k % len(EVAL_CHANGED)], EVAL_STATIC[(k * 3) % len(EVAL_STATIC)], "eval", suite, k, assignment_for_labels=TRUE_ASSIGNMENT))
            k += 1
    return rows


def eval_name_permutation(n: int) -> List[Dict]:
    rows: List[Dict] = []
    for i in range(n):
        h = HELD_KEYS[i % len(HELD_KEYS)]
        a, b = choose_pair(EVAL_NAMES, i + 3000)
        rows.extend(state_rows(f"eval_perm_base_{i:04d}", h, a, b, EVAL_CHANGED[(i + 4) % len(EVAL_CHANGED)], EVAL_STATIC[(i + 5) % len(EVAL_STATIC)], "eval", "name_permutation_base", i, TRUE_ASSIGNMENT))
        rows.extend(state_rows(f"eval_perm_swap_{i:04d}", h, b, a, EVAL_CHANGED[(i + 4) % len(EVAL_CHANGED)], EVAL_STATIC[(i + 9) % len(EVAL_STATIC)], "eval", "name_permutation_swapped", i, TRUE_ASSIGNMENT))
    return rows


def build_eval(n_eval: int) -> Dict[str, List[Dict]]:
    return {
        "heldheld_unseen_edge_closure": eval_heldheld_closure(n_eval),
        "mixed_held_seen_orientation": eval_mixed(n_eval),
        "paired_state_conservation": eval_state(n_eval, "paired_state_conservation"),
        "cross_template_state_readout": eval_state(max(4, n_eval // 2), "cross_template_state_readout", start=1000),
        "name_permutation_counterfactual": eval_name_permutation(max(8, n_eval // 2)),
    }


def predict_label(row: Dict, assignment: Dict[str, int]) -> bool:
    if row["task"] == "relation_comparison":
        r1, r2 = row["relation1"], row["relation2"]
        a1, b1 = row["arg_order1"]
        a2, b2 = row["arg_order2"]
        return final_owner(r1, a1, b1, assignment) == final_owner(r2, a2, b2, assignment)
    if row["task"] == "state_query":
        if row["query_kind"] == "unchanged":
            return row["candidate"] == row["static_owner"]
        cause = row.get("cause_relation", row.get("relation"))
        if REL[cause].component == "seen":
            return row["candidate_slot"] == REL[cause].final_slot
        return row["candidate_slot"] == assignment[cause]
    raise ValueError(row["task"])


def satisfying_assignments(rows: Sequence[Dict]) -> List[Dict[str, int]]:
    labeled = [r for r in rows if r.get("task") in {"relation_comparison", "state_query"}]
    sats = []
    for bits in itertools.product([0, 1], repeat=len(HELD_KEYS)):
        ass = dict(zip(HELD_KEYS, bits))
        if all(predict_label(r, ass) == bool(r["label"]) for r in labeled):
            sats.append(ass)
    return sats


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def label_balance(rows: Sequence[Dict]) -> Dict[str, float]:
    labs = [bool(r["label"]) for r in rows if "label" in r]
    if not labs:
        return {"n_labeled": 0}
    return {"n_labeled": len(labs), "true_frac": sum(labs) / len(labs), "false_frac": 1 - sum(labs) / len(labs)}


def true_state_rows(rows: Sequence[Dict], query_kind: str | None = None) -> List[Dict]:
    return [r for r in rows if r.get("task") == "state_query" and r.get("label") is True and (query_kind is None or r.get("query_kind") == query_kind)]


def frac(xs: Iterable[bool]) -> float | None:
    xs = list(xs)
    if not xs:
        return None
    return sum(xs) / len(xs)


def state_balance(rows: Sequence[Dict]) -> Dict[str, object]:
    trs = true_state_rows(rows)
    changed = true_state_rows(rows, "changed")
    unchanged = true_state_rows(rows, "unchanged")
    def slot_frac(rs: Sequence[Dict]) -> float | None:
        return frac(r["correct_slot"] == 0 for r in rs)
    def surface_frac(rs: Sequence[Dict]) -> float | None:
        vals = []
        for r in rs:
            cand = r["candidate"]
            so = r.get("surface_order") or []
            vals.append(bool(so and so[0] == cand))
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


def held_surface_leak(rows: Sequence[Dict]) -> Dict[str, object]:
    """Scan only held-relation event text for transfer/to-from leaks.

    Previous version scanned full premises including intentional seen events,
    producing false positives.  This version extracts only the event field
    whose generating relation is held.
    """
    texts = []
    for r in rows:
        task = r.get("task")
        if task == "relation_comparison":
            if r.get("relation1", "") in HELD and "event1" in r:
                texts.append(str(r["event1"]))
            if r.get("relation2", "") in HELD and "event2" in r:
                texts.append(str(r["event2"]))
        elif task == "state_query":
            cause = r.get("cause_relation", r.get("relation", ""))
            if cause in HELD and "cause_event" in r:
                texts.append(str(r["cause_event"]))
            if r.get("held_distractor_relation", "") in HELD and "held_distractor_event" in r:
                texts.append(str(r["held_distractor_event"]))
        elif task == "unsupervised_event_text":
            if r.get("relation", "") in HELD and "text" in r:
                texts.append(str(r["text"]))
    pat = re.compile(r"\b(to|from|gave|give|given|handed|received|got|bequeathed|ceded|acquired|inherited)\b", re.I)
    bad = [t for t in texts if pat.search(t)]
    return {"held_related_texts_scanned": len(texts), "known_transfer_or_to_from_hits": len(bad), "example_hits": bad[:5]}


def pair_state_shortcuts(rows: Sequence[Dict]) -> Dict[str, float | int | None]:
    # For true changed/unchanged rows, first/second participant shortcut and same-owner fraction.
    changed = true_state_rows(rows, "changed")
    unchanged = true_state_rows(rows, "unchanged")
    if not changed and not unchanged:
        return {"n_true": 0}
    all_true = changed + unchanged
    by_pair = defaultdict(dict)
    for r in rows:
        if r.get("task") == "state_query" and r.get("label") is True:
            by_pair[r["pair_id"]][r["query_kind"]] = r
    same_owner = [d["changed"]["candidate"] == d["unchanged"]["candidate"] for d in by_pair.values() if "changed" in d and "unchanged" in d]
    return {
        "n_true": len(all_true),
        "always_slot0_accuracy": frac(r["correct_slot"] == 0 for r in all_true),
        "always_slot1_accuracy": frac(r["correct_slot"] == 1 for r in all_true),
        "changed_static_same_owner_frac": frac(same_owner),
    }


def audit(common_seen: Sequence[Dict], arms: Dict[str, Dict[str, List[Dict]]], evals: Dict[str, List[Dict]]) -> Dict:
    arm_summaries = {}
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
            "orientation_dependency_counts": dict(Counter(r.get("orientation_dependency") for r in sup)),
            "formal_satisfying_assignment_count": len(sats),
            "formal_satisfying_assignments": sats,
            "true_assignment_satisfies": TRUE_ASSIGNMENT in sats,
            "inverted_assignment_satisfies": INVERTED_ASSIGNMENT in sats,
        }
    eval_summaries = {}
    for name, rows in evals.items():
        eval_summaries[name] = {
            "rows": len(rows),
            "words": sum(row_words(r) for r in rows),
            "label_balance": label_balance(rows),
            "state_balance": state_balance(rows),
            "orientation_dependency_counts": dict(Counter(r.get("orientation_dependency") for r in rows)),
            "pair_state_shortcuts": pair_state_shortcuts(rows),
        }
    all_text_rows = list(common_seen)
    for b in arms.values():
        all_text_rows += b["supervised"] + b["unsupervised"]
    for rows in evals.values():
        all_text_rows += rows
    return {
        "status": "SYMMETRY_SUBSTRATE_V2_AUDIT_COMPLETE",
        "repair_intent": "nonce held relations, no held to/from/known-transfer wording, actual invariant neutral control, balanced anchor slots, graph closure, formal sign enumeration",
        "true_assignment": TRUE_ASSIGNMENT,
        "inverted_assignment": INVERTED_ASSIGNMENT,
        "common_seen_coordinate": {
            "rows": len(common_seen),
            "words": sum(row_words(r) for r in common_seen),
            "label_balance": label_balance(common_seen),
            "state_balance": state_balance(common_seen),
        },
        "arms": arm_summaries,
        "eval_sets": eval_summaries,
        "global_checks": {
            "train_eval_names_disjoint": set(TRAIN_NAMES).isdisjoint(EVAL_NAMES),
            "train_eval_changed_objects_disjoint": set(TRAIN_CHANGED).isdisjoint(EVAL_CHANGED),
            "train_eval_static_objects_disjoint": set(TRAIN_STATIC).isdisjoint(EVAL_STATIC),
            "changed_static_objects_disjoint_within_train": set(TRAIN_CHANGED).isdisjoint(TRAIN_STATIC),
            "changed_static_objects_disjoint_within_eval": set(EVAL_CHANGED).isdisjoint(EVAL_STATIC),
            "held_surface_leak_scan": held_surface_leak(all_text_rows),
            "heldheld_only_has_exactly_global_Z2": arm_summaries["heldheld_only"]["formal_satisfying_assignment_count"] == 2,
            "aligned_state_identifies_true": arm_summaries["aligned_state_bridge"]["formal_satisfying_assignments"] == [TRUE_ASSIGNMENT],
            "inverted_state_identifies_inverted": arm_summaries["inverted_state_bridge"]["formal_satisfying_assignments"] == [INVERTED_ASSIGNMENT],
            "neutral_decoupled_preserves_Z2": arm_summaries["neutral_decoupled"]["formal_satisfying_assignment_count"] == 2,
            "mixed_event_identifies_true": arm_summaries["mixed_event_bridge"]["formal_satisfying_assignments"] == [TRUE_ASSIGNMENT],
        },
    }


def write_summary(path: Path, aud: Dict, files: Dict[str, str]) -> None:
    lines = []
    lines.append("# research v2 repaired symmetry-identification substrate")
    lines.append("")
    lines.append("CPU/file-only rebuilt substrate after independent_review found the natural-verb prototype unsafe. No model loading, training, official evaluation, GPU work, upload, or leaderboard action occurred.")
    lines.append("")
    lines.append("## Scientific object")
    lines.append("")
    lines.append("Held relation words are nonce lexical components in otherwise natural possession-state language. Held-held training rows constrain only relative orientation and admit exactly the two global assignments related by a role swap. Sparse aligned or mixed bridges identify the true assignment; inverted bridges identify the opposite assignment; the neutral decoupled control leaves the same two assignments.")
    lines.append("")
    lines.append("## Formal sign assignments")
    lines.append("")
    lines.append(f"- true assignment: `{aud['true_assignment']}`")
    lines.append(f"- inverted assignment: `{aud['inverted_assignment']}`")
    lines.append("")
    lines.append("| arm | supervised rows | words | satisfying assignments | true? | inverted? | slot0 true changed | surface-first true changed |")
    lines.append("|---|---:|---:|---:|---|---|---:|---:|")
    for arm, s in aud["arms"].items():
        sb = s["state_balance"]
        lines.append(f"| {arm} | {s['supervised_rows']} | {s['supervised_words']} | {s['formal_satisfying_assignment_count']} | {s['true_assignment_satisfies']} | {s['inverted_assignment_satisfies']} | {fmt(sb.get('slot0_true_changed'))} | {fmt(sb.get('surface_first_true_changed'))} |")
    lines.append("")
    lines.append("## Evaluation suites")
    lines.append("")
    lines.append("| suite | rows | true frac | slot0 true changed | surface-first true changed | slot shortcut all | same changed/static owner |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for suite, s in aud["eval_sets"].items():
        lb = s["label_balance"]
        sb = s["state_balance"]
        ps = s["pair_state_shortcuts"]
        lines.append(f"| {suite} | {s['rows']} | {fmt(lb.get('true_frac'))} | {fmt(sb.get('slot0_true_changed'))} | {fmt(sb.get('surface_first_true_changed'))} | {fmt(ps.get('always_slot0_accuracy'))} | {fmt(ps.get('changed_static_same_owner_frac'))} |")
    lines.append("")
    lines.append("## Repaired shortcut checks")
    lines.append("")
    for k, v in aud["global_checks"].items():
        if k == "held_surface_leak_scan":
            lines.append(f"- held surface leak scan: {v['known_transfer_or_to_from_hits']} hits across {v['held_related_texts_scanned']} held-related text fields")
            if v["example_hits"]:
                for e in v["example_hits"]:
                    lines.append(f"  - hit example: {e}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Remaining scientific caution")
    lines.append("")
    lines.append("This v2 substrate removes the main natural-verb leakage by using nonce held relations. It therefore tests a controlled lexical-component identifiability law, not full natural verb understanding. A positive learned result would justify a later ecological shadow with rare natural verbs only if zero-shot/exposure-only natural orientation is measured and residual ambiguity remains.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for k, v in files.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(x) -> str:
    if x is None:
        return "nan"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--common-seen-worlds", type=int, default=48)
    ap.add_argument("--heldheld-per-edge", type=int, default=16)
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
    write_jsonl(p, common); files["common_seen_train_jsonl"] = str(p.relative_to(PROJECT_ROOT))
    for arm, blocks in arms.items():
        ps = out / "arms" / arm / "train_supervised.jsonl"
        pu = out / "arms" / arm / "train_unsup_text.jsonl"
        write_jsonl(ps, blocks["supervised"]); write_jsonl(pu, blocks["unsupervised"])
        files[f"arm_{arm}_supervised_jsonl"] = str(ps.relative_to(PROJECT_ROOT))
        files[f"arm_{arm}_unsup_jsonl"] = str(pu.relative_to(PROJECT_ROOT))
    for suite, rows in evals.items():
        pe = out / "eval" / f"{suite}.jsonl"
        write_jsonl(pe, rows); files[f"eval_{suite}_jsonl"] = str(pe.relative_to(PROJECT_ROOT))

    aud = audit(common, arms, evals)
    pa = out / "symmetry_substrate_v2_audit.json"
    pa.write_text(json.dumps(aud, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files["audit_json"] = str(pa.relative_to(PROJECT_ROOT))

    arm_csv = []
    for arm, s in aud["arms"].items():
        row = {"arm": arm, "supervised_rows": s["supervised_rows"], "supervised_words": s["supervised_words"], "formal_satisfying_assignment_count": s["formal_satisfying_assignment_count"], "true_assignment_satisfies": s["true_assignment_satisfies"], "inverted_assignment_satisfies": s["inverted_assignment_satisfies"]}
        row.update({f"label_{k}": v for k, v in s["label_balance"].items()})
        row.update({f"state_{k}": v for k, v in s["state_balance"].items()})
        arm_csv.append(row)
    pc = out / "arm_summary.csv"; write_csv(pc, arm_csv); files["arm_summary_csv"] = str(pc.relative_to(PROJECT_ROOT))

    eval_csv = []
    for suite, s in aud["eval_sets"].items():
        row = {"suite": suite, "rows": s["rows"], "words": s["words"]}
        row.update({f"label_{k}": v for k, v in s["label_balance"].items()})
        row.update({f"state_{k}": v for k, v in s["state_balance"].items()})
        row.update({f"shortcut_{k}": v for k, v in s["pair_state_shortcuts"].items()})
        eval_csv.append(row)
    pec = out / "eval_summary.csv"; write_csv(pec, eval_csv); files["eval_summary_csv"] = str(pec.relative_to(PROJECT_ROOT))

    psum = out / "symmetry_substrate_v2_summary.md"
    write_summary(psum, aud, files); files["summary_md"] = str(psum.relative_to(PROJECT_ROOT))

    print(json.dumps({
        "status": aud["status"],
        "out": str(out.relative_to(PROJECT_ROOT)),
        "summary_md": files["summary_md"],
        "formal_counts": {k: v["formal_satisfying_assignment_count"] for k, v in aud["arms"].items()},
        "global_checks": aud["global_checks"],
        "no_model_loading_training_evaluation_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
