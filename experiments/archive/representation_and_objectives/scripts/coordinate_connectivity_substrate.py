#!/usr/bin/env python3
"""research: coordinate-connectivity substrate.

Tests whether learner-usable filler connectivity enables role-coordinate induction
when formally identical information content is present in both conditions.

Connected: bridge state rows share name pairs with held-held comparison rows,
creating a constraint chain from seen-coordinate reference through held-held
orientation to bridge state labels.

Disconnected: bridge state rows use disjoint name pairs from held-held comparisons.
Formal information is identical (same relations, labels, initial patterns), but no
filler-mediated constraint path connects bridge evidence to held-held orientation.

Both conditions share identical:
- common seen-coordinate state rows (all 8 training pairs)
- held-held comparison rows (using CONN pairs)
- eval suites (using EVAL names, disjoint from all training)
- row types, templates, label distributions, initial-pattern balance
- total supervised row counts per arm

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
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/coordinate_connectivity_substrate')

# ── relation model (identical to research) ──────────────────────────────────

@dataclass(frozen=True)
class Rel:
    key: str; component: str; final_slot: int; lex: str; family: str

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

# ── name/object pools ──────────────────────────────────────────────────────

TRAIN_NAMES = ["Mira", "Noel", "Iris", "Omar", "Lena", "Pavel", "Rina", "Tomas",
               "Nia", "Felix", "Ava", "Jonas", "Keira", "Milo", "Sara", "Theo"]
EVAL_NAMES = ["Zara", "Eli", "Nora", "Caleb", "Vera", "Hugo", "Maya", "Luca",
              "June", "Arun", "Leah", "Mateo", "Tara", "Simon", "Yara", "Ben"]
TRAIN_CHANGED = ["lantern", "parcel", "compass", "violin", "notebook", "teapot",
                 "camera", "blanket", "basket", "tablet", "wallet", "sketchbook",
                 "helmet", "map", "jacket", "medal"]
EVAL_CHANGED = ["goblet", "key", "telescope", "bracelet", "vase", "satchel",
                "radio", "painting", "flute", "badge", "ticket", "scarf",
                "hammer", "shell", "drum", "booklet"]
TRAIN_STATIC = ["mug", "rope", "spoon", "bottle", "pillow", "clock", "broom",
                "coin", "brush", "kite", "stone", "candle", "plate", "glove",
                "bell", "saddle"]
EVAL_STATIC = ["ruler", "mirror", "bucket", "comb", "ladder", "needle", "pencil",
               "towel", "anchor", "fork", "whistle", "basketball", "bowl", "rug",
               "lamp", "button"]

# ── explicit pair pools for connectivity manipulation ──────────────────────

CONN_PAIRS: List[Tuple[str, str]] = [
    ("Mira", "Omar"), ("Noel", "Iris"), ("Lena", "Pavel"), ("Rina", "Tomas")]
DISC_PAIRS: List[Tuple[str, str]] = [
    ("Nia", "Felix"), ("Ava", "Jonas"), ("Keira", "Milo"), ("Sara", "Theo")]
ALL_TRAIN_PAIRS = CONN_PAIRS + DISC_PAIRS
ANCHOR_RELS = ["h0_dax", "h2_norp"]
HH_EDGES = [("h0_dax", "h1_mep"), ("h2_norp", "h3_ziv"),
             ("h1_mep", "h2_norp"), ("h0_dax", "h2_norp")]

# ── row-building functions (from research, self-contained) ──────────────────

def held_event(rel_key: str, a: str, b: str, obj: str, voice: int) -> Tuple[str, List[str], str, str]:
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

def event(rel_key: str, a: str, b: str, obj: str, voice: int):
    if REL[rel_key].component == "held":
        return held_event(rel_key, a, b, obj, voice)
    return seen_event(rel_key, a, b, obj, voice)

def slot(rel_key: str, assignment: Dict[str, int]) -> int:
    if rel_key in HELD:
        return int(assignment[rel_key])
    return REL[rel_key].final_slot

def owner_from_slot(slot_idx: int, a: str, b: str) -> str:
    return (a, b)[slot_idx]

def final_owner(rel_key: str, a: str, b: str, assignment: Dict[str, int] = None) -> str:
    if assignment is None: assignment = TRUE_ASSIGNMENT
    return owner_from_slot(slot(rel_key, assignment), a, b)

def old_owner(rel_key: str, a: str, b: str, assignment: Dict[str, int] = None) -> str:
    if assignment is None: assignment = TRUE_ASSIGNMENT
    return owner_from_slot(1 - slot(rel_key, assignment), a, b)

def comparison_row(row_id, rel1, rel2, a, b, obj, same, split, suite, v1, v2):
    s1 = slot(rel1, TRUE_ASSIGNMENT); s2 = slot(rel2, TRUE_ASSIGNMENT)
    a2, b2 = ((a, b) if s1 == s2 else (b, a)) if same else ((b, a) if s1 == s2 else (a, b))
    e1, o1, t1, vv1 = event(rel1, a, b, obj, v1)
    e2, o2, t2, vv2 = event(rel2, a2, b2, obj, v2)
    assert (final_owner(rel1, a, b) == final_owner(rel2, a2, b2)) == same
    comps = (REL[rel1].component, REL[rel2].component)
    dep = "held_seen_breaks_symmetry" if comps.count("held") == 1 else \
          "heldheld_invariant" if comps == ("held", "held") else "seen_seen_fixed"
    return {
        "row_id": row_id, "task": "relation_comparison", "split": split, "suite": suite,
        "relation1": rel1, "relation2": rel2,
        "component1": REL[rel1].component, "component2": REL[rel2].component,
        "arg_order1": [a, b], "arg_order2": [a2, b2],
        "surface_order1": o1, "surface_order2": o2,
        "template1": t1, "template2": t2, "voice1": vv1, "voice2": vv2,
        "object": obj, "event1": e1, "event2": e2,
        "text": f"Event A: {e1} Event B: {e2} Did Event A and Event B leave the {obj} with the same person?",
        "label": bool(same), "orientation_dependency": dep,
        "global_swap_changes_label": dep == "held_seen_breaks_symmetry",
    }

def comparison_orbit(pfx, r1, r2, a, b, obj, same, split, suite):
    return [comparison_row(f"{pfx}_v{v1}{v2}", r1, r2, a, b, obj, same, split, suite, v1, v2)
            for v1 in [0, 1] for v2 in [0, 1]]

def state_rows(row_id, rel_key, a, b, changed_obj, static_obj, split, suite, voice, static_slot,
               assignment=None, initial_pattern=None):
    if assignment is None: assignment = TRUE_ASSIGNMENT
    new = final_owner(rel_key, a, b, assignment)
    old_ow = owner_from_slot(1 - slot(rel_key, assignment), a, b)
    # If initial_pattern == "same", initial owner = final owner (same as new).
    # Default (opposite or None): initial owner = old_ow (complement of final).
    if initial_pattern == "same":
        premise_initial = new
    else:
        premise_initial = old_ow
    static = (a, b)[static_slot % 2]
    ev, surface_order, tmpl, vv = event(rel_key, a, b, changed_obj, voice)
    premise = f"At first, {premise_initial} had the {changed_obj}, and {static} had the {static_obj}. The event was this: {ev}"
    rows = []
    for qk, obj, true_owner, dep in [
        ("changed", changed_obj, new, "held_seen_breaks_symmetry" if rel_key in HELD else "seen_seen_fixed"),
        ("unchanged", static_obj, static, "unaffected_fact_should_be_preserved"),
    ]:
        for cand in [a, b]:
            rows.append({
                "row_id": f"{row_id}_{qk}_{0 if cand == a else 1}", "pair_id": row_id,
                "task": "state_query", "split": split, "suite": suite,
                "relation": rel_key, "component": REL[rel_key].component,
                "arg_order": [a, b], "surface_order": surface_order,
                "template": tmpl, "voice": vv, "static_slot": static_slot,
                "changed_object": changed_obj, "static_object": static_obj,
                "query_kind": qk, "candidate": cand,
                "candidate_slot": 0 if cand == a else 1,
                "correct_slot": 0 if true_owner == a else 1,
                "premise": premise,
                "hypothesis": f"After the event, {cand} had the {obj}.",
                "label": bool(cand == true_owner),
                "orientation_dependency": dep,
                "global_swap_changes_label": dep == "held_seen_breaks_symmetry",
                "cause_relation": rel_key, "cause_event": ev,
                "static_owner": static, "supervised_changed_owner": new,
                "inverted_bridge_label": assignment != TRUE_ASSIGNMENT,
                "initial_pattern": initial_pattern or "opposite",
                "initial_owner": premise_initial,
            })
    return rows

def state_orbit(pfx, rel_key, a, b, changed_obj, static_obj, split, suite,
                assignment=None, initial_pattern=None):
    rows = []
    for voice in [0, 1]:
        for ss in [0, 1]:
            rows.extend(state_rows(f"{pfx}_v{voice}_st{ss}", rel_key, a, b,
                                   changed_obj, static_obj, split, suite, voice, ss,
                                   assignment, initial_pattern))
    return rows

def unsup_row(row_id, rel, a, b, obj, split, suite, voice):
    ev, so, tmpl, vv = event(rel, a, b, obj, voice)
    return {
        "row_id": row_id, "task": "unsupervised_event_text", "split": split, "suite": suite,
        "relation": rel, "component": REL[rel].component,
        "arg_order": [a, b], "surface_order": so, "template": tmpl, "voice": vv,
        "object": obj, "text": ev,
        "orientation_dependency": "none_unsupervised", "global_swap_changes_label": False,
    }

# ── explicit construction with controlled pair assignment ──────────────────

def pick_obj(pool: List[str], idx: int) -> str:
    return pool[idx % len(pool)]

def common_seen_train_explicit(pairs: List[Tuple[str,str]], n_per_pair_per_rel: int = 4) -> List[Dict[str, Any]]:
    """All pairs × both seen relations, using explicit pair assignment."""
    rows = []
    oi = 0
    for pi, (a, b) in enumerate(pairs):
        for ri, rel in enumerate(SEEN_KEYS):
            for j in range(n_per_pair_per_rel):
                rows.extend(state_orbit(
                    f"common_seen_p{pi}_{rel}_{j:02d}", rel, a, b,
                    pick_obj(TRAIN_CHANGED, oi), pick_obj(TRAIN_STATIC, oi * 3 + 1),
                    "train", "common_seen_coordinate"))
                oi += 1
    return rows

def heldheld_train_explicit(pairs: List[Tuple[str,str]], n_per_edge: int = 12) -> List[Dict[str, Any]]:
    """Held-held comparison using only the specified pairs."""
    rows = []; oi = 0
    for ei, (r1, r2) in enumerate(HH_EDGES):
        for j in range(n_per_edge):
            a, b = pairs[j % len(pairs)]
            # flip pair order every len(pairs) to balance
            if (j // len(pairs)) % 2:
                a, b = b, a
            rows.extend(comparison_orbit(
                f"train_hh_e{ei}_{j:03d}", r1, r2, a, b,
                pick_obj(TRAIN_CHANGED, oi + 100), same=(j % 2 == 0),
                split="train", suite="heldheld_path_train"))
            oi += 1
    return rows

def bridge_state_explicit(pairs: List[Tuple[str,str]], assignment: Dict[str,int],
                          suite: str) -> List[Dict[str, Any]]:
    """Bridge state orbits with k16-style 50% opposite / 50% same initial balance.

    2 anchor rels × 2 orbits per pair = 4 per pair; k16 balance: per relation,
    half pairs get opposite, half get same, cross-balanced with static_slot.
    """
    rows = []; oi = 0
    for ri, rel in enumerate(ANCHOR_RELS):
        for pi, (a, b) in enumerate(pairs):
            # k16 balance: first half opposite, second half same per relation
            # Cross-balance across relations: if even rel index, first 2 pairs opposite;
            # if odd, last 2 pairs opposite. This avoids the research diagonal.
            if ri % 2 == 0:
                ip = "opposite" if pi < len(pairs) // 2 else "same"
            else:
                ip = "same" if pi < len(pairs) // 2 else "opposite"
            rows.extend(state_orbit(
                f"train_{suite}_{rel}_p{pi:02d}", rel, a, b,
                pick_obj(TRAIN_CHANGED, oi + 200),
                pick_obj(TRAIN_STATIC, oi * 3 + 201),
                "train", suite, assignment, initial_pattern=ip))
            oi += 1
    return rows

def unsup_held_exposure(pairs: List[Tuple[str,str]]) -> List[Dict[str, Any]]:
    """Unsupervised nonce-verb event text for all specified pairs × all held rels."""
    rows = []; oi = 0
    for pi, (a, b) in enumerate(pairs):
        for rel in HELD_KEYS:
            for voice in [0, 1]:
                rows.append(unsup_row(
                    f"train_held_exposure_p{pi}_{rel}_v{voice}", rel, a, b,
                    pick_obj(TRAIN_CHANGED, oi + 300), "train", "held_event_exposure", voice))
                oi += 1
    return rows

# ── condition builder ──────────────────────────────────────────────────────

def build_condition(condition: str) -> Dict[str, Any]:
    """Build one condition (connected or disconnected)."""
    bridge_pairs = CONN_PAIRS if condition == "connected" else DISC_PAIRS
    common = common_seen_train_explicit(ALL_TRAIN_PAIRS)
    hh = heldheld_train_explicit(CONN_PAIRS)
    unsup = unsup_held_exposure(ALL_TRAIN_PAIRS)
    arms = {}
    for arm_name, assignment in [
        ("aligned_state_bridge", TRUE_ASSIGNMENT),
        ("inverted_state_bridge", INVERTED_ASSIGNMENT),
    ]:
        bridge = bridge_state_explicit(bridge_pairs, assignment, f"{arm_name}_sparse")
        arms[arm_name] = {"supervised": hh + bridge, "unsupervised": unsup}
    arms["heldheld_only"] = {"supervised": hh, "unsupervised": unsup}
    return {"common": common, "arms": arms, "bridge_pairs": bridge_pairs}

# ── eval (shared, from research) ────────────────────────────────────────────

def choose_pair_eval(i: int) -> Tuple[str, str]:
    n = len(EVAL_NAMES)
    a = EVAL_NAMES[i % n]; b = EVAL_NAMES[(i * 5 + 3) % n]
    if a == b: b = EVAL_NAMES[(i + 7) % n]
    if (i // n) % 2: a, b = b, a
    return a, b

def build_eval(n_eval: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    evals: Dict[str, List[Dict[str, Any]]] = {}
    # heldheld closure
    hh_pairs = [("h0_dax", "h1_mep"), ("h2_norp", "h3_ziv"), ("h3_ziv", "h0_dax"), ("h1_mep", "h3_ziv")]
    rows = []; k = 0
    for pi, (r1, r2) in enumerate(hh_pairs):
        for j in range(n_eval):
            a, b = choose_pair_eval(k)
            rows.extend(comparison_orbit(f"eval_hh_closure_p{pi}_{j:04d}", r1, r2, a, b,
                                         pick_obj(EVAL_CHANGED, k), same=(j % 2 == 0),
                                         split="eval", suite="heldheld_unseen_edge_closure"))
            k += 1
    evals["heldheld_unseen_edge_closure"] = rows
    # mixed held-seen
    rows = []; k = 0
    for h in HELD_KEYS:
        for s in SEEN_KEYS:
            for j in range(n_eval):
                a, b = choose_pair_eval(k)
                same = (j % 2 == 0)
                rows.extend(comparison_orbit(f"eval_mixed_hs_{h}_{s}_{j:04d}", h, s, a, b,
                                             pick_obj(EVAL_CHANGED, k), same=same,
                                             split="eval", suite="mixed_held_seen_orientation"))
                rows.extend(comparison_orbit(f"eval_mixed_sh_{s}_{h}_{j:04d}", s, h, a, b,
                                             pick_obj(EVAL_CHANGED, k + 5), same=same,
                                             split="eval", suite="mixed_held_seen_orientation"))
                k += 1
    evals["mixed_held_seen_orientation"] = rows
    # paired state conservation — both initial patterns to distinguish anti-copy vs event-role
    rows = []; k = 0
    for h in HELD_KEYS:
        for j in range(n_eval):
            a, b = choose_pair_eval(k)
            for ip in ["opposite", "same"]:
                rows.extend(state_orbit(f"eval_psc_{h}_{j:04d}_{ip}", h, a, b,
                                        pick_obj(EVAL_CHANGED, k), pick_obj(EVAL_STATIC, k * 3),
                                        "eval", "paired_state_conservation",
                                        initial_pattern=ip))
            k += 1
    evals["paired_state_conservation"] = rows
    # cross template — both initial patterns
    rows = []; k = 1000
    for h in HELD_KEYS:
        for j in range(max(4, n_eval // 2)):
            a, b = choose_pair_eval(k)
            for ip in ["opposite", "same"]:
                rows.extend(state_orbit(f"eval_xtempl_{h}_{j:04d}_{ip}", h, a, b,
                                        pick_obj(EVAL_CHANGED, k), pick_obj(EVAL_STATIC, k * 3),
                                        "eval", "cross_template_state_readout",
                                        initial_pattern=ip))
            k += 1
    evals["cross_template_state_readout"] = rows
    return evals

# ── formal verification ────────────────────────────────────────────────────

def predict_label(row: Dict[str, Any], assignment: Dict[str, int]) -> bool:
    if row["task"] == "relation_comparison":
        r1, r2 = row["relation1"], row["relation2"]
        a1, b1 = row["arg_order1"]; a2, b2 = row["arg_order2"]
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
    sats = []
    for bits in itertools.product([0, 1], repeat=len(HELD_KEYS)):
        ass = dict(zip(HELD_KEYS, bits))
        if all(predict_label(r, ass) == bool(r["label"]) for r in labeled):
            sats.append(ass)
    return sats

# ── connectivity graph audit ───────────────────────────────────────────────

def connectivity_graph(common: List[Dict], supervised: List[Dict]) -> Dict[str, Any]:
    """Compute the bipartite entity-pair × relation-row-type graph."""
    pair_rel_types: Dict[str, set] = defaultdict(set)
    all_rows = common + supervised
    for r in all_rows:
        if r.get("task") == "relation_comparison":
            pair_key = tuple(sorted(r["arg_order1"]))
            pair_rel_types[str(pair_key)].add(f"comparison_{r['relation1']}_{r['relation2']}")
        elif r.get("task") == "state_query":
            pair_key = tuple(sorted(r["arg_order"]))
            suite = r.get("suite", "")
            if "common_seen" in suite:
                pair_rel_types[str(pair_key)].add(f"seen_state_{r['relation']}")
            elif "bridge" in suite or "sparse" in suite:
                pair_rel_types[str(pair_key)].add(f"bridge_state_{r['relation']}")
    # Path analysis: is there a pair that connects seen_state to comparison to bridge_state?
    chains = []
    for pair, types in pair_rel_types.items():
        has_seen = any("seen_state" in t for t in types)
        has_comparison = any("comparison" in t for t in types)
        has_bridge = any("bridge_state" in t for t in types)
        if has_seen and has_comparison and has_bridge:
            chains.append({"pair": pair, "types": sorted(types), "chain": "seen→comparison→bridge"})
        elif has_comparison and has_bridge:
            chains.append({"pair": pair, "types": sorted(types), "chain": "comparison→bridge"})
    return {
        "n_pairs": len(pair_rel_types),
        "pair_type_counts": {p: len(t) for p, t in sorted(pair_rel_types.items())},
        "n_full_chains": sum(1 for c in chains if c["chain"] == "seen→comparison→bridge"),
        "n_partial_chains": sum(1 for c in chains if c["chain"] == "comparison→bridge"),
        "chains": chains,
    }

# ── baseline checks ───────────────────────────────────────────────────────

def anti_copy_accuracy(rows: List[Dict]) -> Dict[str, Any]:
    """Anti-copy: predict the non-initial owner for changed, initial owner for unchanged."""
    changed = [r for r in rows if r.get("task") == "state_query" and r.get("query_kind") == "changed" and r.get("label")]
    by_ip = defaultdict(list)
    for r in changed:
        ip = r.get("initial_pattern", "opposite")
        pred = r["candidate"] != r.get("initial_owner", "")
        by_ip[ip].append(float(pred == r["label"]))
    return {ip: sum(v) / len(v) if v else None for ip, v in sorted(by_ip.items())}

def order_rule_best(rows: List[Dict], task: str) -> Dict[str, float]:
    labeled = [r for r in rows if r.get("task") == task and "label" in r]
    if not labeled: return {"n": 0}
    if task == "relation_comparison":
        features = ["surface_order_same", "voice_same", "arg_order_same"]
        def feat(r):
            so1 = r.get("surface_order1", []);  so2 = r.get("surface_order2", [])
            return {"surface_order_same": so1 == so2,
                    "voice_same": r.get("voice1") == r.get("voice2"),
                    "arg_order_same": r.get("arg_order1") == r.get("arg_order2")}
    else:
        features = ["candidate_surface_first", "candidate_slot0", "active_voice"]
        def feat(r):
            so = r.get("surface_order", [])
            return {"candidate_surface_first": bool(so and so[0] == r["candidate"]),
                    "candidate_slot0": r.get("candidate_slot") == 0,
                    "active_voice": r.get("voice") == "active"}
    feats = [feat(r) for r in labeled]; labs = [bool(r["label"]) for r in labeled]
    best_acc = 0.5
    for f in features:
        vals = [bool(fr.get(f)) for fr in feats]
        acc = max(sum(v == y for v, y in zip(vals, labs)), sum((not v) == y for v, y in zip(vals, labs))) / len(labs)
        best_acc = max(best_acc, acc)
    return {"n": len(labeled), "best_accuracy": best_acc}

# ── I/O ────────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

def label_balance(rows):
    labs = [bool(r["label"]) for r in rows if "label" in r]
    return {"n": len(labs), "true_frac": sum(labs) / len(labs)} if labs else {"n": 0}

def words(text: str) -> int:
    return len(re.findall(r"\b\\w+\\b", text))

def row_words(r):
    keys = ("text", "premise", "hypothesis", "event1", "event2", "cause_event")
    return sum(words(str(r.get(k, ""))) for k in keys)

def held_surface_leak(rows):
    texts = []
    for r in rows:
        if r.get("task") == "relation_comparison":
            for k in ["event1", "event2"]:
                if any(r.get(f"relation{i}") in HELD for i in [1, 2]) and k in r:
                    texts.append(str(r[k]))
        elif r.get("task") == "state_query" and r.get("cause_relation", r.get("relation")) in HELD:
            if "cause_event" in r: texts.append(str(r["cause_event"]))
    pat = re.compile(r"\\b(to|from|gave|give|given|handed|received|got)\\b", re.I)
    return {"scanned": len(texts), "leaks": sum(1 for t in texts if pat.search(t))}

# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)

    evals = build_eval(8)
    conditions = {}
    for cond in ["connected", "disconnected"]:
        cd = build_condition(cond)
        cond_out = out / cond
        # Write common seen
        write_jsonl(cond_out / "common_seen_train.jsonl", cd["common"])
        # Write arms
        for arm, blocks in cd["arms"].items():
            write_jsonl(cond_out / "arms" / arm / "train_supervised.jsonl", blocks["supervised"])
            write_jsonl(cond_out / "arms" / arm / "train_unsup_text.jsonl", blocks["unsupervised"])
        conditions[cond] = cd

    # Write shared eval
    for suite, rows in evals.items():
        write_jsonl(out / "eval" / f"{suite}.jsonl", rows)

    # ── Audit ──
    manifest = {"conditions": {}, "eval": {}, "global_checks": {}}
    for cond in ["connected", "disconnected"]:
        cd = conditions[cond]
        cond_report = {"bridge_pairs": [list(p) for p in cd["bridge_pairs"]], "arms": {}}
        for arm, blocks in cd["arms"].items():
            sup = blocks["supervised"]
            all_train = cd["common"] + sup
            sats = satisfying_assignments(sup)
            conn_graph = connectivity_graph(cd["common"], sup)
            cond_report["arms"][arm] = {
                "common_seen_rows": len(cd["common"]),
                "supervised_rows": len(sup),
                "unsupervised_rows": len(blocks["unsupervised"]),
                "total_train_rows": len(all_train),
                "supervised_words": sum(row_words(r) for r in sup),
                "label_balance_supervised": label_balance(sup),
                "label_balance_common": label_balance(cd["common"]),
                "formal_satisfying_assignments": len(sats),
                "satisfying_assignments": sats,
                "true_satisfies": TRUE_ASSIGNMENT in sats,
                "inverted_satisfies": INVERTED_ASSIGNMENT in sats,
                "anti_copy_accuracy": anti_copy_accuracy(all_train),
                "order_baseline_comparison": order_rule_best(all_train, "relation_comparison"),
                "order_baseline_state_changed": order_rule_best(
                    [r for r in all_train if r.get("query_kind") == "changed"], "state_query"),
                "held_surface_leak": held_surface_leak(all_train),
                "connectivity_graph": conn_graph,
            }
        cond_report["name_exposure"] = {}
        for arm, blocks in cd["arms"].items():
            all_train = cd["common"] + blocks["supervised"]
            counts = Counter()
            for r in all_train:
                for k in ["arg_order", "arg_order1", "arg_order2"]:
                    for name in r.get(k, []):
                        counts[name] += 1
            cond_report["name_exposure"][arm] = dict(counts.most_common())
        conditions[cond] = cond_report
        manifest["conditions"][cond] = cond_report

    for suite, rows in evals.items():
        manifest["eval"][suite] = {
            "rows": len(rows),
            "label_balance": label_balance(rows),
            "order_baseline": order_rule_best(rows, "relation_comparison") if "comparison" in rows[0].get("task", "") or "mixed" in suite else order_rule_best(rows, "state_query"),
        }

    # Global checks
    gc = {}
    gc["train_eval_names_disjoint"] = set(TRAIN_NAMES).isdisjoint(EVAL_NAMES)
    gc["conn_disc_pairs_disjoint"] = set(CONN_PAIRS).isdisjoint(DISC_PAIRS)
    gc["all_train_names_covered"] = set(n for p in ALL_TRAIN_PAIRS for n in p) == set(TRAIN_NAMES)
    # Check formal information equivalence
    for arm in ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]:
        c_sats = manifest["conditions"]["connected"]["arms"][arm]["formal_satisfying_assignments"]
        d_sats = manifest["conditions"]["disconnected"]["arms"][arm]["formal_satisfying_assignments"]
        gc[f"formal_equiv_{arm}"] = c_sats == d_sats
    # Check connectivity difference
    for arm in ["aligned_state_bridge", "inverted_state_bridge"]:
        c_chains = manifest["conditions"]["connected"]["arms"][arm]["connectivity_graph"]["n_full_chains"]
        d_chains = manifest["conditions"]["disconnected"]["arms"][arm]["connectivity_graph"]["n_full_chains"]
        gc[f"connected_has_chains_{arm}"] = c_chains > 0
        gc[f"disconnected_no_chains_{arm}"] = d_chains == 0
    manifest["global_checks"] = gc
    manifest["status"] = "COORDINATE_CONNECTIVITY_SUBSTRATE_COMPLETE"

    # Write manifest
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    # Write summary
    lines = ["# research coordinate-connectivity substrate", "",
             "Tests whether learner-usable filler connectivity enables role-coordinate induction.",
             "Both conditions have identical formal information (same satisfying assignments).", "",
             "## Conditions", ""]
    for cond in ["connected", "disconnected"]:
        cr = manifest["conditions"][cond]
        lines.append(f"### {cond.upper()}")
        lines.append(f"- bridge pairs: {cr['bridge_pairs']}")
        for arm, ar in cr["arms"].items():
            lines.append(f"- {arm}: {ar['supervised_rows']} supervised + {ar['common_seen_rows']} common = {ar['total_train_rows']} total")
            lines.append(f"  formal assignments: {ar['formal_satisfying_assignments']}, true={ar['true_satisfies']}, inv={ar['inverted_satisfies']}")
            cg = ar["connectivity_graph"]
            lines.append(f"  full chains (seen→comparison→bridge): {cg['n_full_chains']}, partial: {cg['n_partial_chains']}")
            ac = ar["anti_copy_accuracy"]
            lines.append(f"  anti-copy: opposite={ac.get('opposite','n/a'):.3f}, same={ac.get('same','n/a') if ac.get('same') is not None else 'n/a'}")
        lines.append("")
    lines.append("## Global checks")
    for k, v in gc.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Eval suites (shared)")
    for suite, er in manifest["eval"].items():
        lines.append(f"- {suite}: {er['rows']} rows")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- manifest: `{out.relative_to(PROJECT_ROOT) / 'manifest.json'}`")
    lines.append(f"- connected: `{out.relative_to(PROJECT_ROOT) / 'connected/'}`")
    lines.append(f"- disconnected: `{out.relative_to(PROJECT_ROOT) / 'disconnected/'}`")
    lines.append(f"- eval: `{out.relative_to(PROJECT_ROOT) / 'eval/'}`")
    (out / "connectivity_substrate_summary.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "status": manifest["status"],
        "out": str(out.relative_to(PROJECT_ROOT)),
        "summary": str((out / "connectivity_substrate_summary.md").relative_to(PROJECT_ROOT)),
        "manifest": str((out / "manifest.json").relative_to(PROJECT_ROOT)),
        "global_checks": gc,
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
