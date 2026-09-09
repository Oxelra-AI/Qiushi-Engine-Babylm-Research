#!/usr/bin/env python3
"""research: neutral-block-repaired pair-binding connectivity substrate.

The strategist rejected the research connected/disconnected run because connected
and disconnected changed which names/pairs received bridge supervision.  independent review then
identified the closest clean repair: keep the same names, exact bridge rows,
common seen-coordinate rows, input text multiset, exact pair degrees, token counts,
and per-name role/label counts, while changing only which exact participant pairs
carry orientation-informative held-held labels.

This construction uses two exact-pair sets:
  B = bridge/base pairs, which always receive identical bridge state rows;
  R = degree-matched rewired pairs, which share all individual names but no exact
      ordered pair with B.

Both conditions contain the same held-held input texts on B and R.  In
`pair_connected`, B held-held rows are informative and R rows are orientation-
neutral.  In `pair_rewired`, B rows are neutral and R rows are informative.  The
neutral rows have the same event texts, relation tokens, voices, objects, pair
incidence, and label marginals as informative rows, but their labels are balanced
against the relation-orientation parity and therefore add rank-zero orientation
information.  Thus exact text exposure is matched while pair-binding alignment is
rewired.

No model loading, no GPU training, no BabyLM evaluation, and no upload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import coordinate_connectivity_substrate as base  # noqa: E402

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/pair_binding_neutral_substrate')

B_PAIRS: List[Tuple[str, str]] = list(base.ALL_TRAIN_PAIRS)
N = len(B_PAIRS)
R_PAIRS: List[Tuple[str, str]] = [(B_PAIRS[i][0], B_PAIRS[(i + 1) % N][1]) for i in range(N)]
ALL_PAIRS: List[Tuple[str, str]] = []
for p in B_PAIRS + R_PAIRS:
    if p not in ALL_PAIRS:
        ALL_PAIRS.append(p)

CONDITIONS = ["pair_connected", "pair_rewired"]
ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]
HELD_EDGES = list(base.HH_EDGES)
NAMES = set(base.TRAIN_NAMES)


def pick_obj(pool: List[str], idx: int) -> str:
    return pool[idx % len(pool)]


def assignment_for_arm(arm: str) -> Dict[str, int] | None:
    if arm == "aligned_state_bridge":
        return base.TRUE_ASSIGNMENT
    if arm == "inverted_state_bridge":
        return base.INVERTED_ASSIGNMENT
    if arm == "heldheld_only":
        return None
    raise ValueError(arm)


def common_seen_train(pairs: Sequence[Tuple[str, str]], n_per_pair_per_rel: int = 4) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oi = 0
    for pi, (a, b) in enumerate(pairs):
        for rel in base.SEEN_KEYS:
            for j in range(n_per_pair_per_rel):
                rows.extend(base.state_orbit(
                    f"common_seen_p{pi:02d}_{rel}_{j:02d}", rel, a, b,
                    pick_obj(base.TRAIN_CHANGED, oi), pick_obj(base.TRAIN_STATIC, oi * 3 + 1),
                    "train", "common_seen_coordinate"))
                oi += 1
    return rows


def make_hh_row(prefix: str, rel1: str, rel2: str, a: str, b: str, obj: str,
                geom_same: bool, v1: int, v2: int, neutral: bool) -> Dict[str, Any]:
    row = base.comparison_row(prefix, rel1, rel2, a, b, obj, geom_same,
                              split="train", suite="heldheld_pair_binding_train", v1=v1, v2=v2)
    row["geom_same_final_owner_under_true"] = bool(geom_same)
    if neutral:
        # Half the voice configurations align with the true parity and half invert it.
        # For each rel-edge/pair/voice combination there remains exactly one true
        # and one false label across the two geometry rows, matching informative
        # label-by-voice marginals while contributing zero net parity information.
        flip = (v1 + v2) % 2
        neutral_label = bool(geom_same) ^ bool(flip)
        row["label"] = bool(neutral_label)
        row["orientation_dependency"] = "neutral_rank_zero_pair_degree_control"
        row["global_swap_changes_label"] = False
        row["neutral_flip_bit"] = int(flip)
    else:
        row["orientation_dependency"] = "heldheld_informative_parity"
        row["neutral_flip_bit"] = None
    row["neutral_control"] = bool(neutral)
    row["pair_set"] = prefix.split("_")[0]
    return row


def heldheld_block(pairs: Sequence[Tuple[str, str]], pair_set: str, informative: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oi = 0
    for ei, (r1, r2) in enumerate(HELD_EDGES):
        for pi, (a, b) in enumerate(pairs):
            for geom_same in [False, True]:
                for v1 in [0, 1]:
                    for v2 in [0, 1]:
                        obj = pick_obj(base.TRAIN_CHANGED, oi + 100)
                        rows.append(make_hh_row(
                            f"{pair_set}_hh_e{ei:02d}_p{pi:02d}_{'same' if geom_same else 'diff'}_v{v1}{v2}",
                            r1, r2, a, b, obj, geom_same, v1, v2, neutral=(not informative)))
                        oi += 1
    return rows


def bridge_state_rows(assignment: Dict[str, int], suite: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oi = 0
    for rel in base.ANCHOR_RELS:
        for pi, (a, b) in enumerate(B_PAIRS):
            for ip in ["opposite", "same"]:
                rows.extend(base.state_orbit(
                    f"train_{suite}_{rel}_p{pi:02d}_{ip}", rel, a, b,
                    pick_obj(base.TRAIN_CHANGED, oi + 200), pick_obj(base.TRAIN_STATIC, oi * 3 + 201),
                    "train", suite, assignment=assignment, initial_pattern=ip))
                oi += 1
    return rows


def unsup_held_exposure(pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oi = 0
    for pi, (a, b) in enumerate(pairs):
        for rel in base.HELD_KEYS:
            for voice in [0, 1]:
                rows.append(base.unsup_row(
                    f"train_held_exposure_p{pi:02d}_{rel}_v{voice}", rel, a, b,
                    pick_obj(base.TRAIN_CHANGED, oi + 300), "train", "held_event_exposure", voice))
                oi += 1
    return rows


def build_condition(condition: str) -> Dict[str, Any]:
    common = common_seen_train(ALL_PAIRS)
    if condition == "pair_connected":
        b_info, r_info = True, False
    elif condition == "pair_rewired":
        b_info, r_info = False, True
    else:
        raise ValueError(condition)
    hh_b = heldheld_block(B_PAIRS, "B", informative=b_info)
    hh_r = heldheld_block(R_PAIRS, "R", informative=r_info)
    hh_all = hh_b + hh_r
    unsup = unsup_held_exposure(ALL_PAIRS)
    arms: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for arm in ARMS:
        ass = assignment_for_arm(arm)
        bridge = [] if ass is None else bridge_state_rows(ass, f"{arm}_bridge")
        arms[arm] = {"supervised": hh_all + bridge, "heldheld": hh_all, "bridge": bridge, "unsupervised": unsup}
    return {"common": common, "arms": arms, "B_informative": b_info, "R_informative": r_info}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def text_parts(row: Dict[str, Any]) -> Tuple[str, ...]:
    if row.get("task") == "relation_comparison":
        return (str(row.get("event1", "")), str(row.get("event2", "")))
    if row.get("task") == "state_query":
        return (str(row.get("premise", "")), str(row.get("hypothesis", "")))
    if "text" in row:
        return (str(row.get("text", "")),)
    return tuple()


def input_text_sig(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (row.get("task"),) + text_parts(row)


def input_text_label_sig(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return input_text_sig(row) + ((bool(row.get("label")) if "label" in row else None),)


def pair_key_from_row(row: Dict[str, Any]) -> str | None:
    vals = row.get("arg_order1") if row.get("task") == "relation_comparison" else row.get("arg_order")
    if vals and len(vals) == 2:
        return f"{vals[0]}::{vals[1]}"
    return None


def token_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        for part in text_parts(r):
            c.update(tok.lower() for tok in re.findall(r"\b\w+\b", part))
    return c


def pair_degree_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        pk = pair_key_from_row(r)
        if pk:
            c[pk] += 1
    return c


def pair_hist_counts(rows: Sequence[Dict[str, Any]], include_geom: bool = False) -> Counter:
    c: Counter = Counter()
    for r in rows:
        pk = pair_key_from_row(r)
        if not pk:
            continue
        label = "T" if bool(r.get("label")) else "F" if "label" in r else "NA"
        if r.get("task") == "relation_comparison":
            key = (pk, r.get("task"), r.get("relation1"), r.get("relation2"), r.get("voice1"), r.get("voice2"), label)
            if include_geom:
                key += (r.get("geom_same_final_owner_under_true"), r.get("neutral_control"))
        elif r.get("task") == "state_query":
            key = (pk, r.get("task"), r.get("relation") or r.get("cause_relation"), r.get("voice"), r.get("static_slot"), r.get("initial_pattern"), r.get("query_kind"), r.get("candidate_slot"), label)
        else:
            key = (pk, r.get("task"), r.get("relation"), r.get("voice"), label)
        c[key] += 1
    return c


def name_role_label_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        label = "T" if bool(r.get("label")) else "F" if "label" in r else "NA"
        task = r.get("task")
        if task == "relation_comparison":
            for side in [1, 2]:
                rel = r.get(f"relation{side}")
                for idx, nm in enumerate(r.get(f"arg_order{side}", [])):
                    if nm in NAMES:
                        c[(nm, "cmp_arg", side, idx, rel, r.get(f"voice{side}"), label)] += 1
                for idx, nm in enumerate(r.get(f"surface_order{side}", [])):
                    if nm in NAMES:
                        c[(nm, "cmp_surface", side, idx, rel, r.get(f"voice{side}"), label)] += 1
        elif task == "state_query":
            rel = r.get("relation") or r.get("cause_relation")
            for idx, nm in enumerate(r.get("arg_order", [])):
                if nm in NAMES:
                    c[(nm, "state_arg", idx, rel, r.get("voice"), r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot"), label)] += 1
            for idx, nm in enumerate(r.get("surface_order", [])):
                if nm in NAMES:
                    c[(nm, "state_surface", idx, rel, r.get("voice"), r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot"), label)] += 1
            for role in ["candidate", "initial_owner", "initial_changed_owner", "static_owner", "supervised_changed_owner"]:
                nm = r.get(role)
                if nm in NAMES:
                    c[(nm, role, rel, r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot"), r.get("candidate_slot") if role == "candidate" else None, label)] += 1
    return c


def structural_counts(rows: Sequence[Dict[str, Any]], include_neutral: bool = False) -> Counter:
    c: Counter = Counter()
    for r in rows:
        label = "T" if bool(r.get("label")) else "F" if "label" in r else "NA"
        if r.get("task") == "relation_comparison":
            key = (r.get("task"), r.get("relation1"), r.get("relation2"), r.get("voice1"), r.get("voice2"), label)
            if include_neutral:
                key += (r.get("neutral_control"), r.get("geom_same_final_owner_under_true"))
        elif r.get("task") == "state_query":
            key = (r.get("task"), r.get("relation") or r.get("cause_relation"), r.get("voice"), r.get("static_slot"), r.get("initial_pattern"), r.get("query_kind"), r.get("candidate_slot"), label)
        else:
            key = (r.get("task"), r.get("relation"), r.get("voice"), label)
        c[key] += 1
    return c


def count_equal(a: Counter, b: Counter) -> Dict[str, Any]:
    d = a.copy(); d.subtract(b)
    nz = {str(k): int(v) for k, v in d.items() if v != 0}
    return {"equal": not nz, "n_differences": len(nz), "sample": dict(list(sorted(nz.items()))[:30])}


def multiset_equal(xs: Sequence[Any], ys: Sequence[Any]) -> Dict[str, Any]:
    return count_equal(Counter(xs), Counter(ys))


def informative_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if not (r.get("task") == "relation_comparison" and r.get("neutral_control"))]


def satisfying_assignments_informative(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    return base.satisfying_assignments(informative_rows(rows))


def neutral_orientation_scores(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    neutral = [r for r in rows if r.get("task") == "relation_comparison" and r.get("neutral_control")]
    out: Dict[str, Any] = {"n": len(neutral), "assignment_accuracy": {}}
    for bits in itertools.product([0, 1], repeat=len(base.HELD_KEYS)):
        ass = dict(zip(base.HELD_KEYS, bits))
        vals = [float(base.predict_label(r, ass) == bool(r["label"])) for r in neutral]
        out["assignment_accuracy"][json.dumps(ass, sort_keys=True)] = sum(vals) / len(vals) if vals else None
    accs = [v for v in out["assignment_accuracy"].values() if v is not None]
    out["min_accuracy_over_assignments"] = min(accs) if accs else None
    out["max_accuracy_over_assignments"] = max(accs) if accs else None
    out["all_assignments_at_half"] = all(abs(v - 0.5) < 1e-12 for v in accs)
    return out


def duplicate_text_label_conflicts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[Tuple[Any, ...], set] = defaultdict(set)
    for r in rows:
        if "label" in r:
            by[input_text_sig(r)].add(bool(r["label"]))
    conflicts = {str(k): sorted(v) for k, v in by.items() if len(v) > 1}
    return {"n_unique_inputs": len(by), "n_conflicting_inputs": len(conflicts), "sample_conflicts": dict(list(conflicts.items())[:10])}


def label_balance(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [bool(r["label"]) for r in rows if "label" in r]
    return {"n": len(vals), "true": int(sum(vals)), "false": int(len(vals) - sum(vals)), "true_frac": (sum(vals) / len(vals) if vals else None)}


def row_words(r: Dict[str, Any]) -> int:
    return sum(len(re.findall(r"\b\w+\b", p)) for p in text_parts(r))


def summarize_arm(common: List[Dict[str, Any]], blocks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    sup = blocks["supervised"]
    train = common + sup
    sats = satisfying_assignments_informative(sup)
    hh = blocks["heldheld"]
    info = [r for r in hh if not r.get("neutral_control")]
    neutral = [r for r in hh if r.get("neutral_control")]
    return {
        "common_seen_rows": len(common),
        "heldheld_rows_total": len(hh),
        "heldheld_informative_rows": len(info),
        "heldheld_neutral_rows": len(neutral),
        "bridge_rows": len(blocks["bridge"]),
        "supervised_rows": len(sup),
        "total_model_train_rows": len(train),
        "unsupervised_rows_not_used_by_probe": len(blocks["unsupervised"]),
        "train_token_total": sum(row_words(r) for r in train),
        "label_balance_supervised": label_balance(sup),
        "label_balance_model_train": label_balance(train),
        "formal_satisfying_assignment_count_informative_only": len(sats),
        "formal_satisfying_assignments_informative_only": sats,
        "true_satisfies_informative_only": base.TRUE_ASSIGNMENT in sats,
        "inverted_satisfies_informative_only": base.INVERTED_ASSIGNMENT in sats,
        "neutral_orientation_scores": neutral_orientation_scores(sup),
        "anti_copy_accuracy_by_initial_pattern_on_true_changed": base.anti_copy_accuracy(train),
        "order_baseline_comparison": base.order_rule_best(train, "relation_comparison"),
        "order_baseline_state_changed": base.order_rule_best([r for r in train if r.get("query_kind") == "changed"], "state_query"),
        "held_surface_leak": base.held_surface_leak(train),
        "duplicate_text_label_conflicts": duplicate_text_label_conflicts(train),
        "exact_pair_degrees": dict(sorted(pair_degree_counts(train).items())),
    }


def write_outputs(out: Path, conditions: Dict[str, Dict[str, Any]], evals: Dict[str, List[Dict[str, Any]]]) -> None:
    for cond, cd in conditions.items():
        cdir = out / cond
        write_jsonl(cdir / "common_seen_train.jsonl", cd["common"])
        for arm, blocks in cd["arms"].items():
            write_jsonl(cdir / "arms" / arm / "train_supervised.jsonl", blocks["supervised"])
            write_jsonl(cdir / "arms" / arm / "train_unsup_text.jsonl", blocks["unsupervised"])
    for suite, rows in evals.items():
        write_jsonl(out / "eval" / f"{suite}.jsonl", rows)


def compare_conditions(conditions: Dict[str, Dict[str, Any]], arm: str) -> Dict[str, Any]:
    c = conditions["pair_connected"]["common"] + conditions["pair_connected"]["arms"][arm]["supervised"]
    r = conditions["pair_rewired"]["common"] + conditions["pair_rewired"]["arms"][arm]["supervised"]
    return {
        "input_text_multiset": multiset_equal([input_text_sig(x) for x in c], [input_text_sig(x) for x in r]),
        "input_text_plus_label_multiset": multiset_equal([input_text_label_sig(x) for x in c], [input_text_label_sig(x) for x in r]),
        "token_unigram_counts": count_equal(token_counts(c), token_counts(r)),
        "name_role_label_counts": count_equal(name_role_label_counts(c), name_role_label_counts(r)),
        "exact_pair_degree_counts": count_equal(pair_degree_counts(c), pair_degree_counts(r)),
        "pair_label_voice_relation_counts": count_equal(pair_hist_counts(c, include_geom=False), pair_hist_counts(r, include_geom=False)),
        "pair_geometry_label_counts_expected_intervention": count_equal(pair_hist_counts(c, include_geom=True), pair_hist_counts(r, include_geom=True)),
        "structural_counts_without_neutral_flag": count_equal(structural_counts(c, include_neutral=False), structural_counts(r, include_neutral=False)),
        "structural_counts_with_neutral_flag_expected_intervention": count_equal(structural_counts(c, include_neutral=True), structural_counts(r, include_neutral=True)),
    }


def eval_report(evals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for suite, rows in evals.items():
        out[suite] = {
            "rows": len(rows),
            "label_balance": label_balance(rows),
            "order_baseline": base.order_rule_best(rows, "relation_comparison") if any(r.get("task") == "relation_comparison" for r in rows) else base.order_rule_best(rows, "state_query"),
        }
    return out


def build_manifest(conditions: Dict[str, Dict[str, Any]], evals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "status": "PAIR_BINDING_NEUTRAL_SUBSTRATE_COMPLETE",
        "B_bridge_pairs": [list(p) for p in B_PAIRS],
        "R_rewired_pairs": [list(p) for p in R_PAIRS],
        "all_common_pairs": [list(p) for p in ALL_PAIRS],
        "conditions": {},
        "matched_checks": {},
        "eval": eval_report(evals),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }
    for cond, cd in conditions.items():
        m["conditions"][cond] = {"B_informative": cd["B_informative"], "R_informative": cd["R_informative"], "arms": {}}
        for arm, blocks in cd["arms"].items():
            m["conditions"][cond]["arms"][arm] = summarize_arm(cd["common"], blocks)
    for arm in ARMS:
        m["matched_checks"][arm] = compare_conditions(conditions, arm)

    gc: Dict[str, Any] = {}
    gc["B_R_exact_pair_overlap_zero"] = set(B_PAIRS).isdisjoint(set(R_PAIRS))
    gc["B_R_same_individual_name_set"] = sorted(n for p in B_PAIRS for n in p) == sorted(n for p in R_PAIRS for n in p)
    for arm, chk in m["matched_checks"].items():
        for key in ["input_text_multiset", "token_unigram_counts", "name_role_label_counts", "exact_pair_degree_counts", "pair_label_voice_relation_counts", "structural_counts_without_neutral_flag"]:
            gc[f"{arm}_{key}_equal"] = bool(chk[key]["equal"])
        # Expected intervention: same texts receive different labels in orientation-geometry cells.
        gc[f"{arm}_input_text_plus_label_multiset_not_equal_expected"] = not bool(chk["input_text_plus_label_multiset"]["equal"])
        gc[f"{arm}_pair_geometry_label_counts_differ_expected"] = not bool(chk["pair_geometry_label_counts_expected_intervention"]["equal"])
        for cond in CONDITIONS:
            ar = m["conditions"][cond]["arms"][arm]
            gc[f"{cond}_{arm}_no_duplicate_input_label_conflicts"] = ar["duplicate_text_label_conflicts"]["n_conflicting_inputs"] == 0
            gc[f"{cond}_{arm}_neutral_rank_zero"] = bool(ar["neutral_orientation_scores"]["all_assignments_at_half"])
        ca = m["conditions"]["pair_connected"]["arms"][arm]
        ra = m["conditions"]["pair_rewired"]["arms"][arm]
        gc[f"{arm}_formal_assignment_count_equal"] = ca["formal_satisfying_assignment_count_informative_only"] == ra["formal_satisfying_assignment_count_informative_only"]
        gc[f"{arm}_rows_equal"] = ca["total_model_train_rows"] == ra["total_model_train_rows"]
        gc[f"{arm}_train_token_total_equal"] = ca["train_token_total"] == ra["train_token_total"]
    m["global_checks"] = gc
    return m


def write_summary(out: Path, m: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research neutral-block pair-binding substrate")
    lines.append("")
    lines.append("This is the corrected no-GPU substrate after the research exposure flaw.  Both conditions have identical model-input text multisets, common seen-coordinate rows, bridge rows, token counts, exact pair degrees, per-name role/label counts, relation/voice/label counts, and no duplicate input with conflicting labels.  The only intended difference is which exact participant pairs carry orientation-informative held-held labels; the complementary pairs carry rank-zero neutral labels with the same surface exposure.")
    lines.append("")
    lines.append("## Pair rewiring")
    lines.append(f"- B bridge/base pairs: {m['B_bridge_pairs']}")
    lines.append(f"- R rewired pairs: {m['R_rewired_pairs']}")
    lines.append("- pair_connected: B informative, R neutral")
    lines.append("- pair_rewired: B neutral, R informative")
    lines.append("")
    lines.append("## Central readout")
    lines.append("")
    lines.append("| arm | condition | rows | heldheld info | heldheld neutral | bridge | formal assignments | neutral rank-zero | anti-copy same | anti-copy opposite |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for cond in CONDITIONS:
            ar = m["conditions"][cond]["arms"][arm]
            ac = ar["anti_copy_accuracy_by_initial_pattern_on_true_changed"]
            lines.append(f"| {arm} | {cond} | {ar['total_model_train_rows']} | {ar['heldheld_informative_rows']} | {ar['heldheld_neutral_rows']} | {ar['bridge_rows']} | {ar['formal_satisfying_assignment_count_informative_only']} | {ar['neutral_orientation_scores']['all_assignments_at_half']} | {ac.get('same', 'n/a') if ac.get('same') is not None else 'n/a'} | {ac.get('opposite', 'n/a') if ac.get('opposite') is not None else 'n/a'} |")
    lines.append("")
    lines.append("## Matched checks")
    lines.append("")
    lines.append("| arm | input text multiset | token unigrams | name role/label | exact pair degree | pair relation/voice/label | structural no-neutral-flag | input+label differs | geometry-label differs |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        chk = m["matched_checks"][arm]
        lines.append(f"| {arm} | {chk['input_text_multiset']['equal']} | {chk['token_unigram_counts']['equal']} | {chk['name_role_label_counts']['equal']} | {chk['exact_pair_degree_counts']['equal']} | {chk['pair_label_voice_relation_counts']['equal']} | {chk['structural_counts_without_neutral_flag']['equal']} | {not chk['input_text_plus_label_multiset']['equal']} | {not chk['pair_geometry_label_counts_expected_intervention']['equal']} |")
    lines.append("")
    lines.append("The final two columns are expected to differ: they are the intervention assigning orientation-informative versus rank-zero labels to B or R while preserving the same inputs and simple exposure counts.")
    lines.append("")
    lines.append("## Global checks")
    for k, v in m["global_checks"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Eval suites")
    for suite, er in m["eval"].items():
        lines.append(f"- {suite}: {er['rows']} rows, true_frac={er['label_balance']['true_frac']}")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("This substrate no longer supports the research criticism that bridge-supervised names or dyads differ in exposure.  A future learned comparison would be the minimum-cost test of pair-binding alignment: whether the same orientation information becomes reusable when the informative held-held labels sit on the exact bridge-supervised dyads rather than on degree-matched rewired dyads.  The result would be narrower than full graph connectivity, but cleaner: it addresses whether finite experience must align discriminative relation constraints with the learner's pair-binding interface.  A positive result still needs three simultaneous readouts: same-initial changed exact choice, pair-both conservation, and aligned-vs-inverted signed mixed margins.  A null after local train fit would again push toward architectural role/entity/state factorization.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- manifest: `{(out / 'manifest.json').relative_to(PROJECT_ROOT)}`")
    lines.append(f"- data root: `{out.relative_to(PROJECT_ROOT)}`")
    (out / "pair_binding_neutral_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    conditions = {cond: build_condition(cond) for cond in CONDITIONS}
    evals = base.build_eval(8)
    write_outputs(out, conditions, evals)
    manifest = build_manifest(conditions, evals)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(out, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "out": str(out.relative_to(PROJECT_ROOT)),
        "summary": str((out / "pair_binding_neutral_summary.md").relative_to(PROJECT_ROOT)),
        "manifest": str((out / "manifest.json").relative_to(PROJECT_ROOT)),
        "global_checks_all_true": all(bool(v) for v in manifest["global_checks"].values()),
        "failed_global_checks": {k: v for k, v in manifest["global_checks"].items() if not bool(v)},
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
