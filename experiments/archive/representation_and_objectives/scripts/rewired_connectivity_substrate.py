#!/usr/bin/env python3
"""research: exposure-preserving rewired coordinate-connectivity substrate.

This rebuilds the research connected/disconnected contrast after the strategist
flagged a causal flaw: research changed which names/pairs received bridge
supervision, so a positive result would mix connectivity with known filler-support
effects.

The present construction keeps bridge supervision, common seen-coordinate rows,
name exposure, logical-role counts, relation counts, labels, voices, initial-owner
patterns, and token unigram exposure exactly matched across the two conditions.
The only intended difference is an edge rewiring of the held-held comparison
pairing:

- connected: held-held comparison rows use the same ordered name pairs as the
  bridge state rows, so each bridge-supervised exact pair also carries held-held
  relation constraints.
- rewired: held-held comparison rows use a degree-preserving permutation of the
  second-side names. Every individual name has the same role/label/relation
  exposure as in connected, but no held-held comparison edge uses an exact bridge
  pair.

No model loading, no GPU training, no BabyLM evaluation, and no upload.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import coordinate_connectivity_substrate as base  # noqa: E402

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/rewired_connectivity_substrate')

# Use all eight research train pairs as the bridge set, so no condition changes
# which names/pairs receive bridge state supervision.
BASE_PAIRS: List[Tuple[str, str]] = list(base.ALL_TRAIN_PAIRS)
N = len(BASE_PAIRS)
# Degree-preserving rewiring: preserve first-side names and cyclically permute
# second-side names.  No exact bridge pair remains a comparison pair.
REWIRED_PAIRS: List[Tuple[str, str]] = [(BASE_PAIRS[i][0], BASE_PAIRS[(i + 1) % N][1]) for i in range(N)]
COMMON_PAIRS: List[Tuple[str, str]] = []
for p in BASE_PAIRS + REWIRED_PAIRS:
    if p not in COMMON_PAIRS:
        COMMON_PAIRS.append(p)

CONDITIONS = ["connected", "rewired"]
ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]


def rel_assignment_for_arm(arm: str) -> Dict[str, int] | None:
    if arm == "aligned_state_bridge":
        return base.TRUE_ASSIGNMENT
    if arm == "inverted_state_bridge":
        return base.INVERTED_ASSIGNMENT
    if arm == "heldheld_only":
        return None
    raise ValueError(arm)


def pick_obj(pool: List[str], idx: int) -> str:
    return pool[idx % len(pool)]


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


def heldheld_comparisons_degree_matched(pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Held-held comparison rows with exact per-pair/per-relation label balance.

    For every pair and every held-held edge, emit one same=True and one same=False
    orbit.  Each orbit expands to four active/passive voice combinations.  This
    makes per-name relation, role, and label exposure independent of which
    second-side names are rewired.  The metadata suite string is deliberately
    identical across conditions so postbuild structural counts compare the
    scientific row type rather than the condition name.
    """
    rows: List[Dict[str, Any]] = []
    oi = 0
    suite = "heldheld_edge_rewired_train"
    for ei, (r1, r2) in enumerate(base.HH_EDGES):
        for pi, (a, b) in enumerate(pairs):
            for same in [False, True]:
                rows.extend(base.comparison_orbit(
                    f"train_hh_e{ei:02d}_p{pi:02d}_{'same' if same else 'diff'}",
                    r1, r2, a, b, pick_obj(base.TRAIN_CHANGED, oi + 100),
                    same=same, split="train", suite=suite))
                oi += 1
    return rows


def bridge_state_balanced(pairs: Sequence[Tuple[str, str]], assignment: Dict[str, int], suite: str) -> List[Dict[str, Any]]:
    """Bridge state rows identical across conditions for each arm.

    Each bridge pair sees both anchor held relations and both initial ownership
    patterns.  State_orbit supplies both voices and both static slots, so the
    research static-slot diagonal is absent by construction.
    """
    rows: List[Dict[str, Any]] = []
    oi = 0
    for rel in base.ANCHOR_RELS:
        for pi, (a, b) in enumerate(pairs):
            for ip in ["opposite", "same"]:
                rows.extend(base.state_orbit(
                    f"train_{suite}_{rel}_p{pi:02d}_{ip}", rel, a, b,
                    pick_obj(base.TRAIN_CHANGED, oi + 200),
                    pick_obj(base.TRAIN_STATIC, oi * 3 + 201),
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
    if condition == "connected":
        comparison_pairs = BASE_PAIRS
    elif condition == "rewired":
        comparison_pairs = REWIRED_PAIRS
    else:
        raise ValueError(condition)

    common = common_seen_train(COMMON_PAIRS)
    hh = heldheld_comparisons_degree_matched(comparison_pairs)
    unsup = unsup_held_exposure(COMMON_PAIRS)
    arms: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for arm in ARMS:
        assignment = rel_assignment_for_arm(arm)
        if assignment is None:
            bridge: List[Dict[str, Any]] = []
        else:
            bridge = bridge_state_balanced(BASE_PAIRS, assignment, f"{arm}_bridge")
        arms[arm] = {"supervised": hh + bridge, "bridge": bridge, "heldheld": hh, "unsupervised": unsup}
    return {"common": common, "arms": arms, "comparison_pairs": comparison_pairs, "bridge_pairs": BASE_PAIRS}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def load_text_parts(row: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    if row.get("task") == "relation_comparison":
        parts.extend([str(row.get("event1", "")), str(row.get("event2", "")), str(row.get("text", ""))])
    elif row.get("task") == "state_query":
        parts.extend([str(row.get("premise", "")), str(row.get("hypothesis", "")), str(row.get("cause_event", ""))])
    elif "text" in row:
        parts.append(str(row.get("text", "")))
    return [p for p in parts if p]


def token_counter(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        for p in load_text_parts(r):
            c.update(tok.lower() for tok in re.findall(r"\b\w+\b", p))
    return c


def name_role_counter(rows: Sequence[Dict[str, Any]]) -> Counter:
    """Count name exposure by visible/logical role and label context."""
    c: Counter = Counter()
    names = set(base.TRAIN_NAMES)
    for r in rows:
        task = r.get("task")
        label = "T" if bool(r.get("label")) else "F" if "label" in r else "NA"
        suite = str(r.get("suite"))
        if task == "state_query":
            rel = r.get("relation") or r.get("cause_relation")
            for idx, nm in enumerate(r.get("arg_order", [])):
                if nm in names:
                    c[(nm, "state_arg_slot", idx, rel, r.get("query_kind"), label)] += 1
            for idx, nm in enumerate(r.get("surface_order", [])):
                if nm in names:
                    c[(nm, "state_surface_pos", idx, rel, r.get("voice"), r.get("query_kind"), label)] += 1
            cand = r.get("candidate")
            if cand in names:
                c[(cand, "state_candidate", r.get("candidate_slot"), rel, r.get("query_kind"), label)] += 1
            init = r.get("initial_owner") or r.get("initial_changed_owner")
            if init in names:
                c[(init, "state_initial_owner", rel, r.get("initial_pattern"), r.get("query_kind"), label)] += 1
            st = r.get("static_owner")
            if st in names:
                c[(st, "state_static_owner", rel, r.get("static_slot"), r.get("query_kind"), label)] += 1
        elif task == "relation_comparison":
            for side in [1, 2]:
                rel = r.get(f"relation{side}")
                for idx, nm in enumerate(r.get(f"arg_order{side}", [])):
                    if nm in names:
                        c[(nm, f"comparison_arg{side}_slot", idx, rel, label)] += 1
                for idx, nm in enumerate(r.get(f"surface_order{side}", [])):
                    if nm in names:
                        c[(nm, f"comparison_surface{side}_pos", idx, rel, r.get(f"voice{side}"), label)] += 1
        elif task == "unsupervised_event_text":
            rel = r.get("relation")
            for idx, nm in enumerate(r.get("arg_order", [])):
                if nm in names:
                    c[(nm, "unsup_arg_slot", idx, rel, suite)] += 1
            for idx, nm in enumerate(r.get("surface_order", [])):
                if nm in names:
                    c[(nm, "unsup_surface_pos", idx, rel, r.get("voice"), suite)] += 1
    return c


def structural_counter(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        task = r.get("task")
        label = "T" if bool(r.get("label")) else "F" if "label" in r else "NA"
        if task == "state_query":
            c[(task, r.get("suite"), r.get("relation") or r.get("cause_relation"), r.get("voice"),
               r.get("static_slot"), r.get("initial_pattern"), r.get("query_kind"), r.get("candidate_slot"), label)] += 1
        elif task == "relation_comparison":
            c[(task, r.get("suite"), r.get("relation1"), r.get("relation2"), r.get("voice1"), r.get("voice2"), label)] += 1
        else:
            c[(task, r.get("suite"), r.get("relation"), r.get("voice"), label)] += 1
    return c


def pair_key(pair: Tuple[str, str]) -> str:
    return f"{pair[0]}::{pair[1]}"


def condition_topology(common: Sequence[Dict[str, Any]], bridge: Sequence[Dict[str, Any]], hh: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bridge_pairs = Counter()
    comparison_pairs = Counter()
    common_pairs = Counter()
    relation_by_pair: Dict[str, Counter] = defaultdict(Counter)

    def ordered_arg_pair(r: Dict[str, Any], which: str = "arg_order") -> Tuple[str, str] | None:
        vals = r.get(which, [])
        if len(vals) != 2:
            return None
        return (str(vals[0]), str(vals[1]))

    for r in common:
        if r.get("task") == "state_query":
            p = ordered_arg_pair(r)
            if p:
                common_pairs[pair_key(p)] += 1
    for r in bridge:
        if r.get("task") == "state_query":
            p = ordered_arg_pair(r)
            if p:
                bridge_pairs[pair_key(p)] += 1
                relation_by_pair[pair_key(p)][f"bridge_{r.get('relation')}"] += 1
    for r in hh:
        if r.get("task") == "relation_comparison":
            p = ordered_arg_pair(r, "arg_order1")
            if p:
                comparison_pairs[pair_key(p)] += 1
                relation_by_pair[pair_key(p)][f"cmp_{r.get('relation1')}_{r.get('relation2')}"] += 1

    bset = set(bridge_pairs)
    cset = set(comparison_pairs)
    # Graph over exact ordered pair nodes and relation-type nodes.
    graph: Dict[str, set] = defaultdict(set)
    for pk, ct in bridge_pairs.items():
        for reltag, n in relation_by_pair[pk].items():
            if reltag.startswith("bridge_") and n > 0:
                rn = f"REL::{reltag}"
                pn = f"PAIR::{pk}"
                graph[pn].add(rn); graph[rn].add(pn)
    for pk, ct in comparison_pairs.items():
        for reltag, n in relation_by_pair[pk].items():
            if reltag.startswith("cmp_") and n > 0:
                rn = f"REL::{reltag}"
                pn = f"PAIR::{pk}"
                graph[pn].add(rn); graph[rn].add(pn)
    seen = set()
    comps = []
    for node in sorted(graph):
        if node in seen:
            continue
        q = deque([node]); seen.add(node); comp = []
        while q:
            x = q.popleft(); comp.append(x)
            for y in graph[x]:
                if y not in seen:
                    seen.add(y); q.append(y)
        comps.append(sorted(comp))

    return {
        "n_common_exact_pairs": len(common_pairs),
        "n_bridge_exact_pairs": len(bridge_pairs),
        "n_comparison_exact_pairs": len(comparison_pairs),
        "bridge_comparison_exact_pair_overlap": len(bset & cset),
        "bridge_only_exact_pairs": sorted(bset - cset),
        "comparison_only_exact_pairs": sorted(cset - bset),
        "overlap_exact_pairs": sorted(bset & cset),
        "bridge_pair_row_counts": dict(sorted(bridge_pairs.items())),
        "comparison_pair_row_counts": dict(sorted(comparison_pairs.items())),
        "common_pair_row_counts": dict(sorted(common_pairs.items())),
        "pair_relation_type_counts": {k: dict(v) for k, v in sorted(relation_by_pair.items())},
        "exact_pair_relation_components": comps,
    }


def counter_equal(a: Counter, b: Counter) -> Dict[str, Any]:
    diff = a.copy()
    diff.subtract(b)
    nz = {str(k): v for k, v in diff.items() if v != 0}
    return {"equal": not nz, "n_differences": len(nz), "sample_differences": dict(list(sorted(nz.items()))[:20])}


def label_balance(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [bool(r["label"]) for r in rows if "label" in r]
    return {"n": len(vals), "true": int(sum(vals)), "false": int(len(vals) - sum(vals)), "true_frac": (sum(vals) / len(vals) if vals else None)}


def row_words(row: Dict[str, Any]) -> int:
    return sum(len(re.findall(r"\b\w+\b", p)) for p in load_text_parts(row))


def summarize_arm(common: Sequence[Dict[str, Any]], blocks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    sup = blocks["supervised"]
    all_train = list(common) + list(sup)
    sats = base.satisfying_assignments(sup)
    topo = condition_topology(common, blocks["bridge"], blocks["heldheld"])
    ac = base.anti_copy_accuracy(all_train)
    return {
        "common_seen_rows": len(common),
        "supervised_rows": len(sup),
        "bridge_rows": len(blocks["bridge"]),
        "heldheld_rows": len(blocks["heldheld"]),
        "unsupervised_rows_not_used_by_probe": len(blocks["unsupervised"]),
        "total_model_train_rows": len(all_train),
        "label_balance_supervised": label_balance(sup),
        "label_balance_model_train": label_balance(all_train),
        "train_token_total": sum(row_words(r) for r in all_train),
        "formal_satisfying_assignments": len(sats),
        "satisfying_assignments": sats,
        "true_satisfies": base.TRUE_ASSIGNMENT in sats,
        "inverted_satisfies": base.INVERTED_ASSIGNMENT in sats,
        "anti_copy_accuracy_by_initial_pattern_on_true_changed": ac,
        "order_baseline_comparison": base.order_rule_best(all_train, "relation_comparison"),
        "order_baseline_state_changed": base.order_rule_best([r for r in all_train if r.get("query_kind") == "changed"], "state_query"),
        "held_surface_leak": base.held_surface_leak(all_train),
        "topology": topo,
    }


def write_outputs(out: Path, conditions: Dict[str, Dict[str, Any]], evals: Dict[str, List[Dict[str, Any]]]) -> None:
    for cond, cd in conditions.items():
        cond_out = out / cond
        write_jsonl(cond_out / "common_seen_train.jsonl", cd["common"])
        for arm, blocks in cd["arms"].items():
            write_jsonl(cond_out / "arms" / arm / "train_supervised.jsonl", blocks["supervised"])
            write_jsonl(cond_out / "arms" / arm / "train_unsup_text.jsonl", blocks["unsupervised"])
    for suite, rows in evals.items():
        write_jsonl(out / "eval" / f"{suite}.jsonl", rows)


def compare_condition_counters(conditions: Dict[str, Dict[str, Any]], arm: str) -> Dict[str, Any]:
    c_rows = conditions["connected"]["common"] + conditions["connected"]["arms"][arm]["supervised"]
    r_rows = conditions["rewired"]["common"] + conditions["rewired"]["arms"][arm]["supervised"]
    return {
        "token_unigram_counts": counter_equal(token_counter(c_rows), token_counter(r_rows)),
        "name_role_label_relation_counts": counter_equal(name_role_counter(c_rows), name_role_counter(r_rows)),
        "structural_relation_voice_label_counts": counter_equal(structural_counter(c_rows), structural_counter(r_rows)),
    }


def eval_report(evals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    rep: Dict[str, Any] = {}
    for suite, rows in evals.items():
        rep[suite] = {
            "rows": len(rows),
            "label_balance": label_balance(rows),
            "order_baseline": base.order_rule_best(rows, "relation_comparison") if any(r.get("task") == "relation_comparison" for r in rows) else base.order_rule_best(rows, "state_query"),
        }
    return rep


def global_checks(manifest: Dict[str, Any]) -> Dict[str, Any]:
    gc: Dict[str, Any] = {}
    gc["base_rewired_exact_pair_overlap_zero"] = set(BASE_PAIRS).isdisjoint(REWIRED_PAIRS)
    gc["same_training_names"] = sorted(set(n for p in BASE_PAIRS for n in p)) == sorted(set(n for p in REWIRED_PAIRS for n in p))
    gc["first_side_names_preserved"] = [p[0] for p in BASE_PAIRS] == [p[0] for p in REWIRED_PAIRS]
    gc["second_side_names_permuted"] = sorted(p[1] for p in BASE_PAIRS) == sorted(p[1] for p in REWIRED_PAIRS)
    for arm in ARMS:
        eq = manifest["matched_exposure_checks"][arm]
        for name, rec in eq.items():
            gc[f"{arm}_{name}_equal"] = bool(rec["equal"])
        c_ar = manifest["conditions"]["connected"]["arms"][arm]
        r_ar = manifest["conditions"]["rewired"]["arms"][arm]
        gc[f"{arm}_formal_assignment_count_equal"] = c_ar["formal_satisfying_assignments"] == r_ar["formal_satisfying_assignments"]
        gc[f"{arm}_supervised_rows_equal"] = c_ar["supervised_rows"] == r_ar["supervised_rows"]
        gc[f"{arm}_model_train_rows_equal"] = c_ar["total_model_train_rows"] == r_ar["total_model_train_rows"]
        gc[f"{arm}_connected_exact_pair_overlap_positive"] = c_ar["topology"]["bridge_comparison_exact_pair_overlap"] == len(BASE_PAIRS) if c_ar["bridge_rows"] else True
        gc[f"{arm}_rewired_exact_pair_overlap_zero"] = r_ar["topology"]["bridge_comparison_exact_pair_overlap"] == 0 if r_ar["bridge_rows"] else True
    return gc


def write_summary(out: Path, manifest: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research exposure-preserving rewired connectivity substrate")
    lines.append("")
    lines.append("This rebuild removes the research name-exposure flaw.  Bridge state supervision is identical in the two conditions; held-held comparison exposure is rewired so individual names keep the same role/relation/label counts while exact bridge-pair overlap changes.")
    lines.append("")
    lines.append("## Pair construction")
    lines.append("")
    lines.append(f"- bridge/base pairs: {manifest['base_pairs']}")
    lines.append(f"- rewired comparison pairs: {manifest['rewired_pairs']}")
    lines.append("- connected condition: held-held comparisons use the bridge/base pairs")
    lines.append("- rewired condition: held-held comparisons use the degree-preserving second-name permutation")
    lines.append("")
    lines.append("## Central structural readout")
    lines.append("")
    lines.append("| arm | condition | common | held-held | bridge | model-train rows | exact bridge/comparison pair overlap | formal assignments | anti-copy opposite | anti-copy same |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for cond in CONDITIONS:
            ar = manifest["conditions"][cond]["arms"][arm]
            ac = ar["anti_copy_accuracy_by_initial_pattern_on_true_changed"]
            lines.append(f"| {arm} | {cond} | {ar['common_seen_rows']} | {ar['heldheld_rows']} | {ar['bridge_rows']} | {ar['total_model_train_rows']} | {ar['topology']['bridge_comparison_exact_pair_overlap']} | {ar['formal_satisfying_assignments']} | {ac.get('opposite', 'n/a') if ac.get('opposite') is not None else 'n/a'} | {ac.get('same', 'n/a') if ac.get('same') is not None else 'n/a'} |")
    lines.append("")
    lines.append("## Matched exposure checks")
    lines.append("")
    lines.append("| arm | token unigrams | name role/label/relation counts | structural relation/voice/label counts |")
    lines.append("|---|---:|---:|---:|")
    for arm in ARMS:
        chk = manifest["matched_exposure_checks"][arm]
        lines.append(f"| {arm} | {chk['token_unigram_counts']['equal']} | {chk['name_role_label_relation_counts']['equal']} | {chk['structural_relation_voice_label_counts']['equal']} |")
    lines.append("")
    lines.append("## Global checks")
    for k, v in manifest["global_checks"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Eval suites")
    for suite, er in manifest["eval"].items():
        lines.append(f"- {suite}: {er['rows']} rows, true_frac={er['label_balance']['true_frac']}")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("This file-only construction is the corrected substrate for a future minimal learned comparison.  A useful positive learned result would require the connected condition, but not the rewired condition, to show same-initial changed exact choice, pair-both conservation, and signed mixed held-seen orientation.  If both fail after local train fit, the evidence points away from data topology on this surface and toward an explicit role/entity/state interface.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- manifest: `{(out / 'manifest.json').relative_to(PROJECT_ROOT)}`")
    lines.append(f"- connected data: `{(out / 'connected').relative_to(PROJECT_ROOT)}`")
    lines.append(f"- rewired data: `{(out / 'rewired').relative_to(PROJECT_ROOT)}`")
    lines.append(f"- eval data: `{(out / 'eval').relative_to(PROJECT_ROOT)}`")
    (out / "rewired_connectivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    conditions = {cond: build_condition(cond) for cond in CONDITIONS}
    evals = base.build_eval(8)
    write_outputs(out, conditions, evals)

    manifest: Dict[str, Any] = {
        "status": "REWIRED_CONNECTIVITY_SUBSTRATE_COMPLETE",
        "base_pairs": [list(p) for p in BASE_PAIRS],
        "rewired_pairs": [list(p) for p in REWIRED_PAIRS],
        "common_pairs": [list(p) for p in COMMON_PAIRS],
        "conditions": {},
        "matched_exposure_checks": {},
        "eval": eval_report(evals),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }

    for cond, cd in conditions.items():
        manifest["conditions"][cond] = {
            "comparison_pairs": [list(p) for p in cd["comparison_pairs"]],
            "bridge_pairs": [list(p) for p in cd["bridge_pairs"]],
            "arms": {},
        }
        for arm, blocks in cd["arms"].items():
            manifest["conditions"][cond]["arms"][arm] = summarize_arm(cd["common"], blocks)

    for arm in ARMS:
        manifest["matched_exposure_checks"][arm] = compare_condition_counters(conditions, arm)

    manifest["global_checks"] = global_checks(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(out, manifest)

    print(json.dumps({
        "status": manifest["status"],
        "out": str(out.relative_to(PROJECT_ROOT)),
        "summary": str((out / "rewired_connectivity_summary.md").relative_to(PROJECT_ROOT)),
        "manifest": str((out / "manifest.json").relative_to(PROJECT_ROOT)),
        "global_checks_all_true": all(bool(v) for v in manifest["global_checks"].values()),
        "failed_global_checks": {k: v for k, v in manifest["global_checks"].items() if not bool(v)},
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
