#!/usr/bin/env python3
"""research: relation-graph degree-preserving connectivity substrate.

This is a stricter rebuild after the original full-run design was rejected.
It removes the known filler exposure confound by keeping bridge supervision,
common seen-coordinate rows, exact name pairs, pair degrees, token unigrams, and
per-name role/label counts identical across conditions.  The only intended
rewiring is in the held-held relation graph:

- rel_connected: held-held comparison edges form one degree-2 component linking
  bridge-anchor relations h0/h2 to target held relations h1/h3.
- rel_disconnected: held-held comparison edges preserve every relation's degree
  and side-position counts but split anchors {h0,h2} from targets {h1,h3}.

This substrate tests a relation-coordinate connectivity necessity.  Unlike the
research filler-overlap construction, the disconnected arm is formally less
identifying for h1/h3 because the anchor-to-target relation path is removed; this
is intentional and explicit.

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
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import coordinate_connectivity_substrate as base  # noqa: E402

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/relation_graph_connectivity_substrate')

PAIRS: List[Tuple[str, str]] = list(base.ALL_TRAIN_PAIRS)
CONDITIONS = ["rel_connected", "rel_disconnected"]
ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]

CONNECTED_EDGES: List[Tuple[str, str]] = [
    ("h0_dax", "h1_mep"),
    ("h1_mep", "h2_norp"),
    ("h2_norp", "h3_ziv"),
    ("h3_ziv", "h0_dax"),
]
DISCONNECTED_EDGES: List[Tuple[str, str]] = [
    ("h0_dax", "h2_norp"),
    ("h2_norp", "h0_dax"),
    ("h1_mep", "h3_ziv"),
    ("h3_ziv", "h1_mep"),
]


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


def common_seen_train(n_per_pair_per_rel: int = 4) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oi = 0
    for pi, (a, b) in enumerate(PAIRS):
        for rel in base.SEEN_KEYS:
            for j in range(n_per_pair_per_rel):
                rows.extend(base.state_orbit(
                    f"common_seen_p{pi:02d}_{rel}_{j:02d}", rel, a, b,
                    pick_obj(base.TRAIN_CHANGED, oi), pick_obj(base.TRAIN_STATIC, oi * 3 + 1),
                    "train", "common_seen_coordinate"))
                oi += 1
    return rows


def heldheld_comparisons(edges: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oi = 0
    for ei, (r1, r2) in enumerate(edges):
        for pi, (a, b) in enumerate(PAIRS):
            for same in [False, True]:
                rows.extend(base.comparison_orbit(
                    f"train_hh_e{ei:02d}_p{pi:02d}_{'same' if same else 'diff'}",
                    r1, r2, a, b, pick_obj(base.TRAIN_CHANGED, oi + 100),
                    same=same, split="train", suite="heldheld_relation_graph_train"))
                oi += 1
    return rows


def bridge_state_rows(assignment: Dict[str, int], suite: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oi = 0
    for rel in base.ANCHOR_RELS:
        for pi, (a, b) in enumerate(PAIRS):
            for ip in ["opposite", "same"]:
                rows.extend(base.state_orbit(
                    f"train_{suite}_{rel}_p{pi:02d}_{ip}", rel, a, b,
                    pick_obj(base.TRAIN_CHANGED, oi + 200),
                    pick_obj(base.TRAIN_STATIC, oi * 3 + 201),
                    "train", suite, assignment=assignment, initial_pattern=ip))
                oi += 1
    return rows


def unsup_held_exposure() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oi = 0
    for pi, (a, b) in enumerate(PAIRS):
        for rel in base.HELD_KEYS:
            for voice in [0, 1]:
                rows.append(base.unsup_row(
                    f"train_held_exposure_p{pi:02d}_{rel}_v{voice}", rel, a, b,
                    pick_obj(base.TRAIN_CHANGED, oi + 300), "train", "held_event_exposure", voice))
                oi += 1
    return rows


def build_condition(condition: str) -> Dict[str, Any]:
    edges = CONNECTED_EDGES if condition == "rel_connected" else DISCONNECTED_EDGES
    common = common_seen_train()
    hh = heldheld_comparisons(edges)
    unsup = unsup_held_exposure()
    arms: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for arm in ARMS:
        ass = assignment_for_arm(arm)
        bridge = [] if ass is None else bridge_state_rows(ass, f"{arm}_bridge")
        arms[arm] = {"supervised": hh + bridge, "heldheld": hh, "bridge": bridge, "unsupervised": unsup}
    return {"common": common, "arms": arms, "edges": edges}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def text_parts(r: Dict[str, Any]) -> List[str]:
    if r.get("task") == "relation_comparison":
        return [str(r.get("event1", "")), str(r.get("event2", "")), str(r.get("text", ""))]
    if r.get("task") == "state_query":
        return [str(r.get("premise", "")), str(r.get("hypothesis", "")), str(r.get("cause_event", ""))]
    if "text" in r:
        return [str(r.get("text", ""))]
    return []


def token_counter(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        for p in text_parts(r):
            c.update(tok.lower() for tok in re.findall(r"\b\w+\b", p))
    return c


def pair_key(vals: Sequence[Any]) -> str | None:
    if len(vals) != 2:
        return None
    return f"{vals[0]}::{vals[1]}"


def pair_degree(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        if r.get("task") == "relation_comparison":
            pk = pair_key(r.get("arg_order1", []))
        else:
            pk = pair_key(r.get("arg_order", []))
        if pk:
            c[pk] += 1
    return c


def name_role_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    names = set(base.TRAIN_NAMES)
    for r in rows:
        label = "T" if bool(r.get("label")) else "F" if "label" in r else "NA"
        if r.get("task") == "state_query":
            rel = r.get("relation") or r.get("cause_relation")
            for idx, nm in enumerate(r.get("arg_order", [])):
                if nm in names:
                    c[(nm, "state_arg", idx, rel, r.get("voice"), r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot"), label)] += 1
            for idx, nm in enumerate(r.get("surface_order", [])):
                if nm in names:
                    c[(nm, "state_surface", idx, rel, r.get("voice"), r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot"), label)] += 1
            for role in ["candidate", "initial_owner", "initial_changed_owner", "static_owner", "supervised_changed_owner"]:
                nm = r.get(role)
                if nm in names:
                    c[(nm, role, rel, r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot"), r.get("candidate_slot") if role == "candidate" else None, label)] += 1
        elif r.get("task") == "relation_comparison":
            for side in [1, 2]:
                rel = r.get(f"relation{side}")
                for idx, nm in enumerate(r.get(f"arg_order{side}", [])):
                    if nm in names:
                        c[(nm, f"cmp_arg{side}", idx, rel, r.get(f"voice{side}"), label)] += 1
                for idx, nm in enumerate(r.get(f"surface_order{side}", [])):
                    if nm in names:
                        c[(nm, f"cmp_surface{side}", idx, rel, r.get(f"voice{side}"), label)] += 1
    return c


def structural_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        label = "T" if bool(r.get("label")) else "F" if "label" in r else "NA"
        if r.get("task") == "relation_comparison":
            c[(r.get("task"), r.get("relation1"), r.get("relation2"), r.get("voice1"), r.get("voice2"), label)] += 1
        elif r.get("task") == "state_query":
            c[(r.get("task"), r.get("relation") or r.get("cause_relation"), r.get("voice"), r.get("static_slot"), r.get("initial_pattern"), r.get("query_kind"), r.get("candidate_slot"), label)] += 1
        else:
            c[(r.get("task"), r.get("relation"), r.get("voice"), label)] += 1
    return c


def count_equal(a: Counter, b: Counter) -> Dict[str, Any]:
    d = a.copy(); d.subtract(b)
    nz = {str(k): int(v) for k, v in d.items() if v != 0}
    return {"equal": not nz, "n_differences": len(nz), "sample": dict(list(sorted(nz.items()))[:30])}


def rel_graph_components(edges: Sequence[Tuple[str, str]]) -> List[List[str]]:
    g: Dict[str, set] = defaultdict(set)
    for a, b in edges:
        g[a].add(b); g[b].add(a)
    for h in base.HELD_KEYS:
        g.setdefault(h, set())
    seen = set(); comps = []
    for h in base.HELD_KEYS:
        if h in seen:
            continue
        q = deque([h]); seen.add(h); comp = []
        while q:
            x = q.popleft(); comp.append(x)
            for y in g[x]:
                if y not in seen:
                    seen.add(y); q.append(y)
        comps.append(sorted(comp))
    return sorted(comps)


def relation_degree_side_counts(edges: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    deg = Counter(); side1 = Counter(); side2 = Counter()
    for a, b in edges:
        deg[a] += 1; deg[b] += 1; side1[a] += 1; side2[b] += 1
    return {"degree": dict(sorted(deg.items())), "side1": dict(sorted(side1.items())), "side2": dict(sorted(side2.items()))}


def label_balance(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [bool(r["label"]) for r in rows if "label" in r]
    return {"n": len(vals), "true": int(sum(vals)), "false": int(len(vals) - sum(vals)), "true_frac": (sum(vals) / len(vals) if vals else None)}


def row_words(r: Dict[str, Any]) -> int:
    return sum(len(re.findall(r"\b\w+\b", p)) for p in text_parts(r))


def summarize_arm(common: List[Dict[str, Any]], blocks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    sup = blocks["supervised"]
    train = common + sup
    sats = base.satisfying_assignments(sup)
    return {
        "common_seen_rows": len(common),
        "heldheld_rows": len(blocks["heldheld"]),
        "bridge_rows": len(blocks["bridge"]),
        "supervised_rows": len(sup),
        "total_model_train_rows": len(train),
        "unsupervised_rows_not_used_by_probe": len(blocks["unsupervised"]),
        "train_token_total": sum(row_words(r) for r in train),
        "label_balance_supervised": label_balance(sup),
        "label_balance_model_train": label_balance(train),
        "formal_satisfying_assignment_count": len(sats),
        "formal_satisfying_assignments": sats,
        "true_satisfies": base.TRUE_ASSIGNMENT in sats,
        "inverted_satisfies": base.INVERTED_ASSIGNMENT in sats,
        "anti_copy_accuracy_by_initial_pattern_on_true_changed": base.anti_copy_accuracy(train),
        "order_baseline_comparison": base.order_rule_best(train, "relation_comparison"),
        "order_baseline_state_changed": base.order_rule_best([r for r in train if r.get("query_kind") == "changed"], "state_query"),
        "held_surface_leak": base.held_surface_leak(train),
        "exact_pair_degrees": dict(sorted(pair_degree(train).items())),
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
    c = conditions["rel_connected"]["common"] + conditions["rel_connected"]["arms"][arm]["supervised"]
    d = conditions["rel_disconnected"]["common"] + conditions["rel_disconnected"]["arms"][arm]["supervised"]
    return {
        "token_unigram_counts": count_equal(token_counter(c), token_counter(d)),
        "name_role_label_counts": count_equal(name_role_counts(c), name_role_counts(d)),
        "exact_pair_degree_counts": count_equal(pair_degree(c), pair_degree(d)),
        "structural_counts": count_equal(structural_counts(c), structural_counts(d)),
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
    manifest: Dict[str, Any] = {
        "status": "RELATION_GRAPH_CONNECTIVITY_SUBSTRATE_COMPLETE",
        "pairs": [list(p) for p in PAIRS],
        "edge_sets": {
            "rel_connected": [list(e) for e in CONNECTED_EDGES],
            "rel_disconnected": [list(e) for e in DISCONNECTED_EDGES],
        },
        "relation_graph": {
            "rel_connected_components": rel_graph_components(CONNECTED_EDGES),
            "rel_disconnected_components": rel_graph_components(DISCONNECTED_EDGES),
            "rel_connected_degree_side_counts": relation_degree_side_counts(CONNECTED_EDGES),
            "rel_disconnected_degree_side_counts": relation_degree_side_counts(DISCONNECTED_EDGES),
        },
        "conditions": {},
        "matched_checks": {},
        "eval": eval_report(evals),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }
    for cond, cd in conditions.items():
        manifest["conditions"][cond] = {"edges": [list(e) for e in cd["edges"]], "arms": {}}
        for arm, blocks in cd["arms"].items():
            manifest["conditions"][cond]["arms"][arm] = summarize_arm(cd["common"], blocks)
    for arm in ARMS:
        manifest["matched_checks"][arm] = compare_conditions(conditions, arm)
    gc: Dict[str, Any] = {}
    gc["same_exact_pairs"] = True
    gc["connected_one_relation_component"] = len(manifest["relation_graph"]["rel_connected_components"]) == 1
    gc["disconnected_two_relation_components"] = len(manifest["relation_graph"]["rel_disconnected_components"]) == 2
    gc["relation_degrees_equal"] = manifest["relation_graph"]["rel_connected_degree_side_counts"]["degree"] == manifest["relation_graph"]["rel_disconnected_degree_side_counts"]["degree"]
    gc["relation_side1_counts_equal"] = manifest["relation_graph"]["rel_connected_degree_side_counts"]["side1"] == manifest["relation_graph"]["rel_disconnected_degree_side_counts"]["side1"]
    gc["relation_side2_counts_equal"] = manifest["relation_graph"]["rel_connected_degree_side_counts"]["side2"] == manifest["relation_graph"]["rel_disconnected_degree_side_counts"]["side2"]
    for arm, chk in manifest["matched_checks"].items():
        for name, rec in chk.items():
            # structural_counts differs by relation-pair topology and is reported, not required.
            if name != "structural_counts":
                gc[f"{arm}_{name}_equal"] = bool(rec["equal"])
        ca = manifest["conditions"]["rel_connected"]["arms"][arm]
        da = manifest["conditions"]["rel_disconnected"]["arms"][arm]
        gc[f"{arm}_rows_equal"] = ca["total_model_train_rows"] == da["total_model_train_rows"]
        gc[f"{arm}_supervised_rows_equal"] = ca["supervised_rows"] == da["supervised_rows"]
        gc[f"{arm}_tokens_total_equal"] = ca["train_token_total"] == da["train_token_total"]
    manifest["global_checks"] = gc
    return manifest


def write_summary(out: Path, m: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research relation-graph connectivity substrate")
    lines.append("")
    lines.append("This construction preserves the names, exact bridge state rows, common seen-coordinate rows, exact pair degrees, token exposure, and per-name role/label counts.  It rewires only the held-held relation graph while preserving each held relation's degree and side-position counts.")
    lines.append("")
    lines.append("## Relation edge sets")
    lines.append(f"- rel_connected: {m['edge_sets']['rel_connected']} -> components {m['relation_graph']['rel_connected_components']}")
    lines.append(f"- rel_disconnected: {m['edge_sets']['rel_disconnected']} -> components {m['relation_graph']['rel_disconnected_components']}")
    lines.append(f"- degree counts connected/disconnected: {m['relation_graph']['rel_connected_degree_side_counts']['degree']} / {m['relation_graph']['rel_disconnected_degree_side_counts']['degree']}")
    lines.append("")
    lines.append("## Central readout")
    lines.append("")
    lines.append("| arm | condition | rows | held-held | bridge | formal assignments | true satisfies | inverted satisfies | anti-copy same | anti-copy opposite |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for cond in CONDITIONS:
            ar = m["conditions"][cond]["arms"][arm]
            ac = ar["anti_copy_accuracy_by_initial_pattern_on_true_changed"]
            lines.append(f"| {arm} | {cond} | {ar['total_model_train_rows']} | {ar['heldheld_rows']} | {ar['bridge_rows']} | {ar['formal_satisfying_assignment_count']} | {ar['true_satisfies']} | {ar['inverted_satisfies']} | {ac.get('same', 'n/a') if ac.get('same') is not None else 'n/a'} | {ac.get('opposite', 'n/a') if ac.get('opposite') is not None else 'n/a'} |")
    lines.append("")
    lines.append("## Matched checks")
    lines.append("")
    lines.append("| arm | token unigrams | name role/label | exact pair degree | structural row counts |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm in ARMS:
        chk = m["matched_checks"][arm]
        lines.append(f"| {arm} | {chk['token_unigram_counts']['equal']} | {chk['name_role_label_counts']['equal']} | {chk['exact_pair_degree_counts']['equal']} | {chk['structural_counts']['equal']} |")
    lines.append("")
    lines.append("Structural row counts are intentionally not equal because the relation-pair edge set is the intervention.  The degree/side counts of individual relations are equal.")
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
    lines.append("This is a clean CPU substrate for a future minimal learned test of relation-coordinate connectivity.  Because rel_disconnected has no relation path from bridge anchors h0/h2 to targets h1/h3, target transfer is not formally determined there; the useful contrast is whether rel_connected can exploit the path without any extra filler exposure, while rel_disconnected should at most learn direct anchors.  A positive result still requires exact same-initial changed choice and pair-both conservation, plus signed mixed held-seen orientation.  If rel_connected fails after local fit, the evidence points toward missing architectural factorization rather than more counterexamples.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- manifest: `{(out / 'manifest.json').relative_to(PROJECT_ROOT)}`")
    lines.append(f"- data root: `{out.relative_to(PROJECT_ROOT)}`")
    (out / "relation_graph_connectivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "summary": str((out / "relation_graph_connectivity_summary.md").relative_to(PROJECT_ROOT)),
        "manifest": str((out / "manifest.json").relative_to(PROJECT_ROOT)),
        "global_checks_all_true": all(bool(v) for v in manifest["global_checks"].values()),
        "failed_global_checks": {k: v for k, v in manifest["global_checks"].items() if not bool(v)},
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
