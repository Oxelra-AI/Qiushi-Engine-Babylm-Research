#!/usr/bin/env python3
"""research: counterallocated low-k information-budget substrate.

CPU/file-only construction.  It prepares, but does not run, the next experiment
that would be justified only if the balanced k16 learned probe shows joint state
success plus signed mixed-relation transfer.

Why separate from research?
--------------------------
research prepared a useful k16 balanced binary condition, but its low-k spread
conditions used deterministic early-cell selection.  For a scientific dose law,
low-k conditions must be interpreted across matched allocations so that cell,
relation, voice, static-slot, name/object identity, and pair order are not
silently confused with the amount of independently informative experience.

This script produces panels of fixed-total conditions.  Each condition keeps the
same number of state pairs as the redundant k0 arm; it only changes which pairs
are independently informative same-initial worlds.  Across allocations within a
panel, the selected cells cycle over relation × voice × static_slot, and selected
within-cell pair offsets rotate.

No model loading, training, official BabyLM evaluation, upload, or leaderboard.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path("experiments/archive/representation_and_objectives")
ROOT = PROJECT_ROOT / "data/factorial_initial_ownership"
OUT_DEFAULT = PROJECT_ROOT / "data/counterallocated_budget_substrate"

ARMS = ["heldheld_only", "aligned_state_bridge", "inverted_state_bridge"]
BRIDGE_ARMS = {"aligned_state_bridge", "inverted_state_bridge"}
EVAL_SPLIT = ROOT / "underdetermined" / "eval"

SPREAD_K_TO_ALLOCATIONS = {0: 1, 1: 16, 2: 8, 4: 8, 8: 4, 16: 4}
SINGLE_K_TO_ALLOCATIONS = {1: 4, 2: 4, 4: 4, 8: 4, 16: 1}
SINGLE_RELATIONS = ["h0_dax", "h2_norp"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def save_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def group_state_pairs(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    other: List[Dict[str, Any]] = []
    pairs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("task") == "state_query":
            pairs[str(r["pair_id"])].append(r)
        else:
            other.append(r)
    return other, dict(pairs)


def get_initial_opposite_owner(r: Dict[str, Any]) -> str:
    final_owner = r["supervised_changed_owner"]
    a, b = r["arg_order"]
    return a if final_owner == b else b


def reconstruct_premise(initial_changed_owner: str, changed_obj: str, static_owner: str, static_obj: str, cause_event: str) -> str:
    return (
        f"At first, {initial_changed_owner} had the {changed_obj}, "
        f"and {static_owner} had the {static_obj}. "
        f"The event was this: {cause_event}"
    )


def annotate_row(r: Dict[str, Any], pattern: str, initial_owner: str, suffix: str = "", new_premise: str | None = None) -> Dict[str, Any]:
    out = dict(r)
    out["initial_pattern"] = pattern
    out["initial_changed_owner"] = initial_owner
    if suffix:
        out["pair_id"] = str(r["pair_id"]) + suffix
        out["row_id"] = str(r["row_id"]) + suffix
    if new_premise is not None:
        out["premise"] = new_premise
    return out


def make_variant(pair_rows: Sequence[Dict[str, Any]], pattern: str, suffix: str = "") -> List[Dict[str, Any]]:
    if pattern == "opposite":
        return [annotate_row(r, "opposite", get_initial_opposite_owner(r), suffix="") for r in pair_rows]
    if pattern != "same":
        raise ValueError(pattern)
    base = pair_rows[0]
    final_owner = base["supervised_changed_owner"]
    new_premise = reconstruct_premise(
        final_owner,
        base["changed_object"],
        base["static_owner"],
        base["static_object"],
        base["cause_event"],
    )
    return [annotate_row(r, "same", final_owner, suffix=suffix or "_ca_same", new_premise=new_premise) for r in pair_rows]


def cell_key(pair_rows: Sequence[Dict[str, Any]]) -> Tuple[str, str, int]:
    r = pair_rows[0]
    return str(r.get("relation")), str(r.get("voice")), int(r.get("static_slot"))


def pair_digest(pair_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    r = pair_rows[0]
    return {
        "pair_id": r.get("pair_id"),
        "relation": r.get("relation"),
        "voice": r.get("voice"),
        "static_slot": r.get("static_slot"),
        "arg_order": r.get("arg_order"),
        "changed_object": r.get("changed_object"),
        "static_object": r.get("static_object"),
        "supervised_changed_owner": r.get("supervised_changed_owner"),
        "initial_opposite_owner": get_initial_opposite_owner(r),
    }


def ordered_buckets(pairs: Dict[str, List[Dict[str, Any]]], relation_filter: str | None = None) -> Dict[Tuple[str, str, int], List[str]]:
    buckets: Dict[Tuple[str, str, int], List[str]] = defaultdict(list)
    for pid, rows in pairs.items():
        ck = cell_key(rows)
        if relation_filter is not None and ck[0] != relation_filter:
            continue
        buckets[ck].append(pid)
    for ck in list(buckets):
        buckets[ck] = sorted(buckets[ck])
    return dict(sorted(buckets.items()))


def select_allocated_pairs(pairs: Dict[str, List[Dict[str, Any]]], k: int, alloc_index: int, relation_filter: str | None = None) -> List[str]:
    buckets = ordered_buckets(pairs, relation_filter=relation_filter)
    cells = list(buckets)
    if not cells:
        raise ValueError("no cells")
    capacity = sum(len(v) for v in buckets.values())
    if k > capacity:
        raise ValueError(f"k={k} exceeds capacity={capacity} for relation_filter={relation_filter}")
    if k == 0:
        return []

    n_cells = len(cells)
    q, rem = divmod(k, n_cells)
    selected: List[str] = []
    rotated_cells = cells[alloc_index % n_cells:] + cells[:alloc_index % n_cells]
    # Rotate the within-cell pair offset on every allocation, not only after a
    # full cycle over cells.  Otherwise budgets divisible by the cell count
    # (e.g. spread k8/k16) repeat exactly the same pair identities across
    # allocations, confounding dose with name/object/pair identity.
    pair_offset = alloc_index % max(len(v) for v in buckets.values())

    # Base quota in every cell.
    for ci, ck in enumerate(cells):
        ids = buckets[ck]
        if q > len(ids):
            raise ValueError(f"cell {ck} has {len(ids)} pairs, q={q}")
        for j in range(q):
            selected.append(ids[(pair_offset + j + ci) % len(ids)])

    # Remainder cycles across cells over allocation index.
    for extra_i, ck in enumerate(rotated_cells[:rem]):
        ids = buckets[ck]
        used = set(selected)
        for shift in range(len(ids)):
            cand = ids[(pair_offset + q + extra_i + shift) % len(ids)]
            if cand not in used:
                selected.append(cand)
                used.add(cand)
                break
        else:
            raise ValueError(f"could not choose extra pair in {ck}")
    if len(selected) != k:
        raise AssertionError((k, selected))
    return sorted(selected)


def build_arm(base_rows: Sequence[Dict[str, Any]], selected_same: set[str], condition_suffix: str) -> List[Dict[str, Any]]:
    other, pairs = group_state_pairs(base_rows)
    out: List[Dict[str, Any]] = list(other)
    for pid in sorted(pairs):
        if pid in selected_same:
            out.extend(make_variant(pairs[pid], "same", suffix=condition_suffix))
        else:
            out.extend(make_variant(pairs[pid], "opposite"))
    return out


def owner_slot(owner: Any, row: Dict[str, Any]) -> int | None:
    order = row.get("arg_order") or []
    for i, x in enumerate(order):
        if x == owner:
            return i
    return None


def state_rule_pred(row: Dict[str, Any], rule: str) -> bool | None:
    if row.get("task") != "state_query":
        return None
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    if row.get("query_kind") != "changed":
        return None
    if rule == "event_role":
        return row.get("candidate") == row.get("supervised_changed_owner")
    init_slot = owner_slot(row.get("initial_changed_owner"), row)
    if init_slot is None:
        return None
    st = row.get("static_slot")
    if rule == "anti_copy":
        target_slot = 1 - init_slot
    elif rule == "copy_initial":
        target_slot = init_slot
    elif rule == "static_slot_switch":
        if st is None:
            return None
        target_slot = 1 - init_slot if int(st) == 0 else init_slot
    elif rule == "opposite_static_slot_switch":
        if st is None:
            return None
        target_slot = init_slot if int(st) == 0 else 1 - init_slot
    else:
        raise ValueError(rule)
    return row.get("candidate_slot") == target_slot


def mean(vals: Sequence[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def score_rule(rows: Sequence[Dict[str, Any]], rule: str, only_true: bool = False, query_kind: str | None = None) -> Dict[str, Any]:
    vals: List[float] = []
    for r in rows:
        if r.get("task") != "state_query":
            continue
        if only_true and not bool(r.get("label")):
            continue
        if query_kind is not None and r.get("query_kind") != query_kind:
            continue
        pred = state_rule_pred(r, rule)
        if pred is None:
            continue
        vals.append(float(bool(pred) == bool(r.get("label"))))
    return {"n": len(vals), "acc": mean(vals)}


def group_count(rows: Sequence[Dict[str, Any]], fields: Sequence[str], only_true_changed: bool = False) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        if r.get("task") != "state_query":
            continue
        if only_true_changed and not (r.get("query_kind") == "changed" and bool(r.get("label"))):
            continue
        key = "|".join(str(r.get(f)) for f in fields)
        c[key] += 1
    return dict(sorted(c.items()))


def pair_both(rows: Sequence[Dict[str, Any]], rule: str) -> Dict[str, Any]:
    true_state = [r for r in rows if r.get("task") == "state_query" and bool(r.get("label"))]
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in true_state:
        by_pair[str(r.get("pair_id"))][str(r.get("query_kind"))] = r
    vals: List[float] = []
    for d in by_pair.values():
        if "changed" not in d or "unchanged" not in d:
            continue
        pc = state_rule_pred(d["changed"], rule)
        pu = state_rule_pred(d["unchanged"], rule)
        if pc is None or pu is None:
            continue
        vals.append(float(bool(pc) == bool(d["changed"].get("label")) and bool(pu) == bool(d["unchanged"].get("label"))))
    return {"n": len(vals), "acc": mean(vals)}


def arm_summary(rows: Sequence[Dict[str, Any]], selected_same: Sequence[str]) -> Dict[str, Any]:
    state = [r for r in rows if r.get("task") == "state_query"]
    changed_true = [r for r in state if r.get("query_kind") == "changed" and bool(r.get("label"))]
    return {
        "rows": len(rows),
        "state_rows": len(state),
        "state_pairs": len({r.get("pair_id") for r in state}),
        "same_pair_count": len(selected_same),
        "selected_same_pair_ids": list(selected_same),
        "pattern_static_counts_rows": group_count(state, ["initial_pattern", "static_slot"]),
        "changed_true_cell_counts": group_count(changed_true, ["relation", "voice", "static_slot", "initial_pattern"], only_true_changed=False),
        "rule_scores_changed_true": {rule: score_rule(changed_true, rule, only_true=False) for rule in ["anti_copy", "copy_initial", "static_slot_switch", "opposite_static_slot_switch", "event_role"]},
        "rule_scores_all_state": {rule: score_rule(state, rule, only_true=False) for rule in ["anti_copy", "copy_initial", "static_slot_switch", "opposite_static_slot_switch", "event_role"]},
        "rule_pair_both": {rule: pair_both(state, rule) for rule in ["anti_copy", "copy_initial", "static_slot_switch", "opposite_static_slot_switch", "event_role"]},
    }


def write_condition(cond_dir: Path, base_arm_rows: Dict[str, List[Dict[str, Any]]], common_seen: Sequence[Dict[str, Any]], eval_rows: Dict[str, List[Dict[str, Any]]], condition_name: str, k: int, alloc_index: int, coverage: str, relation_filter: str | None) -> Dict[str, Any]:
    cond_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(cond_dir / "common_seen_train.jsonl", common_seen)
    for suite, rows in eval_rows.items():
        save_jsonl(cond_dir / "eval" / f"{suite}.jsonl", rows)
    crep: Dict[str, Any] = {
        "condition": condition_name,
        "k": k,
        "alloc_index": alloc_index,
        "coverage": coverage,
        "relation_filter": relation_filter,
        "arms": {},
    }
    for arm in ARMS:
        if arm in BRIDGE_ARMS:
            _, pairs = group_state_pairs(base_arm_rows[arm])
            selected = select_allocated_pairs(pairs, k, alloc_index, relation_filter=relation_filter)
            suffix = f"_ca_{coverage}_k{k:02d}_a{alloc_index:02d}_same"
            rows = build_arm(base_arm_rows[arm], set(selected), suffix)
            crep["arms"][arm] = arm_summary(rows, selected)
            crep["arms"][arm]["selected_same_pair_digest"] = [pair_digest(pairs[pid]) for pid in selected]
        else:
            selected = []
            rows = list(base_arm_rows[arm])
            crep["arms"][arm] = arm_summary(rows, selected)
        save_jsonl(cond_dir / "arms" / arm / "train_supervised.jsonl", rows)
        unsup = load_jsonl(ROOT / "underdetermined" / "arms" / arm / "train_unsup_text.jsonl")
        save_jsonl(cond_dir / "arms" / arm / "train_unsup_text.jsonl", unsup)
    return crep


def build(out: Path, clean: bool = False) -> Dict[str, Any]:
    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    common_seen = load_jsonl(ROOT / "underdetermined" / "common_seen_train.jsonl")
    eval_rows = {p.stem: load_jsonl(p) for p in sorted(EVAL_SPLIT.glob("*.jsonl"))}
    base_arm_rows = {arm: load_jsonl(ROOT / "underdetermined" / "arms" / arm / "train_supervised.jsonl") for arm in ARMS}

    conditions: List[Tuple[str, int, int, str, str | None]] = []
    for k, n_alloc in SPREAD_K_TO_ALLOCATIONS.items():
        for a in range(n_alloc):
            conditions.append((f"ca_k{k:02d}_spread_a{a:02d}", k, a, "spread", None))
    for rel in SINGLE_RELATIONS:
        rel_tag = rel.replace("_", "")
        for k, n_alloc in SINGLE_K_TO_ALLOCATIONS.items():
            for a in range(n_alloc):
                conditions.append((f"ca_k{k:02d}_single_{rel_tag}_a{a:02d}", k, a, f"single_{rel}", rel))

    report: Dict[str, Any] = {
        "status": "COUNTERALLOCATED_BUDGET_SUBSTRATE_COMPLETE",
        "source": str(ROOT),
        "out": str(out),
        "purpose": "Prepare counterallocated low-k conditions after balanced k16 confirms whether independent same-initial worlds can induce event-role structure.",
        "construction": {
            "fixed_total_state_pairs_per_bridge_arm": 32,
            "spread_allocations_by_k": SPREAD_K_TO_ALLOCATIONS,
            "single_relation_allocations_by_k": SINGLE_K_TO_ALLOCATIONS,
            "single_relations": SINGLE_RELATIONS,
            "eval_suites_copied_from_step282_underdetermined": sorted(eval_rows),
            "common_seen_rows": len(common_seen),
        },
        "conditions": {},
        "panel_summaries": {},
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }

    for cname, k, alloc, coverage, rel_filter in conditions:
        crep = write_condition(out / cname, base_arm_rows, common_seen, eval_rows, cname, k, alloc, coverage, rel_filter)
        report["conditions"][cname] = crep

    report["panel_summaries"] = panel_summary(report["conditions"])
    write_json(out / "counterallocated_budget_report.json", report)
    write_summary(out / "counterallocated_budget_summary.md", report)
    return report


def panel_summary(conditions: Dict[str, Any]) -> Dict[str, Any]:
    panels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cname, crep in conditions.items():
        panel = f"{crep['coverage']}|k{crep['k']:02d}"
        panels[panel].append(crep)
    out: Dict[str, Any] = {}
    for panel, reps in sorted(panels.items()):
        # Count selected cells across allocations for aligned arm; inverted is constructed identically by cell.
        cell_counts: Counter[str] = Counter()
        pair_counts: Counter[str] = Counter()
        anti_vals: List[float] = []
        event_vals: List[float] = []
        switch_vals: List[float] = []
        for rep in reps:
            arm = rep["arms"].get("aligned_state_bridge", {})
            for d in arm.get("selected_same_pair_digest", []):
                cell_counts[f"{d.get('relation')}|{d.get('voice')}|st{d.get('static_slot')}"] += 1
                pair_counts[str(d.get("pair_id"))] += 1
            anti_vals.append(arm.get("rule_scores_changed_true", {}).get("anti_copy", {}).get("acc"))
            event_vals.append(arm.get("rule_scores_changed_true", {}).get("event_role", {}).get("acc"))
            switch_vals.append(arm.get("rule_scores_changed_true", {}).get("static_slot_switch", {}).get("acc"))
        out[panel] = {
            "n_allocations": len(reps),
            "selected_cell_counts_across_allocations_aligned": dict(sorted(cell_counts.items())),
            "selected_pair_minmax_across_allocations_aligned": [min(pair_counts.values()) if pair_counts else 0, max(pair_counts.values()) if pair_counts else 0],
            "anti_copy_changed_true_acc_values": anti_vals,
            "event_role_changed_true_acc_values": event_vals,
            "switch_changed_true_acc_values": switch_vals,
        }
    return out


def fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_summary(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research counterallocated information-budget substrate")
    lines.append("")
    lines.append("This CPU-only construction prepares a future information-efficiency experiment only if the balanced k16 learned probe supports the mechanism.  Each condition keeps total state-pair count fixed and changes only which state worlds are independently informative (`initial=same`).")
    lines.append("")
    lines.append("## Panels")
    lines.append("")
    lines.append("| panel | allocations | selected cell counts across allocations | selected pair min..max | anti-copy changed acc values | event-role changed acc values | research-switch changed acc values |")
    lines.append("|---|---:|---|---|---|---|---|")
    for panel, ps in sorted(report["panel_summaries"].items()):
        lines.append(
            f"| `{panel}` | {ps['n_allocations']} | `{ps['selected_cell_counts_across_allocations_aligned']}` | "
            f"`{ps['selected_pair_minmax_across_allocations_aligned']}` | "
            f"`{[round(float(x), 3) for x in ps['anti_copy_changed_true_acc_values']]}` | "
            f"`{[round(float(x), 3) for x in ps['event_role_changed_true_acc_values']]}` | "
            f"`{[round(float(x), 3) for x in ps['switch_changed_true_acc_values']]}` |"
        )
    lines.append("")
    lines.append("## How to use scientifically")
    lines.append("")
    lines.append("- Do not launch these panels unless the research balanced k0/k16 learned run shows that k16 has genuine joint state success and signed mixed-relation transfer.")
    lines.append("- For spread panels, interpret k through allocation-averaged behavior, because individual low-k files cannot cover every relation × voice × static-slot cell.")
    lines.append("- The single-relation panels test whether an informative state bridge in one relation can propagate through the held-held comparison graph to another relation; failure there with spread success means relation coverage matters.")
    lines.append("- Compare every learned result against the transparent shortcut values in this summary and the research heuristic-baseline file.  Aggregate accuracy alone is not meaningful.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- full JSON report: `{path.parent / 'counterallocated_budget_report.json'}`")
    lines.append(f"- condition directories: `{path.parent}/ca_kXX_*`")
    lines.append("- source redundant substrate: `data/factorial_initial_ownership/underdetermined/`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--clean", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report = build(args.out, clean=args.clean)
    print(json.dumps({
        "status": report["status"],
        "out": report["out"],
        "summary": str(args.out / "counterallocated_budget_summary.md"),
        "n_conditions": len(report["conditions"]),
        "panels": sorted(report["panel_summaries"]),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
