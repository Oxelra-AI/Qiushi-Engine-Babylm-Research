#!/usr/bin/env python3
"""Transparent heuristic baselines for research pair-binding neutral substrate.

CPU-only; no model loading or training.  Scores state shortcuts and relation-comparison
surface rules on the exact files used by the learned pair-binding pilot.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

PROJECT = Path("experiments/archive/representation_and_objectives")
DATA_ROOT = PROJECT / "data/pair_binding_neutral_substrate"
OUT = PROJECT / "data/pair_binding_heuristic_baselines"
CONDITIONS = ["pair_connected", "pair_rewired"]
ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]
EVAL_SUITES = ["paired_state_conservation", "cross_template_state_readout", "mixed_held_seen_orientation", "heldheld_unseen_edge_closure"]
B_PREFIX = "B_hh"
R_PREFIX = "R_hh"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def slot_of_owner(owner: Any, row: Dict[str, Any]) -> int | None:
    order = row.get("arg_order") or []
    for i, x in enumerate(order):
        if x == owner:
            return i
    return None


def pred_anticopy(row: Dict[str, Any]) -> bool | None:
    if row.get("task") != "state_query": return None
    if row.get("query_kind") == "changed":
        init = row.get("initial_owner") or row.get("initial_changed_owner")
        s = slot_of_owner(init, row)
        if s is None: return None
        return row.get("candidate_slot") == 1 - s
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    return None


def pred_copy_initial(row: Dict[str, Any]) -> bool | None:
    if row.get("task") != "state_query": return None
    if row.get("query_kind") == "changed":
        return row.get("candidate") == (row.get("initial_owner") or row.get("initial_changed_owner"))
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    return None


def pred_static_slot_switch(row: Dict[str, Any]) -> bool | None:
    if row.get("task") != "state_query": return None
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    init = row.get("initial_owner") or row.get("initial_changed_owner")
    s = slot_of_owner(init, row); st = row.get("static_slot")
    if s is None or st is None: return None
    target = 1 - s if int(st) == 0 else s
    return row.get("candidate_slot") == target


def pred_static_slot_opposite(row: Dict[str, Any]) -> bool | None:
    if row.get("task") != "state_query": return None
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    init = row.get("initial_owner") or row.get("initial_changed_owner")
    s = slot_of_owner(init, row); st = row.get("static_slot")
    if s is None or st is None: return None
    target = s if int(st) == 0 else 1 - s
    return row.get("candidate_slot") == target


def pred_event_role_oracle(row: Dict[str, Any]) -> bool | None:
    if row.get("task") != "state_query": return None
    if row.get("query_kind") == "changed":
        return row.get("candidate") == row.get("supervised_changed_owner")
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    return None


STATE_RULES: Dict[str, Callable[[Dict[str, Any]], bool | None]] = {
    "anti_copy": pred_anticopy,
    "copy_initial": pred_copy_initial,
    "static_slot_step282_switch": pred_static_slot_switch,
    "static_slot_opposite_switch": pred_static_slot_opposite,
    "event_role_oracle": pred_event_role_oracle,
}


def comp_geom_same(row: Dict[str, Any]) -> bool | None:
    if row.get("task") != "relation_comparison": return None
    if "geom_same_final_owner_under_true" in row:
        return bool(row.get("geom_same_final_owner_under_true"))
    return None


def comp_neutral_voice(row: Dict[str, Any]) -> bool | None:
    if row.get("task") != "relation_comparison": return None
    if "geom_same_final_owner_under_true" not in row: return None
    v1 = 0 if row.get("voice1") == "active" else 1
    v2 = 0 if row.get("voice2") == "active" else 1
    return bool(row.get("geom_same_final_owner_under_true")) ^ bool((v1 + v2) % 2)


def comp_pair_set_oracle(row: Dict[str, Any], condition: str) -> bool | None:
    if row.get("task") != "relation_comparison": return None
    rid = str(row.get("row_id", ""))
    if condition == "pair_connected":
        return comp_geom_same(row) if rid.startswith(B_PREFIX) else comp_neutral_voice(row)
    if condition == "pair_rewired":
        return comp_neutral_voice(row) if rid.startswith(B_PREFIX) else comp_geom_same(row)
    return None


COMP_RULES = {
    "geom_same_true_parity": comp_geom_same,
    "neutral_voice_xor": comp_neutral_voice,
}


def score(rows: Sequence[Dict[str, Any]], fn: Callable[[Dict[str, Any]], bool | None]) -> Dict[str, Any]:
    vals = []
    skipped = 0
    pred_true = []
    for r in rows:
        if "label" not in r:
            continue
        p = fn(r)
        if p is None:
            skipped += 1
            continue
        vals.append(float(bool(p) == bool(r["label"])))
        pred_true.append(float(bool(p)))
    return {"n": len(vals), "skipped": skipped, "acc": mean(vals), "pred_true_frac": mean(pred_true)}


def group_key(r: Dict[str, Any], fields: Sequence[str]) -> str:
    return "|".join("None" if r.get(f) is None else str(r.get(f)) for f in fields)


def grouped(rows: Sequence[Dict[str, Any]], fn: Callable[[Dict[str, Any]], bool | None], fields: Sequence[str]) -> Dict[str, Any]:
    d: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows: d[group_key(r, fields)].append(r)
    return {k: score(v, fn) for k, v in sorted(d.items())}


def choice_for_state_rule(rows: Sequence[Dict[str, Any]], fn: Callable[[Dict[str, Any]], bool | None]) -> Dict[str, Any]:
    true_rows = [r for r in rows if r.get("task") == "state_query" and bool(r.get("label"))]
    vals = []
    by_pattern = defaultdict(list)
    by_pattern_static = defaultdict(list)
    pair_both = []
    by_pair = defaultdict(dict)
    for r in true_rows:
        p = fn(r)
        if p is None: continue
        ok = bool(p) == bool(r["label"])
        vals.append(float(ok))
        by_pattern[str(r.get("initial_pattern"))].append(float(ok))
        by_pattern_static[f"{r.get('initial_pattern')}|{r.get('static_slot')}"].append(float(ok))
        by_pair[str(r.get("pair_id"))][str(r.get("query_kind"))] = ok
    for d in by_pair.values():
        if "changed" in d and "unchanged" in d:
            pair_both.append(float(d["changed"] and d["unchanged"]))
    return {"true_state_acc": mean(vals), "by_pattern": {k: mean(v) for k, v in sorted(by_pattern.items())}, "by_pattern_static": {k: mean(v) for k, v in sorted(by_pattern_static.items())}, "pair_both": mean(pair_both), "n_pair_both": len(pair_both)}


def analyze_train(condition: str, arm: str) -> Dict[str, Any]:
    root = DATA_ROOT / condition
    rows = load_jsonl(root / "common_seen_train.jsonl") + load_jsonl(root / "arms" / arm / "train_supervised.jsonl")
    state = [r for r in rows if r.get("task") == "state_query"]
    comp = [r for r in rows if r.get("task") == "relation_comparison"]
    out = {"n_rows": len(rows), "state_rules": {}, "comparison_rules": {}}
    for name, fn in STATE_RULES.items():
        out["state_rules"][name] = {"all_rows": score(state, fn), "true_choices": choice_for_state_rule(state, fn)}
    for name, fn in COMP_RULES.items():
        out["comparison_rules"][name] = {"all_rows": score(comp, fn), "by_neutral": grouped(comp, fn, ["neutral_control"])}
    out["comparison_rules"]["pair_set_training_oracle"] = {"all_rows": score(comp, lambda r: comp_pair_set_oracle(r, condition))}
    return out


def analyze_eval(suite: str) -> Dict[str, Any]:
    rows = load_jsonl(DATA_ROOT / "eval" / f"{suite}.jsonl")
    state = [r for r in rows if r.get("task") == "state_query"]
    comp = [r for r in rows if r.get("task") == "relation_comparison"]
    out: Dict[str, Any] = {"n_rows": len(rows), "state_rules": {}, "comparison_rules": {}}
    for name, fn in STATE_RULES.items():
        out["state_rules"][name] = {"all_rows": score(state, fn), "true_choices": choice_for_state_rule(state, fn)}
    for name, fn in COMP_RULES.items():
        out["comparison_rules"][name] = {"all_rows": score(comp, fn)}
    return out


def build_report() -> Dict[str, Any]:
    rep: Dict[str, Any] = {"status": "PAIR_BINDING_HEURISTIC_BASELINES_COMPLETE", "train": {}, "eval": {}}
    for cond in CONDITIONS:
        rep["train"][cond] = {}
        for arm in ARMS:
            rep["train"][cond][arm] = analyze_train(cond, arm)
    for suite in EVAL_SUITES:
        rep["eval"][suite] = analyze_eval(suite)
    return rep


def fmt(x: Any) -> str:
    return "n/a" if x is None else f"{float(x):.3f}"


def write_summary(path: Path, rep: Dict[str, Any]) -> None:
    lines = ["# research pair-binding heuristic baselines", ""]
    lines.append("CPU-only transparent rules on the corrected neutral-block pair-binding substrate.")
    lines.append("")
    lines.append("## Train state shortcuts: true-choice and pair-both")
    lines.append("")
    lines.append("| condition | arm | rule | true state acc | same | opposite | pair-both |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for cond in CONDITIONS:
        for arm in ARMS:
            for rule in ["anti_copy", "copy_initial", "static_slot_step282_switch", "event_role_oracle"]:
                d = rep["train"][cond][arm]["state_rules"][rule]["true_choices"]
                lines.append(f"| {cond} | {arm} | {rule} | {fmt(d.get('true_state_acc'))} | {fmt(d.get('by_pattern',{}).get('same'))} | {fmt(d.get('by_pattern',{}).get('opposite'))} | {fmt(d.get('pair_both'))} |")
    lines.append("")
    lines.append("## Train held-held comparison rules")
    lines.append("")
    lines.append("| condition | arm | rule | all acc | neutral=False | neutral=True |")
    lines.append("|---|---|---|---:|---:|---:|")
    for cond in CONDITIONS:
        for arm in ARMS:
            cr = rep["train"][cond][arm]["comparison_rules"]
            for rule in ["geom_same_true_parity", "neutral_voice_xor", "pair_set_training_oracle"]:
                d = cr[rule]
                byn = d.get("by_neutral", {})
                lines.append(f"| {cond} | {arm} | {rule} | {fmt(d.get('all_rows',{}).get('acc'))} | {fmt(byn.get('False',{}).get('acc'))} | {fmt(byn.get('True',{}).get('acc'))} |")
    lines.append("")
    lines.append("## Eval state shortcuts")
    lines.append("")
    lines.append("| suite | rule | true state acc | same | opposite | pair-both |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for suite in ["paired_state_conservation", "cross_template_state_readout"]:
        for rule in ["anti_copy", "copy_initial", "static_slot_step282_switch", "event_role_oracle"]:
            d = rep["eval"][suite]["state_rules"][rule]["true_choices"]
            lines.append(f"| {suite} | {rule} | {fmt(d.get('true_state_acc'))} | {fmt(d.get('by_pattern',{}).get('same'))} | {fmt(d.get('by_pattern',{}).get('opposite'))} | {fmt(d.get('pair_both'))} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("Anti-copy remains the expected shortcut: perfect on opposite-initial changed rows and wrong on same-initial changed rows.  The event-role oracle is the only state rule that solves both.  The comparison rows contain both informative parity labels and neutral rank-zero labels; a pair-set-specific training oracle can fit them by using whether a row belongs to the B or R dyad set.  Because the B/R exposure, text, label marginals, and exact pair degrees are matched across conditions, this oracle reflects the intended intervention rather than a simple exposure imbalance.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- full JSON: `{path.parent / 'heuristic_baselines.json'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    DATA_ROOT = args.data_root
    rep = build_report()
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "heuristic_baselines.json", rep)
    write_summary(args.out / "heuristic_baselines_summary.md", rep)
    print(json.dumps({"status": rep["status"], "summary": str(args.out / "heuristic_baselines_summary.md"), "json": str(args.out / "heuristic_baselines.json"), "no_model_loading_training_evaluation_upload_or_leaderboard": True}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
