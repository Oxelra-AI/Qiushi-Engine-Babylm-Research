#!/usr/bin/env python3
"""research: transparent heuristic baselines for balanced k0/k16 substrates.

This CPU-only analysis scores shortcut and event-role rules on the exact train/eval
rows used by the research learned probe.  It does not load models, train, evaluate
BabyLM, or write leaderboard artifacts.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

PROJECT = Path("experiments/archive/representation_and_objectives")
DATA_ROOT = PROJECT / "data/information_budget_substrate"
OUT = PROJECT / "data/balanced_heuristic_baselines"
CONDITIONS = ["replace_k00_spread", "replace_k16_spread"]
ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]
EVAL_SUITES = [
    "paired_state_conservation",
    "cross_template_state_readout",
    "name_permutation_counterfactual",
    "mixed_held_seen_orientation",
    "heldheld_unseen_edge_closure",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def mean(xs: Sequence[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def owner_from_slot(slot: int | None, row: Dict[str, Any]) -> Any:
    order = row.get("arg_order") or []
    if slot is None or slot < 0 or slot >= len(order):
        return None
    return order[slot]


def slot_of_owner(owner: Any, row: Dict[str, Any]) -> int | None:
    order = row.get("arg_order") or []
    for i, x in enumerate(order):
        if x == owner:
            return i
    return None


def candidate_is_owner(owner: Any, row: Dict[str, Any]) -> bool | None:
    if owner is None:
        return None
    return row.get("candidate") == owner


def pred_anticopy(row: Dict[str, Any]) -> bool | None:
    """Changed object goes to the non-initial owner; unchanged object stays with initial/static owner."""
    if row.get("task") != "state_query":
        return None
    if row.get("query_kind") == "changed":
        init = row.get("initial_changed_owner")
        init_slot = slot_of_owner(init, row)
        if init_slot is None:
            return None
        return row.get("candidate_slot") == 1 - init_slot
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    return None


def pred_copy_initial(row: Dict[str, Any]) -> bool | None:
    """Changed object stays with its initial owner; unchanged object stays with static owner."""
    if row.get("task") != "state_query":
        return None
    if row.get("query_kind") == "changed":
        return row.get("candidate") == row.get("initial_changed_owner")
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    return None


def pred_static_slot_step282_switch(row: Dict[str, Any]) -> bool | None:
    """The rule that can fit research's diagonal disambiguated arm.

    For changed queries: if static_slot=0 choose complement(initial), if static_slot=1 choose initial.
    For unchanged queries: choose static owner.  This is not event-role induction.
    """
    if row.get("task") != "state_query":
        return None
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    if row.get("query_kind") == "changed":
        init_slot = slot_of_owner(row.get("initial_changed_owner"), row)
        st = row.get("static_slot")
        if init_slot is None or st is None:
            return None
        target_slot = 1 - init_slot if int(st) == 0 else init_slot
        return row.get("candidate_slot") == target_slot
    return None


def pred_static_slot_opposite_switch(row: Dict[str, Any]) -> bool | None:
    """Opposite diagonal switch: if static_slot=0 choose initial; if static_slot=1 choose complement(initial)."""
    if row.get("task") != "state_query":
        return None
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    if row.get("query_kind") == "changed":
        init_slot = slot_of_owner(row.get("initial_changed_owner"), row)
        st = row.get("static_slot")
        if init_slot is None or st is None:
            return None
        target_slot = init_slot if int(st) == 0 else 1 - init_slot
        return row.get("candidate_slot") == target_slot
    return None


def pred_event_role(row: Dict[str, Any]) -> bool | None:
    """Use the dataset's intended supervised changed owner and static owner.

    This is an oracle symbolic event-role rule, used as a positive ceiling for state rows.
    """
    if row.get("task") != "state_query":
        return None
    if row.get("query_kind") == "changed":
        return row.get("candidate") == row.get("supervised_changed_owner")
    if row.get("query_kind") == "unchanged":
        return row.get("candidate") == row.get("static_owner")
    return None


def pred_always_true(row: Dict[str, Any]) -> bool | None:
    return True


def pred_candidate_slot0(row: Dict[str, Any]) -> bool | None:
    if row.get("candidate_slot") is None:
        return None
    return int(row.get("candidate_slot")) == 0


def pred_candidate_slot1(row: Dict[str, Any]) -> bool | None:
    if row.get("candidate_slot") is None:
        return None
    return int(row.get("candidate_slot")) == 1


STATE_RULES: Dict[str, Callable[[Dict[str, Any]], bool | None]] = {
    "anti_copy_changed__static_copy_unchanged": pred_anticopy,
    "copy_initial_changed__static_copy_unchanged": pred_copy_initial,
    "static_slot_step282_switch": pred_static_slot_step282_switch,
    "static_slot_opposite_switch": pred_static_slot_opposite_switch,
    "oracle_event_role_state": pred_event_role,
    "always_true": pred_always_true,
    "candidate_slot0": pred_candidate_slot0,
    "candidate_slot1": pred_candidate_slot1,
}


def rule_score(rows: Sequence[Dict[str, Any]], fn: Callable[[Dict[str, Any]], bool | None]) -> Dict[str, Any]:
    scored: List[Dict[str, Any]] = []
    skipped = 0
    for r in rows:
        pred = fn(r)
        if pred is None:
            skipped += 1
            continue
        scored.append({"row": r, "pred": bool(pred), "correct": bool(pred) == bool(r.get("label"))})
    if not scored:
        return {"n": 0, "skipped": skipped}
    return {"n": len(scored), "skipped": skipped, "acc": mean([float(x["correct"]) for x in scored]), "pred_true_frac": mean([float(x["pred"]) for x in scored])}


def group_key(row: Dict[str, Any], fields: Sequence[str]) -> str:
    return "|".join("None" if row.get(f) is None else str(row.get(f)) for f in fields)


def grouped_rule_scores(rows: Sequence[Dict[str, Any]], fn: Callable[[Dict[str, Any]], bool | None], fields: Sequence[str]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r, fields)].append(r)
    return {k: rule_score(v, fn) for k, v in sorted(groups.items())}


def pair_both_for_rule(rows: Sequence[Dict[str, Any]], fn: Callable[[Dict[str, Any]], bool | None]) -> Dict[str, Any]:
    true_rows = [r for r in rows if r.get("task") == "state_query" and bool(r.get("label"))]
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in true_rows:
        pred = fn(r)
        if pred is None:
            continue
        rr = dict(r)
        rr["rule_correct"] = bool(pred) == bool(r.get("label"))
        by_pair[str(r.get("pair_id"))][str(r.get("query_kind"))] = rr
    pairs = []
    for pid, d in by_pair.items():
        if "changed" in d and "unchanged" in d:
            c, u = d["changed"], d["unchanged"]
            pairs.append({
                "pair_id": pid,
                "correct": bool(c["rule_correct"] and u["rule_correct"]),
                "initial_pattern": c.get("initial_pattern"),
                "static_slot": c.get("static_slot"),
                "relation": c.get("relation"),
                "voice": c.get("voice"),
            })
    out = {"all": rule_score([dict(p, label=True) for p in pairs], lambda r: r["correct"])}
    # The lambda trick above is awkward for pred_true_frac; overwrite with clearer values.
    if pairs:
        out["all"] = {"n": len(pairs), "acc": mean([float(p["correct"]) for p in pairs])}
        for fields in [["initial_pattern"], ["initial_pattern", "static_slot"], ["initial_pattern", "static_slot", "relation", "voice"]]:
            gg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for p in pairs:
                gg[group_key(p, fields)].append(p)
            out["by_" + "_".join(fields)] = {k: {"n": len(v), "acc": mean([float(x["correct"]) for x in v])} for k, v in sorted(gg.items())}
    return out


def state_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    state = [r for r in rows if r.get("task") == "state_query"]
    changed_true = [r for r in state if r.get("query_kind") == "changed" and bool(r.get("label"))]
    unchanged_true = [r for r in state if r.get("query_kind") == "unchanged" and bool(r.get("label"))]
    return {
        "n_state": len(state),
        "n_changed_true": len(changed_true),
        "n_unchanged_true": len(unchanged_true),
        "pattern_static_counts_all_state": dict(sorted(Counter(group_key(r, ["initial_pattern", "static_slot"]) for r in state).items())),
        "pattern_static_counts_changed_true": dict(sorted(Counter(group_key(r, ["initial_pattern", "static_slot"]) for r in changed_true).items())),
        "relation_voice_static_pattern_changed_true": dict(sorted(Counter(group_key(r, ["relation", "voice", "static_slot", "initial_pattern"]) for r in changed_true).items())),
    }


def analyze_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"row_summary": state_summary(rows), "rules": {}}
    state_rows = [r for r in rows if r.get("task") == "state_query"]
    for name, fn in STATE_RULES.items():
        true_state = [r for r in state_rows if bool(r.get("label"))]
        changed_true = [r for r in true_state if r.get("query_kind") == "changed"]
        unchanged_true = [r for r in true_state if r.get("query_kind") == "unchanged"]
        out["rules"][name] = {
            "state_all_rows": rule_score(state_rows, fn),
            "true_state_choices": rule_score(true_state, fn),
            "changed_true": rule_score(changed_true, fn),
            "unchanged_true": rule_score(unchanged_true, fn),
            "changed_true_by_pattern": grouped_rule_scores(changed_true, fn, ["initial_pattern"]),
            "changed_true_by_pattern_static": grouped_rule_scores(changed_true, fn, ["initial_pattern", "static_slot"]),
            "changed_true_by_pattern_static_relation_voice": grouped_rule_scores(changed_true, fn, ["initial_pattern", "static_slot", "relation", "voice"]),
            "pair_both": pair_both_for_rule(state_rows, fn),
        }
    return out


def analyze_train(condition: str, arm: str) -> Dict[str, Any]:
    cdir = DATA_ROOT / condition
    rows = load_jsonl(cdir / "common_seen_train.jsonl") + load_jsonl(cdir / "arms" / arm / "train_supervised.jsonl")
    return analyze_split(rows)


def analyze_eval(condition: str, suite: str) -> Dict[str, Any]:
    rows = load_jsonl(DATA_ROOT / condition / "eval" / f"{suite}.jsonl")
    return analyze_split(rows)


def compact_table(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("# research balanced heuristic baselines")
    lines.append("")
    lines.append("CPU-only transparent rules on the exact `replace_k00_spread` and `replace_k16_spread` files used by the research learned probe.  These are not learned results; they define shortcut ceilings/floors for interpreting the learned run.")
    lines.append("")
    lines.append("## Training state rows: changed-true accuracy")
    lines.append("")
    lines.append("| condition | arm | rule | changed true acc | pair-both acc | changed same acc | changed opposite acc | offdiag same/st0 | offdiag opposite/st1 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for condition in CONDITIONS:
        for arm in ARMS:
            rules = report["train"][condition][arm]["rules"]
            for rule in ["anti_copy_changed__static_copy_unchanged", "copy_initial_changed__static_copy_unchanged", "static_slot_step282_switch", "oracle_event_role_state"]:
                d = rules[rule]
                def acc(path: List[str]) -> Any:
                    x: Any = d
                    for p in path:
                        x = x.get(p, {}) if isinstance(x, dict) else {}
                    return x.get("acc") if isinstance(x, dict) else None
                def fmt(x: Any) -> str:
                    return "n/a" if x is None else f"{float(x):.3f}"
                lines.append(
                    f"| {condition} | {arm} | {rule} | "
                    f"{fmt(acc(['changed_true']))} | "
                    f"{fmt(acc(['pair_both','all']))} | "
                    f"{fmt(acc(['changed_true_by_pattern','same']))} | "
                    f"{fmt(acc(['changed_true_by_pattern','opposite']))} | "
                    f"{fmt(acc(['changed_true_by_pattern_static','same|0']))} | "
                    f"{fmt(acc(['changed_true_by_pattern_static','opposite|1']))} |"
                )
    lines.append("")
    lines.append("## Evaluation paired-state-conservation: changed-true accuracy")
    lines.append("")
    lines.append("| condition | rule | changed true acc | pair-both acc | same | opposite | same/st0 | same/st1 | opposite/st0 | opposite/st1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for condition in CONDITIONS:
        rules = report["eval"][condition]["paired_state_conservation"]["rules"]
        for rule in ["anti_copy_changed__static_copy_unchanged", "copy_initial_changed__static_copy_unchanged", "static_slot_step282_switch", "static_slot_opposite_switch", "oracle_event_role_state"]:
            d = rules[rule]
            def get(path: List[str]) -> Any:
                x: Any = d
                for p in path:
                    x = x.get(p, {}) if isinstance(x, dict) else {}
                return x.get("acc") if isinstance(x, dict) else None
            def fmt(x: Any) -> str:
                return "n/a" if x is None else f"{float(x):.3f}"
            lines.append(
                f"| {condition} | {rule} | "
                f"{fmt(get(['changed_true']))} | {fmt(get(['pair_both','all']))} | "
                f"{fmt(get(['changed_true_by_pattern','same']))} | {fmt(get(['changed_true_by_pattern','opposite']))} | "
                f"{fmt(get(['changed_true_by_pattern_static','same|0']))} | {fmt(get(['changed_true_by_pattern_static','same|1']))} | "
                f"{fmt(get(['changed_true_by_pattern_static','opposite|0']))} | {fmt(get(['changed_true_by_pattern_static','opposite|1']))} |"
            )
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("- `replace_k00_spread` is a redundant observational-equivalence condition: anti-copy and oracle event-role agree on opposite-initial changed training rows.")
    lines.append("- `replace_k16_spread` breaks the research static-slot diagonal: in train and eval, both `same|0` and `opposite|1` off-diagonal cells exist.  The research static-slot switch scores 0.5 on changed rows rather than 1.0.")
    lines.append("- The oracle event-role state rule is 1.0 on all state cells; anti-copy and copy-initial are 0.5 on balanced k16 changed rows.  Learned k16 success must therefore be compared to all four pattern×static cells and pair-both conservation, not aggregate state accuracy.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- full JSON: `{OUT / 'heuristic_baselines_report.json'}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"train": {}, "eval": {}, "conditions": CONDITIONS, "arms": ARMS, "eval_suites": EVAL_SUITES}
    for condition in CONDITIONS:
        report["train"][condition] = {}
        for arm in ARMS:
            report["train"][condition][arm] = analyze_train(condition, arm)
        report["eval"][condition] = {}
        for suite in EVAL_SUITES:
            report["eval"][condition][suite] = analyze_eval(condition, suite)
    write_json(OUT / "heuristic_baselines_report.json", report)
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/balanced_heuristic_baselines/heuristic_baselines_summary.md')).write_text(compact_table(report), encoding="utf-8")
    print(json.dumps({
        "status": "HEURISTIC_BASELINES_COMPLETE",
        "summary": str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/balanced_heuristic_baselines/heuristic_baselines_summary.md')),
        "report": str(OUT / "heuristic_baselines_report.json"),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
