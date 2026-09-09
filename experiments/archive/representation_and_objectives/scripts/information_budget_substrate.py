#!/usr/bin/env python3
"""research: information-budget variants for the initial-ownership disambiguation route.

CPU/file-only construction. No model loading, training, official BabyLM evaluation,
upload, or leaderboard work.

Purpose
-------
research built a binary underdetermined-vs-disambiguated factorial substrate.  The
running learned probe will decide whether breaking the non-initial-owner shortcut
can induce event-role behavior.  This script prepares the next *cheap* research
asset: a balanced family that asks how much independently informative evidence is
needed, and whether relation coverage matters, against equal-sized redundant
opposite-initial evidence.

Important repair
----------------
A post-hoc audit of the research disambiguated construction shows that the simple
sorted-pair alternation made initial_pattern perfectly correlated with
static_slot in the held-state bridge rows.  The binary run is still useful as a
first learned probe, but it is not the final clean construction for an
information-budget law.  The variants produced here select initial=same worlds
with explicit balance over relation, voice, and static_slot whenever the budget
allows it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path("experiments/archive/representation_and_objectives")
ROOT = PROJECT_ROOT / "data" / "factorial_initial_ownership"
OUT_DEFAULT = PROJECT_ROOT / "data" / "information_budget_substrate"

ARMS = ["heldheld_only", "aligned_state_bridge", "inverted_state_bridge"]
BRIDGE_ARMS = {"aligned_state_bridge", "inverted_state_bridge"}
BUDGET_KS = [0, 1, 2, 4, 8, 16, 24]
COVERAGE_SPECS = [
    ("spread", None),
    ("single_h0_dax", "h0_dax"),
    ("single_h2_norp", "h2_norp"),
]


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def group_state_pairs(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    other: List[Dict[str, Any]] = []
    pairs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("task") == "state_query":
            pairs[r["pair_id"]].append(r)
        else:
            other.append(r)
    return other, dict(pairs)


def get_initial_opposite_owner(r: Dict[str, Any]) -> str:
    final_owner = r["supervised_changed_owner"]
    a, b = r["arg_order"]
    return a if final_owner == b else b


def reconstruct_premise(initial_changed_owner: str, changed_obj: str,
                        static_owner: str, static_obj: str,
                        cause_event: str) -> str:
    return (f"At first, {initial_changed_owner} had the {changed_obj}, "
            f"and {static_owner} had the {static_obj}. "
            f"The event was this: {cause_event}")


def annotate_row(r: Dict[str, Any], pattern: str, initial_owner: str,
                 suffix: str = "", new_premise: str | None = None) -> Dict[str, Any]:
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
    if not pair_rows:
        return []
    if pattern == "opposite":
        return [annotate_row(r, "opposite", get_initial_opposite_owner(r), suffix=suffix) for r in pair_rows]
    if pattern != "same":
        raise ValueError(pattern)
    final_owner = pair_rows[0]["supervised_changed_owner"]
    changed_obj = pair_rows[0]["changed_object"]
    static_owner = pair_rows[0]["static_owner"]
    static_obj = pair_rows[0]["static_object"]
    cause_event = pair_rows[0]["cause_event"]
    new_premise = reconstruct_premise(final_owner, changed_obj, static_owner, static_obj, cause_event)
    return [annotate_row(r, "same", final_owner, suffix=suffix or "_isame", new_premise=new_premise) for r in pair_rows]


def pair_meta(pair_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    r = pair_rows[0]
    return {
        "relation": r.get("relation"),
        "voice": r.get("voice"),
        "static_slot": r.get("static_slot"),
        "arg_order": tuple(r.get("arg_order", [])),
        "changed_object": r.get("changed_object"),
        "static_object": r.get("static_object"),
    }


def round_robin_select(pairs: Dict[str, List[Dict[str, Any]]], k: int,
                       coverage: str, relation_filter: str | None) -> List[str]:
    """Select k pair_ids, balancing relation×voice×static_slot when possible."""
    metas = {pid: pair_meta(rs) for pid, rs in pairs.items()}
    candidates = []
    for pid, m in metas.items():
        if relation_filter is not None and m["relation"] != relation_filter:
            continue
        candidates.append(pid)
    if k > len(candidates):
        raise ValueError(f"coverage={coverage} asks k={k}, only {len(candidates)} candidates")
    if k == 0:
        return []

    buckets: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
    for pid in candidates:
        m = metas[pid]
        if relation_filter is None:
            key = (m["relation"], m["voice"], m["static_slot"])
        else:
            key = (m["voice"], m["static_slot"])
        buckets[key].append(pid)
    for key in buckets:
        buckets[key] = sorted(buckets[key])

    selected: List[str] = []
    keys = sorted(buckets)
    while len(selected) < k:
        moved = False
        for key in keys:
            if buckets[key] and len(selected) < k:
                selected.append(buckets[key].pop(0))
                moved = True
        if not moved:
            break
    assert len(selected) == k
    return selected


def build_replace_arm(base_rows: Sequence[Dict[str, Any]], selected_same: set[str]) -> List[Dict[str, Any]]:
    other, pairs = group_state_pairs(base_rows)
    out: List[Dict[str, Any]] = list(other)
    for pid in sorted(pairs):
        if pid in selected_same:
            out.extend(make_variant(pairs[pid], "same", suffix="_ibsame"))
        else:
            out.extend(make_variant(pairs[pid], "opposite"))
    return out


def state_rule_acc(rows: Sequence[Dict[str, Any]], rule: str) -> Dict[str, Any]:
    """Row and true-choice accuracy for simple state rules.

    The rules use the correct static owner for unchanged queries; they differ only
    on the changed object:
    - anticopy: choose complement(initial_changed_owner)
    - copy_initial: choose initial_changed_owner
    - event_role: choose supervised_changed_owner
    """
    state = [r for r in rows if r.get("task") == "state_query"]
    out: Dict[str, Any] = {}
    for pattern in [None, "opposite", "same"]:
        for qk in [None, "changed", "unchanged"]:
            subset = [r for r in state]
            if pattern is not None:
                subset = [r for r in subset if r.get("initial_pattern") == pattern]
            if qk is not None:
                subset = [r for r in subset if r.get("query_kind") == qk]
            if not subset:
                continue
            corr = 0
            for r in subset:
                if r["query_kind"] == "unchanged":
                    pred_owner = r["static_owner"]
                elif rule == "anticopy":
                    a, b = r["arg_order"]
                    init = r.get("initial_changed_owner")
                    pred_owner = a if init == b else b
                elif rule == "copy_initial":
                    pred_owner = r.get("initial_changed_owner")
                elif rule == "event_role":
                    pred_owner = r["supervised_changed_owner"]
                else:
                    raise ValueError(rule)
                pred_label = (r["candidate"] == pred_owner)
                corr += int(pred_label == bool(r["label"]))
            prefix = "all" if pattern is None else pattern
            qprefix = "state" if qk is None else qk
            out[f"{prefix}_{qprefix}_row_acc"] = corr / len(subset)
            out[f"{prefix}_{qprefix}_row_n"] = len(subset)

    # True-choice accuracy on changed rows; this matches the most important eval diagnostic.
    true_changed = [r for r in state if r.get("query_kind") == "changed" and r.get("label") is True]
    for pattern in [None, "opposite", "same"]:
        subset = true_changed
        if pattern is not None:
            subset = [r for r in subset if r.get("initial_pattern") == pattern]
        if not subset:
            continue
        corr = 0
        for r in subset:
            if rule == "anticopy":
                a, b = r["arg_order"]
                init = r.get("initial_changed_owner")
                pred_owner = a if init == b else b
            elif rule == "copy_initial":
                pred_owner = r.get("initial_changed_owner")
            elif rule == "event_role":
                pred_owner = r["supervised_changed_owner"]
            else:
                raise ValueError(rule)
            corr += int(r["candidate"] == pred_owner)
        prefix = "all" if pattern is None else pattern
        out[f"{prefix}_changed_true_choice_acc"] = corr / len(subset)
        out[f"{prefix}_changed_true_choice_n"] = len(subset)
    return out


def pair_both_acc(rows: Sequence[Dict[str, Any]], rule: str) -> Dict[str, Any]:
    state_true = [r for r in rows if r.get("task") == "state_query" and r.get("label") is True]
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in state_true:
        by_pair[r["pair_id"]][r["query_kind"]] = r
    out: Dict[str, Any] = {}
    for pattern in [None, "opposite", "same"]:
        vals: List[int] = []
        for d in by_pair.values():
            if "changed" not in d or "unchanged" not in d:
                continue
            rc, ru = d["changed"], d["unchanged"]
            if pattern is not None and rc.get("initial_pattern") != pattern:
                continue
            if rule == "anticopy":
                a, b = rc["arg_order"]
                init = rc.get("initial_changed_owner")
                changed_pred = a if init == b else b
            elif rule == "copy_initial":
                changed_pred = rc.get("initial_changed_owner")
            elif rule == "event_role":
                changed_pred = rc["supervised_changed_owner"]
            else:
                raise ValueError(rule)
            unchanged_pred = ru["static_owner"]
            vals.append(int(rc["candidate"] == changed_pred and ru["candidate"] == unchanged_pred))
        if vals:
            prefix = "all" if pattern is None else pattern
            out[f"{prefix}_pair_both"] = sum(vals) / len(vals)
            out[f"{prefix}_pair_both_n"] = len(vals)
    return out


def arm_audit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    state = [r for r in rows if r.get("task") == "state_query"]
    comp = [r for r in rows if r.get("task") == "relation_comparison"]
    report: Dict[str, Any] = {
        "rows": len(rows),
        "comparison_rows": len(comp),
        "state_rows": len(state),
        "state_pairs": len({r["pair_id"] for r in state}),
        "initial_pattern_counts_rows": dict(Counter(r.get("initial_pattern") for r in state)),
        "relation_counts_rows": dict(Counter(r.get("relation") for r in state)),
        "voice_counts_rows": dict(Counter(r.get("voice") for r in state)),
        "static_slot_counts_rows": dict(Counter(str(r.get("static_slot")) for r in state)),
        "rel_voice_static_pattern_counts_rows": dict(Counter(
            f"{r.get('relation')}|{r.get('voice')}|st{r.get('static_slot')}|{r.get('initial_pattern')}" for r in state
        )),
    }
    for rule in ["anticopy", "copy_initial", "event_role"]:
        report[f"{rule}_state"] = state_rule_acc(rows, rule)
        report[f"{rule}_pair_both"] = pair_both_acc(rows, rule)
    return report


def disambiguated_audit() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm in ["aligned_state_bridge", "inverted_state_bridge"]:
        rows = load_jsonl(ROOT / "disambiguated" / "arms" / arm / "train_supervised.jsonl")
        state = [r for r in rows if r.get("task") == "state_query"]
        out[arm] = {
            "pattern_by_static_slot_rows": dict(Counter(f"{r.get('initial_pattern')}|st{r.get('static_slot')}" for r in state)),
            "pattern_by_relation_voice_static_rows": dict(Counter(
                f"{r.get('relation')}|{r.get('voice')}|st{r.get('static_slot')}|{r.get('initial_pattern')}" for r in state
            )),
        }
    return out


def build_budget_family(out: Path) -> Dict[str, Any]:
    common_seen = load_jsonl(ROOT / "underdetermined" / "common_seen_train.jsonl")
    eval_rows: Dict[str, List[Dict[str, Any]]] = {}
    for ef in sorted((ROOT / "underdetermined" / "eval").iterdir()):
        if ef.suffix == ".jsonl":
            eval_rows[ef.stem] = load_jsonl(ef)

    # Base rows are the underdetermined/opposite construction so every replacement
    # variant starts from the same exposure count and only changes which worlds are
    # independently informative.
    base_arm_rows: Dict[str, List[Dict[str, Any]]] = {
        arm: load_jsonl(ROOT / "underdetermined" / "arms" / arm / "train_supervised.jsonl")
        for arm in ARMS
    }

    report: Dict[str, Any] = {
        "status": "INFORMATION_BUDGET_SUBSTRATE_COMPLETE",
        "source": str(ROOT),
        "important_audit": {
            "disambiguated_static_slot_confound": disambiguated_audit(),
            "interpretation": "research disambiguated selected initial=same by sorted pair_id, making pattern correlate with static_slot. The replacement variants here explicitly balance relation/voice/static_slot when K permits.",
        },
        "conditions": {},
    }

    for k in BUDGET_KS:
        for coverage, rel_filter in COVERAGE_SPECS:
            # Skip single-relation budgets larger than the available relation capacity.
            try:
                _, probe_pairs = group_state_pairs(base_arm_rows["aligned_state_bridge"])
                selected_probe = round_robin_select(probe_pairs, k, coverage, rel_filter)
            except ValueError:
                continue
            condition = f"replace_k{k:02d}_{coverage}"
            cond_out = out / condition
            save_jsonl(cond_out / "common_seen_train.jsonl", common_seen)
            for suite, rows in eval_rows.items():
                save_jsonl(cond_out / "eval" / f"{suite}.jsonl", rows)

            crep: Dict[str, Any] = {
                "mode": "fixed_total_replacement",
                "k_informative_same_pairs_per_bridge_arm": k,
                "coverage": coverage,
                "relation_filter": rel_filter,
                "same_pair_ids_probe_aligned": selected_probe,
                "arms": {},
            }
            for arm in ARMS:
                if arm in BRIDGE_ARMS:
                    other, pairs = group_state_pairs(base_arm_rows[arm])
                    selected = set(round_robin_select(pairs, k, coverage, rel_filter))
                    arm_rows = build_replace_arm(base_arm_rows[arm], selected)
                    crep["arms"][arm] = {
                        "same_pair_ids": sorted(selected),
                        **arm_audit(arm_rows),
                    }
                else:
                    arm_rows = list(base_arm_rows[arm])
                    crep["arms"][arm] = arm_audit(arm_rows)
                save_jsonl(cond_out / "arms" / arm / "train_supervised.jsonl", arm_rows)
                # Keep unsupervised text identical to research/research.
                unsup = load_jsonl(ROOT / "underdetermined" / "arms" / arm / "train_unsup_text.jsonl")
                save_jsonl(cond_out / "arms" / arm / "train_unsup_text.jsonl", unsup)
            report["conditions"][condition] = crep

    (out / "information_budget_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    write_summary(out / "information_budget_summary.md", report)
    return report


def fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_summary(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research information-budget substrate")
    lines.append("")
    lines.append("## Why this construction exists")
    lines.append("")
    lines.append("The running research learned probe tests the binary transition from an underdetermined initial-owner surface to a half-opposite/half-same disambiguated surface.  If that transition is positive, the next scientific question is not another large score: it is how much independently informative evidence is sufficient, and whether it must cover relations/templates rather than merely add redundant exposure.  This CPU-only file family prepares that test without starting new GPU work.")
    lines.append("")
    lines.append("## research construction audit")
    lines.append("")
    lines.append("The research disambiguated state rows selected same/opposite by sorted `pair_id`.  In the held-state bridge rows this made `initial_pattern` perfectly correlated with `static_slot`:")
    lines.append("")
    for arm, audit in report["important_audit"]["disambiguated_static_slot_confound"].items():
        lines.append(f"- `{arm}`: `{audit['pattern_by_static_slot_rows']}`")
    lines.append("")
    lines.append("The currently running learned run can still reveal whether the model moves away from the non-initial-owner rule, but any positive result needs confirmation on a pattern-balanced construction before being promoted into a data-efficiency law.")
    lines.append("")
    lines.append("## Fixed-total replacement family")
    lines.append("")
    lines.append("Every condition keeps the same total number of bridge state pairs as research.  `k` pairs are independently informative (`initial=same`, where anti-copy fails); the remaining pairs are redundant opposite-initial worlds (`initial=opposite`, where anti-copy and event-role agree).  `replace_k00_spread` is the equal-sized fully redundant control.")
    lines.append("")
    lines.append("| condition | k | coverage | arm | same pairs | anticopy changed row acc | copy-initial changed row acc | event-role changed row acc | same pattern balance note |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---|")
    for cname, crep in sorted(report["conditions"].items()):
        if not (cname.endswith("spread") or cname.endswith("single_h0_dax") or cname.endswith("single_h2_norp")):
            continue
        k = crep["k_informative_same_pairs_per_bridge_arm"]
        coverage = crep["coverage"]
        for arm in ["aligned_state_bridge", "inverted_state_bridge"]:
            a = crep["arms"][arm]
            ac = a["anticopy_state"].get("all_changed_row_acc")
            cp = a["copy_initial_state"].get("all_changed_row_acc")
            er = a["event_role_state"].get("all_changed_row_acc")
            counts = a.get("rel_voice_static_pattern_counts_rows", {})
            # For large balanced budgets, the summary table would be unreadable; give a compact note.
            same_cells = [key for key, val in counts.items() if key.endswith("|same") and val]
            note = f"same cells={len(same_cells)}"
            lines.append(f"| `{cname}` | {k} | {coverage} | {arm} | {len(a.get('same_pair_ids', []))} | {fmt(ac)} | {fmt(cp)} | {fmt(er)} | {note} |")
    lines.append("")
    lines.append("## Interpretation before any new expensive run")
    lines.append("")
    lines.append("In the symbolic hypothesis class, one informative state world on a connected held-held graph is enough to choose a global orientation; redundant opposite-initial worlds never distinguish event-role from anti-copy.  In a neural learner, the useful quantity is statistical: the anti-copy rule loses only on the changed-object rows of the informative worlds.  For `k` informative same pairs out of 32, anti-copy changed-row accuracy is `1 - k/32`, and its accuracy over all state rows is diluted further by unchanged facts, held-held comparisons, and common-seen rows.  Therefore a threshold may appear even though the formal information is present at k=1.")
    lines.append("")
    lines.append("## Recommended use after research result arrives")
    lines.append("")
    lines.append("1. First read the authoritative research probe result.  If state improvement appears, do not treat it as a general principle until `replace_k16_spread` confirms it without the static-slot confound.")
    lines.append("2. If balanced k=16 gives both initial-same conservation and mixed held-seen orientation, run a minimal information-budget sweep: k=0,2,4,8,16 spread, plus k=8 single-relation controls.  The comparison asks whether independent disambiguating worlds beat equal-sized redundant evidence and whether one relation anchor is enough to propagate through the held-held graph.")
    lines.append("3. If balanced k=16 improves state rows but mixed orientation remains chance, the result is a task-local event/state rule rather than coordinate propagation; the next work should inspect why comparison format does not reuse the induced relation slot.")
    lines.append("4. If research fails and training fit is adequate, this route may need a simpler surface or another architecture; do not launch the budget sweep just because these files exist.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Full report: `{path.parent / 'information_budget_report.json'}`")
    lines.append(f"- Condition directories: `{path.parent}/replace_kXX_*`")
    lines.append("- Source binary substrate: `data/factorial_initial_ownership/`")
    lines.append("- research verified confound: `notes/verified_initial_owner_shortcut_and_open_controls.md`")
    lines.append("")
    write_text(path, "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    report = build_budget_family(out)
    compact: Dict[str, Any] = {
        "status": report["status"],
        "out": str(out),
        "summary": str(out / "information_budget_summary.md"),
        "n_conditions": len(report["conditions"]),
        "static_slot_confound": report["important_audit"]["disambiguated_static_slot_confound"],
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }
    # Add the central clean repair audit numbers.
    key = "replace_k16_spread"
    if key in report["conditions"]:
        compact[key] = {
            arm: {
                "same_pairs": len(report["conditions"][key]["arms"][arm].get("same_pair_ids", [])),
                "anticopy_changed_row_acc": report["conditions"][key]["arms"][arm]["anticopy_state"].get("all_changed_row_acc"),
                "event_role_changed_row_acc": report["conditions"][key]["arms"][arm]["event_role_state"].get("all_changed_row_acc"),
                "rel_voice_static_pattern_counts_rows": report["conditions"][key]["arms"][arm].get("rel_voice_static_pattern_counts_rows", {}),
            }
            for arm in ["aligned_state_bridge", "inverted_state_bridge"]
        }
    print(json.dumps(compact, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
