#!/usr/bin/env python3
"""research: Factorial initial-ownership × event-role substrate.

Preserves the research substrate as the UNDERDETERMINED condition (event-role and
anti-copy observationally equivalent) and builds a matched DISAMBIGUATED condition
where initial ownership varies independently of event role.

Predicted transition:
- Underdetermined training → anti-copy behavior → no mixed orientation separation
- Disambiguated training → event-role inference → orientation may appear

Formal equivalence: On research data, initial_owner = participant(1 - s(R)),
so complement(initial_owner) = participant(s(R)) = event-role prediction.
On disambiguated data with initial_owner = participant(s(R)),
complement(initial_owner) = participant(1 - s(R)) ≠ event-role → anti-copy wrong.

CPU/file-only. No model loading, training, evaluation, upload, or leaderboard.
"""
import json, csv, hashlib, argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path("experiments/archive/representation_and_objectives")
DATA = PROJECT_ROOT / "data" / "equivariant_symmetry_substrate"
OUT_DEFAULT = PROJECT_ROOT / "data" / "factorial_initial_ownership"

ARMS_WITH_HELD_STATE = {"aligned_state_bridge", "inverted_state_bridge"}


# ---------- IO helpers ----------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------- Row manipulation ----------

def group_by_pair(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[r["pair_id"]].append(r)
    return dict(groups)


def reconstruct_premise(initial_changed_owner: str, changed_obj: str,
                        static_owner: str, static_obj: str,
                        cause_event: str) -> str:
    return (f"At first, {initial_changed_owner} had the {changed_obj}, "
            f"and {static_owner} had the {static_obj}. "
            f"The event was this: {cause_event}")


def annotate_row(r: Dict, initial_pattern: str, initial_changed_owner: str,
                 new_premise: Optional[str] = None) -> Dict:
    """Return a copy of r with initial-pattern annotation and optionally modified premise."""
    out = dict(r)
    out["initial_pattern"] = initial_pattern
    out["initial_changed_owner"] = initial_changed_owner
    if new_premise is not None:
        out["premise"] = new_premise
    if initial_pattern == "same":
        out["pair_id"] = r["pair_id"] + "_isame"
        out["row_id"] = r["row_id"] + "_isame"
    return out


def get_initial_opposite_owner(r: Dict) -> str:
    """Identify the non-final owner from the original research row."""
    new = r["supervised_changed_owner"]
    a, b = r["arg_order"]
    return a if new == b else b


def make_opposite_variant(pair_rows: List[Dict]) -> List[Dict]:
    """Mark original rows as initial=opposite (no text change)."""
    out = []
    for r in pair_rows:
        old = get_initial_opposite_owner(r)
        out.append(annotate_row(r, "opposite", old))
    return out


def make_same_variant(pair_rows: List[Dict]) -> List[Dict]:
    """Create initial=same variant: change premise so changed object starts with final owner."""
    new_owner = pair_rows[0]["supervised_changed_owner"]
    changed_obj = pair_rows[0]["changed_object"]
    static_owner = pair_rows[0]["static_owner"]
    static_obj = pair_rows[0]["static_object"]
    cause_event = pair_rows[0]["cause_event"]
    new_premise = reconstruct_premise(new_owner, changed_obj, static_owner, static_obj, cause_event)
    out = []
    for r in pair_rows:
        out.append(annotate_row(r, "same", new_owner, new_premise))
    return out


# ---------- Condition building ----------

def build_condition_arm(arm_name: str, supervised_rows: List[Dict],
                        condition: str) -> Tuple[List[Dict], Dict]:
    """Build arm data for a given condition.
    
    For ARMS_WITH_HELD_STATE:
    - underdetermined: all state pairs initial=opposite
    - disambiguated: alternating opposite/same
    Other arms: all state pairs marked initial=opposite (unchanged).
    """
    state = [r for r in supervised_rows if r.get("task") == "state_query"]
    other = [r for r in supervised_rows if r.get("task") != "state_query"]
    pairs = group_by_pair(state)
    pair_ids = sorted(pairs.keys())

    new_state = []
    counts = Counter()

    if arm_name in ARMS_WITH_HELD_STATE and condition == "disambiguated":
        for i, pid in enumerate(pair_ids):
            if i % 2 == 0:
                new_state.extend(make_opposite_variant(pairs[pid]))
                counts["opposite"] += 1
            else:
                new_state.extend(make_same_variant(pairs[pid]))
                counts["same"] += 1
    else:
        for pid in pair_ids:
            new_state.extend(make_opposite_variant(pairs[pid]))
            counts["opposite"] += 1

    info = {
        "state_pairs": len(pair_ids),
        "pattern_counts": dict(counts),
        "state_rows": len(new_state),
        "comparison_rows": len(other),
    }
    return other + new_state, info


def build_eval_both_patterns(eval_rows: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Augment eval state rows with both initial patterns.
    Comparison rows pass through unchanged. State pairs get both variants."""
    state = [r for r in eval_rows if r.get("task") == "state_query"]
    other = [r for r in eval_rows if r.get("task") != "state_query"]
    pairs = group_by_pair(state)

    new_state = []
    for pid in sorted(pairs.keys()):
        new_state.extend(make_opposite_variant(pairs[pid]))
        new_state.extend(make_same_variant(pairs[pid]))

    info = {
        "original_state_pairs": len(pairs),
        "augmented_state_rows": len(new_state),
        "comparison_rows": len(other),
        "total": len(other) + len(new_state),
    }
    return other + new_state, info


# ---------- Analysis ----------

def anticopy_baseline(rows: List[Dict]) -> Dict[str, Any]:
    """Deterministic non-initial-owner baseline on true-labeled state rows."""
    state_true = [r for r in rows
                  if r.get("task") == "state_query" and r.get("label") is True]
    results: Dict[str, Any] = {}
    for pattern in [None, "opposite", "same"]:
        for qk in ["changed", "unchanged"]:
            subset = [r for r in state_true if r["query_kind"] == qk]
            if pattern is not None:
                subset = [r for r in subset if r.get("initial_pattern") == pattern]
            if not subset:
                continue
            correct = 0
            for r in subset:
                initial = r.get("initial_changed_owner")
                a, b = r["arg_order"]
                if qk == "changed":
                    predicted = a if initial == b else b
                    correct += int(r["candidate"] == predicted)
                else:
                    correct += int(r["candidate"] == r["static_owner"])
            key_prefix = "all" if pattern is None else pattern
            results[f"{key_prefix}_{qk}_acc"] = correct / len(subset) if subset else None
            results[f"{key_prefix}_{qk}_n"] = len(subset)
    return results


def event_role_baseline(rows: List[Dict]) -> Dict[str, Any]:
    """Deterministic event-role baseline: for changed, pick supervised_changed_owner."""
    state_true = [r for r in rows
                  if r.get("task") == "state_query" and r.get("label") is True]
    results: Dict[str, Any] = {}
    for pattern in [None, "opposite", "same"]:
        for qk in ["changed", "unchanged"]:
            subset = [r for r in state_true if r["query_kind"] == qk]
            if pattern is not None:
                subset = [r for r in subset if r.get("initial_pattern") == pattern]
            if not subset:
                continue
            correct = 0
            for r in subset:
                if qk == "changed":
                    correct += int(r["candidate"] == r["supervised_changed_owner"])
                else:
                    correct += int(r["candidate"] == r["static_owner"])
            key_prefix = "all" if pattern is None else pattern
            results[f"{key_prefix}_{qk}_acc"] = correct / len(subset) if subset else None
            results[f"{key_prefix}_{qk}_n"] = len(subset)
    return results


def pair_both_baseline(rows: List[Dict], rule: str) -> Dict[str, Any]:
    """Compute pair-both accuracy for a deterministic rule."""
    state_true = [r for r in rows
                  if r.get("task") == "state_query" and r.get("label") is True]
    by_pair: Dict[str, Dict] = defaultdict(dict)
    for r in state_true:
        by_pair[r["pair_id"]][r["query_kind"]] = r

    results: Dict[str, Any] = {}
    for pattern in [None, "opposite", "same"]:
        pairs_ok = []
        for pid, d in by_pair.items():
            if "changed" not in d or "unchanged" not in d:
                continue
            rc = d["changed"]
            ru = d["unchanged"]
            if pattern is not None and rc.get("initial_pattern") != pattern:
                continue
            a, b = rc["arg_order"]
            initial = rc.get("initial_changed_owner")
            if rule == "anticopy":
                changed_pred = a if initial == b else b
            elif rule == "event_role":
                changed_pred = rc["supervised_changed_owner"]
            else:
                changed_pred = None
            unchanged_pred = ru["static_owner"]
            c_ok = changed_pred == rc["candidate"]
            u_ok = unchanged_pred == ru["candidate"]
            pairs_ok.append(int(c_ok and u_ok))
        key_prefix = "all" if pattern is None else pattern
        results[f"{key_prefix}_pair_both"] = sum(pairs_ok) / len(pairs_ok) if pairs_ok else None
        results[f"{key_prefix}_pair_n"] = len(pairs_ok)
    return results


def balance_check(rows: List[Dict]) -> Dict[str, Any]:
    """Check label, voice, and initial-pattern balance in state rows."""
    state = [r for r in rows if r.get("task") == "state_query"]
    if not state:
        return {"n_state": 0}
    return {
        "n_state": len(state),
        "label_true_frac": sum(r["label"] for r in state) / len(state),
        "voice_counts": dict(Counter(r.get("voice") for r in state)),
        "query_kind_counts": dict(Counter(r["query_kind"] for r in state)),
        "initial_pattern_counts": dict(Counter(r.get("initial_pattern") for r in state)),
        "relation_counts": dict(Counter(r.get("relation") for r in state)),
    }


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load research data ----
    common_seen = load_jsonl(DATA / "common_seen_train.jsonl")
    arms_data: Dict[str, Dict[str, List[Dict]]] = {}
    for arm_dir in sorted((DATA / "arms").iterdir()):
        if not arm_dir.is_dir():
            continue
        arms_data[arm_dir.name] = {
            "supervised": load_jsonl(arm_dir / "train_supervised.jsonl"),
            "unsupervised": load_jsonl(arm_dir / "train_unsup_text.jsonl"),
        }
    eval_data: Dict[str, List[Dict]] = {}
    for ef in sorted((DATA / "eval").iterdir()):
        if ef.suffix == ".jsonl":
            eval_data[ef.stem] = load_jsonl(ef)

    # ---- Build conditions ----
    full_report: Dict[str, Any] = {
        "status": "FACTORIAL_SUBSTRATE_COMPLETE",
        "source": str(DATA),
        "conditions": {},
    }

    for condition in ["underdetermined", "disambiguated"]:
        cond_out = out / condition
        save_jsonl(cond_out / "common_seen_train.jsonl", common_seen)

        arm_reports: Dict[str, Any] = {}
        for arm_name, data in sorted(arms_data.items()):
            new_sup, info = build_condition_arm(arm_name, data["supervised"], condition)
            save_jsonl(cond_out / "arms" / arm_name / "train_supervised.jsonl", new_sup)
            save_jsonl(cond_out / "arms" / arm_name / "train_unsup_text.jsonl", data["unsupervised"])

            # Baselines on training data
            info["train_anticopy"] = anticopy_baseline(new_sup)
            info["train_event_role"] = event_role_baseline(new_sup)
            info["train_balance"] = balance_check(new_sup)
            arm_reports[arm_name] = info

        # ---- Eval: augment state rows with both initial patterns ----
        eval_reports: Dict[str, Any] = {}
        for suite, rows in sorted(eval_data.items()):
            has_state = any(r.get("task") == "state_query" for r in rows)
            if has_state and any(r.get("supervised_changed_owner") for r in rows):
                new_rows, einfo = build_eval_both_patterns(rows)
            else:
                new_rows = list(rows)
                einfo = {"original_rows": len(rows), "augmented": False}

            save_jsonl(cond_out / "eval" / f"{suite}.jsonl", new_rows)

            # Baselines on eval data
            einfo["eval_anticopy"] = anticopy_baseline(new_rows)
            einfo["eval_event_role"] = event_role_baseline(new_rows)
            einfo["eval_anticopy_pair_both"] = pair_both_baseline(new_rows, "anticopy")
            einfo["eval_event_role_pair_both"] = pair_both_baseline(new_rows, "event_role")
            einfo["eval_balance"] = balance_check(new_rows)
            eval_reports[suite] = einfo

        full_report["conditions"][condition] = {
            "arms": arm_reports,
            "eval": eval_reports,
        }

    # ---- Save reports ----
    (out / "factorial_report.json").write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    # ---- Summary markdown ----
    md_lines = ["# research factorial initial-ownership substrate", ""]
    md_lines.append("## Design")
    md_lines.append("Two matched training conditions built from research data:")
    md_lines.append("- **Underdetermined**: all state pairs initial=opposite (anti-copy works)")
    md_lines.append("- **Disambiguated**: alternating opposite/same (anti-copy fails on half)")
    md_lines.append("- Comparison rows, unsupervised text, and common-seen train identical")
    md_lines.append("- Eval includes both initial patterns for all state suites")
    md_lines.append("")

    for cond, cr in full_report["conditions"].items():
        md_lines.append(f"## Condition: {cond}")
        md_lines.append("")
        md_lines.append("### Training arms")
        md_lines.append("")
        md_lines.append("| arm | state pairs | opp | same | comp rows | anticopy changed | event-role changed |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm, info in sorted(cr["arms"].items()):
            pc = info["pattern_counts"]
            ac_chg = info["train_anticopy"].get("all_changed_acc")
            er_chg = info["train_event_role"].get("all_changed_acc")
            md_lines.append(f"| {arm} | {info['state_pairs']} | {pc.get('opposite',0)} | {pc.get('same',0)} | {info['comparison_rows']} | {ac_chg:.3f} | {er_chg:.3f} |" if ac_chg is not None and er_chg is not None else f"| {arm} | {info['state_pairs']} | {pc.get('opposite',0)} | {pc.get('same',0)} | {info['comparison_rows']} | n/a | n/a |")
        md_lines.append("")

        md_lines.append("### Eval baselines")
        md_lines.append("")
        for suite, einfo in sorted(cr["eval"].items()):
            ac = einfo.get("eval_anticopy", {})
            er = einfo.get("eval_event_role", {})
            acpb = einfo.get("eval_anticopy_pair_both", {})
            erpb = einfo.get("eval_event_role_pair_both", {})

            md_lines.append(f"**{suite}**")
            md_lines.append("")
            if ac.get("opposite_changed_acc") is not None:
                md_lines.append("| pattern | anticopy changed | event-role changed | anticopy pair-both | event-role pair-both |")
                md_lines.append("|---|---:|---:|---:|---:|")
                for pat in ["opposite", "same", "all"]:
                    ac_c = ac.get(f"{pat}_changed_acc")
                    er_c = er.get(f"{pat}_changed_acc")
                    ac_pb = acpb.get(f"{pat}_pair_both")
                    er_pb = erpb.get(f"{pat}_pair_both")
                    if ac_c is not None:
                        md_lines.append(f"| {pat} | {ac_c:.3f} | {er_c:.3f} | {ac_pb:.3f} | {er_pb:.3f} |" if er_c is not None and ac_pb is not None and er_pb is not None else f"| {pat} | {ac_c:.3f} | - | - | - |")
                md_lines.append("")
            else:
                md_lines.append(f"(no state rows or no initial-pattern annotation)")
                md_lines.append("")

    md_lines.append("## Predicted transition")
    md_lines.append("")
    md_lines.append("If disambiguation forces event-role learning:")
    md_lines.append("1. Disambiguated models should achieve high changed accuracy on BOTH initial patterns")
    md_lines.append("2. Underdetermined models should fail on initial=same (anti-copy predicts wrong)")
    md_lines.append("3. Mixed held-seen orientation should separate (aligned > inverted) in disambiguated")
    md_lines.append("4. Pair-both on initial=same should transition from low to high")
    md_lines.append("")
    md_lines.append("## Files")
    md_lines.append(f"- Report JSON: `{out / 'factorial_report.json'}`")
    md_lines.append(f"- Formal derivation: `notes/formal_derivation_factorial_disambiguation.md`")
    md_lines.append(f"- research base: `{DATA}`")
    md_lines.append(f"- research confound: `notes/verified_initial_owner_shortcut_and_open_controls.md`")
    md_lines.append("")

    summary_path = out / "factorial_substrate_summary.md"
    write_md(summary_path, "\n".join(md_lines))

    # ---- Print compact JSON result ----
    compact = {
        "status": "FACTORIAL_SUBSTRATE_COMPLETE",
        "out": str(out.relative_to(PROJECT_ROOT)),
        "summary_md": str(summary_path.relative_to(PROJECT_ROOT)),
    }
    # Add key baselines
    for cond in ["underdetermined", "disambiguated"]:
        cr = full_report["conditions"][cond]
        for arm in ["aligned_state_bridge", "inverted_state_bridge"]:
            ai = cr["arms"].get(arm, {})
            ac = ai.get("train_anticopy", {})
            compact[f"{cond}_{arm}_anticopy_changed"] = ac.get("all_changed_acc")
        # Eval key numbers
        for suite in ["paired_state_conservation"]:
            ei = cr["eval"].get(suite, {})
            ac = ei.get("eval_anticopy", {})
            compact[f"{cond}_{suite}_anticopy_same_changed"] = ac.get("same_changed_acc")
            compact[f"{cond}_{suite}_anticopy_opposite_changed"] = ac.get("opposite_changed_acc")
            acpb = ei.get("eval_anticopy_pair_both", {})
            compact[f"{cond}_{suite}_anticopy_same_pair_both"] = acpb.get("same_pair_both")
            compact[f"{cond}_{suite}_anticopy_opposite_pair_both"] = acpb.get("opposite_pair_both")
            erpb = ei.get("eval_event_role_pair_both", {})
            compact[f"{cond}_{suite}_event_role_same_pair_both"] = erpb.get("same_pair_both")
    compact["no_model_loading_training_evaluation_upload_or_leaderboard"] = True
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
