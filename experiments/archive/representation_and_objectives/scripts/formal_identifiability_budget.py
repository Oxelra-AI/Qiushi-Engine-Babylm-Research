#!/usr/bin/env python3
"""research: formal identifiability analysis for k-budget substrates.

This CPU-only script enumerates held-relation role assignments for research and
research k-budget training files.  It asks what the labels determine in the
intended symbolic hypothesis class, independently of neural optimization.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT = Path("experiments/archive/representation_and_objectives")
OUT_DEFAULT = PROJECT / "data/formal_identifiability_budget"
ROOTS_DEFAULT = [
    PROJECT / "data/information_budget_substrate",
    PROJECT / "data/counterallocated_budget_substrate",
]
HELD_KEYS = ["h0_dax", "h1_mep", "h2_norp", "h3_ziv"]
TRUE_ASSIGNMENT = {"h0_dax": 1, "h1_mep": 1, "h2_norp": 0, "h3_ziv": 0}
INVERTED_ASSIGNMENT = {k: 1 - v for k, v in TRUE_ASSIGNMENT.items()}
SEEN_FINAL_SLOT = {"s_give": 1, "s_receive": 0}
REL_COMPONENT = {**{k: "held" for k in HELD_KEYS}, **{k: "seen" for k in SEEN_FINAL_SLOT}}
ARMS = ["heldheld_only", "aligned_state_bridge", "inverted_state_bridge"]


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


def owner_from_slot(slot: int, a: str, b: str) -> str:
    return (a, b)[slot]


def slot(rel_key: str, assignment: Dict[str, int]) -> int:
    if rel_key in TRUE_ASSIGNMENT:
        return int(assignment[rel_key])
    return int(SEEN_FINAL_SLOT[rel_key])


def final_owner(rel_key: str, a: str, b: str, assignment: Dict[str, int]) -> str:
    return owner_from_slot(slot(rel_key, assignment), a, b)


def predict_label(row: Dict[str, Any], assignment: Dict[str, int]) -> bool:
    if row.get("task") == "relation_comparison":
        r1, r2 = row["relation1"], row["relation2"]
        a1, b1 = row["arg_order1"]
        a2, b2 = row["arg_order2"]
        return final_owner(r1, a1, b1, assignment) == final_owner(r2, a2, b2, assignment)
    if row.get("task") == "state_query":
        if row.get("query_kind") == "unchanged":
            return row["candidate"] == row["static_owner"]
        cause = row.get("cause_relation", row.get("relation"))
        if cause in SEEN_FINAL_SLOT:
            return row["candidate_slot"] == SEEN_FINAL_SLOT[cause]
        return row["candidate_slot"] == assignment[cause]
    raise ValueError(row.get("task"))


def satisfying_assignments(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    labeled = [r for r in rows if r.get("task") in {"relation_comparison", "state_query"}]
    sats = []
    for bits in itertools.product([0, 1], repeat=len(HELD_KEYS)):
        ass = dict(zip(HELD_KEYS, bits))
        if all(predict_label(r, ass) == bool(r["label"]) for r in labeled):
            sats.append(ass)
    return sats


def assignment_name(ass: Dict[str, int]) -> str:
    if ass == TRUE_ASSIGNMENT:
        return "true"
    if ass == INVERTED_ASSIGNMENT:
        return "inverted"
    return "mixed_" + "".join(str(ass[k]) for k in HELD_KEYS)


def hamming_to_true(ass: Dict[str, int]) -> int:
    return sum(int(ass[k] != TRUE_ASSIGNMENT[k]) for k in HELD_KEYS)


def row_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    state = [r for r in rows if r.get("task") == "state_query"]
    changed_true = [r for r in state if r.get("query_kind") == "changed" and bool(r.get("label"))]
    comp = [r for r in rows if r.get("task") == "relation_comparison"]
    return {
        "rows": len(rows),
        "state_rows": len(state),
        "comparison_rows": len(comp),
        "state_pairs": len({r.get("pair_id") for r in state}),
        "same_state_pairs": len({r.get("pair_id") for r in state if r.get("initial_pattern") == "same"}),
        "opposite_state_pairs": len({r.get("pair_id") for r in state if r.get("initial_pattern") == "opposite"}),
        "changed_true_relation_counts": dict(sorted(Counter(r.get("relation") for r in changed_true).items())),
        "changed_true_relation_voice_static_pattern_counts": dict(sorted(Counter(
            f"{r.get('relation')}|{r.get('voice')}|st{r.get('static_slot')}|{r.get('initial_pattern')}" for r in changed_true
        ).items())),
        "comparison_pair_counts": dict(sorted(Counter(f"{r.get('relation1')}->{r.get('relation2')}" for r in comp).items())),
    }


def analyze_arm(root: Path, condition: str, arm: str) -> Dict[str, Any]:
    path = root / condition / "arms" / arm / "train_supervised.jsonl"
    rows = load_jsonl(path)
    sats = satisfying_assignments(rows)
    return {
        "condition": condition,
        "arm": arm,
        "path": str(path),
        "counts": row_counts(rows),
        "satisfying_assignment_count": len(sats),
        "satisfying_assignment_names": [assignment_name(a) for a in sats],
        "satisfying_assignments": sats,
        "true_satisfies": TRUE_ASSIGNMENT in sats,
        "inverted_satisfies": INVERTED_ASSIGNMENT in sats,
        "hamming_to_true_values": [hamming_to_true(a) for a in sats],
    }


def condition_dirs(root: Path) -> List[str]:
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "arms").exists() and (p / "common_seen_train.jsonl").exists():
            out.append(p.name)
    return out


def analyze_root(root: Path) -> Dict[str, Any]:
    reps: Dict[str, Any] = {}
    for cond in condition_dirs(root):
        reps[cond] = {}
        for arm in ARMS:
            p = root / cond / "arms" / arm / "train_supervised.jsonl"
            if p.exists():
                reps[cond][arm] = analyze_arm(root, cond, arm)
    return reps


def summarize(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# research formal identifiability analysis for budget substrates")
    lines.append("")
    lines.append("This CPU-only enumeration checks what the training labels determine in the intended held-relation role-assignment class.  It does not say what a neural learner will learn; it establishes whether the substrate itself contains enough information for the intended assignment.")
    lines.append("")
    for root, reps in report["roots"].items():
        lines.append(f"## Root `{root}`")
        lines.append("")
        lines.append("| condition | arm | state pairs | same pairs | satisfying assignments | true? | inverted? | assignment names |")
        lines.append("|---|---|---:|---:|---:|---|---|---|")
        for cond, arms in sorted(reps.items()):
            for arm, rec in sorted(arms.items()):
                c = rec["counts"]
                names = rec["satisfying_assignment_names"]
                short = names if len(names) <= 4 else names[:4] + [f"...{len(names)} total"]
                lines.append(f"| `{cond}` | {arm} | {c['state_pairs']} | {c['same_state_pairs']} | {rec['satisfying_assignment_count']} | {rec['true_satisfies']} | {rec['inverted_satisfies']} | `{short}` |")
        lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("- `heldheld_only` should retain the two global assignments because held-held comparisons define only relative constraints.")
    lines.append("- Any state bridge with true state labels should formally identify the true assignment in this symbolic class, even when anti-copy is observationally equivalent in the raw state-row surface.  Therefore the learned k0 versus k16 question is not formal identifiability alone; it is whether the neural learner selects the intended assignment rather than a lower-complexity shortcut.")
    lines.append("- If single-relation bridge conditions formally identify all held relations through the held-held comparison graph, but learned transfer stays relation-local, then the limiting factor is failure to use the comparison graph/role interface, not absence of symbolic information.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- full JSON: `{Path(report['out']) / 'formal_identifiability_budget_report.json'}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path, default=ROOTS_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    report: Dict[str, Any] = {
        "status": "FORMAL_IDENTIFIABILITY_BUDGET_COMPLETE",
        "true_assignment": TRUE_ASSIGNMENT,
        "inverted_assignment": INVERTED_ASSIGNMENT,
        "roots": {},
        "out": str(args.out),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }
    for root in args.roots:
        report["roots"][str(root)] = analyze_root(root)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "formal_identifiability_budget_report.json", report)
    (args.out / "formal_identifiability_budget_summary.md").write_text(summarize(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "summary": str(args.out / "formal_identifiability_budget_summary.md"),
        "report": str(args.out / "formal_identifiability_budget_report.json"),
        "roots": list(report["roots"]),
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
