#!/usr/bin/env python3
"""Correct assignment-set analysis for research slot-orbit solver.

The first slot-orbit summary printed true-assignment and inverted-assignment
accuracies, which is useful but can be misread.  This script reports what the
training evidence actually identifies: if the satisfying assignment set is a
singleton, use that selected assignment; if it has two or more members, report
the eval range across train-consistent assignments and mark orientation as
unidentified.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/slot_orbit_solver')
SUBSTRATE_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate')

sys.path.insert(0, str(_public_path('experiments/archive/representation_and_objectives/scripts')))
import slot_orbit_solver as sol

ARMS = sol.ARMS
EVAL_SUITES = sol.EVAL_SUITES
TRUE_ASSIGNMENT = sol.TRUE_ASSIGNMENT
INVERTED_ASSIGNMENT = sol.INVERTED_ASSIGNMENT


def row_acc(rows, ass):
    labeled = [r for r in rows if r.get("task") in {"relation_comparison", "state_query"} and "label" in r]
    if not labeled:
        return None
    return sum(sol.predict_label(r, ass) == bool(r["label"]) for r in labeled) / len(labeled)


def split_acc(rows, ass, query_kind):
    labeled = [r for r in rows if r.get("task") == "state_query" and r.get("query_kind") == query_kind and "label" in r]
    if not labeled:
        return None
    return sum(sol.predict_label(r, ass) == bool(r["label"]) for r in labeled) / len(labeled)


def pair_both(rows, ass):
    labeled = [r for r in rows if r.get("task") == "state_query" and "label" in r]
    if not labeled:
        return None
    preds = [sol.predict_label(r, ass) for r in labeled]
    labs = [bool(r["label"]) for r in labeled]
    ps = sol.pair_state(labeled, preds, labs, prefix="x")
    return ps["x_pair_both"]


def analyze(substrate: Path, include_common_seen: bool) -> List[Dict[str, Any]]:
    common = sol.load_jsonl(substrate / "common_seen_train.jsonl") if include_common_seen else []
    evals = {s: sol.load_jsonl(substrate / "eval" / f"{s}.jsonl") for s in EVAL_SUITES}
    records: List[Dict[str, Any]] = []
    for arm in ARMS:
        train = sol.load_jsonl(substrate / "arms" / arm / "train_supervised.jsonl") + common
        sats = sol.satisfying(train)
        singleton = len(sats) == 1
        selected = sats[0] if singleton else None
        for suite, rows in evals.items():
            vals = [row_acc(rows, ass) for ass in sats]
            vals = [v for v in vals if v is not None]
            ch_vals = [split_acc(rows, ass, "changed") for ass in sats]
            ch_vals = [v for v in ch_vals if v is not None]
            un_vals = [split_acc(rows, ass, "unchanged") for ass in sats]
            un_vals = [v for v in un_vals if v is not None]
            both_vals = [pair_both(rows, ass) for ass in sats]
            both_vals = [v for v in both_vals if v is not None]
            rec: Dict[str, Any] = {
                "arm": arm,
                "include_common_seen": include_common_seen,
                "suite": suite,
                "n_satisfying": len(sats),
                "selected_assignment": selected,
                "true_in_satisfying": TRUE_ASSIGNMENT in sats,
                "inverted_in_satisfying": INVERTED_ASSIGNMENT in sats,
                "accuracy_range_min": min(vals) if vals else None,
                "accuracy_range_max": max(vals) if vals else None,
                "true_assignment_accuracy": row_acc(rows, TRUE_ASSIGNMENT),
                "inverted_assignment_accuracy": row_acc(rows, INVERTED_ASSIGNMENT),
                "selected_accuracy": row_acc(rows, selected) if selected else None,
                "selected_changed_accuracy": split_acc(rows, selected, "changed") if selected else None,
                "selected_unchanged_accuracy": split_acc(rows, selected, "unchanged") if selected else None,
                "selected_pair_both": pair_both(rows, selected) if selected else None,
                "changed_range_min": min(ch_vals) if ch_vals else None,
                "changed_range_max": max(ch_vals) if ch_vals else None,
                "unchanged_range_min": min(un_vals) if un_vals else None,
                "unchanged_range_max": max(un_vals) if un_vals else None,
                "pair_both_range_min": min(both_vals) if both_vals else None,
                "pair_both_range_max": max(both_vals) if both_vals else None,
            }
            records.append(rec)
    return records


def fmt(x):
    if x is None:
        return "nan"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", type=Path, default=SUBSTRATE_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--include-common-seen", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = analyze(args.substrate, args.include_common_seen)
    suffix = "with_common" if args.include_common_seen else "no_common"
    jp = args.out / f"slot_orbit_assignment_analysis_{suffix}.json"
    jp.write_text(json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md = args.out / f"slot_orbit_assignment_analysis_{suffix}_summary.md"
    lines = [f"# research slot-orbit assignment-set analysis ({suffix})", ""]
    lines.append("This is the corrected reading of the slot-orbit positive control. Singleton train-consistent assignment means the bridge identifies an orientation; two or more train-consistent assignments means orientation is not identified by that arm.")
    lines.append("")
    lines.append("| arm | sat. | selected | mixed selected | mixed range | state changed selected | changed range | state unchanged selected | pair both selected | pair both range |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        mx = next(r for r in records if r["arm"] == arm and r["suite"] == "mixed_held_seen_orientation")
        st = next(r for r in records if r["arm"] == arm and r["suite"] == "paired_state_conservation")
        selected = "singleton" if mx["selected_assignment"] is not None else "ambiguous"
        lines.append(
            f"| {arm} | {mx['n_satisfying']} | {selected} | {fmt(mx['selected_accuracy'])} | {fmt(mx['accuracy_range_min'])}-{fmt(mx['accuracy_range_max'])} | "
            f"{fmt(st['selected_changed_accuracy'])} | {fmt(st['changed_range_min'])}-{fmt(st['changed_range_max'])} | {fmt(st['selected_unchanged_accuracy'])} | {fmt(st['selected_pair_both'])} | {fmt(st['pair_both_range_min'])}-{fmt(st['pair_both_range_max'])} |"
        )
    lines.append("")
    lines.append("Reading: heldheld_only and neutral_decoupled preserve the true/inverted ambiguity, so their mixed/changed ranges span 0 to 1 but no orientation is selected. Aligned selects the true orientation; inverted selects the opposite orientation and therefore scores 0 on true-labeled mixed/changed rows while preserving unchanged rows. This is the intended positive control for the repaired text surface.")
    lines.append("")
    lines.append(f"- results_json: `{jp.relative_to(PROJECT_ROOT)}`")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status":"SLOT_ORBIT_ASSIGNMENT_ANALYSIS_COMPLETE","summary_md":str(md.relative_to(PROJECT_ROOT)),"results_json":str(jp.relative_to(PROJECT_ROOT)),"include_common_seen":args.include_common_seen}, indent=2), flush=True)


if __name__ == "__main__":
    main()
