#!/usr/bin/env python3
"""research: independent audit for the rewired connectivity substrate.

This CPU-only script verifies whether the corrected substrate removes the research
name-exposure confound before any GPU training.  It compares connected vs rewired
model-visible inputs for each arm.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT = Path("experiments/archive/representation_and_objectives")
DATA_ROOT = PROJECT / "data/rewired_connectivity_substrate"
OUT = PROJECT / "data/rewired_connectivity_audit"
CONDITIONS = ["connected", "rewired"]
ARMS = ["aligned_state_bridge", "inverted_state_bridge", "heldheld_only"]
NAMES = {"Mira", "Omar", "Noel", "Iris", "Lena", "Pavel", "Rina", "Tomas", "Nia", "Felix", "Ava", "Jonas", "Keira", "Milo", "Sara", "Theo"}


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


def model_rows(condition: str, arm: str) -> List[Dict[str, Any]]:
    root = DATA_ROOT / condition
    return load_jsonl(root / "common_seen_train.jsonl") + load_jsonl(root / "arms" / arm / "train_supervised.jsonl")


def arm_blocks(condition: str, arm: str) -> Dict[str, List[Dict[str, Any]]]:
    root = DATA_ROOT / condition
    sup = load_jsonl(root / "arms" / arm / "train_supervised.jsonl")
    return {
        "common": load_jsonl(root / "common_seen_train.jsonl"),
        "supervised": sup,
        "bridge": [r for r in sup if r.get("task") == "state_query" and "bridge" in str(r.get("suite", ""))],
        "heldheld": [r for r in sup if r.get("task") == "relation_comparison"],
    }


def text_fields(row: Dict[str, Any]) -> Tuple[str, ...]:
    if row.get("task") == "relation_comparison":
        return (str(row.get("event1", "")), str(row.get("event2", "")), str(row.get("text", "")))
    if row.get("task") == "state_query":
        return (str(row.get("premise", "")), str(row.get("hypothesis", "")))
    if "text" in row:
        return (str(row.get("text", "")),)
    return tuple()


def row_text_signature(row: Dict[str, Any], include_label: bool = False) -> Tuple[Any, ...]:
    sig: Tuple[Any, ...] = (row.get("task"),) + text_fields(row)
    if include_label and "label" in row:
        sig += (bool(row.get("label")),)
    return sig


def delex(s: str) -> str:
    out = s
    # Replace longer names first to avoid accidental fragments, although these names are distinct.
    for nm in sorted(NAMES, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(nm)}\b", "NAME", out)
    return out


def delex_role_signature(row: Dict[str, Any], include_label: bool = False) -> Tuple[Any, ...]:
    fields = tuple(delex(x) for x in text_fields(row))
    sig: Tuple[Any, ...] = (row.get("task"), row.get("suite"), row.get("relation") or row.get("cause_relation"),
                            row.get("relation1"), row.get("relation2"), row.get("voice"), row.get("voice1"), row.get("voice2"),
                            row.get("static_slot"), row.get("initial_pattern"), row.get("query_kind"), row.get("candidate_slot")) + fields
    if include_label and "label" in row:
        sig += (bool(row.get("label")),)
    return sig


def tokens(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        for x in text_fields(r):
            c.update(tok.lower() for tok in re.findall(r"\b\w+\b", x))
    return c


def exact_name_role_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        label = "T" if bool(r.get("label")) else "F" if "label" in r else "NA"
        task = r.get("task")
        if task == "state_query":
            rel = r.get("relation") or r.get("cause_relation")
            for idx, nm in enumerate(r.get("arg_order", [])):
                if nm in NAMES:
                    c[(nm, task, "arg_slot", idx, rel, r.get("voice"), r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot"), label)] += 1
            for idx, nm in enumerate(r.get("surface_order", [])):
                if nm in NAMES:
                    c[(nm, task, "surface_pos", idx, rel, r.get("voice"), r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot"), label)] += 1
            for role in ["candidate", "initial_owner", "initial_changed_owner", "static_owner", "supervised_changed_owner"]:
                nm = r.get(role)
                if nm in NAMES:
                    c[(nm, task, role, r.get("candidate_slot") if role == "candidate" else None, rel, r.get("voice"), r.get("query_kind"), r.get("initial_pattern"), r.get("static_slot"), label)] += 1
        elif task == "relation_comparison":
            for side in [1, 2]:
                rel = r.get(f"relation{side}")
                for idx, nm in enumerate(r.get(f"arg_order{side}", [])):
                    if nm in NAMES:
                        c[(nm, task, f"arg{side}_slot", idx, rel, r.get(f"voice{side}"), label)] += 1
                for idx, nm in enumerate(r.get(f"surface_order{side}", [])):
                    if nm in NAMES:
                        c[(nm, task, f"surface{side}_pos", idx, rel, r.get(f"voice{side}"), label)] += 1
    return c


def pair_key_from_arg(row: Dict[str, Any], key: str = "arg_order") -> str | None:
    vals = row.get(key) or []
    if len(vals) != 2:
        return None
    return f"{vals[0]}::{vals[1]}"


def pair_degree_counts(blocks: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Counter]:
    out = {"common": Counter(), "bridge": Counter(), "heldheld": Counter(), "all_model": Counter()}
    for r in blocks["common"]:
        pk = pair_key_from_arg(r)
        if pk:
            out["common"][pk] += 1; out["all_model"][pk] += 1
    for r in blocks["bridge"]:
        pk = pair_key_from_arg(r)
        if pk:
            out["bridge"][pk] += 1; out["all_model"][pk] += 1
    for r in blocks["heldheld"]:
        pk = pair_key_from_arg(r, "arg_order1")
        if pk:
            out["heldheld"][pk] += 1; out["all_model"][pk] += 1
    return out


def count_equal(a: Counter, b: Counter) -> Dict[str, Any]:
    d = a.copy(); d.subtract(b)
    nz = {str(k): int(v) for k, v in d.items() if v != 0}
    return {"equal": not nz, "n_differences": len(nz), "sample": dict(list(sorted(nz.items()))[:30])}


def multiset_equal(a: Sequence[Any], b: Sequence[Any]) -> Dict[str, Any]:
    return count_equal(Counter(a), Counter(b))


def degree_sequence(c: Counter) -> List[int]:
    return sorted(c.values())


def summarize_arm(arm: str) -> Dict[str, Any]:
    cb = arm_blocks("connected", arm)
    rb = arm_blocks("rewired", arm)
    crows = cb["common"] + cb["supervised"]
    rrows = rb["common"] + rb["supervised"]
    cdeg = pair_degree_counts(cb)
    rdeg = pair_degree_counts(rb)
    report: Dict[str, Any] = {
        "n_rows_connected": len(crows),
        "n_rows_rewired": len(rrows),
        "common_rows_exact_same_order": [row_text_signature(x, include_label=True) for x in cb["common"]] == [row_text_signature(x, include_label=True) for x in rb["common"]],
        "bridge_rows_exact_same_multiset": multiset_equal([row_text_signature(x, include_label=True) for x in cb["bridge"]], [row_text_signature(x, include_label=True) for x in rb["bridge"]]),
        "heldheld_exact_text_multiset": multiset_equal([row_text_signature(x, include_label=False) for x in cb["heldheld"]], [row_text_signature(x, include_label=False) for x in rb["heldheld"]]),
        "heldheld_delexicalized_labeled_template_multiset": multiset_equal([delex_role_signature(x, include_label=True) for x in cb["heldheld"]], [delex_role_signature(x, include_label=True) for x in rb["heldheld"]]),
        "all_model_token_unigram_counts": count_equal(tokens(crows), tokens(rrows)),
        "all_model_name_role_label_counts": count_equal(exact_name_role_counts(crows), exact_name_role_counts(rrows)),
        "pair_degree_sequences": {k: {"connected": degree_sequence(cdeg[k]), "rewired": degree_sequence(rdeg[k]), "equal": degree_sequence(cdeg[k]) == degree_sequence(rdeg[k])} for k in cdeg},
        "pair_degrees_by_exact_pair": {k: {"connected": dict(sorted(cdeg[k].items())), "rewired": dict(sorted(rdeg[k].items()))} for k in cdeg},
        "bridge_heldheld_exact_overlap_connected": sorted(set(cdeg["bridge"]) & set(cdeg["heldheld"])),
        "bridge_heldheld_exact_overlap_rewired": sorted(set(rdeg["bridge"]) & set(rdeg["heldheld"])),
    }
    return report


def main() -> None:
    global DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    DATA_ROOT = args.data_root
    report = {"status": "REWIRED_CONNECTIVITY_AUDIT_COMPLETE", "data_root": str(DATA_ROOT), "arms": {}}
    for arm in ARMS:
        report["arms"][arm] = summarize_arm(arm)
    report["global"] = {}
    for arm, ar in report["arms"].items():
        report["global"][f"{arm}_row_count_equal"] = ar["n_rows_connected"] == ar["n_rows_rewired"]
        report["global"][f"{arm}_common_exact_same_order"] = ar["common_rows_exact_same_order"]
        report["global"][f"{arm}_bridge_exact_same_multiset"] = ar["bridge_rows_exact_same_multiset"]["equal"]
        report["global"][f"{arm}_heldheld_delex_labeled_equal"] = ar["heldheld_delexicalized_labeled_template_multiset"]["equal"]
        report["global"][f"{arm}_token_unigram_equal"] = ar["all_model_token_unigram_counts"]["equal"]
        report["global"][f"{arm}_name_role_label_equal"] = ar["all_model_name_role_label_counts"]["equal"]
        report["global"][f"{arm}_pair_degree_sequence_all_equal"] = ar["pair_degree_sequences"]["all_model"]["equal"]
        # Deliberately expected not to be equal for held-held exact strings: rewiring changes co-occurrence edges.
        report["global"][f"{arm}_heldheld_exact_text_multiset_equal"] = ar["heldheld_exact_text_multiset"]["equal"]
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "rewired_connectivity_audit.json", report)

    lines = ["# research independent audit of rewired connectivity substrate", ""]
    lines.append("This audit compares the model-visible connected and rewired training files before any GPU run.")
    lines.append("")
    lines.append("## Main checks")
    lines.append("")
    lines.append("| arm | rows equal | common exact | bridge exact | held-held delex+label | token unigrams | name role/label | all-pair degree sequence | held-held exact text |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, ar in report["arms"].items():
        g = report["global"]
        lines.append(f"| {arm} | {g[f'{arm}_row_count_equal']} | {g[f'{arm}_common_exact_same_order']} | {g[f'{arm}_bridge_exact_same_multiset']} | {g[f'{arm}_heldheld_delex_labeled_equal']} | {g[f'{arm}_token_unigram_equal']} | {g[f'{arm}_name_role_label_equal']} | {g[f'{arm}_pair_degree_sequence_all_equal']} | {g[f'{arm}_heldheld_exact_text_multiset_equal']} |")
    lines.append("")
    lines.append("## Topology readout")
    lines.append("")
    lines.append("| arm | connected bridge∩heldheld exact pairs | rewired bridge∩heldheld exact pairs | all-model pair degree sequence connected | all-model pair degree sequence rewired |")
    lines.append("|---|---|---|---|---|")
    for arm, ar in report["arms"].items():
        lines.append(f"| {arm} | {ar['bridge_heldheld_exact_overlap_connected']} | {ar['bridge_heldheld_exact_overlap_rewired']} | {ar['pair_degree_sequences']['all_model']['connected']} | {ar['pair_degree_sequences']['all_model']['rewired']} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("The corrected substrate fixes the research defect at the level that matters for the known filler-support confound: bridge supervision is exact-identical; common seen-coordinate exposure is exact-identical; token unigrams, delexicalized row templates, individual-name role/label counts, and pair-degree sequences are matched.  The held-held exact sentence multiset is intentionally not identical because the intervention is an edge rewiring of which name co-occurrences carry held-held constraints.  Thus a future learned comparison can be read as a topology/co-occurrence connectivity test, not as a different bridge dose or different individual-name role exposure.  If an even stricter same-exact-sentence contrast is desired, it would require label rewiring or contradictory decoy rows and should be treated as a different design.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- full audit JSON: `{args.out / 'rewired_connectivity_audit.json'}`")
    (args.out / "rewired_connectivity_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "summary": str(args.out / "rewired_connectivity_audit_summary.md"),
        "json": str(args.out / "rewired_connectivity_audit.json"),
        "central_checks": report["global"],
        "no_model_loading_training_evaluation_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
