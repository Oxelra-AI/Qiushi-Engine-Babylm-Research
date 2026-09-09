#!/usr/bin/env python3
"""research slot-orbit solver over the actual repaired text surface.

This is a transparent positive control: can the research surface instantiate the
research symmetry law once natural active/passive argument slots are parsed from
text, while order-only and BoW solvers remain at chance?

The parser reads event strings, not hidden row-coordinate fields. It extracts a
relation key plus two latent syntactic roles:
  slot0 = active subject / by-phrase source / receiver-subject role
  slot1 = direct object / passive subject / to-recipient role
It then enumerates the 16 held-relation orientation assignments and retains the
ones satisfying each arm's labels.  Eval uses the same parsed fields.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
SUBSTRATE_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/slot_orbit_solver')

HELD_LEX_TO_KEY = {"dax": "h0_dax", "mep": "h1_mep", "norp": "h2_norp", "ziv": "h3_ziv"}
HELD_KEYS = ["h0_dax", "h1_mep", "h2_norp", "h3_ziv"]
SEEN_FINAL_SLOT = {"s_give": 1, "s_receive": 0}
TRUE_ASSIGNMENT = {"h0_dax": 1, "h1_mep": 1, "h2_norp": 0, "h3_ziv": 0}
INVERTED_ASSIGNMENT = {k: 1 - v for k, v in TRUE_ASSIGNMENT.items()}
ARMS = ["exposure_only", "heldheld_only", "aligned_state_bridge", "inverted_state_bridge", "neutral_decoupled", "mixed_event_bridge"]
EVAL_SUITES = ["heldheld_unseen_edge_closure", "mixed_held_seen_orientation", "paired_state_conservation", "cross_template_state_readout", "name_permutation_counterfactual"]

NAME = r"([A-Z][a-z]+)"
OBJ = r"([a-z]+)"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_event(text: str) -> Tuple[str, str, str]:
    """Return (relation_key, slot0_name, slot1_name) from an event sentence."""
    t = text.strip().rstrip(".")
    # held active: During the lantern episode, Mira daxed Omar.
    m = re.search(rf"During the {OBJ} episode, {NAME} (dax|mep|norp|ziv)ed {NAME}$", t)
    if m:
        _, a, lex, b = m.groups()
        return HELD_LEX_TO_KEY[lex], a, b
    # held passive: During the lantern episode, Omar was daxed by Mira.
    m = re.search(rf"During the {OBJ} episode, {NAME} was (dax|mep|norp|ziv)ed by {NAME}$", t)
    if m:
        _, b, lex, a = m.groups()
        return HELD_LEX_TO_KEY[lex], a, b
    # give active
    m = re.search(rf"During the {OBJ} episode, {NAME} gave the {OBJ} to {NAME}$", t)
    if m:
        _, a, _obj2, b = m.groups()
        return "s_give", a, b
    # give passive: object was given to B by A
    m = re.search(rf"During the {OBJ} episode, the {OBJ} was given to {NAME} by {NAME}$", t)
    if m:
        _, _obj2, b, a = m.groups()
        return "s_give", a, b
    # receive active
    m = re.search(rf"During the {OBJ} episode, {NAME} received the {OBJ} from {NAME}$", t)
    if m:
        _, a, _obj2, b = m.groups()
        return "s_receive", a, b
    # receive fronted/passive-like
    m = re.search(rf"During the {OBJ} episode, from {NAME}, the {OBJ} was received by {NAME}$", t)
    if m:
        _, b, _obj2, a = m.groups()
        return "s_receive", a, b
    raise ValueError(f"cannot parse event: {text!r}")


def final_slot(rel: str, ass: Dict[str, int]) -> int:
    if rel in ass:
        return int(ass[rel])
    return int(SEEN_FINAL_SLOT[rel])


def final_owner_from_event(event_text: str, ass: Dict[str, int]) -> str:
    rel, a, b = parse_event(event_text)
    return (a, b)[final_slot(rel, ass)]


def event_relation(event_text: str) -> str:
    return parse_event(event_text)[0]


def predict_comparison(row: Dict[str, Any], ass: Dict[str, int]) -> bool:
    return final_owner_from_event(row["event1"], ass) == final_owner_from_event(row["event2"], ass)


def parse_state_query(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Return event text, candidate name, object, static owner from a state row."""
    hyp = row["hypothesis"].strip().rstrip(".")
    m = re.match(rf"After the event, {NAME} had the {OBJ}$", hyp)
    if not m:
        raise ValueError(f"cannot parse hypothesis: {row['hypothesis']!r}")
    candidate, obj = m.groups()
    # We use row-provided cause_event field as the actual event string written in the text.
    ev = row.get("cause_event")
    if not ev:
        # fallback extraction from premise suffix
        pm = re.search(r"The event was this: (.*)$", row["premise"])
        if not pm:
            raise ValueError(f"cannot parse premise event: {row['premise']!r}")
        ev = pm.group(1).strip()
    # Static owner is explicitly mentioned in the natural premise.
    sm = re.search(rf", and {NAME} had the {re.escape(str(row.get('static_object', obj)))}", row["premise"])
    static_owner = sm.group(1) if sm else row.get("static_owner")
    return ev, candidate, obj, static_owner


def predict_state(row: Dict[str, Any], ass: Dict[str, int]) -> bool:
    ev, cand, obj, static_owner = parse_state_query(row)
    if row.get("query_kind") == "unchanged":
        return cand == static_owner
    return cand == final_owner_from_event(ev, ass)


def predict_label(row: Dict[str, Any], ass: Dict[str, int]) -> bool:
    if row.get("task") == "relation_comparison":
        return predict_comparison(row, ass)
    if row.get("task") == "state_query":
        return predict_state(row, ass)
    raise ValueError(row.get("task"))


def assignments() -> List[Dict[str, int]]:
    outs: List[Dict[str, int]] = []
    for bits in __import__("itertools").product([0, 1], repeat=len(HELD_KEYS)):
        outs.append(dict(zip(HELD_KEYS, bits)))
    return outs


def satisfying(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, int]]:
    labeled = [r for r in rows if r.get("task") in {"relation_comparison", "state_query"} and "label" in r]
    sats: List[Dict[str, int]] = []
    for ass in assignments():
        if all(predict_label(r, ass) == bool(r["label"]) for r in labeled):
            sats.append(ass)
    return sats


def eval_with_assignments(rows: Sequence[Dict[str, Any]], sats: Sequence[Dict[str, int]]) -> Dict[str, Any]:
    labeled = [r for r in rows if r.get("task") in {"relation_comparison", "state_query"} and "label" in r]
    if not labeled:
        return {"n": 0}
    # If multiple assignments remain, report both: true-choice and majority over remaining assignments.
    true_ass = TRUE_ASSIGNMENT if TRUE_ASSIGNMENT in sats else (sats[0] if sats else TRUE_ASSIGNMENT)
    inv_ass = INVERTED_ASSIGNMENT if INVERTED_ASSIGNMENT in sats else (sats[-1] if sats else INVERTED_ASSIGNMENT)
    def score(ass: Dict[str, int]) -> List[bool]:
        return [predict_label(r, ass) for r in labeled]
    labels = [bool(r["label"]) for r in labeled]
    pred_true = score(TRUE_ASSIGNMENT)
    pred_inv = score(INVERTED_ASSIGNMENT)
    if sats:
        votes = []
        for i, r in enumerate(labeled):
            vals = [predict_label(r, ass) for ass in sats]
            votes.append(sum(vals) >= len(vals) / 2)
    else:
        votes = [False] * len(labeled)
    out: Dict[str, Any] = {
        "n": len(labels),
        "n_satisfying_train_assignments": len(sats),
        "true_assignment_accuracy": acc(pred_true, labels),
        "inverted_assignment_accuracy": acc(pred_inv, labels),
        "remaining_assignment_majority_accuracy": acc(votes, labels),
    }
    for q in ["changed", "unchanged"]:
        idx = [i for i, r in enumerate(labeled) if r.get("query_kind") == q]
        if idx:
            out[f"true_acc_{q}"] = acc([pred_true[i] for i in idx], [labels[i] for i in idx])
            out[f"inverted_acc_{q}"] = acc([pred_inv[i] for i in idx], [labels[i] for i in idx])
    # Pair conservation for true and inverted assignments.
    if any(r.get("task") == "state_query" for r in labeled):
        for name, preds in [("true", pred_true), ("inverted", pred_inv), ("majority", votes)]:
            out.update(pair_state(labeled, preds, labels, prefix=name))
    return out


def acc(pred: Sequence[bool], y: Sequence[bool]) -> float:
    return sum(bool(p) == bool(t) for p, t in zip(pred, y)) / max(len(y), 1)


def pair_state(rows: Sequence[Dict[str, Any]], preds: Sequence[bool], labels: Sequence[bool], prefix: str) -> Dict[str, float | int]:
    by_pair: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("task") == "state_query":
            by_pair[str(r.get("pair_id"))].append(i)
    c = Counter()
    for idxs in by_pair.values():
        ch = [i for i in idxs if rows[i].get("query_kind") == "changed"]
        un = [i for i in idxs if rows[i].get("query_kind") == "unchanged"]
        cok = bool(ch) and all(preds[i] == labels[i] for i in ch)
        uok = bool(un) and all(preds[i] == labels[i] for i in un)
        c["both" if cok and uok else "changed_only" if cok else "unchanged_only" if uok else "neither"] += 1
    denom = max(sum(c.values()), 1)
    return {f"{prefix}_pair_{k}": c[k] / denom for k in ["both", "changed_only", "unchanged_only", "neither"]} | {f"{prefix}_pair_n": denom}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", type=Path, default=SUBSTRATE_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--include-common-seen", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    common = load_jsonl(args.substrate / "common_seen_train.jsonl") if args.include_common_seen else []
    arms = {a: load_jsonl(args.substrate / "arms" / a / "train_supervised.jsonl") for a in ARMS}
    evals = {s: load_jsonl(args.substrate / "eval" / f"{s}.jsonl") for s in EVAL_SUITES}
    records: List[Dict[str, Any]] = []
    for arm, rows in arms.items():
        train = rows + common
        sats = satisfying(train)
        rec_base = {
            "arm": arm,
            "include_common_seen": args.include_common_seen,
            "train_rows": len(train),
            "satisfying_assignment_count": len(sats),
            "satisfying_assignments": sats,
            "true_assignment_satisfies": TRUE_ASSIGNMENT in sats,
            "inverted_assignment_satisfies": INVERTED_ASSIGNMENT in sats,
        }
        for suite, erows in evals.items():
            rec = dict(rec_base)
            rec["suite"] = suite
            rec.update(eval_with_assignments(erows, sats))
            records.append(rec)
    suffix = "with_common" if args.include_common_seen else "no_common"
    jp = args.out / f"slot_orbit_solver_{suffix}.json"
    jp.write_text(json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md = args.out / f"slot_orbit_solver_{suffix}_summary.md"
    lines: List[str] = []
    lines.append(f"# research slot-orbit solver ({suffix})")
    lines.append("")
    lines.append("Transparent positive control over the actual text: parse active/passive/give/receive events into reusable argument slots, then enumerate held-relation orientation assignments from train labels.")
    lines.append("")
    lines.append("| arm | sat. assignments | true? | inverted? | hh true | mixed true | mixed inverted | state true changed | state inverted changed | state true both | state inverted both |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        rs = [r for r in records if r["arm"] == arm]
        hh = next(r for r in rs if r["suite"] == "heldheld_unseen_edge_closure")
        mx = next(r for r in rs if r["suite"] == "mixed_held_seen_orientation")
        st = next(r for r in rs if r["suite"] == "paired_state_conservation")
        lines.append(
            f"| {arm} | {hh['satisfying_assignment_count']} | {hh['true_assignment_satisfies']} | {hh['inverted_assignment_satisfies']} | "
            f"{hh['true_assignment_accuracy']:.3f} | {mx['true_assignment_accuracy']:.3f} | {mx['inverted_assignment_accuracy']:.3f} | "
            f"{st.get('true_acc_changed', float('nan')):.3f} | {st.get('inverted_acc_changed', float('nan')):.3f} | "
            f"{st.get('true_pair_both', float('nan')):.3f} | {st.get('inverted_pair_both', float('nan')):.3f} |"
        )
    lines.append("")
    lines.append("Reading: heldheld/neutral leave two train-consistent orientations; aligned state keeps only true; inverted keeps only inverted; mixed event keeps only true. On the repaired surface, success therefore requires a role-slot interface plus sparse orientation, not name order or lexical BoW statistics.")
    lines.append("")
    lines.append(f"- results_json: `{jp.relative_to(PROJECT_ROOT)}`")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "SLOT_ORBIT_SOLVER_COMPLETE", "summary_md": str(md.relative_to(PROJECT_ROOT)), "results_json": str(jp.relative_to(PROJECT_ROOT)), "include_common_seen": args.include_common_seen}, indent=2), flush=True)


if __name__ == "__main__":
    main()
