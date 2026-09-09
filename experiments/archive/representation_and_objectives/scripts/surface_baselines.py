#!/usr/bin/env python3
"""research transparent surface baselines for the equivariant substrate.

This script tests whether the repaired research surface still admits simple
non-semantic solutions.  It trains deterministic pattern-majority classifiers on
visible features and evaluates them on the held-out suites.

Feature groups:
  order_only:    visible order, voice, event side, candidate position; no relation IDs.
  order_reltype: order_only plus component/relation-pair type, but not nonce identities.
  order_relid:  order_only plus relation identities/lexemes (a stronger transparent check).

The scientific requirement before model pilots is that order_only is at chance
on mixed held-seen and changed-state readouts; relation identity may solve
held-held consistency but should not pick an absolute mixed orientation without
bridge evidence.
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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
PROJECT_ROOT = _public_path('.')
SUBSTRATE_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/equivariant_symmetry_substrate')
OUT_DEFAULT = _public_path('experiments/archive/representation_and_objectives/data/surface_baselines')

ARMS = [
    "exposure_only",
    "heldheld_only",
    "aligned_state_bridge",
    "inverted_state_bridge",
    "neutral_decoupled",
    "mixed_event_bridge",
]
EVAL_SUITES = [
    "heldheld_unseen_edge_closure",
    "mixed_held_seen_orientation",
    "paired_state_conservation",
    "cross_template_state_readout",
    "name_permutation_counterfactual",
]
FEATURE_GROUPS = ["order_only", "order_reltype", "order_relid"]


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


def canonical_pair(x: str, y: str) -> str:
    return "|".join(sorted([str(x), str(y)]))


def comp_features(r: Dict[str, Any], group: str) -> Tuple[Tuple[str, Any], ...]:
    so1 = tuple(r.get("surface_order1") or [])
    so2 = tuple(r.get("surface_order2") or [])
    ao1 = tuple(r.get("arg_order1") or [])
    ao2 = tuple(r.get("arg_order2") or [])
    feats: List[Tuple[str, Any]] = [
        ("task", "relation_comparison"),
        ("surface_first_same", bool(so1 and so2 and so1[0] == so2[0])),
        ("surface_order_exact_same", so1 == so2),
        ("surface_order_reversed", len(so1) == 2 and len(so2) == 2 and so1[0] == so2[1] and so1[1] == so2[0]),
        ("latent_arg_order_same", ao1 == ao2),
        ("latent_arg_order_reversed", len(ao1) == 2 and len(ao2) == 2 and ao1[0] == ao2[1] and ao1[1] == ao2[0]),
        ("voice1", r.get("voice1")),
        ("voice2", r.get("voice2")),
        ("voice_same", r.get("voice1") == r.get("voice2")),
        ("held_first", r.get("component1") == "held"),
    ]
    if group in {"order_reltype", "order_relid"}:
        rel1 = str(r.get("relation1"))
        rel2 = str(r.get("relation2"))
        c1 = str(r.get("component1"))
        c2 = str(r.get("component2"))
        # Family/type available from relation naming but no absolute held orientation.
        held_family_1 = rel1.split("_")[0] if rel1.startswith("h") else rel1
        held_family_2 = rel2.split("_")[0] if rel2.startswith("h") else rel2
        feats += [
            ("component_pair", f"{c1}>{c2}"),
            ("component_unordered", canonical_pair(c1, c2)),
            ("seen_relation_present", rel1 if c1 == "seen" else rel2 if c2 == "seen" else "none"),
            ("held_family_pair", canonical_pair(held_family_1, held_family_2)),
        ]
    if group == "order_relid":
        feats += [
            ("relation_pair_ordered", f"{r.get('relation1')}>{r.get('relation2')}"),
            ("relation_pair_unordered", canonical_pair(r.get("relation1"), r.get("relation2"))),
            ("relation1", r.get("relation1")),
            ("relation2", r.get("relation2")),
            ("template_pair", f"{r.get('template1')}>{r.get('template2')}"),
        ]
    return tuple(sorted(feats))


def state_features(r: Dict[str, Any], group: str) -> Tuple[Tuple[str, Any], ...]:
    so = tuple(r.get("surface_order") or [])
    cand = r.get("candidate")
    feats: List[Tuple[str, Any]] = [
        ("task", "state_query"),
        ("query_kind", r.get("query_kind")),
        ("candidate_surface_first", bool(so and so[0] == cand)),
        ("candidate_surface_second", bool(len(so) > 1 and so[1] == cand)),
        ("candidate_slot0", r.get("candidate_slot") == 0),
        ("candidate_slot1", r.get("candidate_slot") == 1),
        ("voice", r.get("voice")),
        ("active_voice", r.get("voice") == "active"),
    ]
    if group in {"order_reltype", "order_relid"}:
        rel = str(r.get("relation", r.get("cause_relation")))
        comp = str(r.get("component"))
        cause = str(r.get("cause_relation", rel))
        feats += [
            ("component", comp),
            ("cause_component", "held" if cause.startswith("h") else "seen" if cause.startswith("s") else "other"),
            ("has_held_distractor", bool(r.get("held_distractor_relation"))),
        ]
    if group == "order_relid":
        feats += [
            ("relation", r.get("relation")),
            ("cause_relation", r.get("cause_relation", r.get("relation"))),
            ("held_distractor_relation", r.get("held_distractor_relation", "none")),
            ("template", r.get("template")),
        ]
    return tuple(sorted(feats))


def row_features(r: Dict[str, Any], group: str) -> Tuple[Tuple[str, Any], ...]:
    if r.get("task") == "relation_comparison":
        return comp_features(r, group)
    if r.get("task") == "state_query":
        return state_features(r, group)
    return (("task", r.get("task")),)


def fit_pattern_majority(rows: Sequence[Dict[str, Any]], group: str, task: str | None = None) -> Dict[str, Any]:
    labeled = [r for r in rows if "label" in r and (task is None or r.get("task") == task)]
    counts: Dict[Tuple[Tuple[str, Any], ...], Counter] = defaultdict(Counter)
    global_counts: Counter = Counter()
    for r in labeled:
        y = bool(r["label"])
        counts[row_features(r, group)][y] += 1
        global_counts[y] += 1
    default = bool(global_counts[True] >= global_counts[False]) if global_counts else False
    mapping = {k: (v[True] >= v[False]) for k, v in counts.items()}
    train_preds = [mapping[row_features(r, group)] for r in labeled]
    train_labels = [bool(r["label"]) for r in labeled]
    train_acc = acc(train_preds, train_labels)
    return {"mapping": mapping, "default": default, "n_train": len(labeled), "n_patterns": len(mapping), "train_acc": train_acc}


def predict(model: Dict[str, Any], rows: Sequence[Dict[str, Any]], group: str, task: str | None = None) -> List[bool]:
    labeled = [r for r in rows if "label" in r and (task is None or r.get("task") == task)]
    return [model["mapping"].get(row_features(r, group), model["default"]) for r in labeled]


def acc(preds: Sequence[bool], labels: Sequence[bool]) -> float | None:
    if not labels:
        return None
    return sum(p == y for p, y in zip(preds, labels)) / len(labels)


def eval_model(model: Dict[str, Any], rows: Sequence[Dict[str, Any]], group: str, task: str | None = None) -> Dict[str, Any]:
    labeled = [r for r in rows if "label" in r and (task is None or r.get("task") == task)]
    preds = predict(model, labeled, group)
    labels = [bool(r["label"]) for r in labeled]
    out: Dict[str, Any] = {
        "n": len(labels),
        "accuracy": acc(preds, labels),
        "pred_true_frac": (sum(preds) / len(preds) if preds else None),
        "label_true_frac": (sum(labels) / len(labels) if labels else None),
        "unseen_pattern_frac": (sum(row_features(r, group) not in model["mapping"] for r in labeled) / len(labeled) if labeled else None),
    }
    # Changed/unchanged split for state queries
    for q in ["changed", "unchanged"]:
        idx = [i for i, r in enumerate(labeled) if r.get("query_kind") == q]
        if idx:
            out[f"acc_{q}"] = sum(preds[i] == labels[i] for i in idx) / len(idx)
            out[f"pred_true_frac_{q}"] = sum(preds[i] for i in idx) / len(idx)
    # Orientation dependency split
    for dep in sorted(set(r.get("orientation_dependency") for r in labeled)):
        if dep is None:
            continue
        idx = [i for i, r in enumerate(labeled) if r.get("orientation_dependency") == dep]
        if idx:
            out[f"acc_dep_{dep}"] = sum(preds[i] == labels[i] for i in idx) / len(idx)
    return out


def load_all(substrate: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    common = load_jsonl(substrate / "common_seen_train.jsonl")
    arms = {a: load_jsonl(substrate / "arms" / a / "train_supervised.jsonl") for a in ARMS}
    evals = {s: load_jsonl(substrate / "eval" / f"{s}.jsonl") for s in EVAL_SUITES}
    return arms, evals, common


def fmt(x: Any) -> str:
    if x is None:
        return "nan"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", type=Path, default=SUBSTRATE_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--include-common-seen", action="store_true", help="include common seen state rows in train baselines")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    arms, evals, common = load_all(args.substrate)
    records: List[Dict[str, Any]] = []
    for arm, arm_rows in arms.items():
        train_rows = list(arm_rows) + (list(common) if args.include_common_seen else [])
        for group in FEATURE_GROUPS:
            for task in [None, "relation_comparison", "state_query"]:
                model = fit_pattern_majority(train_rows, group, task)
                for suite, rows in evals.items():
                    ev = eval_model(model, rows, group, task)
                    rec = {
                        "arm": arm,
                        "feature_group": group,
                        "task_fit": task or "all_labeled",
                        "suite": suite,
                        "n_train": model["n_train"],
                        "n_patterns": model["n_patterns"],
                        "train_acc": model["train_acc"],
                    }
                    rec.update(ev)
                    records.append(rec)

    suffix = "with_common" if args.include_common_seen else "no_common"
    json_path = args.out / f"surface_baselines_{suffix}.json"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Compact scientific summary for critical suites.
    md = args.out / f"surface_baselines_{suffix}_summary.md"
    lines: List[str] = []
    lines.append(f"# research transparent surface baselines ({suffix})")
    lines.append("")
    lines.append("Pattern-majority classifiers over visible features. `order_only` excludes relation IDs; `order_relid` is the strongest transparent lexical check.")
    lines.append("")
    lines.append("## Critical comparison transfer")
    lines.append("")
    lines.append("| arm | feature group | task fit | train acc | hh eval | mixed eval | mixed unseen-pattern |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for arm in ARMS:
        for group in FEATURE_GROUPS:
            recs = [r for r in records if r["arm"] == arm and r["feature_group"] == group and r["task_fit"] == "relation_comparison"]
            hh = next((r for r in recs if r["suite"] == "heldheld_unseen_edge_closure"), None)
            mx = next((r for r in recs if r["suite"] == "mixed_held_seen_orientation"), None)
            if hh and mx:
                lines.append(f"| {arm} | {group} | relation_comparison | {fmt(mx['train_acc'])} | {fmt(hh['accuracy'])} | {fmt(mx['accuracy'])} | {fmt(mx['unseen_pattern_frac'])} |")
    lines.append("")
    lines.append("## Critical changed-state transfer")
    lines.append("")
    lines.append("| arm | feature group | task fit | train acc | paired changed | paired unchanged | cross-template changed |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for arm in ARMS:
        for group in FEATURE_GROUPS:
            recs = [r for r in records if r["arm"] == arm and r["feature_group"] == group and r["task_fit"] == "state_query"]
            ps = next((r for r in recs if r["suite"] == "paired_state_conservation"), None)
            ct = next((r for r in recs if r["suite"] == "cross_template_state_readout"), None)
            if ps and ct:
                lines.append(f"| {arm} | {group} | state_query | {fmt(ps['train_acc'])} | {fmt(ps.get('acc_changed'))} | {fmt(ps.get('acc_unchanged'))} | {fmt(ct.get('acc_changed'))} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("A repaired interface is acceptable for low-cost model pilots only if order_only is at chance on mixed held-seen comparison and changed-state readouts. The order_relid group is intentionally stronger; if it exceeds chance without bridge evidence, the lexical surface itself still leaks orientation.")
    lines.append("")
    lines.append(f"- results_json: `{json_path.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "SURFACE_BASELINES_COMPLETE",
        "include_common_seen": args.include_common_seen,
        "summary_md": str(md.relative_to(PROJECT_ROOT)),
        "results_json": str(json_path.relative_to(PROJECT_ROOT)),
        "n_records": len(records),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
