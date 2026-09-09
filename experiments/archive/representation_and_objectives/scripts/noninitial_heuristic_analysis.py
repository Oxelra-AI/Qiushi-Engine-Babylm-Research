#!/usr/bin/env python3
"""Analyze research counterfactual predictions against a non-initial-owner heuristic.

The key question is whether the learned state-query behavior follows the latent
relation role or the simpler transition rule: for the changed object, pick the
participant who did not initially own it. Unchanged rows are scored by static-fact
preservation. This file is CPU-only and reads saved per-row logits.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path("experiments/archive/representation_and_objectives")
PRED_PATH = ROOT / "data" / "initial_owner_counterfactual_probe" / "counterfactual_per_row_predictions.jsonl"
OUTDIR = ROOT / "data" / "noninitial_heuristic_analysis"

STATE_SUITES = [
    "paired_state_conservation",
    "cf_initial_true_final",
    "cf_initial_opposite_true_final",
    "cf_initial_slot0",
    "cf_initial_slot1",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_mean(vals: Iterable[float]) -> float | None:
    vals = [float(v) for v in vals]
    return sum(vals) / len(vals) if vals else None


def safe_std(vals: Iterable[float]) -> float | None:
    vals = [float(v) for v in vals]
    if not vals:
        return None
    m = safe_mean(vals)
    assert m is not None
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def slot_int(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, str) and x in {"", "None", "null"}:
        return None
    return int(x)


def heuristic_slot(rows: List[Dict[str, Any]]) -> int | None:
    # All rows in the group share query_kind and initial/correct slots.
    r0 = rows[0]
    qk = str(r0.get("query_kind"))
    correct = slot_int(r0.get("correct_slot"))
    if correct is None:
        return None
    if qk == "unchanged":
        # The non-initial-owner heuristic is only about changed objects; unchanged
        # evidence should preserve the static owner.
        return correct
    if qk != "changed":
        return None
    init = slot_int(r0.get("initial_owner_slot"))
    if init is None:
        # In ordinary research state rows the changed object was constructed to
        # start with the complement of the true final owner. Therefore the
        # non-initial heuristic equals the true label by construction.
        return correct
    return 1 - init


def deterministic_heuristic_true_accuracy(rows: List[Dict[str, Any]]) -> float | None:
    hs = heuristic_slot(rows)
    if hs is None:
        return None
    correct = slot_int(rows[0].get("correct_slot"))
    if correct is None:
        return None
    return 1.0 if hs == correct else 0.0


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pred_rows = load_jsonl(PRED_PATH)
    pred_rows = [r for r in pred_rows if r.get("task") == "state_query" and r.get("suite_eval") in STATE_SUITES]

    # Group candidate rows into two-candidate choices.
    groups: Dict[Tuple[str, int, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in pred_rows:
        groups[(str(r["arm"]), int(r["seed"]), str(r["suite_eval"]), str(r["pair_id"]), str(r["query_kind"]))].append(r)

    choice_rows: List[Dict[str, Any]] = []
    for (arm, seed, suite, pid, qk), rows in sorted(groups.items()):
        if len(rows) != 2:
            continue
        by_slot = {slot_int(r.get("candidate_slot")): r for r in rows}
        if set(by_slot) != {0, 1}:
            continue
        pred_slot = 0 if float(by_slot[0]["logit_true"]) >= float(by_slot[1]["logit_true"]) else 1
        correct = slot_int(rows[0].get("correct_slot"))
        hs = heuristic_slot(rows)
        initial = slot_int(rows[0].get("initial_owner_slot"))
        choice_rows.append({
            "arm": arm,
            "seed": seed,
            "suite": suite,
            "pair_id": pid,
            "query_kind": qk,
            "relation": str(rows[0].get("relation")),
            "voice": str(rows[0].get("voice")),
            "correct_slot_true": correct,
            "initial_owner_slot": initial,
            "initial_equals_true_final": rows[0].get("initial_equals_true_final"),
            "pred_slot": pred_slot,
            "heuristic_slot_noninitial_or_static": hs,
            "model_true_correct": None if correct is None else int(pred_slot == correct),
            "model_heuristic_agreement": None if hs is None else int(pred_slot == hs),
            "deterministic_heuristic_true_correct": deterministic_heuristic_true_accuracy(rows),
            "slot0_logit_true": float(by_slot[0]["logit_true"]),
            "slot1_logit_true": float(by_slot[1]["logit_true"]),
            "slot1_minus_slot0": float(by_slot[1]["logit_true"]) - float(by_slot[0]["logit_true"]),
        })

    # Aggregate by arm/suite/query_kind/seed first, then mean over seeds.
    seed_metrics: Dict[Tuple[str, str, str, int], Dict[str, float]] = {}
    by_seed: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in choice_rows:
        by_seed[(r["arm"], r["suite"], r["query_kind"], r["seed"])].append(r)
    for key, rows in by_seed.items():
        seed_metrics[key] = {
            "n": len(rows),
            "model_true_acc": safe_mean(r["model_true_correct"] for r in rows if r["model_true_correct"] is not None),
            "model_heuristic_agree": safe_mean(r["model_heuristic_agreement"] for r in rows if r["model_heuristic_agreement"] is not None),
            "heuristic_true_acc": safe_mean(r["deterministic_heuristic_true_correct"] for r in rows if r["deterministic_heuristic_true_correct"] is not None),
            "mean_slot1_minus_slot0": safe_mean(r["slot1_minus_slot0"] for r in rows),
        }

    summary_rows: List[Dict[str, Any]] = []
    key2vals: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (arm, suite, qk, seed), m in seed_metrics.items():
        for metric, val in m.items():
            if metric == "n" or val is None:
                continue
            key2vals[(arm, suite, qk)][metric].append(float(val))
    for (arm, suite, qk), metrics in sorted(key2vals.items()):
        rec: Dict[str, Any] = {"arm": arm, "suite": suite, "query_kind": qk, "n_seeds": len(next(iter(metrics.values()))) if metrics else 0}
        for metric, vals in sorted(metrics.items()):
            rec[f"{metric}_mean"] = safe_mean(vals)
            rec[f"{metric}_std"] = safe_std(vals)
        summary_rows.append(rec)

    # More compact changed-only table across key suites.
    compact: Dict[str, Dict[str, Dict[str, float | None]]] = defaultdict(dict)
    for rec in summary_rows:
        if rec["query_kind"] != "changed":
            continue
        compact[rec["arm"]][rec["suite"]] = {
            "model_true_acc": rec.get("model_true_acc_mean"),
            "model_heuristic_agree": rec.get("model_heuristic_agree_mean"),
            "heuristic_true_acc": rec.get("heuristic_true_acc_mean"),
        }

    with (OUTDIR / "choice_rows.jsonl").open("w") as f:
        for r in choice_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    with (OUTDIR / "noninitial_heuristic_summary.json").open("w") as f:
        json.dump({"summary_rows": summary_rows, "compact_changed": compact}, f, indent=2, sort_keys=True)
    with (OUTDIR / "noninitial_heuristic_summary.csv").open("w", newline="") as f:
        fieldnames = sorted({k for r in summary_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(summary_rows)

    def fmt(x: Any) -> str:
        if x is None:
            return "nan"
        return f"{float(x):.3f}"

    lines: List[str] = []
    lines.append("# research non-initial-owner heuristic analysis")
    lines.append("")
    lines.append("This CPU-only file compares saved counterfactual model choices with the deterministic state heuristic: for changed objects, choose the participant who did **not** initially own the changed object; for unchanged objects, preserve the static owner. In ordinary research state rows the changed object always starts with the complement of the true final owner, so this heuristic is indistinguishable from true role-based state inference unless the initial owner is counterfactually varied.")
    lines.append("")
    lines.append("## Changed-object choices")
    lines.append("")
    lines.append("| arm | ordinary true acc | ordinary heuristic agree | true-final-init true acc | true-final-init heuristic agree | opposite-init true acc | opposite-init heuristic agree |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in ["heldheld_only", "aligned_matched", "inverted_matched", "neutral_matched"]:
        c = compact.get(arm, {})
        vals = [
            c.get("paired_state_conservation", {}).get("model_true_acc"),
            c.get("paired_state_conservation", {}).get("model_heuristic_agree"),
            c.get("cf_initial_true_final", {}).get("model_true_acc"),
            c.get("cf_initial_true_final", {}).get("model_heuristic_agree"),
            c.get("cf_initial_opposite_true_final", {}).get("model_true_acc"),
            c.get("cf_initial_opposite_true_final", {}).get("model_heuristic_agree"),
        ]
        lines.append(f"| {arm} | " + " | ".join(fmt(v) for v in vals) + " |")
    lines.append("")
    lines.append("## Deterministic heuristic accuracy against the true target")
    lines.append("")
    lines.append("For changed rows this baseline is 1.0 on ordinary/opposite-initial rows and 0.0 on true-final-initial rows by construction. The important comparison is whether model choices agree with the heuristic even when the heuristic is false.")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("Aligned and inverted state-trained arms agree with the non-initial-owner heuristic on the changed object at roughly the same level whether the heuristic is true (ordinary/opposite-initial rows) or false (true-final-initial rows). This explains the research/281 dissociation: state-query performance can improve without role-coordinate orientation because the corpus lets the model learn a transition-away rule tied to initial ownership. The unchanged half remains much easier, so pair-both can look high while changed-role semantics is wrong on the counterfactual rows.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- choice rows: `{OUTDIR / 'choice_rows.jsonl'}`")
    lines.append(f"- JSON summary: `{OUTDIR / 'noninitial_heuristic_summary.json'}`")
    lines.append(f"- CSV summary: `{OUTDIR / 'noninitial_heuristic_summary.csv'}`")
    ((OUTDIR.parents[4] / 'research/documents/representation_and_objectives/data/noninitial_heuristic_analysis/noninitial_heuristic_summary.md')).write_text("\n".join(lines))

    print(json.dumps({
        "status": "NONINITIAL_HEURISTIC_ANALYSIS_COMPLETE",
        "summary": str((OUTDIR.parents[4] / 'research/documents/representation_and_objectives/data/noninitial_heuristic_analysis/noninitial_heuristic_summary.md')),
        "json": str(OUTDIR / "noninitial_heuristic_summary.json"),
        "compact_changed": compact,
        "no_model_loading_training_official_eval_upload": True,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
