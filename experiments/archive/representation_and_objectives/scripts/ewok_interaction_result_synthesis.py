#!/usr/bin/env python3
"""research: synthesize full-EWoK interaction-specificity results.

CPU-only.  Reads the completed research records and turns them into compact
scientific evidence for the next construction decision.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import collections
import csv
import json
import math
import pathlib
import statistics
import time
from typing import Any, Iterable

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
RECORDS = _public_path('experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_records_fullcpu.csv')
SUMMARY_JSON = _public_path('experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_fullcpu.json')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis')
NOTE = _public_path('research/notes/representation_and_objectives/full_ewok_interaction_synthesis.md')

MODELS = ["legal40_depth_12x384_43022", "legal40_8x480_43022"]


def as_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def finite(x: float) -> bool:
    return math.isfinite(float(x))


def qstats(vals: Iterable[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if finite(float(v)))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p95": q(0.95), "max": xs[-1]}


def read_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with RECORDS.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    wrong = [r for r in rows if as_bool(r.get("saved_model_wrong_flag"))]
    stable = [r for r in rows if as_bool(r.get("conditional_reversal_failure_stable"))]
    return {
        "n": n,
        "saved_accuracy": sum(as_bool(r.get("saved_model_correct_flag")) for r in rows) / max(1, n),
        "saved_wrong": len(wrong),
        "stable_failure": len(stable),
        "stable_failure_frac_all": len(stable) / max(1, n),
        "stable_failure_frac_wrong": len(stable) / max(1, len(wrong)),
        "interaction_sum": qstats(as_float(r.get("interaction_sum")) for r in rows),
        "interaction_sum_saved_wrong": qstats(as_float(r.get("interaction_sum")) for r in wrong),
        "interaction_mean_saved_wrong": qstats(as_float(r.get("interaction_mean")) for r in wrong),
        "within_both_positive_wrong_frac": sum(as_bool(r.get("both_within_context_sum_positive")) for r in wrong) / max(1, len(wrong)),
        "local_both_actual_over_swapped_positive_wrong_frac": sum(as_bool(r.get("local_both_actual_over_swapped_positive")) for r in wrong) / max(1, len(wrong)),
        "deletion_interaction_sum_wrong": qstats(as_float(r.get("deletion_interaction_sum")) for r in wrong),
        "interaction_minus_deletion_sum_wrong": qstats(as_float(r.get("interaction_minus_deletion_sum")) for r in wrong),
        "local_min_actual_over_swapped_sum_wrong": qstats(as_float(r.get("local_min_actual_over_swapped_sum")) for r in wrong),
    }


def group_count(rows: list[dict[str, Any]], field: str, take: int = 30) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[str(r.get(field) or "")].append(r)
    out = []
    for k, rs in groups.items():
        wrong = [r for r in rs if as_bool(r.get("saved_model_wrong_flag"))]
        stable = [r for r in rs if as_bool(r.get("conditional_reversal_failure_stable"))]
        out.append({
            field: k,
            "n": len(rs),
            "accuracy": sum(as_bool(r.get("saved_model_correct_flag")) for r in rs) / max(1, len(rs)),
            "stable_failure": len(stable),
            "stable_failure_frac_all": len(stable) / max(1, len(rs)),
            "stable_failure_frac_wrong": len(stable) / max(1, len(wrong)),
            "wrong": len(wrong),
            "interaction_sum_median": qstats(as_float(r.get("interaction_sum")) for r in rs).get("median"),
        })
    return sorted(out, key=lambda x: (-x["stable_failure"], x[field]))[:take]


def pairwise_overlap(by_index: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    both = []
    either = []
    depth_only = []
    shallow_only = []
    both_wrong = []
    pattern_counts = collections.Counter()
    for idx, mm in by_index.items():
        if not all(m in mm for m in MODELS):
            continue
        d = mm[MODELS[0]]
        s = mm[MODELS[1]]
        d_fail = as_bool(d.get("conditional_reversal_failure_stable"))
        s_fail = as_bool(s.get("conditional_reversal_failure_stable"))
        if d_fail and s_fail:
            both.append(idx)
        if d_fail or s_fail:
            either.append(idx)
        if d_fail and not s_fail:
            depth_only.append(idx)
        if s_fail and not d_fail:
            shallow_only.append(idx)
        if as_bool(d.get("saved_model_wrong_flag")) and as_bool(s.get("saved_model_wrong_flag")):
            both_wrong.append(idx)
        pattern_counts[(as_bool(d.get("saved_model_correct_flag")), as_bool(s.get("saved_model_correct_flag")), d_fail, s_fail)] += 1
    return {
        "common_rows": len(by_index),
        "both_stable_failure": len(both),
        "either_stable_failure": len(either),
        "depth_only_stable_failure": len(depth_only),
        "shallow_only_stable_failure": len(shallow_only),
        "both_wrong": len(both_wrong),
        "both_stable_failure_frac_all": len(both) / max(1, len(by_index)),
        "either_stable_failure_frac_all": len(either) / max(1, len(by_index)),
        "both_stable_failure_frac_both_wrong": len(both) / max(1, len(both_wrong)),
        "first_100_both_indices": both[:100],
        "first_100_either_indices": either[:100],
        "pattern_counts_correct_fail": {str(k): v for k, v in pattern_counts.items()},
    }


def make_examples(by_index: dict[int, dict[str, dict[str, Any]]], indices: list[int], n: int = 20) -> list[dict[str, Any]]:
    examples = []
    for idx in indices[:n]:
        mm = by_index[idx]
        d = mm[MODELS[0]]
        s = mm[MODELS[1]]
        examples.append({
            "global_index": idx,
            "domain": d.get("domain"),
            "ContextType": d.get("ContextType"),
            "ContextDiff": d.get("ContextDiff"),
            "TargetDiff": d.get("TargetDiff"),
            "ConceptA": d.get("ConceptA"),
            "ConceptB": d.get("ConceptB"),
            "context_diff_c1": d.get("context_diff_c1_texts_joined"),
            "context_diff_c2": d.get("context_diff_c2_texts_joined"),
            "target_diff_words": d.get("target_diff_words"),
            "depth": {
                "correct": as_bool(d.get("saved_model_correct_flag")),
                "stable_failure": as_bool(d.get("conditional_reversal_failure_stable")),
                "interaction_sum": as_float(d.get("interaction_sum")),
                "within_c1": as_float(d.get("within_context_margin_c1_sum")),
                "within_c2": as_float(d.get("within_context_margin_c2_sum")),
                "local_min_actual_over_swapped_sum": as_float(d.get("local_min_actual_over_swapped_sum")),
            },
            "legal40_8x480": {
                "correct": as_bool(s.get("saved_model_correct_flag")),
                "stable_failure": as_bool(s.get("conditional_reversal_failure_stable")),
                "interaction_sum": as_float(s.get("interaction_sum")),
                "within_c1": as_float(s.get("within_context_margin_c1_sum")),
                "within_c2": as_float(s.get("within_context_margin_c2_sum")),
                "local_min_actual_over_swapped_sum": as_float(s.get("local_min_actual_over_swapped_sum")),
            },
        })
    return examples


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = read_records()
    source_summary = read_json(SUMMARY_JSON)
    by_model: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_index: dict[int, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in records:
        model = str(r.get("model"))
        by_model[model].append(r)
        by_index[int(r["global_index"])] [model] = r

    per_model = {m: summarize_rows(rs) for m, rs in by_model.items()}
    overlap = pairwise_overlap(by_index)
    both_indices = overlap["first_100_both_indices"]
    either_indices = overlap["first_100_either_indices"]
    examples = {
        "both_models_stable_failure_first20": make_examples(by_index, both_indices, 20),
        "either_model_stable_failure_first20": make_examples(by_index, either_indices, 20),
    }

    group_tables = {}
    for m, rs in by_model.items():
        group_tables[m] = {
            "by_domain": group_count(rs, "domain", 50),
            "by_context_type": group_count(rs, "ContextType", 20),
            "by_context_diff": group_count(rs, "ContextDiff", 40),
            "by_target_diff": group_count(rs, "TargetDiff", 40),
            "by_pattern": group_count(rs, "pattern", 40),
        }
        for name, table in group_tables[m].items():
            write_csv(OUT_DIR / f"{m}_{name}.csv", table)

    # Rows where both models have the same stable failure are the most useful for construction because they are robust to shape.
    both_full = []
    both_set = set(overlap["first_100_both_indices"])
    # Use the complete both set, not just first 100.
    for idx, mm in by_index.items():
        if all(m in mm for m in MODELS):
            if all(as_bool(mm[m].get("conditional_reversal_failure_stable")) for m in MODELS):
                d = mm[MODELS[0]]
                both_full.append({
                    "global_index": idx,
                    "domain": d.get("domain"),
                    "ContextType": d.get("ContextType"),
                    "ContextDiff": d.get("ContextDiff"),
                    "TargetDiff": d.get("TargetDiff"),
                    "ConceptA": d.get("ConceptA"),
                    "ConceptB": d.get("ConceptB"),
                    "context_diff_c1": d.get("context_diff_c1_texts_joined"),
                    "context_diff_c2": d.get("context_diff_c2_texts_joined"),
                    "target_diff_words": d.get("target_diff_words"),
                    "depth_interaction_sum": as_float(mm[MODELS[0]].get("interaction_sum")),
                    "shallow_interaction_sum": as_float(mm[MODELS[1]].get("interaction_sum")),
                    "depth_local_min_actual_over_swapped_sum": as_float(mm[MODELS[0]].get("local_min_actual_over_swapped_sum")),
                    "shallow_local_min_actual_over_swapped_sum": as_float(mm[MODELS[1]].get("local_min_actual_over_swapped_sum")),
                })
    write_csv(_public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/both_models_stable_failure_rows.csv'), both_full)

    payload = {
        "status": "EWOK_INTERACTION_SYNTHESIS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_summary_json": str(SUMMARY_JSON),
        "source_records_csv": str(RECORDS),
        "source_step092_status": source_summary.get("status"),
        "per_model": per_model,
        "pairwise_overlap": overlap,
        "group_tables": group_tables,
        "examples": examples,
        "files": {
            "summary": str(_public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/ewok_interaction_synthesis.json')),
            "both_stable_rows": str(_public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/both_models_stable_failure_rows.csv')),
            "note": str(NOTE),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (_public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/ewok_interaction_synthesis.json')).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    lines = [
        "# research — full EWoK interaction synthesis\n\n",
        "The completed full-EWoK CPU measurement confirms that the surviving EWoK weakness is large and structurally stable enough to deserve construction work, but not as a broad relation-word replacement. The measured object is conditional target reversal: after the same two target alternatives and their priors are held fixed, the model often fails to make context change reverse the preferred target.\n\n",
        "## Main numbers\n\n",
    ]
    for m in MODELS:
        s = per_model[m]
        lines.append(
            f"- {m}: saved accuracy {s['saved_accuracy']:.4f}; saved wrong {s['saved_wrong']:,}; "
            f"stable conditional-reversal failures {s['stable_failure']:,} ({s['stable_failure_frac_wrong']:.3f} of wrong); "
            f"saved-wrong interaction median {s['interaction_sum_saved_wrong']['median']:.3f}; both-within-context-positive among wrong {s['within_both_positive_wrong_frac']:.3f}.\n"
        )
    lines += [
        f"- Both legal40 models share stable failures on {overlap['both_stable_failure']:,}/{overlap['common_rows']:,} rows ({overlap['both_stable_failure_frac_all']:.3f} of all EWoK; {overlap['both_stable_failure_frac_both_wrong']:.3f} of rows both models get wrong).\n",
        f"- Either model has a stable failure on {overlap['either_stable_failure']:,}/{overlap['common_rows']:,} rows ({overlap['either_stable_failure_frac_all']:.3f}).\n\n",
        "## Structure\n\n",
        "The stable failure mass is concentrated enough to be a real construction target: the first shared-failure rows span agent-properties, material/physical/social/spatial relations, direct and indirect contexts, antonym and other context differences, and concept-swap targets. Deletion interactions are near zero, so the measured reversal is attached to the local context-difference span rather than a target-prior artifact. The local actual-vs-swapped margins are usually not both positive, which is exactly the training signal a future objective would need to repair.\n\n",
        "## What this supports\n\n",
        "A future route should construct corpus-derived two-context/two-target examples where target priors and lexical plausibility cancel by design: the same two target candidates are scored under two minimally different contexts, and only the context difference should flip the target preference. This result does not support a broad sentence-level relation-corruption objective, and it does not by itself justify H100 pretraining before a generator can build a clean non-evaluation-derived corpus analogue.\n\n",
        "## Files\n\n",
        f"- Summary JSON: `{_public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/ewok_interaction_synthesis.json')}`\n",
        f"- Shared stable-failure rows: `{_public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/both_models_stable_failure_rows.csv')}`\n",
    ]
    NOTE.write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "depth_stable_wrong": per_model[MODELS[0]]["stable_failure"],
        "depth_stable_frac_wrong": per_model[MODELS[0]]["stable_failure_frac_wrong"],
        "shallow_stable_wrong": per_model[MODELS[1]]["stable_failure"],
        "shallow_stable_frac_wrong": per_model[MODELS[1]]["stable_failure_frac_wrong"],
        "both_stable": overlap["both_stable_failure"],
        "either_stable": overlap["either_stable_failure"],
        "both_stable_frac_all": overlap["both_stable_failure_frac_all"],
        "summary": str(_public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/ewok_interaction_synthesis.json')),
        "note": str(NOTE),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
