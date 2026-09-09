#!/usr/bin/env python3
"""research: design a bounded compact-savings allocation experiment.

No model training is performed here.  The script materializes the resource arithmetic and
the concrete row sets for the next Execute step: inherited current-view budget versus
compact recurrence versus compact + additional source-supported experience, using only
individually reviewed rows from research.  It makes explicit charged words, epochs,
optimizer presentations, and what each readout would establish.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
S = _public_path('experiments/archive/functional_learning')
LABELS = _public_path('experiments/archive/functional_learning/data/selective_reviewed_labels/selective_semantic_labels.jsonl')
PLAN = _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/plan.json')
DIR = _public_path('experiments/archive/functional_learning/data/allocation_design')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")
STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "into", "about", "over", "under",
    "after", "before", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "may", "might", "must", "should", "shall",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "not", "no", "so", "there", "here", "what", "which", "who", "when", "where", "why", "how",
    "current", "state", "information", "according", "context", "also", "onto", "between",
}

# Hand-authored common targets for the reviewed auxiliary rows that can serve as
# additional source-supported experience.  All are source-supported; omitted payload in
# supported_partial_view rows is deliberately not targeted.
AUX_TARGETS: Dict[str, List[Dict[str, str]]] = {
    "rw2s1_000216": [
        {"task_id": "s01_gas_price", "axis": "quantity", "frame": "Gas prices were rising by [ANS] in March.", "original_answer": "7.5%", "altered_answer": "5%", "replace_old": "7.5%", "replace_new": "5%"},
        {"task_id": "s02_halifax_rates", "axis": "institution_action", "frame": "The Halifax had put up its interest [ANS].", "original_answer": "rates", "altered_answer": "fees", "replace_old": "interest rates", "replace_new": "interest fees"},
        {"task_id": "s03_first_buyers", "axis": "beneficiary", "frame": "The increase spared first-time [ANS].", "original_answer": "buyers", "altered_answer": "sellers", "replace_old": "first time buyers", "replace_new": "first time sellers"},
    ],
    "rw2s1_002259": [
        {"task_id": "s04_moma_city", "axis": "location", "frame": "The piece could go in the Museum of Modern Art in [ANS].", "original_answer": "New York", "altered_answer": "Boston", "replace_old": "New York", "replace_new": "Boston"},
    ],
    "rw2s1_017873": [
        {"task_id": "s05_ramadan_food", "axis": "object", "frame": "At Ramadan's end, Muslims got salt and pepper on a fruit [ANS].", "original_answer": "salad", "altered_answer": "cake", "replace_old": "fruit salad", "replace_new": "fruit cake"},
        {"task_id": "s06_christians_trip", "axis": "destination", "frame": "Christians get a trip to the [ANS] on gossamer wings.", "original_answer": "moon", "altered_answer": "sun", "replace_old": "trip to the moon", "replace_new": "trip to the sun"},
    ],
    "rw2s1_032596": [
        {"task_id": "s07_room_tea", "axis": "event", "frame": "The table was set for afternoon [ANS].", "original_answer": "tea", "altered_answer": "dinner", "replace_old": "afternoon tea", "replace_new": "afternoon dinner"},
        {"task_id": "s08_residence_country", "axis": "location", "frame": "Auntie and her nieces had long residence in [ANS].", "original_answer": "France", "altered_answer": "Spain", "replace_old": "long residence in France", "replace_new": "long residence in Spain"},
    ],
    "rw_024111": [
        {"task_id": "s09_protected_people", "axis": "quantity_group", "frame": "The safeguards protected [ANS] of people.", "original_answer": "millions", "altered_answer": "hundreds", "replace_old": "million of people", "replace_new": "hundreds of people"},
        {"task_id": "s10_money_range", "axis": "quantity_range", "frame": "The amounts ranged from thousands to tens of thousands of [ANS].", "original_answer": "dollars", "altered_answer": "pounds", "replace_old": "dollars being protected", "replace_new": "pounds being protected"},
    ],
    "rw_041960": [
        {"task_id": "s11_bro_singing", "axis": "participant", "frame": "BRO was singing along with [ANS].", "original_answer": "Mot", "altered_answer": "Chi", "replace_old": "BRO is singing along with Mot", "replace_new": "BRO is singing along with Chi"},
        {"task_id": "s12_verifier_belief", "axis": "participant_state", "frame": "The verifier believed Chi was [ANS] singing.", "original_answer": "not", "altered_answer": "also", "replace_old": "verifier believes she is not", "replace_new": "verifier believes she is also"},
    ],
    "rw_000248": [
        {"task_id": "r01_rating", "axis": "quantity", "frame": "Renee Schonfeld rated the film [ANS] out of 5 stars.", "original_answer": "3", "altered_answer": "4", "replace_old": "3 out of 5", "replace_new": "4 out of 5"},
        {"task_id": "r02_company", "axis": "creator_attribution", "frame": "The review credited Spike Brandt, Tony Cerone, and [ANS].", "original_answer": "company", "altered_answer": "director", "replace_old": "and company", "replace_new": "and director"},
        {"task_id": "r03_media_org", "axis": "institution", "frame": "Renee Schonfeld wrote for Common Sense [ANS].", "original_answer": "Media", "altered_answer": "Press", "replace_old": "Common Sense Media", "replace_new": "Common Sense Press"},
    ],
}


def rel(path: pathlib.Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def wc(text: str) -> int:
    return len((text or "").strip().split())


def content_words(text: str) -> List[str]:
    out = []
    for m in WORD_RE.finditer(text or ""):
        w = m.group(0)
        wl = w.lower().strip("'\u2019")
        if len(wl) < 4 or wl in STOP or wl.isdigit():
            continue
        out.append(w)
    return out


def row_record(r: Dict[str, Any], view_kind: str, role: str) -> Dict[str, Any]:
    view = r["current_rewrite"] if view_kind == "current" else r["compact_rewrite"]
    return {
        "pair_id": r["pair_id"],
        "role": role,
        "label": r["label"],
        "source_corpus": r.get("source_corpus", r.get("source", "")),
        "source_text": r["original"],
        "current_rewrite": r["current_rewrite"],
        "compact_rewrite": r["compact_rewrite"],
        "view_kind": view_kind,
        "view_text": view,
        "source_words": wc(r["original"]),
        "view_words": wc(view),
        "row_words": wc(r["original"]) + wc(view),
        "current_words": wc(r["current_rewrite"]),
        "compact_words": wc(r["compact_rewrite"]),
        "content_words": content_words(view),
    }


def make_aux_tasks(rows_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks = []
    for pid, defs in AUX_TARGETS.items():
        r = rows_by_id[pid]
        for d in defs:
            source = r["original"]
            old = d["replace_old"]
            new = d["replace_new"]
            altered = source.replace(old, new, 1)
            tasks.append({
                "task_id": d["task_id"],
                "pair_id": pid,
                "label": r["label"],
                "axis": d["axis"],
                "frame": d["frame"],
                "original_answer": d["original_answer"],
                "altered_answer": d["altered_answer"],
                "source_text": source,
                "altered_source_text": altered,
                "source_edit": {"old": old, "new": new, "found": altered != source},
                "current_rewrite": r["current_rewrite"],
                "compact_rewrite": r["compact_rewrite"],
            })
    return tasks


def build_aux_schedule(aux_rows: List[Dict[str, Any]], per_epoch_capacity: int, epochs: int = 80) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sizes = [int(r["row_words"]) for r in aux_rows]
    names = [r["pair_id"] for r in aux_rows]
    # All feasible nonempty subsets under the per-epoch capacity.
    subsets = []
    for mask in range(1, 1 << len(aux_rows)):
        idx = [i for i in range(len(aux_rows)) if mask & (1 << i)]
        s = sum(sizes[i] for i in idx)
        if s <= per_epoch_capacity:
            subsets.append((idx, s))
    counts = [0] * len(aux_rows)
    schedule = []
    rng = random.Random(59059)
    for epoch in range(epochs):
        best = None
        best_score = -1e18
        for idx, s in subsets:
            projected = counts[:]
            for i in idx:
                projected[i] += 1
            # Favor using the budget, but not by starving the longest rows forever.  The
            # row-word-weighted underexposure term gives long rows comparable total word exposure.
            target_presentations = 40
            under = sum(max(0, target_presentations - counts[i]) * sizes[i] for i in idx)
            imbalance = max(projected) - min(projected)
            over = sum(max(0, projected[i] - 44) for i in idx)
            score = under + 0.35 * s - 25.0 * imbalance - 10000.0 * over + rng.random() * 1e-3
            if score > best_score:
                best_score = score
                best = (idx, s)
        assert best is not None
        idx, s = best
        for i in idx:
            counts[i] += 1
        schedule.append({
            "epoch": epoch + 1,
            "aux_pair_ids": [names[i] for i in idx],
            "aux_row_words": s,
            "unused_capacity_words": per_epoch_capacity - s,
        })
    total_words_by_pair = {names[i]: counts[i] * sizes[i] for i in range(len(aux_rows))}
    summary = {
        "epochs": epochs,
        "per_epoch_capacity": per_epoch_capacity,
        "capacity_total": per_epoch_capacity * epochs,
        "used_aux_words_total": sum(x["aux_row_words"] for x in schedule),
        "unused_capacity_total": sum(x["unused_capacity_words"] for x in schedule),
        "aux_presentations_by_pair": {names[i]: counts[i] for i in range(len(aux_rows))},
        "aux_words_by_pair": total_words_by_pair,
        "schedule_pattern_counts": {" + ".join(k): v for k, v in Counter(tuple(x["aux_pair_ids"]) for x in schedule).items()},
    }
    return schedule, summary


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(LABELS)
    rows_by_id = {r["pair_id"]: r for r in rows}
    plan57 = json.loads(PLAN.read_text(encoding="utf-8"))
    faithful_ids = plan57["common_pair_ids"]
    faithful = [rows_by_id[pid] for pid in faithful_ids]
    aux_ids = list(AUX_TARGETS)
    aux = [rows_by_id[pid] for pid in aux_ids]

    base_current = [row_record(r, "current", "base_faithful") for r in faithful]
    base_compact = [row_record(r, "compact", "base_faithful") for r in faithful]
    aux_compact = [row_record(r, "compact", "additional_source_supported") for r in aux]

    per_epoch_saving = sum(r["row_words"] for r in base_current) - sum(r["row_words"] for r in base_compact)
    base_current_words = sum(r["row_words"] for r in base_current)
    base_compact_words = sum(r["row_words"] for r in base_compact)
    aux_total_once = sum(r["row_words"] for r in aux_compact)
    schedule, schedule_summary = build_aux_schedule(aux_compact, per_epoch_saving, epochs=80)

    arms = [
        {
            "arm": "current_base80",
            "description": "Inherited-view second views for the 17 faithful base rows, 80 epochs.",
            "base_view": "current",
            "base_epochs": 80,
            "base_row_words_per_epoch": base_current_words,
            "aux_policy": "none",
            "charged_words": base_current_words * 80,
            "optimizer_presentations": {"base_rows": 17 * 80, "aux_rows": 0},
        },
        {
            "arm": "compact_base80_unspent",
            "description": "Compact faithful second views for the same 17 rows, 80 epochs, leaving the saved word budget unused.",
            "base_view": "compact",
            "base_epochs": 80,
            "base_row_words_per_epoch": base_compact_words,
            "aux_policy": "unspent",
            "charged_words": base_compact_words * 80,
            "optimizer_presentations": {"base_rows": 17 * 80, "aux_rows": 0},
        },
        {
            "arm": "compact_base80_recurrence92",
            "description": "Compact faithful second views continued to 92 epochs, spending approximately the inherited-view word difference as recurrence on the same source IDs.",
            "base_view": "compact",
            "base_epochs": 92,
            "base_row_words_per_epoch": base_compact_words,
            "aux_policy": "same_source_recurrence",
            "charged_words": base_compact_words * 92,
            "optimizer_presentations": {"base_rows": 17 * 92, "aux_rows": 0},
            "note": "Exact continuation evidence already exists at compact_continuation_control and need not be rerun unless the training harness changes.",
        },
        {
            "arm": "compact_base80_aux_support",
            "description": "Compact faithful second views for 80 epochs, plus reviewed additional source-supported compact rows scheduled within the 159 saved row words per epoch.",
            "base_view": "compact",
            "base_epochs": 80,
            "base_row_words_per_epoch": base_compact_words,
            "aux_policy": "reviewed_additional_source_supported_rows",
            "aux_rows": len(aux_compact),
            "aux_total_words_once": aux_total_once,
            "aux_schedule_summary": schedule_summary,
            "charged_words": base_compact_words * 80 + schedule_summary["used_aux_words_total"],
            "optimizer_presentations": {"base_rows": 17 * 80, "aux_rows": sum(schedule_summary["aux_presentations_by_pair"].values())},
        },
    ]

    base_common_targets = [
        "Use the research 25-item common-target bank to check preservation on trained-content and held-source items; analyze source_original, source_altered, and no_source separately.",
        "Do not average no_source with contextual source directions as evidence use; no_source is a prior movement readout.",
    ]
    aux_tasks = make_aux_tasks(rows_by_id)
    missing_edits = [t for t in aux_tasks if not t["source_edit"]["found"]]

    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design/base_current_rows.jsonl'), base_current)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design/base_compact_rows.jsonl'), base_compact)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design/aux_compact_rows.jsonl'), aux_compact)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design/aux_support_tasks.jsonl'), aux_tasks)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design/aux_support_schedule_epoch80.jsonl'), schedule)
    summary = {
        "status": "ALLOCATION_EXPERIMENT_DESIGN",
        "purpose": "Define the next learner comparison that tests whether faithful shortening can preserve base learning while allocating saved words to additional reviewed source-supported experience.",
        "inputs": {
            "semantic_labels": rel(LABELS),
            "plan": rel(PLAN),
            "directional_summary": "experiments/archive/functional_learning/data/route_synthesis/common_target_directional_summary.json",
        },
        "base_faithful_rows": len(base_compact),
        "additional_reviewed_rows": len(aux_compact),
        "additional_label_counts": dict(Counter(r["label"] for r in aux_compact)),
        "word_arithmetic_per_epoch": {
            "base_current_row_words": base_current_words,
            "base_compact_row_words": base_compact_words,
            "saved_row_words": per_epoch_saving,
            "base_current_view_words": sum(r["view_words"] for r in base_current),
            "base_compact_view_words": sum(r["view_words"] for r in base_compact),
            "saved_view_words": sum(r["current_words"] - r["compact_words"] for r in base_compact),
            "aux_all_once_row_words": aux_total_once,
        },
        "arms": arms,
        "aux_schedule_summary": schedule_summary,
        "readouts": {
            "base_common_preservation": base_common_targets,
            "base_surface": "Reuse research-style current/compact with-source and view-only NLL to detect loss of the base second-view signal.",
            "aux_surface": "Score additional reviewed rows with and without their source to see whether the reinvested examples are learned.",
            "aux_common_targets": f"{len(aux_tasks)} hand-authored common cloze tasks over the additional rows, each with original-source, altered-source, and no-source conditions.",
            "broad_behavior": "For this bounded subset, use parent KL / no-source prior shifts / optional Cheap7 quick score only as preservation signals; it is not a BabyLM endpoint.",
        },
        "aux_task_edit_issues": missing_edits,
        "interpretation": {
            "current_vs_compact_unspent": "Tests preservation of second-view learning under shorter faithful expression at fewer charged words and equal update count.",
            "compact_recurrence_vs_aux_support": "Separates spending saved budget as same-source recurrence from spending it on additional source-supported content. Extra examples, charged words, presentations, and compute must be recorded separately.",
            "history_warning": "The auxiliary rows come from the same inherited Qwen/official ecosystem and may have been seen by coherent86; call them additional reviewed source-supported rows, not definitely novel experience.",
            "promotion_rule": "Only if compact+aux preserves base common/source behavior while learning the auxiliary support better than recurrence should it justify a realistic legal-stream candidate.",
        },
        "artifacts": {
            "base_current_rows": rel(_public_path('experiments/archive/functional_learning/data/allocation_design/base_current_rows.jsonl')),
            "base_compact_rows": rel(_public_path('experiments/archive/functional_learning/data/allocation_design/base_compact_rows.jsonl')),
            "aux_compact_rows": rel(_public_path('experiments/archive/functional_learning/data/allocation_design/aux_compact_rows.jsonl')),
            "aux_support_tasks": rel(_public_path('experiments/archive/functional_learning/data/allocation_design/aux_support_tasks.jsonl')),
            "aux_schedule": rel(_public_path('experiments/archive/functional_learning/data/allocation_design/aux_support_schedule_epoch80.jsonl')),
        },
    }
    (_public_path('experiments/archive/functional_learning/data/allocation_design/allocation_experiment_design.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "design": rel(_public_path('experiments/archive/functional_learning/data/allocation_design/allocation_experiment_design.json')), "base_saved_words_per_epoch": per_epoch_saving, "aux_used_words_total": schedule_summary["used_aux_words_total"], "aux_tasks": len(aux_tasks), "missing_task_edits": len(missing_edits)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
