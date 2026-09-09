#!/usr/bin/env python3
"""Refined allocation design for the next small learner comparison.

Produces a refined executable specification for the next small learner comparison.
It repairs the auxiliary common-target tasks, adds the missing current+aux and
interleaved recurrence comparisons, and records word/token/resource accounting.
No model training is performed here.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import random
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

ROOT = _public_path('.')
S = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))
import corrected_bridge_trainer as bridge  # noqa: E402
import allocation_experiment_design as v1  # noqa: E402

LABELS = _public_path('experiments/archive/functional_learning/data/selective_reviewed_labels/selective_semantic_labels.jsonl')
PLAN = _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/plan.json')
OUT = _public_path('experiments/archive/functional_learning/data/allocation_design_v2')

REFINED_AUX_TARGETS: Dict[str, List[Dict[str, str]]] = {
    "rw2s1_000216": [
        {"task_id": "s01_gas_price", "axis": "quantity", "frame": "Gas prices were rising by [ANS] in March.", "original_answer": "7.5%", "altered_answer": "5%", "replace_old": "7.5%", "replace_new": "5%"},
        {"task_id": "s02_halifax_rates", "axis": "institution_action", "frame": "The Halifax had put up its [ANS] rates.", "original_answer": "interest", "altered_answer": "mortgage", "replace_old": "interest rates", "replace_new": "mortgage rates"},
        {"task_id": "s03_first_buyers", "axis": "beneficiary", "frame": "First-time [ANS] were spared the increase.", "original_answer": "buyers", "altered_answer": "sellers", "replace_old": "first time buyers", "replace_new": "first time sellers"},
    ],
    "rw2s1_002259": [
        {"task_id": "s04_moma_city", "axis": "location", "frame": "The possible museum city was [ANS].", "original_answer": "New York", "altered_answer": "Boston", "replace_old": "New York", "replace_new": "Boston"},
    ],
    "rw2s1_017873": [
        {"task_id": "s05_ramadan_food", "axis": "object", "frame": "At Ramadan's end, Muslims got salt and pepper on a fruit [ANS].", "original_answer": "salad", "altered_answer": "cake", "replace_old": "fruit salad", "replace_new": "fruit cake"},
        {"task_id": "s06_christians_trip", "axis": "destination", "frame": "Christians get a trip to the [ANS] on gossamer wings.", "original_answer": "moon", "altered_answer": "sun", "replace_old": "trip to the moon", "replace_new": "trip to the sun"},
    ],
    "rw2s1_032596": [
        {"task_id": "s07_room_tea", "axis": "event", "frame": "The table was set for afternoon [ANS].", "original_answer": "tea", "altered_answer": "coffee", "replace_old": "afternoon tea", "replace_new": "afternoon coffee"},
        {"task_id": "s08_residence_country", "axis": "location", "frame": "Auntie and her nieces had long residence in [ANS].", "original_answer": "France", "altered_answer": "Spain", "replace_old": "long residence in France", "replace_new": "long residence in Spain"},
    ],
    "rw_024111": [
        {"task_id": "s09_money_range", "axis": "quantity_range", "frame": "The amounts ranged from thousands to tens of thousands of [ANS].", "original_answer": "dollars", "altered_answer": "pounds", "replace_old": "dollars being protected", "replace_new": "pounds being protected"},
    ],
    "rw_041960": [
        {"task_id": "s10_bro_action", "axis": "participant_action", "frame": "BRO was [ANS] along with Mot.", "original_answer": "singing", "altered_answer": "clapping", "replace_old": "BRO is singing along with Mot", "replace_new": "BRO is clapping along with Mot"},
        {"task_id": "s11_verifier_belief", "axis": "participant_state", "frame": "The verifier believed Chi was [ANS].", "original_answer": "not singing", "altered_answer": "singing", "replace_old": "verifier believes she is not", "replace_new": "verifier believes she is singing"},
    ],
    "rw_000248": [
        {"task_id": "r01_rating", "axis": "quantity", "frame": "Renee Schonfeld rated the film [ANS] out of 5 stars.", "original_answer": "3", "altered_answer": "4", "replace_old": "3 out of 5", "replace_new": "4 out of 5"},
        {"task_id": "r02_media_org", "axis": "institution", "frame": "Renee Schonfeld wrote for Common Sense [ANS].", "original_answer": "Media", "altered_answer": "Press", "replace_old": "Common Sense Media", "replace_new": "Common Sense Press"},
        {"task_id": "r03_review_quality", "axis": "evaluation", "frame": "The review said the sequel was funny and [ANS].", "original_answer": "original", "altered_answer": "forgettable", "replace_old": "funny and original", "replace_new": "funny and forgettable"},
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


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(str(text), add_special_tokens=True)["input_ids"])


def add_token_accounting(rows: List[Dict[str, Any]], tokenizer) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        rr["row_tokens_with_specials"] = token_len(tokenizer, rr["source_text"].strip() + " " + rr["view_text"].strip())
        rr["source_tokens_with_specials"] = token_len(tokenizer, rr["source_text"])
        rr["view_tokens_with_specials"] = token_len(tokenizer, rr["view_text"])
        out.append(rr)
    return out


def make_aux_tasks(rows_by_id: Dict[str, Dict[str, Any]], tokenizer) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tasks, issues = [], []
    for pid, defs in REFINED_AUX_TARGETS.items():
        r = rows_by_id[pid]
        for d in defs:
            source = r["original"]
            altered = source.replace(d["replace_old"], d["replace_new"], 1)
            found = altered != source
            if not found:
                issues.append({"task_id": d["task_id"], "pair_id": pid, "old": d["replace_old"]})
            rec = {
                "task_id": d["task_id"],
                "pair_id": pid,
                "label": r["label"],
                "axis": d["axis"],
                "frame": d["frame"],
                "original_answer": d["original_answer"],
                "altered_answer": d["altered_answer"],
                "source_text": source,
                "altered_source_text": altered,
                "source_edit": {"old": d["replace_old"], "new": d["replace_new"], "found": found},
                "current_rewrite": r["current_rewrite"],
                "compact_rewrite": r["compact_rewrite"],
                "original_answer_tokens_with_specials": token_len(tokenizer, d["original_answer"]),
                "altered_answer_tokens_with_specials": token_len(tokenizer, d["altered_answer"]),
                "note": "Score raw and per-token-normalized margins; pair_id is the independent source unit.",
            }
            tasks.append(rec)
    return tasks, issues


def subset_best(rows: List[Dict[str, Any]], target: int, key: str = "row_words", at_least: bool = False) -> Tuple[List[str], int]:
    n = len(rows)
    best_ids: List[str] = []
    best_sum = 0
    best_err = 10**9
    for mask in range(1, 1 << n):
        ids = [i for i in range(n) if mask & (1 << i)]
        s = sum(int(rows[i][key]) for i in ids)
        if at_least and s < target:
            continue
        if (not at_least) and s > target:
            continue
        err = abs(s - target)
        if err < best_err or (err == best_err and len(ids) < len(best_ids)):
            best_err = err
            best_ids = [rows[i]["pair_id"] for i in ids]
            best_sum = s
    return best_ids, best_sum


def build_aux_schedule(aux_rows: List[Dict[str, Any]], capacity: int, epochs: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Reuse v1's balanced schedule, now with repaired readout tasks.
    schedule, summary = v1.build_aux_schedule(aux_rows, capacity, epochs)
    return schedule, summary


def build_matched_recurrence_schedule(base_rows: List[Dict[str, Any]], aux_schedule: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    schedule = []
    counts = Counter()
    used = 0
    for e in aux_schedule:
        ids, s = subset_best(base_rows, int(e["aux_row_words"]), at_least=False)
        for pid in ids:
            counts[pid] += 1
        used += s
        schedule.append({"epoch": e["epoch"], "extra_base_pair_ids": ids, "extra_row_words": s, "matched_aux_words": e["aux_row_words"], "word_gap_vs_aux": s - int(e["aux_row_words"])})
    return schedule, {"used_extra_words_total": used, "target_aux_words_total": sum(int(x["aux_row_words"]) for x in aux_schedule), "word_gap_total": used - sum(int(x["aux_row_words"]) for x in aux_schedule), "extra_presentations_by_base_pair": dict(counts)}


def build_current_substitution_schedule(current_rows: List[Dict[str, Any]], aux_schedule: List[Dict[str, Any]], epoch_capacity: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    schedule = []
    kept_counts = Counter()
    drop_counts = Counter()
    total = 0
    for e in aux_schedule:
        aux_words = int(e["aux_row_words"])
        drop_ids, drop_words = subset_best(current_rows, aux_words, at_least=True)
        kept = [r["pair_id"] for r in current_rows if r["pair_id"] not in set(drop_ids)]
        kept_words = sum(int(r["row_words"]) for r in current_rows if r["pair_id"] in set(kept))
        for pid in kept:
            kept_counts[pid] += 1
        for pid in drop_ids:
            drop_counts[pid] += 1
        epoch_words = kept_words + aux_words
        total += epoch_words
        schedule.append({"epoch": e["epoch"], "kept_current_pair_ids": kept, "dropped_current_pair_ids": drop_ids, "dropped_current_words": drop_words, "aux_pair_ids": e["aux_pair_ids"], "aux_row_words": aux_words, "epoch_row_words": epoch_words, "unused_capacity_words": epoch_capacity - epoch_words})
    return schedule, {"total_row_words": total, "epoch_capacity": epoch_capacity, "capacity_total": epoch_capacity * len(aux_schedule), "unused_capacity_total": epoch_capacity * len(aux_schedule) - total, "kept_presentations_by_pair": dict(kept_counts), "dropped_presentations_by_pair": dict(drop_counts)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(LABELS)
    rows_by_id = {r["pair_id"]: r for r in rows}
    plan57 = json.loads(PLAN.read_text(encoding="utf-8"))
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)

    base_ids = plan57["common_pair_ids"]
    aux_ids = list(REFINED_AUX_TARGETS)
    base_current = add_token_accounting([v1.row_record(rows_by_id[pid], "current", "base_faithful") for pid in base_ids], tokenizer)
    base_compact = add_token_accounting([v1.row_record(rows_by_id[pid], "compact", "base_faithful") for pid in base_ids], tokenizer)
    aux_compact = add_token_accounting([v1.row_record(rows_by_id[pid], "compact", "additional_source_supported") for pid in aux_ids], tokenizer)
    aux_tasks, task_issues = make_aux_tasks(rows_by_id, tokenizer)

    current_words = sum(r["row_words"] for r in base_current)
    compact_words = sum(r["row_words"] for r in base_compact)
    saving = current_words - compact_words
    aux_schedule, aux_summary = build_aux_schedule(aux_compact, saving, 80)
    recurrence_schedule, recurrence_summary = build_matched_recurrence_schedule(base_compact, aux_schedule)
    current_subst_schedule, current_subst_summary = build_current_substitution_schedule(base_current, aux_schedule, current_words)

    arms = [
        {"arm": "current_base80", "role": "inherited base reference", "base_view": "current", "epochs": 80, "row_words": current_words * 80, "row_tokens_with_specials": sum(r["row_tokens_with_specials"] for r in base_current) * 80, "base_presentations": 17 * 80, "aux_presentations": 0},
        {"arm": "compact_base80_unspent", "role": "preservation at lower processed-word count", "base_view": "compact", "epochs": 80, "row_words": compact_words * 80, "row_tokens_with_specials": sum(r["row_tokens_with_specials"] for r in base_compact) * 80, "base_presentations": 17 * 80, "aux_presentations": 0},
        {"arm": "compact_interleaved_recurrence", "role": "saved-budget spent on old source IDs at same epochs as aux rows", "base_view": "compact", "epochs": 80, "row_words": compact_words * 80 + recurrence_summary["used_extra_words_total"], "base_presentations": 17 * 80 + sum(recurrence_summary["extra_presentations_by_base_pair"].values()), "aux_presentations": 0, "schedule": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/compact_interleaved_recurrence_schedule.jsonl'))},
        {"arm": "compact_aux_support", "role": "saved-budget spent on additional reviewed source-supported rows", "base_view": "compact", "epochs": 80, "row_words": compact_words * 80 + aux_summary["used_aux_words_total"], "base_presentations": 17 * 80, "aux_presentations": sum(aux_summary["aux_presentations_by_pair"].values()), "schedule": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/aux_support_schedule_epoch80.jsonl'))},
        {"arm": "current_aux_substitution", "role": "current-view base with base exposures removed to fit the same aux rows", "base_view": "current", "epochs": 80, "row_words": current_subst_summary["total_row_words"], "base_presentations": sum(current_subst_summary["kept_presentations_by_pair"].values()), "aux_presentations": sum(aux_summary["aux_presentations_by_pair"].values()), "schedule": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/current_aux_substitution_schedule.jsonl'))},
    ]

    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/base_current_rows.jsonl'), base_current)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/base_compact_rows.jsonl'), base_compact)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/aux_compact_rows.jsonl'), aux_compact)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/aux_support_tasks_repaired.jsonl'), aux_tasks)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/aux_support_schedule_epoch80.jsonl'), aux_schedule)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/compact_interleaved_recurrence_schedule.jsonl'), recurrence_schedule)
    write_jsonl(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/current_aux_substitution_schedule.jsonl'), current_subst_schedule)

    summary = {
        "status": "ALLOCATION_DESIGN_V2",
        "purpose": "Execute-ready design for testing whether compact faithful views preserve base learning while allowing the same processed-word budget to teach additional reviewed source-supported rows.",
        "human_stage_context": "Stage III needs principle-guided improvement of Qiushi-BabyLM-36M-Strict-Small-v4, not a standalone compaction score. This is a bounded learner test before any legal-stream v5 candidate.",
        "inputs": {"semantic_labels": rel(LABELS), "plan": rel(PLAN), "independent_review_review": "data/external/independent_review01_verifier1_integration.md"},
        "word_accounting_per_base_epoch": {"current_base_row_words": current_words, "compact_base_row_words": compact_words, "saved_row_words": saving, "aux_all_once_row_words": sum(r["row_words"] for r in aux_compact)},
        "token_accounting_per_base_epoch": {"current_base_row_tokens_with_specials": sum(r["row_tokens_with_specials"] for r in base_current), "compact_base_row_tokens_with_specials": sum(r["row_tokens_with_specials"] for r in base_compact), "saved_row_tokens_with_specials": sum(r["row_tokens_with_specials"] for r in base_current) - sum(r["row_tokens_with_specials"] for r in base_compact), "aux_all_once_row_tokens_with_specials": sum(r["row_tokens_with_specials"] for r in aux_compact)},
        "base_rows": len(base_current),
        "additional_rows": len(aux_compact),
        "additional_label_counts": dict(Counter(r["label"] for r in aux_compact)),
        "additional_common_tasks": len(aux_tasks),
        "additional_task_edit_issues": task_issues,
        "aux_schedule_summary": aux_summary,
        "recurrence_schedule_summary": recurrence_summary,
        "current_substitution_schedule_summary": current_subst_summary,
        "arms": arms,
        "primary_metrics_for_execute": {
            "common_target_decomposition": "For each source item compute g=(delta original-source + delta altered-source)/2, b=(delta original-source - delta altered-source)/2, and p=delta no-source prior. Treat g as the source-following movement, b/p as prior or practiced-answer movement.",
            "base_preservation": "research 25-task bank plus research surface NLL on current and compact views; require compact_aux not to erase the base preservation seen in compact_unspent.",
            "additional_support_learning": "Repaired aux tasks and aux surface NLL; judge by source directions and pair-level aggregation, not only compact surface reconstruction.",
            "allocation_contrast": "compact_aux_support must be compared against compact_interleaved_recurrence and current_aux_substitution under the recorded processed-word, token, presentation, and step counts.",
            "broad_preservation": "Use parent KL or a small Cheap7/common broad readout only as early preservation evidence; this experiment is not a BabyLM endpoint.",
        },
        "interpretation_patterns": {
            "supports_next_practical_test": "compact_aux preserves base common-target and surface behavior near compact_unspent/current while improving additional-support g across multiple source IDs more than interleaved recurrence and current_aux_substitution, without large prior-only movement.",
            "does_not_support_next_practical_test": "improvement appears only on trained compact surfaces, comes mainly from no-source/original-answer shifts, is driven by one row, or aux learning damages base common targets or broad preservation signals.",
            "remaining_before_v5": "A positive small result would justify a larger reviewed-policy legal-stream candidate and official-compatible evaluation; it would not itself be v5.",
        },
        "artifacts": {"base_current_rows": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/base_current_rows.jsonl')), "base_compact_rows": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/base_compact_rows.jsonl')), "aux_compact_rows": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/aux_compact_rows.jsonl')), "aux_tasks": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/aux_support_tasks_repaired.jsonl')), "aux_schedule": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/aux_support_schedule_epoch80.jsonl')), "recurrence_schedule": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/compact_interleaved_recurrence_schedule.jsonl')), "current_substitution_schedule": rel(_public_path('experiments/archive/functional_learning/data/allocation_design_v2/current_aux_substitution_schedule.jsonl'))},
    }
    out_path = _public_path('experiments/archive/functional_learning/data/allocation_design_v2/allocation_design_v2.json')
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "design": rel(out_path), "base_saved_words_per_epoch": saving, "arms": [a["arm"] for a in arms], "aux_tasks": len(aux_tasks), "task_issues": len(task_issues)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
