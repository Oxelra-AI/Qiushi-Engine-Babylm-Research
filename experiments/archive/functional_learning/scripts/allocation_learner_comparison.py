#!/usr/bin/env python3
"""research: balanced compact-allocation learner comparison.

This executes the research allocation idea after repairing the deterministic packing
problem. The scientific question is whether faithful compact second views preserve the
base aligned-source signal while freeing enough processed-word budget for additional
reviewed source-supported rows, compared with spending the same saved budget on extra
recurrence or adding the same support to inherited current views by removing base
presentations.

The script keeps the auxiliary row schedule from research, but builds balanced,
seeded recurrence and current-substitution schedules so source identity is not starved
by exact word packing. It trains small private-adapter continuations from the trusted
coherent86 parent and scores each state on common source-grounded contrasts and masked
surface reconstruction.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))
import corrected_bridge_trainer as bridge  # noqa: E402
import concentrated_compact_learning_test as s57  # noqa: E402
import common_target_probe as s58  # noqa: E402

DESIGN_DIR = _public_path('experiments/archive/functional_learning/data/allocation_design_v2')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/allocation_balanced_comparison')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_seed(*parts: Any) -> int:
    txt = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return int.from_bytes(hashlib.blake2b(txt.encode("utf-8"), digest_size=8).digest(), "little") & ((1 << 63) - 1)


def set_global_seed(seed: int) -> None:
    random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed) % (2**63 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) % (2**63 - 1))


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def finite_mean(xs: Iterable[Any]) -> Optional[float]:
    vals: List[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def finite_median(xs: Iterable[Any]) -> Optional[float]:
    vals: List[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    return statistics.median(vals) if vals else None


def row_to_record(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_id": r["pair_id"],
        "label": r.get("label"),
        "source_corpus": r.get("source_corpus", ""),
        "original": r.get("source_text", r.get("original", "")),
        "source_text": r.get("source_text", r.get("original", "")),
        "current_rewrite": r.get("current_rewrite", ""),
        "compact_rewrite": r.get("compact_rewrite", ""),
        "role": r.get("role", ""),
    }


def build_train_examples(row_records: List[Dict[str, Any]], tokenizer, max_length: int, view_kind: str,
                         max_train_targets_per_row: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    examples: Dict[str, Dict[str, Any]] = {}
    skips = Counter()
    for r in row_records:
        source = str(r.get("source_text", r.get("original", ""))).strip()
        current = str(r.get("current_rewrite", "")).strip()
        compact = str(r.get("compact_rewrite", "")).strip()
        view = current if view_kind == "current" else compact
        if not source or not view or view == "KEEP_CURRENT":
            skips["empty"] += 1
            continue
        prefix = source + " "
        text = prefix + view
        view_offset = len(prefix)
        enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=int(max_length), return_offsets_mapping=True)
        ids = [int(x) for x in enc["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        train_groups = []
        spans = s57.content_word_spans(view)
        for a, b, word in spans:
            pos = s57.locate_positions(offsets, view_offset + a, view_offset + b)
            if pos:
                train_groups.append({"word": word, "view_start": a, "view_end": b, "positions": pos})
        if not train_groups:
            skips["no_targets"] += 1
            continue
        if len(train_groups) > int(max_train_targets_per_row):
            rng = random.Random(stable_seed("train-target-subset", r["pair_id"], view_kind))
            train_groups = rng.sample(train_groups, int(max_train_targets_per_row))
            train_groups.sort(key=lambda x: x["view_start"])
        ex = {
            "pair_id": r["pair_id"],
            "label": r.get("label"),
            "role": r.get("role", ""),
            "source_text": source,
            "current_rewrite": current,
            "compact_rewrite": compact,
            "view_kind": view_kind,
            "view_text": view,
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "train_groups": train_groups,
            "source_words": s57.wc(source),
            "view_words": s57.wc(view),
            "row_words": s57.wc(source) + s57.wc(view),
        }
        examples[r["pair_id"]] = ex
    stats = {
        "view_kind": view_kind,
        "n_examples": len(examples),
        "skips": dict(skips),
        "source_words": sum(x["source_words"] for x in examples.values()),
        "view_words": sum(x["view_words"] for x in examples.values()),
        "row_words": sum(x["row_words"] for x in examples.values()),
        "train_groups": sum(len(x["train_groups"]) for x in examples.values()),
    }
    return examples, stats


def build_surface_tasks(row_records: List[Dict[str, Any]], tokenizer, max_length: int, max_targets_per_view: int,
                        group_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_records = [row_to_record(r) for r in row_records]
    tasks, st = s57.build_eval_tasks(base_records, tokenizer, int(max_length), int(max_targets_per_view))
    for t in tasks:
        t["row_group"] = group_name
    return tasks, st


def score_surface_tasks(model, tokenizer, tasks: List[Dict[str, Any]], device: torch.device, batch_size: int) -> List[Dict[str, Any]]:
    return s57.score_mask_tasks(model, tokenizer, tasks, device, int(batch_size))


def summarize_surface(score_rows: List[Dict[str, Any]], parent_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    parent: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    if parent_rows is not None:
        for r in parent_rows:
            key = (r.get("row_group"), r["pair_id"], r["eval_view_kind"], r["condition"], int(r["target_index"]))
            parent[key] = r
    by_group_view_cond: Dict[str, Dict[str, Any]] = {}
    for group_name in sorted(set(str(r.get("row_group", "")) for r in score_rows)):
        for view in sorted(set(str(r["eval_view_kind"]) for r in score_rows if str(r.get("row_group", "")) == group_name)):
            for cond in ["with_source", "view_only"]:
                g = [r for r in score_rows if str(r.get("row_group", "")) == group_name and r["eval_view_kind"] == view and r["condition"] == cond]
                if not g:
                    continue
                deltas = []
                if parent_rows is not None:
                    for r in g:
                        key = (r.get("row_group"), r["pair_id"], r["eval_view_kind"], r["condition"], int(r["target_index"]))
                        p = parent.get(key)
                        if p is not None:
                            deltas.append(float(r["nll"]) - float(p["nll"]))
                by_group_view_cond[f"{group_name}/{view}/{cond}"] = {
                    "n_targets": len(g),
                    "n_pairs": len(set(r["pair_id"] for r in g)),
                    "mean_nll": finite_mean(r["nll"] for r in g),
                    "median_nll": finite_median(r["nll"] for r in g),
                    "mean_delta_nll_vs_parent": finite_mean(deltas) if deltas else None,
                }
    # source help by target key
    paired: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(dict)
    meta: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in score_rows:
        key = (r.get("row_group"), r["pair_id"], r["eval_view_kind"], int(r["target_index"]))
        paired[key][r["condition"]] = float(r["nll"])
        meta[key] = {"row_group": r.get("row_group"), "pair_id": r["pair_id"], "eval_view_kind": r["eval_view_kind"]}
    help_rows = []
    for key, vals in paired.items():
        if "with_source" in vals and "view_only" in vals:
            help_rows.append({**meta[key], "source_help": vals["view_only"] - vals["with_source"]})
    by_help = {}
    for group_name in sorted(set(str(r["row_group"]) for r in help_rows)):
        for view in sorted(set(str(r["eval_view_kind"]) for r in help_rows if str(r["row_group"]) == group_name)):
            g = [r for r in help_rows if str(r["row_group"]) == group_name and str(r["eval_view_kind"]) == view]
            by_help[f"{group_name}/{view}"] = {
                "n_targets": len(g),
                "n_pairs": len(set(r["pair_id"] for r in g)),
                "mean_source_help": finite_mean(r["source_help"] for r in g),
                "median_source_help": finite_median(r["source_help"] for r in g),
            }
    # pair-level delta NLL for independence view.
    by_pair = {}
    if parent_rows is not None:
        for group_name in sorted(set(str(r.get("row_group", "")) for r in score_rows)):
            for pid in sorted(set(r["pair_id"] for r in score_rows if str(r.get("row_group", "")) == group_name)):
                gp = [r for r in score_rows if str(r.get("row_group", "")) == group_name and r["pair_id"] == pid]
                vals = []
                for r in gp:
                    key = (r.get("row_group"), r["pair_id"], r["eval_view_kind"], r["condition"], int(r["target_index"]))
                    p = parent.get(key)
                    if p is not None:
                        vals.append(float(r["nll"]) - float(p["nll"]))
                by_pair[f"{group_name}/{pid}"] = {"n_scores": len(vals), "mean_delta_nll_vs_parent": finite_mean(vals)}
    return {"by_group_view_condition": by_group_view_cond, "by_group_view_source_help": by_help, "by_pair": by_pair}


def build_common_candidate_records(tokenizer, base_labels_path: pathlib.Path, aux_tasks_path: pathlib.Path,
                                   max_length: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_bank, base_summary = s58.materialize_bank(base_labels_path)
    for item in base_bank:
        item["task_group"] = "base_common"
    aux_items = load_jsonl(aux_tasks_path)
    for item in aux_items:
        item["task_group"] = "aux_common"
        item.setdefault("split", "aux_support")
        item.setdefault("semantic_label", item.get("label"))
    all_items = base_bank + aux_items
    records: List[Dict[str, Any]] = []
    skips = Counter()
    for item in all_items:
        for condition in ["source_original", "source_altered", "no_source"]:
            for role, cand in [("original_answer", item["original_answer"]), ("altered_answer", item["altered_answer"] )]:
                rec = s58.build_candidate_record(tokenizer, item, condition, role, str(cand), int(max_length))
                if rec is None:
                    skips[f"{item.get('task_group')}_{condition}_{role}_no_position"] += 1
                    continue
                rec["task_group"] = item.get("task_group", "")
                rec["source_edit"] = item.get("source_edit", {})
                records.append(rec)
    summary = {
        "base_bank_summary": base_summary,
        "n_aux_tasks": len(aux_items),
        "n_candidate_records": len(records),
        "skips": dict(skips),
        "tasks_by_group": dict(Counter(r["task_group"] for r in records if r["candidate_role"] == "original_answer" and r["condition"] == "source_original")),
    }
    return records, summary


def score_common_records(model, tokenizer, records: List[Dict[str, Any]], device: torch.device, batch_size: int) -> List[Dict[str, Any]]:
    return s58.score_records(model, tokenizer, records, device, int(batch_size))


def summarize_common(score_rows: List[Dict[str, Any]], parent_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    # Candidate NLL map by task/condition/candidate.
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in score_rows:
        by_key[(r["task_id"], r["condition"], r["candidate_role"])] = r
    task_meta: Dict[str, Dict[str, Any]] = {}
    for r in score_rows:
        task_meta.setdefault(r["task_id"], {k: r.get(k) for k in ["task_id", "pair_id", "split", "axis", "semantic_label", "frame", "task_group"]})
    margins: List[Dict[str, Any]] = []
    for tid, meta in task_meta.items():
        for condition in ["source_original", "source_altered", "no_source"]:
            orig = by_key.get((tid, condition, "original_answer"))
            alt = by_key.get((tid, condition, "altered_answer"))
            if orig is None or alt is None:
                continue
            if condition == "source_altered":
                expected_margin = float(orig["nll"]) - float(alt["nll"])
                expected_role = "altered_answer"
            else:
                expected_margin = float(alt["nll"]) - float(orig["nll"])
                expected_role = "original_answer"
            margins.append({
                **meta,
                "condition": condition,
                "expected_role": expected_role,
                "original_nll": float(orig["nll"]),
                "altered_nll": float(alt["nll"]),
                "expected_margin": expected_margin,
                "expected_correct": bool(expected_margin > 0.0),
                "orig_candidate_tokens": int(orig.get("n_token_scores", 0)),
                "alt_candidate_tokens": int(alt.get("n_token_scores", 0)),
            })
    m_by_tc = {(m["task_id"], m["condition"]): m for m in margins}
    source_rows: List[Dict[str, Any]] = []
    for tid, meta in task_meta.items():
        mo = m_by_tc.get((tid, "source_original"))
        ma = m_by_tc.get((tid, "source_altered"))
        mn = m_by_tc.get((tid, "no_source"))
        if mo and ma:
            source_rows.append({
                **meta,
                "source_original_margin": mo["expected_margin"],
                "source_altered_margin": ma["expected_margin"],
                "no_source_original_margin": mn["expected_margin"] if mn else None,
                "source_follow_swing": mo["expected_margin"] + ma["expected_margin"],
                "both_source_conditions_correct": bool(mo["expected_margin"] > 0.0 and ma["expected_margin"] > 0.0),
            })
    parent_margins: Dict[Tuple[str, str], float] = {}
    if parent_rows is not None:
        parent_summary = summarize_common(parent_rows, None)
        for m in parent_summary["margins"]:
            parent_margins[(m["task_id"], m["condition"])] = float(m["expected_margin"])
    delta_rows: List[Dict[str, Any]] = []
    if parent_rows is not None:
        for tid, meta in task_meta.items():
            mo = m_by_tc.get((tid, "source_original"))
            ma = m_by_tc.get((tid, "source_altered"))
            mn = m_by_tc.get((tid, "no_source"))
            if not (mo and ma):
                continue
            po = parent_margins.get((tid, "source_original"))
            pa = parent_margins.get((tid, "source_altered"))
            pn = parent_margins.get((tid, "no_source"))
            if po is None or pa is None:
                continue
            delta_o = float(mo["expected_margin"]) - po
            delta_a = float(ma["expected_margin"]) - pa
            delta_n = (float(mn["expected_margin"]) - pn) if (mn is not None and pn is not None) else None
            delta_rows.append({
                **meta,
                "delta_source_original": delta_o,
                "delta_source_altered": delta_a,
                "g_source_follow": 0.5 * (delta_o + delta_a),
                "b_direction_imbalance": 0.5 * (delta_o - delta_a),
                "p_no_source_original_prior": delta_n,
            })

    def group_source(rows: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for val in sorted(set(str(r.get(field)) for r in rows)):
            g = [r for r in rows if str(r.get(field)) == val]
            out[val] = {
                "n_tasks": len(g),
                "n_pairs": len(set(r["pair_id"] for r in g)),
                "both_correct": sum(1 for r in g if r.get("both_source_conditions_correct")),
                "mean_swing": finite_mean(r["source_follow_swing"] for r in g),
                "mean_source_original_margin": finite_mean(r["source_original_margin"] for r in g),
                "mean_source_altered_margin": finite_mean(r["source_altered_margin"] for r in g),
                "mean_no_source_original_margin": finite_mean(r.get("no_source_original_margin") for r in g),
            }
        return out

    def group_delta(rows: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for val in sorted(set(str(r.get(field)) for r in rows)):
            g = [r for r in rows if str(r.get(field)) == val]
            out[val] = {
                "n_tasks": len(g),
                "n_pairs": len(set(r["pair_id"] for r in g)),
                "mean_g_source_follow": finite_mean(r["g_source_follow"] for r in g),
                "median_g_source_follow": finite_median(r["g_source_follow"] for r in g),
                "mean_b_direction_imbalance": finite_mean(r["b_direction_imbalance"] for r in g),
                "mean_p_no_source_original_prior": finite_mean(r.get("p_no_source_original_prior") for r in g),
                "mean_delta_source_original": finite_mean(r["delta_source_original"] for r in g),
                "mean_delta_source_altered": finite_mean(r["delta_source_altered"] for r in g),
            }
        return out

    by_condition = {}
    for cond in ["source_original", "source_altered", "no_source"]:
        g = [m for m in margins if m["condition"] == cond]
        by_condition[cond] = {
            "n": len(g),
            "success": sum(1 for m in g if m["expected_margin"] > 0.0),
            "mean_expected_margin": finite_mean(m["expected_margin"] for m in g),
            "median_expected_margin": finite_median(m["expected_margin"] for m in g),
        }
    return {
        "by_condition": by_condition,
        "source_follow_by_group": group_source(source_rows, "task_group"),
        "source_follow_by_split": group_source(source_rows, "split"),
        "delta_decomposition_by_group": group_delta(delta_rows, "task_group") if parent_rows is not None else {},
        "delta_decomposition_by_split": group_delta(delta_rows, "split") if parent_rows is not None else {},
        "delta_decomposition_by_pair": group_delta(delta_rows, "pair_id") if parent_rows is not None else {},
        "margins": margins,
        "source_follow_rows": source_rows,
        "delta_rows": delta_rows,
    }


def feasible_subsets(rows: List[Dict[str, Any]], budget: int, mode: str) -> List[Tuple[List[str], int, int]]:
    """Return (ids, total_words, unused_or_overdrop) for small subset packing."""
    n = len(rows)
    out: List[Tuple[List[str], int, int]] = []
    # The rows are small (17), so exhaustive enumeration is fine for schedule construction.
    for mask in range(1, 1 << n):
        ids = [rows[i]["pair_id"] for i in range(n) if mask & (1 << i)]
        s = sum(int(rows[i]["row_words"]) for i in range(n) if mask & (1 << i))
        if mode == "under":
            if s <= budget:
                out.append((ids, s, budget - s))
        elif mode == "over":
            if s >= budget:
                out.append((ids, s, s - budget))
        else:
            raise ValueError(mode)
    return out


def balanced_choose(candidates: List[Tuple[List[str], int, int]], counts: Counter, words: Counter,
                    row_word_by_id: Dict[str, int], rng: random.Random, mode: str) -> Tuple[List[str], int, int]:
    """Choose a candidate by cumulative balance first, then reasonable word use.

    `mode='under'` is used for recurrence where unused words are acceptable but should
    not dominate source choice. `mode='over'` is used for current substitution where
    overdrop is the unused part of the inherited-view budget.
    """
    scored = []
    all_ids = list(row_word_by_id)
    for ids, total, slack in candidates:
        c2 = Counter(counts)
        w2 = Counter(words)
        for pid in ids:
            c2[pid] += 1
            w2[pid] += int(row_word_by_id[pid])
        count_vals = [c2[pid] for pid in all_ids]
        word_norm_vals = [w2[pid] / max(1, int(row_word_by_id[pid])) for pid in all_ids]
        count_range = max(count_vals) - min(count_vals)
        count_var = statistics.pvariance(count_vals) if len(count_vals) > 1 else 0.0
        norm_var = statistics.pvariance(word_norm_vals) if len(word_norm_vals) > 1 else 0.0
        # Prefer balanced exposure. Within balanced candidates, prefer less slack; this
        # keeps the resource comparison close without letting exact packing pick sources.
        jitter = rng.random() * 1e-6
        score = (count_range, count_var, norm_var, slack, -len(ids), jitter)
        if mode == "under":
            # For recurrence, also avoid pathologically tiny use after balance.
            score = (count_range, count_var, norm_var, slack, -total, jitter)
        scored.append((score, ids, total, slack))
    scored.sort(key=lambda x: x[0])
    _, ids, total, slack = scored[0]
    return ids, total, slack


def build_balanced_recurrence_schedule(base_rows: List[Dict[str, Any]], aux_schedule: List[Dict[str, Any]],
                                       seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(stable_seed("balanced-recurrence", seed))
    row_word_by_id = {r["pair_id"]: int(r["row_words"]) for r in base_rows}
    counts: Counter = Counter()
    words: Counter = Counter()
    schedule: List[Dict[str, Any]] = []
    candidate_cache: Dict[int, List[Tuple[List[str], int, int]]] = {}
    for e in aux_schedule:
        budget = int(e["aux_row_words"])
        if budget not in candidate_cache:
            # Exclude very small fills when a normal two-row fill is possible. The floor
            # is deliberately soft: fairness can still leave unused words.
            cand = feasible_subsets(base_rows, budget, "under")
            if any(total >= int(0.72 * budget) for _, total, _ in cand):
                cand = [(ids, total, slack) for ids, total, slack in cand if total >= int(0.72 * budget)]
            candidate_cache[budget] = cand
        ids, total, slack = balanced_choose(candidate_cache[budget], counts, words, row_word_by_id, rng, "under")
        for pid in ids:
            counts[pid] += 1
            words[pid] += row_word_by_id[pid]
        schedule.append({
            "epoch": int(e["epoch"]),
            "extra_base_pair_ids": ids,
            "extra_row_words": total,
            "matched_aux_words": budget,
            "unused_saved_words": slack,
        })
    vals = [counts[pid] for pid in row_word_by_id]
    summary = {
        "seed": int(seed),
        "used_extra_words_total": sum(int(x["extra_row_words"]) for x in schedule),
        "target_aux_words_total": sum(int(x["matched_aux_words"]) for x in schedule),
        "unused_saved_words_total": sum(int(x["unused_saved_words"]) for x in schedule),
        "extra_presentations_by_base_pair": {pid: int(counts[pid]) for pid in sorted(row_word_by_id)},
        "extra_words_by_base_pair": {pid: int(words[pid]) for pid in sorted(row_word_by_id)},
        "presentation_count_range": [min(vals), max(vals)] if vals else [0, 0],
        "presentation_count_stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }
    return schedule, summary


def build_balanced_current_substitution_schedule(current_rows: List[Dict[str, Any]], aux_schedule: List[Dict[str, Any]],
                                                 epoch_capacity: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(stable_seed("balanced-current-substitution", seed))
    row_word_by_id = {r["pair_id"]: int(r["row_words"]) for r in current_rows}
    counts: Counter = Counter()
    words: Counter = Counter()
    schedule: List[Dict[str, Any]] = []
    candidate_cache: Dict[int, List[Tuple[List[str], int, int]]] = {}
    for e in aux_schedule:
        aux_words = int(e["aux_row_words"])
        if aux_words not in candidate_cache:
            cand = feasible_subsets(current_rows, aux_words, "over")
            # Drop enough to fit, but do not consider huge over-drops unless they are the
            # only way to keep all sources balanced. This still permits unused words.
            if any(slack <= 80 for _, _, slack in cand):
                cand = [(ids, total, slack) for ids, total, slack in cand if slack <= 80]
            candidate_cache[aux_words] = cand
        drop_ids, drop_words, overdrop = balanced_choose(candidate_cache[aux_words], counts, words, row_word_by_id, rng, "over")
        for pid in drop_ids:
            counts[pid] += 1
            words[pid] += row_word_by_id[pid]
        kept = [r["pair_id"] for r in current_rows if r["pair_id"] not in set(drop_ids)]
        epoch_words = int(epoch_capacity) - overdrop
        schedule.append({
            "epoch": int(e["epoch"]),
            "kept_current_pair_ids": kept,
            "dropped_current_pair_ids": drop_ids,
            "dropped_current_words": drop_words,
            "aux_pair_ids": list(e["aux_pair_ids"]),
            "aux_row_words": aux_words,
            "epoch_row_words": epoch_words,
            "unused_capacity_words": overdrop,
        })
    kept_counts: Dict[str, int] = {}
    drop_counts: Dict[str, int] = {}
    for pid in row_word_by_id:
        drop_counts[pid] = int(counts[pid])
        kept_counts[pid] = len(aux_schedule) - int(counts[pid])
    drop_vals = list(drop_counts.values())
    kept_vals = list(kept_counts.values())
    summary = {
        "seed": int(seed),
        "total_row_words": sum(int(x["epoch_row_words"]) for x in schedule),
        "epoch_capacity": int(epoch_capacity),
        "capacity_total": int(epoch_capacity) * len(aux_schedule),
        "unused_capacity_total": sum(int(x["unused_capacity_words"]) for x in schedule),
        "kept_presentations_by_pair": {pid: kept_counts[pid] for pid in sorted(row_word_by_id)},
        "dropped_presentations_by_pair": {pid: drop_counts[pid] for pid in sorted(row_word_by_id)},
        "drop_count_range": [min(drop_vals), max(drop_vals)] if drop_vals else [0, 0],
        "drop_count_stdev": statistics.pstdev(drop_vals) if len(drop_vals) > 1 else 0.0,
        "kept_count_range": [min(kept_vals), max(kept_vals)] if kept_vals else [0, 0],
    }
    return schedule, summary


def make_presentations_for_epoch(arm_name: str, epoch: int,
                                 base_current: Dict[str, Dict[str, Any]], base_compact: Dict[str, Dict[str, Any]],
                                 aux_compact: Dict[str, Dict[str, Any]], aux_schedule_by_epoch: Dict[int, Dict[str, Any]],
                                 rec_schedule_by_epoch: Dict[int, Dict[str, Any]], subst_schedule_by_epoch: Dict[int, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stats = Counter()

    def add(ex: Dict[str, Any], kind: str, idx: int) -> None:
        rr = dict(ex)
        rr["presentation_kind"] = kind
        rr["presentation_index"] = idx
        rows.append(rr)
        stats[f"{kind}_presentations"] += 1
        stats["row_words"] += int(ex["row_words"])
        stats["view_words"] += int(ex["view_words"])

    if arm_name == "current_base80":
        for i, pid in enumerate(sorted(base_current)):
            add(base_current[pid], "current_base", i)
    elif arm_name == "compact_base80_unspent":
        for i, pid in enumerate(sorted(base_compact)):
            add(base_compact[pid], "compact_base", i)
    elif arm_name == "compact_aux_support":
        for i, pid in enumerate(sorted(base_compact)):
            add(base_compact[pid], "compact_base", i)
        e = aux_schedule_by_epoch[epoch]
        for j, pid in enumerate(e["aux_pair_ids"]):
            add(aux_compact[pid], "aux_common", j)
    elif arm_name == "compact_interleaved_recurrence":
        for i, pid in enumerate(sorted(base_compact)):
            add(base_compact[pid], "compact_base", i)
        e = rec_schedule_by_epoch[epoch]
        for j, pid in enumerate(e["extra_base_pair_ids"]):
            add(base_compact[pid], "compact_recurrence", j)
    elif arm_name == "current_aux_substitution":
        e = subst_schedule_by_epoch[epoch]
        for i, pid in enumerate(e["kept_current_pair_ids"]):
            add(base_current[pid], "current_base", i)
        for j, pid in enumerate(e["aux_pair_ids"]):
            add(aux_compact[pid], "aux_common", j)
    else:
        raise ValueError(arm_name)
    return rows, dict(stats)


def collate_presentations(batch: List[Dict[str, Any]], tokenizer, epoch: int, seed: int,
                          mask_prob: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    mask_id = int(tokenizer.mask_token_id)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    target_groups = 0
    target_tokens = 0
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        ids[i, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
        att[i, :L] = 1
        groups = list(ex["train_groups"])
        mseed = stable_seed("mask", seed, epoch, ex["pair_id"], ex["view_kind"], ex.get("presentation_kind"), ex.get("presentation_index", 0))
        rng = random.Random(mseed)
        chosen = [g for g in groups if rng.random() < float(mask_prob)]
        if not chosen and groups:
            chosen = [rng.choice(groups)]
        for g in chosen:
            target_groups += 1
            for p in g["positions"]:
                if p < L:
                    labels[i, p] = ids[i, p]
                    ids[i, p] = mask_id
                    target_tokens += 1
    return ids.to(device), att.to(device), labels.to(device), {"target_groups": target_groups, "target_tokens": target_tokens}


def train_arm(arm_name: str, rep_seed: int, tokenizer, device: torch.device, args: argparse.Namespace,
              out_dir: pathlib.Path, base_current, base_compact, aux_compact,
              aux_schedule_by_epoch, rec_schedule_by_epoch, subst_schedule_by_epoch,
              surface_tasks, parent_surface_rows, common_records, parent_common_rows) -> Dict[str, Any]:
    arm_dir = out_dir / f"seed_{rep_seed}" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "arm_start", "seed": rep_seed, "arm": arm_name}), flush=True)
    set_global_seed(stable_seed("arm-global", rep_seed, arm_name))
    model, missing, unexpected = bridge.load_model(device, private_scale=float(args.private_scale))
    ident = bridge.model_identity(model)
    if ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(ident.get("private_adapter_params", 0)) != 995584:
        raise RuntimeError(f"bad model identity: {ident}")
    opt, opt_info = bridge.freeze_to_private_optimizer(model, float(args.lr), float(args.weight_decay))
    model.train()
    logs: List[Dict[str, Any]] = []
    t0 = time.time()
    epochs = int(args.epochs)
    for epoch in range(1, epochs + 1):
        presentations, pst = make_presentations_for_epoch(arm_name, epoch, base_current, base_compact, aux_compact,
                                                          aux_schedule_by_epoch, rec_schedule_by_epoch, subst_schedule_by_epoch)
        rng = random.Random(stable_seed("order", rep_seed, arm_name, epoch))
        order = list(range(len(presentations)))
        rng.shuffle(order)
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_targets = 0
        epoch_groups = 0
        for start in range(0, len(order), int(args.batch_size)):
            batch = [presentations[i] for i in order[start:start + int(args.batch_size)]]
            ids, att, labels, st = collate_presentations(batch, tokenizer, epoch, rep_seed, float(args.mask_prob), device)
            if int((labels != -100).sum().item()) == 0:
                continue
            out = model(input_ids=ids, attention_mask=att)
            vocab = out.logits.shape[-1]
            loss = F.cross_entropy(out.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="mean")
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"bad loss {arm_name} seed {rep_seed} epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm)).detach().cpu())
            opt.step()
            epoch_loss_sum += float(loss.detach().cpu())
            epoch_batches += 1
            epoch_targets += int(st["target_tokens"])
            epoch_groups += int(st["target_groups"])
            del ids, att, labels, out, loss
        log = {
            "epoch": epoch,
            "mean_loss": epoch_loss_sum / max(1, epoch_batches),
            "batches": epoch_batches,
            "target_groups": epoch_groups,
            "target_tokens": epoch_targets,
            "presentations": len(presentations),
            "row_words_epoch": int(pst.get("row_words", 0)),
            "view_words_epoch": int(pst.get("view_words", 0)),
            "row_words_cum": sum(int(x.get("row_words_epoch", 0)) for x in logs) + int(pst.get("row_words", 0)),
            "elapsed_sec": round(time.time() - t0, 1),
            **{k: int(v) for k, v in pst.items() if k.endswith("_presentations")},
        }
        logs.append(log)
        if epoch == 1 or epoch % int(args.log_every) == 0 or epoch == epochs:
            print(json.dumps({"event": "train_epoch", "seed": rep_seed, "arm": arm_name, **log}), flush=True)
    model.eval()
    surface_rows = score_surface_tasks(model, tokenizer, surface_tasks, device, int(args.eval_batch_size))
    common_rows = score_common_records(model, tokenizer, common_records, device, int(args.eval_batch_size))
    write_jsonl(arm_dir / "surface_scores.jsonl", surface_rows)
    write_jsonl(arm_dir / "common_scores.jsonl", common_rows)
    surf_summary = summarize_surface(surface_rows, parent_surface_rows)
    common_summary = summarize_common(common_rows, parent_common_rows)
    slim_common = {k: v for k, v in common_summary.items() if k not in {"margins", "source_follow_rows", "delta_rows"}}
    # Save row-heavy material separately for inspection.
    write_jsonl(arm_dir / "common_delta_rows.jsonl", common_summary["delta_rows"])
    write_jsonl(arm_dir / "common_source_follow_rows.jsonl", common_summary["source_follow_rows"])
    if not args.no_save_checkpoints:
        ckpt = arm_dir / "checkpoint_final"
        bridge.save_checkpoint(model, tokenizer, ckpt, {"step": "allocation_learner_comparison", "seed": rep_seed, "arm": arm_name, "model_identity": ident}, float(args.private_scale))
        ckpt_path = rel(ckpt)
    else:
        ckpt_path = None
    summary = {
        "seed": int(rep_seed),
        "arm": arm_name,
        "model_identity": ident,
        "optimizer": opt_info,
        "completed_epochs": epochs,
        "final_train_log": logs[-1] if logs else None,
        "train_log": rel(arm_dir / "train_log.jsonl"),
        "surface_summary": surf_summary,
        "common_summary": slim_common,
        "checkpoint_final": ckpt_path,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    write_jsonl(arm_dir / "train_log.jsonl", logs)
    (arm_dir / "arm_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "arm_done", "seed": rep_seed, "arm": arm_name, "elapsed_sec": summary["elapsed_sec"], "summary": rel(arm_dir / "arm_summary.json")}), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def aggregate_run_results(parent_common_summary: Dict[str, Any], parent_surface_summary: Dict[str, Any],
                          arm_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm = defaultdict(list)
    for a in arm_summaries:
        by_arm[a["arm"]].append(a)

    rows = []
    for arm, arr in sorted(by_arm.items()):
        for a in arr:
            cs = a["common_summary"].get("delta_decomposition_by_group", {})
            ss = a["surface_summary"].get("by_group_view_condition", {})
            row = {
                "seed": a["seed"],
                "arm": arm,
                "base_common_g": cs.get("base_common", {}).get("mean_g_source_follow"),
                "base_common_b": cs.get("base_common", {}).get("mean_b_direction_imbalance"),
                "base_common_p": cs.get("base_common", {}).get("mean_p_no_source_original_prior"),
                "aux_common_g": cs.get("aux_common", {}).get("mean_g_source_follow"),
                "aux_common_b": cs.get("aux_common", {}).get("mean_b_direction_imbalance"),
                "aux_common_p": cs.get("aux_common", {}).get("mean_p_no_source_original_prior"),
                "base_compact_with_source_delta_nll": ss.get("base_surface/compact/with_source", {}).get("mean_delta_nll_vs_parent"),
                "base_current_with_source_delta_nll": ss.get("base_surface/current/with_source", {}).get("mean_delta_nll_vs_parent"),
                "aux_compact_with_source_delta_nll": ss.get("aux_surface/compact/with_source", {}).get("mean_delta_nll_vs_parent"),
                "aux_current_with_source_delta_nll": ss.get("aux_surface/current/with_source", {}).get("mean_delta_nll_vs_parent"),
                "elapsed_sec": a.get("elapsed_sec"),
            }
            rows.append(row)

    def agg(vals: List[Any]) -> Dict[str, Any]:
        nums = []
        for v in vals:
            if v is None:
                continue
            try:
                f = float(v)
            except Exception:
                continue
            if math.isfinite(f):
                nums.append(f)
        return {"n": len(nums), "mean": sum(nums) / len(nums) if nums else None, "median": statistics.median(nums) if nums else None, "stdev": statistics.stdev(nums) if len(nums) > 1 else 0.0}

    by_arm_metric: Dict[str, Dict[str, Any]] = {}
    metric_keys = [k for k in rows[0].keys() if k not in {"seed", "arm"}] if rows else []
    for arm in sorted(by_arm):
        ar = [r for r in rows if r["arm"] == arm]
        by_arm_metric[arm] = {k: agg([r.get(k) for r in ar]) for k in metric_keys}

    def mean_metric(arm: str, key: str) -> Optional[float]:
        return by_arm_metric.get(arm, {}).get(key, {}).get("mean")

    contrasts = {
        "compact_aux_minus_compact_unspent": {
            "base_common_g": (mean_metric("compact_aux_support", "base_common_g") or 0.0) - (mean_metric("compact_base80_unspent", "base_common_g") or 0.0) if mean_metric("compact_aux_support", "base_common_g") is not None and mean_metric("compact_base80_unspent", "base_common_g") is not None else None,
            "aux_common_g": (mean_metric("compact_aux_support", "aux_common_g") or 0.0) - (mean_metric("compact_base80_unspent", "aux_common_g") or 0.0) if mean_metric("compact_aux_support", "aux_common_g") is not None and mean_metric("compact_base80_unspent", "aux_common_g") is not None else None,
            "base_compact_with_source_delta_nll": (mean_metric("compact_aux_support", "base_compact_with_source_delta_nll") or 0.0) - (mean_metric("compact_base80_unspent", "base_compact_with_source_delta_nll") or 0.0) if mean_metric("compact_aux_support", "base_compact_with_source_delta_nll") is not None and mean_metric("compact_base80_unspent", "base_compact_with_source_delta_nll") is not None else None,
        },
        "compact_aux_minus_compact_recurrence": {
            "base_common_g": (mean_metric("compact_aux_support", "base_common_g") or 0.0) - (mean_metric("compact_interleaved_recurrence", "base_common_g") or 0.0) if mean_metric("compact_aux_support", "base_common_g") is not None and mean_metric("compact_interleaved_recurrence", "base_common_g") is not None else None,
            "aux_common_g": (mean_metric("compact_aux_support", "aux_common_g") or 0.0) - (mean_metric("compact_interleaved_recurrence", "aux_common_g") or 0.0) if mean_metric("compact_aux_support", "aux_common_g") is not None and mean_metric("compact_interleaved_recurrence", "aux_common_g") is not None else None,
            "aux_compact_with_source_delta_nll": (mean_metric("compact_aux_support", "aux_compact_with_source_delta_nll") or 0.0) - (mean_metric("compact_interleaved_recurrence", "aux_compact_with_source_delta_nll") or 0.0) if mean_metric("compact_aux_support", "aux_compact_with_source_delta_nll") is not None and mean_metric("compact_interleaved_recurrence", "aux_compact_with_source_delta_nll") is not None else None,
        },
        "compact_aux_minus_current_aux_substitution": {
            "base_common_g": (mean_metric("compact_aux_support", "base_common_g") or 0.0) - (mean_metric("current_aux_substitution", "base_common_g") or 0.0) if mean_metric("compact_aux_support", "base_common_g") is not None and mean_metric("current_aux_substitution", "base_common_g") is not None else None,
            "aux_common_g": (mean_metric("compact_aux_support", "aux_common_g") or 0.0) - (mean_metric("current_aux_substitution", "aux_common_g") or 0.0) if mean_metric("compact_aux_support", "aux_common_g") is not None and mean_metric("current_aux_substitution", "aux_common_g") is not None else None,
            "base_compact_with_source_delta_nll": (mean_metric("compact_aux_support", "base_compact_with_source_delta_nll") or 0.0) - (mean_metric("current_aux_substitution", "base_compact_with_source_delta_nll") or 0.0) if mean_metric("compact_aux_support", "base_compact_with_source_delta_nll") is not None and mean_metric("current_aux_substitution", "base_compact_with_source_delta_nll") is not None else None,
            "base_current_with_source_delta_nll": (mean_metric("compact_aux_support", "base_current_with_source_delta_nll") or 0.0) - (mean_metric("current_aux_substitution", "base_current_with_source_delta_nll") or 0.0) if mean_metric("compact_aux_support", "base_current_with_source_delta_nll") is not None and mean_metric("current_aux_substitution", "base_current_with_source_delta_nll") is not None else None,
            "aux_compact_with_source_delta_nll": (mean_metric("compact_aux_support", "aux_compact_with_source_delta_nll") or 0.0) - (mean_metric("current_aux_substitution", "aux_compact_with_source_delta_nll") or 0.0) if mean_metric("compact_aux_support", "aux_compact_with_source_delta_nll") is not None and mean_metric("current_aux_substitution", "aux_compact_with_source_delta_nll") is not None else None,
        },
    }
    return {
        "parent_common_summary": parent_common_summary,
        "parent_surface_summary": parent_surface_summary,
        "per_seed_arm_metrics": rows,
        "by_arm_metric": by_arm_metric,
        "contrasts_mean_arm_differences": contrasts,
        "interpretation_note": "Compare compact_aux jointly: auxiliary learning plus base preservation. Equal auxiliary learning with stronger base preservation is still useful evidence for compact reinvestment; current_aux matching both outcomes means ordinary substitution may suffice.",
    }


def query_gpu_memory() -> List[Dict[str, Any]]:
    import subprocess
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 4:
            rows.append({"index": int(parts[0]), "name": parts[1], "memory_used_mb": int(parts[2]), "memory_total_mb": int(parts[3])})
    return rows


def wait_for_gpu(max_used_mb: int, interval_sec: float, max_wait_sec: float, out_dir: pathlib.Path) -> int:
    start = time.time(); attempt = 0
    log_path = out_dir / "gpu_wait_log.jsonl"
    while True:
        attempt += 1
        rows = query_gpu_memory()
        idle = [r for r in rows if r["memory_used_mb"] <= int(max_used_mb)]
        idle.sort(key=lambda r: (r["memory_used_mb"], r["index"]))
        selected = idle[0] if idle else None
        event = {"event": "gpu_wait_check", "attempt": attempt, "elapsed_sec": round(time.time() - start, 1), "selected": selected, "gpus": rows}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if selected:
            return int(selected["index"])
        if time.time() - start > float(max_wait_sec):
            raise TimeoutError(f"No idle GPU after {max_wait_sec}s")
        time.sleep(float(interval_sec))


def parse_seeds(txt: str) -> List[int]:
    seeds = []
    for part in str(txt).split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("empty seed list")
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design-dir", type=pathlib.Path, default=DESIGN_DIR)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--rep-seeds", default="60001,60002,60003")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--mask-prob", type=float, default=0.35)
    ap.add_argument("--max-train-targets-per-row", type=int, default=16)
    ap.add_argument("--max-eval-targets-per-view", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--idle-memory-mb", type=int, default=2500)
    ap.add_argument("--gpu-wait-sec", type=float, default=600.0)
    ap.add_argument("--gpu-check-interval-sec", type=float, default=30.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--no-save-checkpoints", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str((out_dir / "hf_cache/hf_home").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((out_dir / "hf_cache/transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((out_dir / "hf_cache/modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (out_dir / "hf_cache/modules").mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.rep_seeds)
    design_dir = args.design_dir
    design = load_json(design_dir / "allocation_design_v2.json")
    base_current_rows = load_jsonl(design_dir / "base_current_rows.jsonl")
    base_compact_rows = load_jsonl(design_dir / "base_compact_rows.jsonl")
    aux_compact_rows = load_jsonl(design_dir / "aux_compact_rows.jsonl")
    aux_schedule = load_jsonl(design_dir / "aux_support_schedule_epoch80.jsonl")
    aux_tasks_path = design_dir / "aux_support_tasks_repaired.jsonl"
    label_path = pathlib.Path(design["inputs"]["semantic_labels"])
    if not label_path.is_absolute():
        label_path = ROOT / label_path

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("tokenizer has no mask token")

    base_current, base_current_stats = build_train_examples(base_current_rows, tokenizer, int(args.max_length), "current", int(args.max_train_targets_per_row))
    base_compact, base_compact_stats = build_train_examples(base_compact_rows, tokenizer, int(args.max_length), "compact", int(args.max_train_targets_per_row))
    aux_compact, aux_compact_stats = build_train_examples(aux_compact_rows, tokenizer, int(args.max_length), "compact", int(args.max_train_targets_per_row))
    if set(base_current) != set(base_compact):
        raise RuntimeError("base current/compact IDs differ")
    if set(aux_compact) != set(r["pair_id"] for r in aux_compact_rows):
        raise RuntimeError("aux compact IDs lost during tokenization")

    surface_base, surface_base_stats = build_surface_tasks(base_current_rows, tokenizer, int(args.max_length), int(args.max_eval_targets_per_view), "base_surface")
    surface_aux, surface_aux_stats = build_surface_tasks(aux_compact_rows, tokenizer, int(args.max_length), int(args.max_eval_targets_per_view), "aux_surface")
    surface_tasks = surface_base + surface_aux
    common_records, common_record_summary = build_common_candidate_records(tokenizer, label_path, aux_tasks_path, int(args.max_length))

    repaired_schedules: Dict[str, Any] = {}
    for seed in seeds:
        rec_schedule, rec_summary = build_balanced_recurrence_schedule(base_compact_rows, aux_schedule, seed)
        subst_schedule, subst_summary = build_balanced_current_substitution_schedule(base_current_rows, aux_schedule, design["word_accounting_per_base_epoch"]["current_base_row_words"], seed)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(seed_dir / "balanced_recurrence_schedule.jsonl", rec_schedule)
        write_jsonl(seed_dir / "balanced_current_substitution_schedule.jsonl", subst_schedule)
        repaired_schedules[str(seed)] = {"recurrence_summary": rec_summary, "current_substitution_summary": subst_summary,
                                         "recurrence_schedule": rel(seed_dir / "balanced_recurrence_schedule.jsonl"),
                                         "current_substitution_schedule": rel(seed_dir / "balanced_current_substitution_schedule.jsonl")}
    plan = {
        "status": "ALLOCATION_BALANCED_PLAN",
        "created_utc": now(),
        "purpose": "Balanced execution of compact preservation plus reinvested additional support against recurrence and current-view substitution.",
        "source_design": rel(design_dir / "allocation_design_v2.json"),
        "rep_seeds": seeds,
        "arms": ["current_base80", "compact_base80_unspent", "compact_interleaved_recurrence", "compact_aux_support", "current_aux_substitution"],
        "base_current_stats": base_current_stats,
        "base_compact_stats": base_compact_stats,
        "aux_compact_stats": aux_compact_stats,
        "surface_eval_stats": {"base": surface_base_stats, "aux": surface_aux_stats, "total_tasks": len(surface_tasks)},
        "common_record_summary": common_record_summary,
        "aux_schedule_preserved": rel(design_dir / "aux_support_schedule_epoch80.jsonl"),
        "repaired_schedules": repaired_schedules,
        "resource_accounting": {
            "current_base_words_per_epoch": design["word_accounting_per_base_epoch"]["current_base_row_words"],
            "compact_base_words_per_epoch": design["word_accounting_per_base_epoch"]["compact_base_row_words"],
            "saved_words_per_epoch": design["word_accounting_per_base_epoch"]["saved_row_words"],
            "aux_used_words_total_original_schedule": design["aux_schedule_summary"]["used_aux_words_total"],
            "note": "Balanced schedules may leave more unused words than exact packing so source starvation does not determine the comparison.",
        },
        "interpretation": "Compact+aux should be read jointly: additional support learning together with preservation of base readouts. Equal aux learning with better preservation supports compact reinvestment; current+aux matching both outcomes supports ordinary substitution.",
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    if args.gpu == "auto":
        gpu_idx = wait_for_gpu(int(args.idle_memory_mb), float(args.gpu_check_interval_sec), float(args.gpu_wait_sec), out_dir)
    else:
        gpu_idx = int(args.gpu)
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "device_selected", "device": str(device), "gpu": gpu_idx}), flush=True)

    # Parent readouts once.
    parent_model, missing, unexpected = bridge.load_model(device, private_scale=float(args.private_scale))
    parent_ident = bridge.model_identity(parent_model)
    if parent_ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(parent_ident.get("private_adapter_params", 0)) != 995584:
        raise RuntimeError(f"bad parent identity: {parent_ident}")
    parent_model.eval()
    parent_surface_rows = score_surface_tasks(parent_model, tokenizer, surface_tasks, device, int(args.eval_batch_size))
    parent_common_rows = score_common_records(parent_model, tokenizer, common_records, device, int(args.eval_batch_size))
    write_jsonl(out_dir / "parent_surface_scores.jsonl", parent_surface_rows)
    write_jsonl(out_dir / "parent_common_scores.jsonl", parent_common_rows)
    parent_surface_summary = summarize_surface(parent_surface_rows, None)
    parent_common_summary = summarize_common(parent_common_rows, None)
    (out_dir / "parent_surface_summary.json").write_text(json.dumps(parent_surface_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "parent_common_summary.json").write_text(json.dumps({k: v for k, v in parent_common_summary.items() if k not in {"margins", "source_follow_rows", "delta_rows"}}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del parent_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    arm_summaries: List[Dict[str, Any]] = []
    arms = plan["arms"]
    aux_by_epoch = {int(e["epoch"]): e for e in aux_schedule}
    for seed in seeds:
        rec_schedule = load_jsonl(out_dir / f"seed_{seed}" / "balanced_recurrence_schedule.jsonl")
        subst_schedule = load_jsonl(out_dir / f"seed_{seed}" / "balanced_current_substitution_schedule.jsonl")
        rec_by_epoch = {int(e["epoch"]): e for e in rec_schedule}
        subst_by_epoch = {int(e["epoch"]): e for e in subst_schedule}
        for arm_name in arms:
            summ = train_arm(arm_name, int(seed), tokenizer, device, args, out_dir,
                             base_current, base_compact, aux_compact, aux_by_epoch, rec_by_epoch, subst_by_epoch,
                             surface_tasks, parent_surface_rows, common_records, parent_common_rows)
            arm_summaries.append(summ)
    agg = aggregate_run_results({k: v for k, v in parent_common_summary.items() if k not in {"margins", "source_follow_rows", "delta_rows"}}, parent_surface_summary, arm_summaries)
    # Write compact CSV of the main metrics.
    metric_rows = agg["per_seed_arm_metrics"]
    if metric_rows:
        with (out_dir / "per_seed_arm_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            w.writeheader()
            for r in metric_rows:
                w.writerow(r)
    summary = {
        "status": "ALLOCATION_BALANCED_COMPARISON_DONE",
        "created_utc": now(),
        "plan": rel(out_dir / "plan.json"),
        "parent_identity": parent_ident,
        "n_arm_runs": len(arm_summaries),
        "arm_summaries": [{k: v for k, v in a.items() if k not in {"optimizer", "model_identity"}} for a in arm_summaries],
        "aggregate": agg,
        "scientific_status": "bounded learner comparison for Stage III route selection; not a legal BabyLM endpoint",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
