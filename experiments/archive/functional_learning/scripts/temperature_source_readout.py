#!/usr/bin/env python3
"""research: temperature-adjusted source-use readout for dense unchanged-Qwen focus.

The dense-focus endpoints show larger source-help and common-source margins but also
worse broad likelihood on CDI and context spans.  This script separates two possible
components without touching official scores: (i) a global logit-confidence scale
estimated on lawful non-benchmark text, and (ii) rank/order changes on source-use
readouts.  The temperature is fitted only on ordinary non-Qwen legal-tail text beyond
the first 80-update dense/sparse prefix, not on CDI, SuperGLUE, Entity, or the 25-item
source-reversal bank.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import corrected_bridge_trainer as bridge  # noqa: E402
import concentrated_compact_learning_test as s57  # noqa: E402
import common_target_probe as common_probe  # noqa: E402
import qwen_view_surface_probe as qview  # noqa: E402
import qwen_source_perturb_probe as source_probe  # noqa: E402

DEFAULT_TAIL = _public_path('experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl')
DEFAULT_LABELS = _public_path('experiments/archive/functional_learning/data/selective_reviewed_labels/selective_semantic_labels.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/temperature_source_readout')
WORDS_PER_UPDATE = 39_533
PREFIX_UPDATES = 80
PREFIX_WORD_LIMIT = WORDS_PER_UPDATE * PREFIX_UPDATES

MODEL_SPECS: Dict[str, Optional[pathlib.Path]] = {
    "coherent86": None,
    "sparse_focus_seed62064": _public_path('experiments/archive/functional_learning/data/unchanged_focus_weighted_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "dense_focus_seed62064": _public_path('experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "dense_focus_seed62065": _public_path('experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080'),
    "densemask_sparselabel_seed62064": _public_path('experiments/archive/functional_learning/data/densemask_sparselabel_train_seed62064/correspondence_focus_weighted/checkpoints/update_0080'),
}


def rel(p: pathlib.Path | str | None) -> str | None:
    if p is None:
        return None
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len((text or "").strip().split())


def stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**31 - 1)


def finite_vals(xs: Iterable[Any]) -> List[float]:
    vals: List[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            vals.append(xf)
    return vals


def mean(xs: Iterable[Any]) -> Optional[float]:
    vals = finite_vals(xs)
    return sum(vals) / len(vals) if vals else None


def median(xs: Iterable[Any]) -> Optional[float]:
    vals = finite_vals(xs)
    return statistics.median(vals) if vals else None


def sem(xs: Iterable[Any]) -> Optional[float]:
    vals = finite_vals(xs)
    if len(vals) <= 1:
        return None
    return statistics.stdev(vals) / math.sqrt(len(vals))


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def locate_positions(offsets: List[Tuple[int, int]], start: int, end: int) -> List[int]:
    pos: List[int] = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if a < end and b > start:
            pos.append(i)
    return pos


def build_calibration_records(tail_jsonl: pathlib.Path, tokenizer, *, n_records: int, seed: int,
                              max_length: int, rows_after_prefix: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select masked content words from ordinary legal-tail rows, away from benchmarks."""
    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    stats = Counter()
    total_words = 0
    row_idx = 0
    with tail_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_words = int(row.get("words", wc(row.get("text", ""))))
            before_prefix = total_words < PREFIX_WORD_LIMIT
            total_words += row_words
            if rows_after_prefix and before_prefix:
                row_idx += 1
                continue
            if row.get("source") == "qwen_pair_packed":
                stats["skip_qwen_pair_packed"] += 1
                row_idx += 1
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                stats["skip_empty"] += 1
                row_idx += 1
                continue
            groups = s57.content_word_spans(text)
            if not groups:
                stats["skip_no_content_groups"] += 1
                row_idx += 1
                continue
            # Deterministically sample one content word per row for broad row coverage.
            grng = random.Random(stable_seed("calibration-row", seed, row_idx, row.get("example_id", "")))
            a, b, word = groups[grng.randrange(len(groups))]
            enc = tokenizer(text, add_special_tokens=True, truncation=True, max_length=int(max_length), return_offsets_mapping=True)
            ids = [int(x) for x in enc["input_ids"]]
            offsets = [(int(x), int(y)) for x, y in enc["offset_mapping"]]
            pos = locate_positions(offsets, a, b)
            if not pos:
                stats["skip_no_token_position"] += 1
                row_idx += 1
                continue
            rec = {
                "task_id": f"calib_r{row_idx:07d}",
                "row_idx": row_idx,
                "source": str(row.get("source", "")),
                "example_id": row.get("example_id"),
                "target_word": word,
                "input_ids": ids,
                "positions": pos,
                "row_words": row_words,
            }
            stats["eligible_records"] += 1
            # Reservoir sample without loading the whole 130 MB tail into memory.
            if len(reservoir) < n_records:
                reservoir.append(rec)
            else:
                j = rng.randrange(stats["eligible_records"])
                if j < n_records:
                    reservoir[j] = rec
            row_idx += 1
    reservoir.sort(key=lambda r: int(r["row_idx"]))
    return reservoir, {
        "n_records": len(reservoir),
        "requested_records": int(n_records),
        "seed": int(seed),
        "rows_after_first_80_update_prefix": bool(rows_after_prefix),
        "prefix_word_limit": int(PREFIX_WORD_LIMIT),
        "stats": dict(stats),
        "row_idx_min": min((int(r["row_idx"]) for r in reservoir), default=None),
        "row_idx_max": max((int(r["row_idx"]) for r in reservoir), default=None),
        "source_counts": dict(Counter(str(r.get("source", "")) for r in reservoir)),
        "scientific_use": "Fit only a global logit-temperature on ordinary legal-tail text outside benchmark/CDI/source-reversal targets.",
    }


def strip_task_for_json(t: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in t.items() if k not in {"input_ids", "positions"}}


def gather_position_logits(model, tokenizer, tasks: List[Dict[str, Any]], device: torch.device,
                           batch_size: int) -> Tuple[List[torch.Tensor], List[int], List[Dict[str, Any]]]:
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    mask_id = int(tokenizer.mask_token_id)
    logits_rows: List[torch.Tensor] = []
    target_ids: List[int] = []
    metas: List[Dict[str, Any]] = []
    for start in range(0, len(tasks), int(batch_size)):
        batch = tasks[start:start + int(batch_size)]
        max_len = max(len(t["input_ids"]) for t in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        orig = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        for i, t in enumerate(batch):
            arr = torch.tensor(t["input_ids"], dtype=torch.long, device=device)
            L = int(arr.numel())
            ids[i, :L] = arr
            orig[i, :L] = arr
            att[i, :L] = 1
            for p in t["positions"]:
                if int(p) < L:
                    ids[i, int(p)] = mask_id
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        for i, t in enumerate(batch):
            for p in t["positions"]:
                p = int(p)
                if p >= logits.shape[1]:
                    continue
                logits_rows.append(logits[i, p].detach().cpu())
                target_ids.append(int(orig[i, p].detach().cpu()))
                metas.append({"task_id": t.get("task_id"), "row_idx": t.get("row_idx"), "target_word": t.get("target_word"), "position": p})
        del ids, att, orig, logits
    return logits_rows, target_ids, metas


def nll_from_logits(logits_row: torch.Tensor, target_id: int, temperature: float) -> float:
    z = logits_row.float() / float(temperature)
    return float(torch.logsumexp(z, dim=-1).item() - z[int(target_id)].item())


def rank_from_logits(logits_row: torch.Tensor, target_id: int) -> int:
    v = logits_row[int(target_id)]
    return int(torch.sum(logits_row > v).item()) + 1


def fit_temperature_from_position_logits(logits_rows: List[torch.Tensor], target_ids: List[int], grid: List[float]) -> Dict[str, Any]:
    if not logits_rows:
        return {"status": "NO_CALIBRATION_LOGITS", "best_temperature": 1.0, "n_positions": 0}
    # Coarse grid is intentional: this is a small readout of global scale, not score tuning.
    scores = []
    ranks = []
    for row, tid in zip(logits_rows, target_ids, strict=False):
        ranks.append(rank_from_logits(row, tid))
    for t in grid:
        vals = [nll_from_logits(row, tid, float(t)) for row, tid in zip(logits_rows, target_ids, strict=False)]
        scores.append({"temperature": float(t), "mean_nll": mean(vals), "median_nll": median(vals)})
    best = min(scores, key=lambda x: float(x["mean_nll"]))
    nll_t1 = next((x for x in scores if abs(float(x["temperature"]) - 1.0) < 1e-9), None)
    return {
        "status": "TEMPERATURE_GRID_FIT_DONE",
        "best_temperature": float(best["temperature"]),
        "best_mean_nll": best["mean_nll"],
        "mean_nll_at_T1": nll_t1["mean_nll"] if nll_t1 else None,
        "delta_mean_nll_best_minus_T1": (float(best["mean_nll"]) - float(nll_t1["mean_nll"])) if nll_t1 and best.get("mean_nll") is not None else None,
        "n_positions": len(logits_rows),
        "mean_rank_at_T1": mean(ranks),
        "median_rank_at_T1": median(ranks),
        "grid": scores,
    }


def score_tasks_with_temperature(model, tokenizer, tasks: List[Dict[str, Any]], device: torch.device,
                                 batch_size: int, temperature: float) -> List[Dict[str, Any]]:
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    mask_id = int(tokenizer.mask_token_id)
    out: List[Dict[str, Any]] = []
    for start in range(0, len(tasks), int(batch_size)):
        batch = tasks[start:start + int(batch_size)]
        max_len = max(len(t["input_ids"]) for t in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        orig = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        for i, t in enumerate(batch):
            arr = torch.tensor(t["input_ids"], dtype=torch.long, device=device)
            L = int(arr.numel())
            ids[i, :L] = arr
            orig[i, :L] = arr
            att[i, :L] = 1
            for p in t["positions"]:
                p = int(p)
                if p < L:
                    ids[i, p] = mask_id
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        for i, t in enumerate(batch):
            vals_t1: List[float] = []
            vals_tfit: List[float] = []
            ranks: List[int] = []
            target_ids: List[int] = []
            for p in t["positions"]:
                p = int(p)
                if p >= logits.shape[1]:
                    continue
                row = logits[i, p].detach().cpu()
                tid = int(orig[i, p].detach().cpu())
                target_ids.append(tid)
                vals_t1.append(nll_from_logits(row, tid, 1.0))
                vals_tfit.append(nll_from_logits(row, tid, float(temperature)))
                ranks.append(rank_from_logits(row, tid))
            rr = strip_task_for_json(t)
            rr["nll_T1"] = mean(vals_t1)
            rr["nll_Tfit"] = mean(vals_tfit)
            rr["rank_mean"] = mean(ranks)
            rr["rank_median"] = median(ranks)
            rr["target_ids"] = target_ids
            rr["n_token_scores"] = len(vals_t1)
            out.append(rr)
        del ids, att, orig, logits
    return out


def summarize_qwen_source(score_rows: List[Dict[str, Any]], parent_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    parent_by_task = {str(r["task_id"]): r for r in parent_rows or []}
    by_condition: Dict[str, Any] = {}
    for cond in sorted(set(str(r["condition"]) for r in score_rows)):
        g = [r for r in score_rows if str(r["condition"]) == cond]
        by_condition[cond] = {
            "n": len(g),
            "mean_nll_T1": mean(r.get("nll_T1") for r in g),
            "mean_nll_Tfit": mean(r.get("nll_Tfit") for r in g),
            "mean_rank": mean(r.get("rank_mean") for r in g),
            "median_rank": median(r.get("rank_mean") for r in g),
            "mean_delta_nll_T1_vs_parent": mean(float(r["nll_T1"]) - float(parent_by_task[str(r["task_id"])]["nll_T1"]) for r in g if parent_by_task and str(r["task_id"]) in parent_by_task),
            "mean_delta_nll_Tfit_vs_parent": mean(float(r["nll_Tfit"]) - float(parent_by_task[str(r["task_id"])]["nll_Tfit"]) for r in g if parent_by_task and str(r["task_id"]) in parent_by_task),
            "mean_delta_rank_vs_parent": mean(float(r["rank_mean"]) - float(parent_by_task[str(r["task_id"])]["rank_mean"]) for r in g if parent_by_task and str(r["task_id"]) in parent_by_task),
        }
    by_base: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in score_rows:
        by_base[str(r["base_task_id"])][str(r["condition"])] = r
    parent_by_base: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in parent_rows or []:
        parent_by_base[str(r["base_task_id"])][str(r["condition"])] = r
    triplets: List[Dict[str, Any]] = []
    for base_id, vals in by_base.items():
        if not all(c in vals for c in ["correct_source", "wrong_source", "view_only"]):
            continue
        row = {
            "base_task_id": base_id,
            "pair_id": vals["correct_source"].get("pair_id"),
            "target_word": vals["correct_source"].get("target_word"),
            "specific_advantage_T1": float(vals["wrong_source"]["nll_T1"]) - float(vals["correct_source"]["nll_T1"]),
            "specific_advantage_Tfit": float(vals["wrong_source"]["nll_Tfit"]) - float(vals["correct_source"]["nll_Tfit"]),
            "specific_rank_advantage": float(vals["wrong_source"]["rank_mean"]) - float(vals["correct_source"]["rank_mean"]),
            "total_advantage_T1": float(vals["view_only"]["nll_T1"]) - float(vals["correct_source"]["nll_T1"]),
            "total_advantage_Tfit": float(vals["view_only"]["nll_Tfit"]) - float(vals["correct_source"]["nll_Tfit"]),
            "total_rank_advantage": float(vals["view_only"]["rank_mean"]) - float(vals["correct_source"]["rank_mean"]),
            "correct_rank": float(vals["correct_source"]["rank_mean"]),
            "wrong_rank": float(vals["wrong_source"]["rank_mean"]),
            "view_rank": float(vals["view_only"]["rank_mean"]),
        }
        pv = parent_by_base.get(base_id, {})
        if all(c in pv for c in ["correct_source", "wrong_source", "view_only"]):
            p_spec_t1 = float(pv["wrong_source"]["nll_T1"]) - float(pv["correct_source"]["nll_T1"])
            p_spec_tf = float(pv["wrong_source"]["nll_Tfit"]) - float(pv["correct_source"]["nll_Tfit"])
            p_spec_rank = float(pv["wrong_source"]["rank_mean"]) - float(pv["correct_source"]["rank_mean"])
            p_total_t1 = float(pv["view_only"]["nll_T1"]) - float(pv["correct_source"]["nll_T1"])
            p_total_tf = float(pv["view_only"]["nll_Tfit"]) - float(pv["correct_source"]["nll_Tfit"])
            p_total_rank = float(pv["view_only"]["rank_mean"]) - float(pv["correct_source"]["rank_mean"])
            row.update({
                "delta_specific_advantage_T1_vs_parent": row["specific_advantage_T1"] - p_spec_t1,
                "delta_specific_advantage_Tfit_vs_parent": row["specific_advantage_Tfit"] - p_spec_tf,
                "delta_specific_rank_advantage_vs_parent": row["specific_rank_advantage"] - p_spec_rank,
                "delta_total_advantage_T1_vs_parent": row["total_advantage_T1"] - p_total_t1,
                "delta_total_advantage_Tfit_vs_parent": row["total_advantage_Tfit"] - p_total_tf,
                "delta_total_rank_advantage_vs_parent": row["total_rank_advantage"] - p_total_rank,
            })
        triplets.append(row)
    return {
        "by_condition": by_condition,
        "source_specificity": {
            "n_complete_triplets": len(triplets),
            "mean_specific_advantage_T1": mean(r["specific_advantage_T1"] for r in triplets),
            "mean_specific_advantage_Tfit": mean(r["specific_advantage_Tfit"] for r in triplets),
            "mean_specific_rank_advantage": mean(r["specific_rank_advantage"] for r in triplets),
            "median_specific_rank_advantage": median(r["specific_rank_advantage"] for r in triplets),
            "share_specific_rank_advantage_positive": (sum(1 for r in triplets if r["specific_rank_advantage"] > 0) / len(triplets)) if triplets else None,
            "mean_delta_specific_advantage_T1_vs_parent": mean(r.get("delta_specific_advantage_T1_vs_parent") for r in triplets),
            "mean_delta_specific_advantage_Tfit_vs_parent": mean(r.get("delta_specific_advantage_Tfit_vs_parent") for r in triplets),
            "mean_delta_specific_rank_advantage_vs_parent": mean(r.get("delta_specific_rank_advantage_vs_parent") for r in triplets),
            "share_delta_specific_rank_positive_vs_parent": (sum(1 for r in triplets if r.get("delta_specific_rank_advantage_vs_parent") is not None and r["delta_specific_rank_advantage_vs_parent"] > 0) / len(triplets)) if triplets else None,
            "mean_total_advantage_T1": mean(r["total_advantage_T1"] for r in triplets),
            "mean_total_advantage_Tfit": mean(r["total_advantage_Tfit"] for r in triplets),
            "mean_total_rank_advantage": mean(r["total_rank_advantage"] for r in triplets),
            "mean_delta_total_advantage_T1_vs_parent": mean(r.get("delta_total_advantage_T1_vs_parent") for r in triplets),
            "mean_delta_total_advantage_Tfit_vs_parent": mean(r.get("delta_total_advantage_Tfit_vs_parent") for r in triplets),
            "mean_delta_total_rank_advantage_vs_parent": mean(r.get("delta_total_rank_advantage_vs_parent") for r in triplets),
        },
        "triplets": triplets,
    }


def summarize_common(score_rows: List[Dict[str, Any]], parent_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    by_key = {(str(r["task_id"]), str(r["condition"]), str(r["candidate_role"])): r for r in score_rows}
    p_by_key = {(str(r["task_id"]), str(r["condition"]), str(r["candidate_role"])): r for r in parent_rows or []}
    task_meta: Dict[str, Dict[str, Any]] = {}
    for r in score_rows:
        task_meta.setdefault(str(r["task_id"]), {k: r.get(k) for k in ["task_id", "pair_id", "split", "axis", "semantic_label", "frame"]})
    margins: List[Dict[str, Any]] = []
    for tid, meta in task_meta.items():
        for cond in ["source_original", "source_altered", "no_source"]:
            orig = by_key.get((tid, cond, "original_answer"))
            alt = by_key.get((tid, cond, "altered_answer"))
            if orig is None or alt is None:
                continue
            if cond == "source_altered":
                expected_role = "altered_answer"
                margin_t1 = float(orig["nll_T1"]) - float(alt["nll_T1"])
                margin_tf = float(orig["nll_Tfit"]) - float(alt["nll_Tfit"])
                rank_adv = float(orig["rank_mean"]) - float(alt["rank_mean"])
            else:
                expected_role = "original_answer"
                margin_t1 = float(alt["nll_T1"]) - float(orig["nll_T1"])
                margin_tf = float(alt["nll_Tfit"]) - float(orig["nll_Tfit"])
                rank_adv = float(alt["rank_mean"]) - float(orig["rank_mean"])
            row = {
                **meta,
                "condition": cond,
                "expected_role": expected_role,
                "expected_margin_T1": margin_t1,
                "expected_margin_Tfit": margin_tf,
                "expected_correct_T1": bool(margin_t1 > 0.0),
                "expected_correct_Tfit": bool(margin_tf > 0.0),
                "expected_rank_advantage": rank_adv,
                "expected_rank_better": bool(rank_adv > 0.0),
                "original_rank": float(orig["rank_mean"]),
                "altered_rank": float(alt["rank_mean"]),
            }
            if parent_rows is not None:
                po = p_by_key.get((tid, cond, "original_answer"))
                pa = p_by_key.get((tid, cond, "altered_answer"))
                if po is not None and pa is not None:
                    if cond == "source_altered":
                        pm1 = float(po["nll_T1"]) - float(pa["nll_T1"])
                        pmf = float(po["nll_Tfit"]) - float(pa["nll_Tfit"])
                        pr = float(po["rank_mean"]) - float(pa["rank_mean"])
                    else:
                        pm1 = float(pa["nll_T1"]) - float(po["nll_T1"])
                        pmf = float(pa["nll_Tfit"]) - float(po["nll_Tfit"])
                        pr = float(pa["rank_mean"]) - float(po["rank_mean"])
                    row.update({
                        "delta_expected_margin_T1_vs_parent": margin_t1 - pm1,
                        "delta_expected_margin_Tfit_vs_parent": margin_tf - pmf,
                        "delta_expected_rank_advantage_vs_parent": rank_adv - pr,
                    })
            margins.append(row)
    m_by_tc = {(str(m["task_id"]), str(m["condition"])): m for m in margins}
    follow_rows: List[Dict[str, Any]] = []
    for tid, meta in task_meta.items():
        mo = m_by_tc.get((tid, "source_original"))
        ma = m_by_tc.get((tid, "source_altered"))
        mn = m_by_tc.get((tid, "no_source"))
        if mo is None or ma is None:
            continue
        follow_rows.append({
            **meta,
            "source_follow_swing_T1": float(mo["expected_margin_T1"]) + float(ma["expected_margin_T1"]),
            "source_follow_swing_Tfit": float(mo["expected_margin_Tfit"]) + float(ma["expected_margin_Tfit"]),
            "source_follow_rank_swing": float(mo["expected_rank_advantage"]) + float(ma["expected_rank_advantage"]),
            "both_correct_T1": bool(mo["expected_margin_T1"] > 0.0 and ma["expected_margin_T1"] > 0.0),
            "both_correct_Tfit": bool(mo["expected_margin_Tfit"] > 0.0 and ma["expected_margin_Tfit"] > 0.0),
            "both_rank_better": bool(mo["expected_rank_advantage"] > 0.0 and ma["expected_rank_advantage"] > 0.0),
            "no_source_margin_T1": float(mn["expected_margin_T1"]) if mn else None,
            "no_source_margin_Tfit": float(mn["expected_margin_Tfit"]) if mn else None,
            "orig_margin_T1": float(mo["expected_margin_T1"]),
            "alt_margin_T1": float(ma["expected_margin_T1"]),
            "orig_margin_Tfit": float(mo["expected_margin_Tfit"]),
            "alt_margin_Tfit": float(ma["expected_margin_Tfit"]),
        })
    # Parent deltas for source-follow rows.
    if parent_rows is not None:
        parent_summary = summarize_common(parent_rows, None)
        p_follow = {str(r["task_id"]): r for r in parent_summary.get("source_follow_rows", [])}
        for r in follow_rows:
            p = p_follow.get(str(r["task_id"]))
            if p:
                r["delta_source_follow_swing_T1_vs_parent"] = float(r["source_follow_swing_T1"]) - float(p["source_follow_swing_T1"])
                r["delta_source_follow_swing_Tfit_vs_parent"] = float(r["source_follow_swing_Tfit"]) - float(p["source_follow_swing_Tfit"])
                r["delta_source_follow_rank_swing_vs_parent"] = float(r["source_follow_rank_swing"]) - float(p["source_follow_rank_swing"])
    by_condition: Dict[str, Any] = {}
    for cond in ["source_original", "source_altered", "no_source"]:
        g = [m for m in margins if m["condition"] == cond]
        by_condition[cond] = {
            "n": len(g),
            "mean_expected_margin_T1": mean(m["expected_margin_T1"] for m in g),
            "mean_expected_margin_Tfit": mean(m["expected_margin_Tfit"] for m in g),
            "success_T1": sum(1 for m in g if m["expected_margin_T1"] > 0.0),
            "success_Tfit": sum(1 for m in g if m["expected_margin_Tfit"] > 0.0),
            "mean_expected_rank_advantage": mean(m["expected_rank_advantage"] for m in g),
            "rank_better_count": sum(1 for m in g if m["expected_rank_advantage"] > 0.0),
            "mean_delta_margin_T1_vs_parent": mean(m.get("delta_expected_margin_T1_vs_parent") for m in g),
            "mean_delta_margin_Tfit_vs_parent": mean(m.get("delta_expected_margin_Tfit_vs_parent") for m in g),
            "mean_delta_rank_advantage_vs_parent": mean(m.get("delta_expected_rank_advantage_vs_parent") for m in g),
        }
    return {
        "by_condition": by_condition,
        "source_follow": {
            "n": len(follow_rows),
            "both_correct_T1": sum(1 for r in follow_rows if r["both_correct_T1"]),
            "both_correct_Tfit": sum(1 for r in follow_rows if r["both_correct_Tfit"]),
            "both_rank_better": sum(1 for r in follow_rows if r["both_rank_better"]),
            "mean_source_follow_swing_T1": mean(r["source_follow_swing_T1"] for r in follow_rows),
            "mean_source_follow_swing_Tfit": mean(r["source_follow_swing_Tfit"] for r in follow_rows),
            "mean_source_follow_rank_swing": mean(r["source_follow_rank_swing"] for r in follow_rows),
            "mean_delta_source_follow_swing_T1_vs_parent": mean(r.get("delta_source_follow_swing_T1_vs_parent") for r in follow_rows),
            "mean_delta_source_follow_swing_Tfit_vs_parent": mean(r.get("delta_source_follow_swing_Tfit_vs_parent") for r in follow_rows),
            "mean_delta_source_follow_rank_swing_vs_parent": mean(r.get("delta_source_follow_rank_swing_vs_parent") for r in follow_rows),
            "by_split": {
                split: {
                    "n": len(g),
                    "both_correct_T1": sum(1 for r in g if r["both_correct_T1"]),
                    "both_correct_Tfit": sum(1 for r in g if r["both_correct_Tfit"]),
                    "both_rank_better": sum(1 for r in g if r["both_rank_better"]),
                    "mean_swing_T1": mean(r["source_follow_swing_T1"] for r in g),
                    "mean_swing_Tfit": mean(r["source_follow_swing_Tfit"] for r in g),
                    "mean_rank_swing": mean(r["source_follow_rank_swing"] for r in g),
                    "mean_delta_swing_T1_vs_parent": mean(r.get("delta_source_follow_swing_T1_vs_parent") for r in g),
                    "mean_delta_swing_Tfit_vs_parent": mean(r.get("delta_source_follow_swing_Tfit_vs_parent") for r in g),
                    "mean_delta_rank_swing_vs_parent": mean(r.get("delta_source_follow_rank_swing_vs_parent") for r in g),
                }
                for split, g in sorted(((s, [r for r in follow_rows if r.get("split") == s]) for s in set(r.get("split") for r in follow_rows)), key=lambda x: str(x[0]))
            },
        },
        "margins": margins,
        "source_follow_rows": follow_rows,
    }


def compact_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in obj.items() if k not in {"triplets", "margins", "source_follow_rows"}}


def write_markdown(path: pathlib.Path, result: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# research temperature-adjusted source-use readout\n")
    lines.append("\nThis readout fits a single logit temperature on ordinary non-Qwen legal-tail text outside the 80-update prefix, then re-scores bounded source-use tasks. It is not official BabyLM scoring and is not fit on CDI, SuperGLUE, Entity, GlobalPIQA, Reading, or the 25-item source-reversal outcomes.\n")
    lines.append("\n## Temperature fits\n")
    for name, fit in result.get("temperature_fits", {}).items():
        lines.append(f"- `{name}`: T={fit.get('best_temperature')}, calib NLL T1={fit.get('mean_nll_at_T1')}, best={fit.get('best_mean_nll')}, positions={fit.get('n_positions')}, mean rank={fit.get('mean_rank_at_T1')}\n")
    lines.append("\n## Qwen source perturbation\n")
    for name, summ in result.get("model_summaries", {}).items():
        q = summ.get("qwen_source", {}).get("source_specificity", {})
        lines.append(f"- `{name}`: triplets={q.get('n_complete_triplets')}, spec_adv T1={q.get('mean_specific_advantage_T1')}, spec_adv Tfit={q.get('mean_specific_advantage_Tfit')}, rank_adv={q.get('mean_specific_rank_advantage')}, Δspec T1={q.get('mean_delta_specific_advantage_T1_vs_parent')}, Δspec Tfit={q.get('mean_delta_specific_advantage_Tfit_vs_parent')}, Δrank={q.get('mean_delta_specific_rank_advantage_vs_parent')}\n")
    lines.append("\n## Common source-reversal bank\n")
    for name, summ in result.get("model_summaries", {}).items():
        c = summ.get("common_source_reversal", {}).get("source_follow", {})
        lines.append(f"- `{name}`: both T1={c.get('both_correct_T1')}/25, both Tfit={c.get('both_correct_Tfit')}/25, both rank={c.get('both_rank_better')}/25, swing T1={c.get('mean_source_follow_swing_T1')}, swing Tfit={c.get('mean_source_follow_swing_Tfit')}, rank swing={c.get('mean_source_follow_rank_swing')}, Δswing T1={c.get('mean_delta_source_follow_swing_T1_vs_parent')}, Δswing Tfit={c.get('mean_delta_source_follow_swing_Tfit_vs_parent')}, Δrank={c.get('mean_delta_source_follow_rank_swing_vs_parent')}\n")
    interp = result.get("interpretation", {})
    lines.append("\n## Interpretation\n")
    for k, v in interp.items():
        lines.append(f"- **{k}**: {v}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=DEFAULT_TAIL)
    ap.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=["coherent86", "sparse_focus_seed62064", "dense_focus_seed62064", "dense_focus_seed62065"], choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--calib-records", type=int, default=192)
    ap.add_argument("--calib-seed", type=int, default=86031)
    ap.add_argument("--source-max-segments", type=int, default=120)
    ap.add_argument("--source-seed", type=int, default=67067)
    ap.add_argument("--max-targets-per-segment", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--temperature-grid", default="0.70,0.80,0.90,1.00,1.10,1.25,1.40,1.60,1.90")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, min(8, int(os.environ.get("QIUSHI_TORCH_THREADS", "8")))))

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    calib_records, calib_summary = build_calibration_records(
        args.tail_jsonl, tokenizer, n_records=int(args.calib_records), seed=int(args.calib_seed), max_length=int(args.max_length)
    )
    source_records, source_segment_stats = source_probe.build_segment_records(
        argparse.Namespace(tail_jsonl=args.tail_jsonl, max_updates=PREFIX_UPDATES, words_per_update=WORDS_PER_UPDATE,
                           max_segments=int(args.source_max_segments), sample_seed=int(args.source_seed))
    )
    qwen_tasks, qwen_task_stats = source_probe.build_tasks(
        source_records, tokenizer,
        argparse.Namespace(sample_seed=int(args.source_seed), max_targets_per_segment=int(args.max_targets_per_segment), max_length=int(args.max_length))
    )
    common_bank, common_bank_summary = common_probe.materialize_bank(args.labels)
    common_records, common_rec_summary = common_probe.build_scoring_records(tokenizer, common_bank, int(args.max_length))

    write_jsonl(out_dir / "calibration_records.jsonl", (strip_task_for_json(t) for t in calib_records))
    write_jsonl(out_dir / "qwen_source_tasks.jsonl", (strip_task_for_json(t) for t in qwen_tasks))
    write_jsonl(out_dir / "common_source_reversal_records.jsonl", (strip_task_for_json(t) for t in common_records))

    plan = {
        "status": "TEMPERATURE_SOURCE_READOUT_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "models": {m: rel(MODEL_SPECS[m] if MODEL_SPECS[m] is not None else bridge.PARENT_PATH) for m in args.models},
        "calibration": calib_summary,
        "qwen_source_task_stats": {"segment_stats": source_segment_stats, "task_stats": qwen_task_stats},
        "common_source_reversal_stats": {"bank": common_bank_summary, "records": common_rec_summary},
        "temperature_grid": [float(x) for x in args.temperature_grid.split(",") if x.strip()],
        "boundaries": [
            "Temperature is fitted only on ordinary non-Qwen legal-tail text beyond the first 80-update prefix.",
            "This file does not alter or replace official BabyLM scores.",
            "Ranks are invariant to positive temperature scaling and therefore mark order changes rather than confidence scale.",
        ],
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    if args.device == "auto":
        device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    temp_grid = [float(x) for x in args.temperature_grid.split(",") if x.strip()]
    if 1.0 not in temp_grid:
        temp_grid.append(1.0)
        temp_grid.sort()

    temperature_fits: Dict[str, Any] = {}
    model_summaries: Dict[str, Any] = {}
    load_infos: Dict[str, Any] = {}
    qwen_parent_scores: Optional[List[Dict[str, Any]]] = None
    common_parent_scores: Optional[List[Dict[str, Any]]] = None
    all_compact_rows: List[Dict[str, Any]] = []

    for model_name in args.models:
        t0 = time.time()
        model_path = MODEL_SPECS[model_name]
        print(json.dumps({"event": "load_model", "model": model_name, "path": rel(model_path if model_path else bridge.PARENT_PATH), "device": str(device)}), flush=True)
        model, load_info = qview.load_scoring_model(model_path, device, float(args.private_scale))
        load_infos[model_name] = load_info

        calib_logits, calib_targets, _calib_metas = gather_position_logits(model, tokenizer, calib_records, device, int(args.batch_size))
        temp_fit = fit_temperature_from_position_logits(calib_logits, calib_targets, temp_grid)
        temperature_fits[model_name] = temp_fit
        best_t = float(temp_fit.get("best_temperature", 1.0) or 1.0)
        # Free calibration logits before source scoring.
        del calib_logits

        q_scores = score_tasks_with_temperature(model, tokenizer, qwen_tasks, device, int(args.batch_size), best_t)
        c_scores = score_tasks_with_temperature(model, tokenizer, common_records, device, int(args.batch_size), best_t)
        write_jsonl(out_dir / f"qwen_scores_{model_name}.jsonl", q_scores)
        write_jsonl(out_dir / f"common_scores_{model_name}.jsonl", c_scores)
        if model_name == "coherent86":
            qwen_parent_scores = q_scores
            common_parent_scores = c_scores
        q_summ = summarize_qwen_source(q_scores, qwen_parent_scores)
        c_summ = summarize_common(c_scores, common_parent_scores)
        model_summaries[model_name] = {
            "temperature": best_t,
            "elapsed_sec": round(time.time() - t0, 2),
            "qwen_source": q_summ,
            "common_source_reversal": c_summ,
        }
        slim = {
            "temperature": best_t,
            "elapsed_sec": round(time.time() - t0, 2),
            "qwen_source": compact_summary(q_summ),
            "common_source_reversal": compact_summary(c_summ),
        }
        (out_dir / f"summary_{model_name}.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        all_compact_rows.append({
            "model": model_name,
            "temperature": best_t,
            "calib_mean_nll_T1": temp_fit.get("mean_nll_at_T1"),
            "calib_best_mean_nll": temp_fit.get("best_mean_nll"),
            "qwen_delta_spec_T1": q_summ["source_specificity"].get("mean_delta_specific_advantage_T1_vs_parent"),
            "qwen_delta_spec_Tfit": q_summ["source_specificity"].get("mean_delta_specific_advantage_Tfit_vs_parent"),
            "qwen_delta_spec_rank": q_summ["source_specificity"].get("mean_delta_specific_rank_advantage_vs_parent"),
            "common_both_T1": c_summ["source_follow"].get("both_correct_T1"),
            "common_both_Tfit": c_summ["source_follow"].get("both_correct_Tfit"),
            "common_both_rank": c_summ["source_follow"].get("both_rank_better"),
            "common_delta_swing_T1": c_summ["source_follow"].get("mean_delta_source_follow_swing_T1_vs_parent"),
            "common_delta_swing_Tfit": c_summ["source_follow"].get("mean_delta_source_follow_swing_Tfit_vs_parent"),
            "common_delta_rank_swing": c_summ["source_follow"].get("mean_delta_source_follow_rank_swing_vs_parent"),
        })
        print(json.dumps({"event": "model_done", **all_compact_rows[-1], "elapsed_sec": round(time.time() - t0, 2)}, ensure_ascii=False), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (out_dir / "compact_model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_compact_rows[0].keys()) if all_compact_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_compact_rows:
            writer.writerow(r)

    interpretation = {
        "confidence_scale_test": "If a model's larger NLL margins mostly come from global logit scale, fitted-temperature margins should shrink toward the parent while rank advantages stay near zero. Rank movement and source-reversal success surviving temperature indicate changed ordering/selection rather than only confidence scale.",
        "dense_status": "Dense focus should be treated as improved contextual selection only for components that survive the non-benchmark temperature adjustment or appear as rank/order changes; enlarged already-correct margins alone are not enough.",
        "official_scoring": "No official BabyLM score is modified here; this readout is an explanatory frozen-checkpoint analysis that must remain separate from leaderboard arithmetic.",
    }
    result = {
        "status": "TEMPERATURE_SOURCE_READOUT_DONE",
        "created_utc": now(),
        "plan": rel(out_dir / "plan.json"),
        "load_infos": load_infos,
        "temperature_fits": temperature_fits,
        "model_summaries": model_summaries,
        "compact_model_comparison": rel(out_dir / "compact_model_comparison.csv"),
        "interpretation": interpretation,
    }
    out_json = out_dir / "temperature_source_readout.json"
    out_md = out_dir / "temperature_source_readout.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
