#!/usr/bin/env python3
"""research: common source-grounded target probe for concentrated compaction states.

The research concentrated learner readout masked independently selected words in the
inherited and compact views.  That established own-surface adaptation but did not say
how much of the same semantic content transferred.  This probe uses a small hand-built
bank of common cloze statements/questions with fixed targets across the parent,
current-trained, compact-trained, and compact-continued checkpoints.

Each item has an original source context, a controlled altered source context, a common
statement frame, and two candidate answers.  The original source should prefer the
original answer; the altered source should prefer the altered answer.  A no-source
condition records the model's prior over the same frame.  The source-following swing
therefore separates practiced wording from use of the supplied evidence.

This is still a bounded learner diagnostic, not a BabyLM endpoint.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import os
import pathlib
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))
import coherent86_continuation_trainer as base_loader  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402

DEFAULT_LABELS = _public_path('experiments/archive/functional_learning/data/selective_reviewed_labels/selective_semantic_labels.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/common_target_probe')
DEFAULT_STEP57 = _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def finite_mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def finite_median(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


# Common target items.  The pair_id determines the original source text, inherited view,
# and compact view from the research semantic labels.  `replace` defines the controlled
# source intervention; altered_answer should be supported by the altered source.
TASK_DEFS: List[Dict[str, Any]] = [
    # Trained-content transfer: these 17 pair_ids are the faithful_shortening rows used
    # in research concentrated training.  The frame is deliberately not one of the two
    # trained second views.
    {"task_id": "t01_skelton_value", "split": "trained_content", "pair_id": "rw2s0_002308", "axis": "property", "frame": "Skelton Village's north part had extreme environmental [ANS].", "original_answer": "value", "altered_answer": "risk", "replace": ["extreme environmental value", "extreme environmental risk"]},
    {"task_id": "t02_baku_institution", "split": "trained_content", "pair_id": "rw2s0_006924", "axis": "institution", "frame": "The theology dean worked at Baku State [ANS].", "original_answer": "University", "altered_answer": "College", "replace": ["Baku State University", "Baku State College"]},
    {"task_id": "t03_oil_object", "split": "trained_content", "pair_id": "rw2s0_010026", "axis": "object", "frame": "Mosaddegh's policy was to nationalize Iran's oil [ANS].", "original_answer": "industry", "altered_answer": "fields", "replace": ["oil industry", "oil fields"]},
    {"task_id": "t04_casualty_uncertainty", "split": "trained_content", "pair_id": "rw2s0_014542", "axis": "qualification", "frame": "The number of Protestants murdered in the outbreak was [ANS].", "original_answer": "uncertain", "altered_answer": "certain", "replace": ["is uncertain", "is certain"]},
    {"task_id": "t05_manning_role", "split": "trained_content", "pair_id": "rw2s0_019304", "axis": "role", "frame": "Ed Manning was a professional and college [ANS].", "original_answer": "coach", "altered_answer": "scout", "replace": ["professional and college coach", "professional and college scout"]},
    {"task_id": "t06_mass_day", "split": "trained_content", "pair_id": "rw2s0_019852", "axis": "time", "frame": "The speaker went to Mass that [ANS].", "original_answer": "Sunday", "altered_answer": "Monday", "replace": ["that Sunday", "that Monday"]},
    {"task_id": "t07_music_change", "split": "trained_content", "pair_id": "rw2s0_020973", "axis": "ordered_change", "frame": "The tune could change from fast to [ANS].", "original_answer": "slow", "altered_answer": "loud", "replace": ["fast to a slow tune", "fast to a loud tune"]},
    {"task_id": "t08_abingdon_refusal", "split": "trained_content", "pair_id": "rw2s0_025051", "axis": "polarity_action", "frame": "The Abingdon workers [ANS] to return.", "original_answer": "refused", "altered_answer": "agreed", "replace": ["refused to go back", "agreed to go back"]},
    {"task_id": "t09_route_opposite", "split": "trained_content", "pair_id": "rw2s0_029078", "axis": "spatial_relation", "frame": "After the barber shop, the route says to take the street [ANS] the barber shop.", "original_answer": "opposite", "altered_answer": "beside", "replace": ["opposite to it", "beside it"]},
    {"task_id": "t10_clavering_number", "split": "trained_content", "pair_id": "rw2s1_005600", "axis": "quantity", "frame": "Clavering believed [ANS] was the perfect number of pupils.", "original_answer": "twenty", "altered_answer": "thirty", "replace": ["twenty", "thirty"]},
    {"task_id": "t11_horsemen_weapon", "split": "trained_content", "pair_id": "rw2s1_015912", "axis": "object", "frame": "The horsemen lowered their [ANS].", "original_answer": "lances", "altered_answer": "swords", "replace": ["lances", "swords"]},
    {"task_id": "t12_unseen_scandal_count", "split": "trained_content", "pair_id": "rw2s1_024034", "axis": "quantity", "frame": "The unseen scandal involved about [ANS] young people.", "original_answer": "120,000", "altered_answer": "80,000", "replace": ["a hundred and twenty thousand", "eighty thousand"]},
    {"task_id": "t13_bank_threshold", "split": "trained_content", "pair_id": "rw_004509", "axis": "quantity", "frame": "Currency transaction reports were required for money worth [ANS] or more.", "original_answer": "$10,000", "altered_answer": "$1,000", "replace": ["$10,000", "$1,000"]},
    {"task_id": "t14_radio_channel", "split": "trained_content", "pair_id": "rw_015472", "axis": "quantity_identifier", "frame": "The Chris Evans Breakfast Show was on BBC Radio [ANS].", "original_answer": "2", "altered_answer": "4", "replace": ["BBC Radio 2", "BBC Radio 4"]},
    {"task_id": "t15_noontide_press", "split": "trained_content", "pair_id": "rw_017379", "axis": "institution", "frame": "Von Brunn briefly worked for [ANS] Press.", "original_answer": "Noontide", "altered_answer": "Sunrise", "replace": ["Noontide Press", "Sunrise Press"]},
    {"task_id": "t16_taylor_location", "split": "trained_content", "pair_id": "rw_031652", "axis": "location", "frame": "Temporary traffic lights were at Taylor's [ANS].", "original_answer": "wood", "altered_answer": "bridge", "replace": ["Taylor's wood", "Taylor's bridge"]},
    {"task_id": "t17_abraham_relation", "split": "trained_content", "pair_id": "rw_041183", "axis": "role_relation", "frame": "Abraham married his beautiful [ANS] Sarai.", "original_answer": "sister", "altered_answer": "mother", "replace": ["beautiful sister Sarai", "beautiful mother Sarai"]},
    # Held-source/common-expression transfer: not used in research training.  These are
    # source-grounded controls for whether the small adaptation improves a reusable
    # source-use procedure beyond the trained source IDs.
    {"task_id": "h01_gas_price", "split": "held_source", "pair_id": "rw2s1_000216", "axis": "quantity", "frame": "Gas prices were rising by [ANS] in March.", "original_answer": "7.5%", "altered_answer": "5%", "replace": ["7.5%", "5%"]},
    {"task_id": "h02_rating", "split": "held_source", "pair_id": "rw_000248", "axis": "quantity", "frame": "Renee Schonfeld rated the film [ANS] out of 5 stars.", "original_answer": "3", "altered_answer": "4", "replace": ["3 out of 5", "4 out of 5"]},
    {"task_id": "h03_protest_duration", "split": "held_source", "pair_id": "rw_001604", "axis": "quantity_time", "frame": "The protest period was [ANS].", "original_answer": "a thousand days", "altered_answer": "a year", "replace": ["a thousand days", "a year"]},
    {"task_id": "h04_cherry_food", "split": "held_source", "pair_id": "rw2s0_009985", "axis": "object", "frame": "On Saturday he ate a piece of cherry [ANS].", "original_answer": "pie", "altered_answer": "cake", "replace": ["cherry pie", "cherry cake"]},
    {"task_id": "h05_eagle_feathers", "split": "held_source", "pair_id": "rw2s1_013643", "axis": "quantity", "frame": "The seizure involved [ANS] eagle feathers.", "original_answer": "50", "altered_answer": "60", "replace": ["50 eagle feathers", "60 eagle feathers"]},
    {"task_id": "h06_ramadan_food", "split": "held_source", "pair_id": "rw2s1_017873", "axis": "object", "frame": "At the end of Ramadan, Muslims received salt and pepper on a fruit [ANS].", "original_answer": "salad", "altered_answer": "cake", "replace": ["fruit salad", "fruit cake"]},
    {"task_id": "h07_modern_art_city", "split": "held_source", "pair_id": "rw2s1_002259", "axis": "location", "frame": "The piece could go in the Museum of Modern Art in [ANS].", "original_answer": "New York", "altered_answer": "Boston", "replace": ["New York", "Boston"]},
    {"task_id": "h08_navarino_year", "split": "held_source", "pair_id": "rw2s0_004620", "axis": "quantity_time", "frame": "The Navarino event occurred on October 20 [ANS].", "original_answer": "1827", "altered_answer": "1828", "replace": ["October 20 1827", "October 20 1828"]},
]


def materialize_bank(labels_path: pathlib.Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = load_jsonl(labels_path)
    by_id: Dict[str, Dict[str, Any]] = {str(r["pair_id"]): r for r in rows}
    bank: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    for d in TASK_DEFS:
        pid = d["pair_id"]
        src_row = by_id.get(pid)
        if src_row is None:
            issues.append({"task_id": d["task_id"], "issue": "missing_pair_id", "pair_id": pid})
            continue
        source = str(src_row.get("original", src_row.get("source_text", ""))).strip()
        old, new = d["replace"]
        altered = source.replace(old, new, 1)
        if altered == source:
            issues.append({"task_id": d["task_id"], "issue": "replacement_not_found", "pair_id": pid, "old": old})
            continue
        item = {
            **{k: v for k, v in d.items() if k != "replace"},
            "source_corpus": src_row.get("source_corpus", src_row.get("source", "")),
            "semantic_label": src_row.get("label"),
            "source_text": source,
            "altered_source_text": altered,
            "current_rewrite": src_row.get("current_rewrite", ""),
            "compact_rewrite": src_row.get("compact_rewrite", ""),
            "source_edit": {"old": old, "new": new},
        }
        bank.append(item)
    summary = {
        "n_task_defs": len(TASK_DEFS),
        "n_materialized": len(bank),
        "issues": issues,
        "by_split": dict(Counter(x["split"] for x in bank)),
        "by_axis": dict(Counter(x["axis"] for x in bank)),
    }
    return bank, summary


def build_candidate_record(tokenizer, item: Dict[str, Any], condition: str, candidate_role: str,
                           candidate: str, max_length: int) -> Optional[Dict[str, Any]]:
    if condition == "source_original":
        context = item["source_text"]
    elif condition == "source_altered":
        context = item["altered_source_text"]
    elif condition == "no_source":
        context = ""
    else:
        raise ValueError(condition)
    frame = item["frame"]
    if "[ANS]" not in frame:
        raise ValueError(f"Frame lacks [ANS]: {frame}")
    before, after = frame.split("[ANS]", 1)
    prefix = ("Passage: " + context.strip() + "\n") if context else ""
    stem = prefix + "Statement: " + before
    full_text = stem + candidate + after
    ans_start = len(stem)
    ans_end = ans_start + len(candidate)
    enc = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=int(max_length), return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    pos: List[int] = []
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if a < ans_end and b > ans_start:
            pos.append(i)
    if not pos:
        return None
    if max(pos) >= len(ids):
        return None
    return {
        "task_id": item["task_id"],
        "pair_id": item["pair_id"],
        "split": item["split"],
        "axis": item["axis"],
        "semantic_label": item.get("semantic_label"),
        "condition": condition,
        "candidate_role": candidate_role,
        "candidate": candidate,
        "input_ids": ids,
        "positions": pos,
        "n_positions": len(pos),
        "frame": item["frame"],
    }


def build_scoring_records(tokenizer, bank: List[Dict[str, Any]], max_length: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    skips = Counter()
    for item in bank:
        for condition in ["source_original", "source_altered", "no_source"]:
            for role, cand in [("original_answer", item["original_answer"]), ("altered_answer", item["altered_answer"] )]:
                rec = build_candidate_record(tokenizer, item, condition, role, str(cand), int(max_length))
                if rec is None:
                    skips[f"{condition}_{role}_no_position"] += 1
                    continue
                records.append(rec)
    return records, {"n_candidate_records": len(records), "skips": dict(skips)}


def score_records(model, tokenizer, records: List[Dict[str, Any]], device: torch.device, batch_size: int) -> List[Dict[str, Any]]:
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    mask_id = int(tokenizer.mask_token_id)
    out_rows: List[Dict[str, Any]] = []
    for start in range(0, len(records), int(batch_size)):
        batch = records[start:start + int(batch_size)]
        max_len = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        orig = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            arr = torch.tensor(r["input_ids"], dtype=torch.long, device=device)
            ids[i, :L] = arr
            orig[i, :L] = arr
            att[i, :L] = 1
            for p in r["positions"]:
                if p < L:
                    ids[i, p] = mask_id
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        for i, r in enumerate(batch):
            vals: List[float] = []
            for p in r["positions"]:
                if p >= logits.shape[1]:
                    continue
                lp = torch.log_softmax(logits[i, p], dim=-1)
                vals.append(-float(lp[int(orig[i, p])].detach().cpu()))
            rr = {k: v for k, v in r.items() if k not in {"input_ids", "positions"}}
            rr["nll"] = sum(vals) / len(vals) if vals else float("nan")
            rr["n_token_scores"] = len(vals)
            out_rows.append(rr)
        del ids, att, orig, logits
    return out_rows


def summarize_model_scores(score_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Candidate NLL map.
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in score_rows:
        by_key[(r["task_id"], r["condition"], r["candidate_role"])] = r
    margins: List[Dict[str, Any]] = []
    task_meta: Dict[str, Dict[str, Any]] = {}
    for r in score_rows:
        task_meta.setdefault(r["task_id"], {k: r.get(k) for k in ["task_id", "pair_id", "split", "axis", "semantic_label", "frame"]})
    for task_id, meta in task_meta.items():
        for condition in ["source_original", "source_altered", "no_source"]:
            orig = by_key.get((task_id, condition, "original_answer"))
            alt = by_key.get((task_id, condition, "altered_answer"))
            if orig is None or alt is None:
                continue
            if condition == "source_altered":
                expected_margin = float(orig["nll"]) - float(alt["nll"])  # altered answer should win.
                expected_role = "altered_answer"
            else:
                expected_margin = float(alt["nll"]) - float(orig["nll"])  # original answer should win.
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
    # Per-task contextual flip: original source prefers original AND altered source prefers altered.
    m_by_tc: Dict[Tuple[str, str], Dict[str, Any]] = {(m["task_id"], m["condition"]): m for m in margins}
    flip_rows = []
    for task_id, meta in task_meta.items():
        mo = m_by_tc.get((task_id, "source_original"))
        ma = m_by_tc.get((task_id, "source_altered"))
        mn = m_by_tc.get((task_id, "no_source"))
        if mo and ma:
            flip_rows.append({
                **meta,
                "source_original_margin": mo["expected_margin"],
                "source_altered_margin": ma["expected_margin"],
                "no_source_original_margin": mn["expected_margin"] if mn else None,
                "source_follow_swing": mo["expected_margin"] + ma["expected_margin"],
                "both_source_conditions_correct": bool(mo["expected_margin"] > 0.0 and ma["expected_margin"] > 0.0),
                "altered_flips_against_no_source": bool(ma["expected_margin"] > 0.0 and (mn is None or mn["expected_margin"] >= 0.0)),
            })

    def group_summary(rows: List[Dict[str, Any]], key: str, value_field: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for val in sorted(set(str(r.get(key)) for r in rows)):
            g = [r for r in rows if str(r.get(key)) == val]
            out[val] = {
                "n": len(g),
                "mean": finite_mean(r.get(value_field) for r in g),
                "median": finite_median(r.get(value_field) for r in g),
                "success": sum(1 for r in g if r.get("expected_correct") or r.get("both_source_conditions_correct")),
            }
        return out

    by_condition = {}
    for cond in ["source_original", "source_altered", "no_source"]:
        g = [m for m in margins if m["condition"] == cond]
        by_condition[cond] = {
            "n": len(g),
            "mean_expected_margin": finite_mean(m["expected_margin"] for m in g),
            "median_expected_margin": finite_median(m["expected_margin"] for m in g),
            "success": sum(1 for m in g if m["expected_margin"] > 0.0),
            "by_split": group_summary(g, "split", "expected_margin"),
            "by_axis": group_summary(g, "axis", "expected_margin"),
        }
    summary = {
        "by_condition": by_condition,
        "source_follow": {
            "n": len(flip_rows),
            "both_source_conditions_correct": sum(1 for r in flip_rows if r["both_source_conditions_correct"]),
            "mean_source_follow_swing": finite_mean(r["source_follow_swing"] for r in flip_rows),
            "median_source_follow_swing": finite_median(r["source_follow_swing"] for r in flip_rows),
            "by_split": {
                split: {
                    "n": len(g),
                    "both_correct": sum(1 for r in g if r["both_source_conditions_correct"]),
                    "mean_swing": finite_mean(r["source_follow_swing"] for r in g),
                    "mean_orig_margin": finite_mean(r["source_original_margin"] for r in g),
                    "mean_altered_margin": finite_mean(r["source_altered_margin"] for r in g),
                    "mean_no_source_original_margin": finite_mean(r["no_source_original_margin"] for r in g),
                }
                for split, g in sorted(((s, [r for r in flip_rows if r["split"] == s]) for s in set(r["split"] for r in flip_rows)), key=lambda x: x[0])
            },
            "by_axis": {
                axis: {
                    "n": len(g),
                    "both_correct": sum(1 for r in g if r["both_source_conditions_correct"]),
                    "mean_swing": finite_mean(r["source_follow_swing"] for r in g),
                }
                for axis, g in sorted(((a, [r for r in flip_rows if r["axis"] == a]) for a in set(r["axis"] for r in flip_rows)), key=lambda x: x[0])
            },
        },
        "margins": margins,
        "source_follow_rows": flip_rows,
    }
    return summary


def load_scoring_model(model_name: str, model_path: Optional[pathlib.Path], device: torch.device, private_scale: float):
    endpoint = pathlib.Path(model_path) if model_path is not None else bridge.PARENT_PATH
    model, missing, unexpected = base_loader.load_model(endpoint, device, 128, float(private_scale))
    # The custom model stores private adapter scale in the module, not only in config.
    try:
        for layer in model.deberta.encoder.layer:
            layer.private_adapter.scale = float(private_scale)
            layer.private_adapter.enabled = True
    except Exception:
        pass
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    model.eval()
    ident = bridge.model_identity(model)
    return model, {"endpoint": rel(endpoint), "missing": list(missing), "unexpected": list(unexpected), "identity": ident}


def compare_to_parent(model_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if "parent" not in model_summaries:
        return {}
    parent_rows = {(m["task_id"], m["condition"]): m for m in model_summaries["parent"]["margins"]}
    out: Dict[str, Any] = {}
    for name, summ in model_summaries.items():
        if name == "parent":
            continue
        deltas = []
        for m in summ["margins"]:
            p = parent_rows.get((m["task_id"], m["condition"]))
            if p is None:
                continue
            deltas.append({
                "task_id": m["task_id"],
                "pair_id": m["pair_id"],
                "split": m["split"],
                "axis": m["axis"],
                "condition": m["condition"],
                "delta_expected_margin_vs_parent": float(m["expected_margin"]) - float(p["expected_margin"]),
            })
        out[name] = {
            "n": len(deltas),
            "mean_delta_expected_margin_vs_parent": finite_mean(d["delta_expected_margin_vs_parent"] for d in deltas),
            "by_condition": {
                cond: {
                    "n": len(g),
                    "mean_delta": finite_mean(d["delta_expected_margin_vs_parent"] for d in g),
                    "median_delta": finite_median(d["delta_expected_margin_vs_parent"] for d in g),
                }
                for cond, g in sorted(((c, [d for d in deltas if d["condition"] == c]) for c in set(d["condition"] for d in deltas)), key=lambda x: x[0])
            },
            "by_split": {
                split: {
                    "n": len(g),
                    "mean_delta": finite_mean(d["delta_expected_margin_vs_parent"] for d in g),
                    "median_delta": finite_median(d["delta_expected_margin_vs_parent"] for d in g),
                }
                for split, g in sorted(((s, [d for d in deltas if d["split"] == s]) for s in set(d["split"] for d in deltas)), key=lambda x: x[0])
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="CPU is enough for this small scoring bank; auto uses cuda if available.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--include-compact-wordmatched", action="store_true")
    ap.add_argument("--extra-model", action="append", default=[], help="name=path for additional checkpoints, e.g. compact92_continuation=Sessions/...")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask token")
    bank, bank_summary = materialize_bank(args.labels)
    scoring_records, rec_summary = build_scoring_records(tokenizer, bank, int(args.max_length))
    write_jsonl(out_dir / "common_target_bank.jsonl", bank)
    write_jsonl(out_dir / "candidate_scoring_records.jsonl", [{k: v for k, v in r.items() if k not in {"input_ids"}} for r in scoring_records])
    plan = {
        "status": "COMMON_TARGET_PROBE_PLAN",
        "created_utc": now(),
        "scientific_purpose": "Evaluate common source-grounded targets, with controlled source changes, across research concentrated compaction states.",
        "bank_summary": bank_summary,
        "scoring_record_summary": rec_summary,
        "model_paths": {
            "parent": rel(bridge.PARENT_PATH),
            "current_equal_epoch": rel(_public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/current_equal_epoch/checkpoint_final')),
            "compact_equal_epoch": rel(_public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/compact_equal_epoch/checkpoint_final')),
            "compact_wordmatched": rel(_public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/compact_wordmatched/checkpoint_final')),
        },
        "interpretation_boundaries": [
            "Trained-content items test expression transfer of the same source IDs used in research, not held-source generalization.",
            "Held-source items test whether the small adaptation changed a reusable source-use procedure; these sources were not in the research concentrated training subset.",
            "Controlled altered-source items distinguish following supplied evidence from a prior or memorized answer.",
            "This probe scores cloze candidates and does not by itself measure broad BabyLM competence or official Overall.",
        ],
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    if args.device == "auto":
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(json.dumps({"event": "device_selected", "device": str(device), "n_scoring_records": len(scoring_records)}), flush=True)

    model_specs: List[Tuple[str, Optional[pathlib.Path]]] = [
        ("parent", None),
        ("current_equal_epoch", _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/current_equal_epoch/checkpoint_final')),
        ("compact_equal_epoch", _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/compact_equal_epoch/checkpoint_final')),
    ]
    if args.include_compact_wordmatched:
        model_specs.append(("compact_wordmatched", _public_path('experiments/archive/functional_learning/data/concentrated_compact_learning_schedulematched/compact_wordmatched/checkpoint_final')))
    for spec in args.extra_model:
        if "=" not in spec:
            raise ValueError(f"--extra-model must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        model_specs.append((name, pathlib.Path(path)))

    model_summaries: Dict[str, Dict[str, Any]] = {}
    all_score_rows: List[Dict[str, Any]] = []
    load_infos: Dict[str, Any] = {}
    for name, path in model_specs:
        t0 = time.time()
        print(json.dumps({"event": "load_model", "model": name, "path": rel(path) if path else rel(bridge.PARENT_PATH)}), flush=True)
        model, load_info = load_scoring_model(name, path, device, float(args.private_scale))
        load_infos[name] = load_info
        scores = score_records(model, tokenizer, scoring_records, device, int(args.batch_size))
        for r in scores:
            r["model"] = name
        write_jsonl(out_dir / f"scores_{name}.jsonl", scores)
        summary = summarize_model_scores(scores)
        model_summaries[name] = summary
        # Write compact per-model summary without row-heavy details.
        slim = {k: v for k, v in summary.items() if k not in {"margins", "source_follow_rows"}}
        (out_dir / f"summary_{name}.json").write_text(json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        all_score_rows.extend(scores)
        print(json.dumps({"event": "model_done", "model": name, "elapsed_sec": round(time.time() - t0, 1), "source_follow": slim.get("source_follow")}, ensure_ascii=False), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(out_dir / "scores_all_models.jsonl", all_score_rows)
    deltas = compare_to_parent(model_summaries)
    # Flatten a concise CSV of source-follow rows.
    with (out_dir / "source_follow_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "task_id", "pair_id", "split", "axis", "source_original_margin", "source_altered_margin", "no_source_original_margin", "source_follow_swing", "both_source_conditions_correct"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for model_name, summ in model_summaries.items():
            for r in summ["source_follow_rows"]:
                w.writerow({k: (model_name if k == "model" else r.get(k)) for k in fieldnames})
    final = {
        "status": "COMMON_TARGET_PROBE_DONE",
        "created_utc": now(),
        "plan": rel(out_dir / "plan.json"),
        "common_target_bank": rel(out_dir / "common_target_bank.jsonl"),
        "scores_all_models": rel(out_dir / "scores_all_models.jsonl"),
        "source_follow_csv": rel(out_dir / "source_follow_summary.csv"),
        "bank_summary": bank_summary,
        "scoring_record_summary": rec_summary,
        "load_infos": load_infos,
        "model_summaries": {name: {k: v for k, v in summ.items() if k not in {"margins", "source_follow_rows"}} for name, summ in model_summaries.items()},
        "deltas_vs_parent": deltas,
        "scientific_status": "common-target learner diagnostic; interpret with research own-view results and do not treat as a legal BabyLM endpoint",
    }
    (out_dir / "summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
