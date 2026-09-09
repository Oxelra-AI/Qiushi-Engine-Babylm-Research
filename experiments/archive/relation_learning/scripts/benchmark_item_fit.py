#!/usr/bin/env python3
"""research: benchmark item-fit, depth, and role stratification.

Scientific purpose
------------------
research/006 left a sharp fork. VIEW beats CLEAN/REPEAT on deep Entity
state-update items while losing shallow repetition recall, but this only supports a
representation/learning-computation account if it is not just lower language-model
loss on the same benchmark text. This script measures an efficient, untempered MLM
fit signal on the benchmark item text itself.

What is measured
----------------
* Entity option-completion margins by operation depth (mask all completion tokens in
  a candidate at once, compare raw log-prob sums; no temperature calibration).
* Entity role fit by operation depth: initial-state sentence, operation sentences,
  and query/gold-answer span. The role split directly tests whether a register-fit
  account can explain a depth-gradient by lower loss specifically on operation
  sentences rather than by an undifferentiated benchmark-register average.
* COMPS candidate margins on a deterministic stratified subset of the four COMPS
  files, and Reading target-word loss on all rows.

This is not an official evaluator and does not upload anything. It is a mechanism
instrument: the same scoring rule is applied to all arms/checkpoints, so relative
V-C/V-R/C-R contrasts are the scientific object.
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
import re
import statistics
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
OUT = WS / "data" / "benchmark_item_fit"
DATA_ROOT = ROOT / "experiments/archive" / 'representation_and_objectives' / "data" / "pristine_official_coordinate" / "babylm-eval" / "strict" / "evaluation_data" / "full_eval"
frontier_consolidation_RUNS = ROOT / "experiments/archive" / 'frontier_consolidation' / "training" / "runs"

ARM_CONFIGS = {
    "D_V_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_C_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_R_43022": frontier_consolidation_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_V_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_C_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_R_43122": frontier_consolidation_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
}
CKS = ["chck_80M", "chck_90M", "chck_100M"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r.")
    return s


def split_entity_prefix(prefix: str) -> tuple[str, list[str], str]:
    parts = [x.strip() for x in re.split(r"(?<=\.)\s+", prefix.strip()) if x.strip()]
    if len(parts) < 2:
        return prefix.strip(), [], ""
    initial = parts[0]
    query = parts[-1]
    ops = parts[1:-1]
    return initial, ops, query


def parse_initial_contents(initial_sentence: str) -> dict[int, str]:
    s = initial_sentence.strip()
    if s.endswith("."):
        s = s[:-1]
    out: dict[int, str] = {}
    for m in re.finditer(r"Box\s+(\d+)\s+contains\s+(.*?)(?=,\s*Box\s+\d+\s+contains\s+|$)", s):
        out[int(m.group(1))] = m.group(2).strip()
    return out


def parse_query_box(query_prefix: str) -> int | None:
    m = re.search(r"Box\s+(\d+)\s+contains\s*$", query_prefix.strip())
    return int(m.group(1)) if m else None


def entity_filtered(path: pathlib.Path):
    for obj in read_jsonl(path):
        if any("nothing" in str(option).lower() for option in obj.get("options", [])):
            continue
        yield obj


def load_entity_items(max_per_depth_type: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    counters: dict[tuple[str, int], int] = defaultdict(int)
    for fn in ["regular.jsonl", "ambiref.jsonl", "move_contents.jsonl"]:
        typ = fn[:-6]
        for obj in entity_filtered(DATA_ROOT / "entity_tracking" / fn):
            depth = int(obj["numops"])
            key = (typ, depth)
            if max_per_depth_type is not None and counters[key] >= max_per_depth_type:
                continue
            uid = f"{typ}_{depth}_ops"
            initial, ops, query = split_entity_prefix(obj["input_prefix"])
            qbox = parse_query_box(query)
            initial_map = parse_initial_contents(initial)
            stale_initial = initial_map.get(qbox) if qbox is not None else None
            options = [str(x) for x in obj["options"]]
            counters[key] += 1
            items.append({
                "task": "Entity",
                "uid": uid,
                "entity_type": typ,
                "numops": depth,
                "uid_index": counters[key] - 1,
                "id": f"{typ}:{depth}:{counters[key]-1}:sample{obj.get('sample_id')}",
                "prefix": obj["input_prefix"],
                "options": options,
                "label": 0,
                "gold": options[0],
                "initial_sentence": initial,
                "operation_text": " ".join(ops).strip(),
                "query_prefix": query,
                "query_box": qbox,
                "stale_initial": stale_initial,
                "stale_available": int(stale_initial is not None and any(norm_text(o) == norm_text(stale_initial) for o in options)),
                "stale_is_gold": int(stale_initial is not None and norm_text(stale_initial) == norm_text(options[0])),
            })
    return items


def load_comps_items(max_total: int | None) -> list[dict[str, Any]]:
    specs = [
        ("comps_base.jsonl", "base"),
        ("comps_wugs.jsonl", "wugs"),
        ("comps_wugs_dist-before.jsonl", "wugs_dist_before"),
        ("comps_wugs_dist-in-between.jsonl", "wugs_dist_in_between"),
    ]
    per_file = None if max_total is None else max(1, max_total // len(specs))
    items: list[dict[str, Any]] = []
    for fn, subset in specs:
        rows = list(read_jsonl(DATA_ROOT / "comps" / fn))
        if per_file is not None and len(rows) > per_file:
            # Deterministic evenly-spaced sample avoids a prefix-only slice.
            idxs = np.linspace(0, len(rows) - 1, per_file).round().astype(int).tolist()
            rows = [rows[i] for i in idxs]
        for i, obj in enumerate(rows):
            acc = " ".join([str(obj["prefix_acceptable"]), str(obj["property_phrase"])]).strip()
            unacc = " ".join([str(obj["prefix_unacceptable"]), str(obj["property_phrase"])]).strip()
            items.append({
                "task": "COMPS",
                "uid": subset,
                "subset": subset,
                "negative_sample_type": obj.get("negative_sample_type"),
                "distraction_type": obj.get("distraction_type", "base"),
                "id": f"{subset}:{obj.get('id', i)}",
                "options": [acc, unacc],
                "label": 0,
                "gold": acc,
            })
    return items


def load_reading_items(max_total: int | None = None) -> list[dict[str, Any]]:
    p = DATA_ROOT / "reading" / "reading_data.csv"
    items: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            context = (row.get("item") or "").strip()
            word = (row.get("word") or "").strip()
            text = (context + " " + word).strip()
            items.append({
                "task": "Reading",
                "uid": "reading",
                "id": f"{row.get('item_id')}:{row.get('sent_id')}:{i}",
                "context": context,
                "word": word,
                "options": [word],
                "label": 0,
                "gold": word,
                "text": text,
                "context_length": int(row.get("context_length") or 0),
                "length": int(row.get("length") or 0),
            })
            if max_total is not None and len(items) >= max_total:
                break
    return items


def encode_mask_span(tokenizer, text: str, start: int, end: int, meta: dict[str, Any]) -> dict[str, Any] | None:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, truncation=True, max_length=512)
    ids = list(enc["input_ids"])
    att = list(enc["attention_mask"])
    offs = enc["offset_mapping"]
    positions: list[int] = []
    targets: list[int] = []
    for i, (a, b) in enumerate(offs):
        # Skip special tokens and any token that does not overlap the requested span.
        if b <= a:
            continue
        if b > start and a < end:
            positions.append(i)
            targets.append(ids[i])
    if not positions:
        return None
    masked = list(ids)
    for p in positions:
        masked[p] = tokenizer.mask_token_id
    return {"input_ids": masked, "attention_mask": att, "positions": positions, "targets": targets, **meta}


def build_records(tokenizer, task: str, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_records: list[dict[str, Any]] = []
    cand_records: list[dict[str, Any]] = []
    for item_i, item in enumerate(items):
        if task == "Entity":
            depth = int(item["numops"])
            typ = item["entity_type"]
            base = {
                "item_i": item_i,
                "task": "Entity",
                "uid": item["uid"],
                "id": item["id"],
                "numops": depth,
                "entity_type": typ,
            }
            # Candidate completions: raw untempered margin on answer options.
            for cand_i, option in enumerate(item["options"]):
                text = item["prefix"] + option
                rec = encode_mask_span(tokenizer, text, len(item["prefix"]), len(text), {**base, "record_type": "candidate", "cand_i": cand_i, "role": "candidate_answer"})
                if rec is not None:
                    cand_records.append(rec)
            # Role-fit records. Initial text is identical register across depths; operation_text
            # exists only for numops>0; query span is the gold answer in the final query.
            init = item["initial_sentence"].strip()
            if init:
                rec = encode_mask_span(tokenizer, init, 0, len(init), {**base, "record_type": "role_fit", "role": "initial_state"})
                if rec is not None:
                    fit_records.append(rec)
            ops = item["operation_text"].strip()
            if ops:
                rec = encode_mask_span(tokenizer, ops, 0, len(ops), {**base, "record_type": "role_fit", "role": "operation_sentences"})
                if rec is not None:
                    fit_records.append(rec)
            query_text = item["query_prefix"] + item["gold"]
            rec = encode_mask_span(tokenizer, query_text, len(item["query_prefix"]), len(query_text), {**base, "record_type": "role_fit", "role": "query_gold_answer"})
            if rec is not None:
                fit_records.append(rec)
        elif task == "COMPS":
            base = {
                "item_i": item_i,
                "task": "COMPS",
                "uid": item["uid"],
                "id": item["id"],
                "subset": item["subset"],
                "negative_sample_type": item.get("negative_sample_type"),
                "distraction_type": item.get("distraction_type"),
            }
            for cand_i, option in enumerate(item["options"]):
                rec = encode_mask_span(tokenizer, option, 0, len(option), {**base, "record_type": "candidate", "cand_i": cand_i, "role": "candidate_sentence"})
                if rec is not None:
                    cand_records.append(rec)
        elif task == "Reading":
            base = {
                "item_i": item_i,
                "task": "Reading",
                "uid": "reading",
                "id": item["id"],
                "context_length": item["context_length"],
                "length": item["length"],
            }
            start = len(item["context"] + " ") if item["context"] else 0
            rec = encode_mask_span(tokenizer, item["text"], start, len(item["text"]), {**base, "record_type": "role_fit", "role": "target_word"})
            if rec is not None:
                fit_records.append(rec)
    return fit_records, cand_records


@torch.no_grad()
def score_records(model, records: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    if not records:
        return []
    pad_id = 0
    try:
        pad_id = int(model.config.pad_token_id or 0)
    except Exception:
        pass
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        max_len = max(len(r["input_ids"]) for r in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            input_ids[i, :L] = torch.tensor(r["input_ids"], dtype=torch.long)
            attn[i, :L] = torch.tensor(r["attention_mask"], dtype=torch.long)
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        for i, r in enumerate(batch):
            pos = torch.tensor(r["positions"], dtype=torch.long, device=device)
            tgt = torch.tensor(r["targets"], dtype=torch.long, device=device)
            vals = log_probs[i, pos, tgt].detach().cpu().numpy().astype(float)
            meta = {k: v for k, v in r.items() if k not in {"input_ids", "attention_mask", "positions", "targets"}}
            meta.update({
                "n_masked_tokens": int(len(vals)),
                "logprob_sum": float(np.sum(vals)),
                "nll_per_token": float(-np.mean(vals)),
            })
            rows.append(meta)
    return rows


def aggregate_candidates(cand_rows: list[dict[str, Any]], items_by_task: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in cand_rows:
        by_key[(r["task"], int(r["item_i"]))][int(r["cand_i"])] = r
    out: list[dict[str, Any]] = []
    for (task, item_i), candmap in sorted(by_key.items()):
        item = items_by_task[task][item_i]
        if 0 not in candmap:
            continue
        cand_sums = {c: row["logprob_sum"] for c, row in candmap.items()}
        pred = max(cand_sums, key=lambda c: cand_sums[c])
        non_gold_best = max([v for c, v in cand_sums.items() if c != 0], default=float("nan"))
        margin = cand_sums[0] - non_gold_best if math.isfinite(non_gold_best) else float("nan")
        base = dict(candmap[0])
        keep = {k: base.get(k) for k in ["task", "uid", "id", "numops", "entity_type", "subset", "negative_sample_type", "distraction_type"] if k in base}
        keep.update({
            "item_i": item_i,
            "gold_nll_per_token": base["nll_per_token"],
            "gold_logprob_sum": base["logprob_sum"],
            "gold_completion_tokens": base["n_masked_tokens"],
            "margin_logprob": float(margin),
            "pred_idx": int(pred),
            "correct": int(pred == 0),
            "n_candidates_scored": len(candmap),
        })
        if task == "Entity":
            stale = item.get("stale_initial")
            keep["stale_available"] = int(item.get("stale_available", 0))
            keep["stale_is_gold"] = int(item.get("stale_is_gold", 0))
            keep["pred_is_stale_initial"] = int(stale is not None and pred < len(item["options"]) and norm_text(item["options"][pred]) == norm_text(stale))
        out.append(keep)
    return out


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return float(statistics.mean(xs)) if xs else float("nan")


def pstdev(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def summarize_fit(rows: list[dict[str, Any]], choice_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        task = r["task"]
        arm = r["arm"]
        ck = r["checkpoint"]
        role = r["role"]
        if task == "Entity":
            for group in ["ALL", f"numops_{r['numops']}", f"type_{r['entity_type']}", f"{r['entity_type']}_numops_{r['numops']}"]:
                groups[(arm, ck, task, role, group)].append(r)
        elif task == "Reading":
            groups[(arm, ck, task, role, "ALL")].append(r)
    # Candidate summaries for Entity/COMPS.
    for r in choice_rows:
        task = r["task"]
        arm = r["arm"]
        ck = r["checkpoint"]
        role = "candidate_margin"
        if task == "Entity":
            for group in ["ALL", f"numops_{r['numops']}", f"type_{r['entity_type']}", f"{r['entity_type']}_numops_{r['numops']}"]:
                groups[(arm, ck, task, role, group)].append(r)
        elif task == "COMPS":
            for group in ["ALL", f"subset_{r['subset']}", f"neg_{r.get('negative_sample_type')}", f"dist_{r.get('distraction_type')}"]:
                groups[(arm, ck, task, role, group)].append(r)
    out: list[dict[str, Any]] = []
    for (arm, ck, task, role, group), vals in sorted(groups.items()):
        nll_key = "gold_nll_per_token" if role == "candidate_margin" else "nll_per_token"
        margins = [float(v.get("margin_logprob", float("nan"))) for v in vals]
        correct = [int(v.get("correct", 0)) for v in vals if "correct" in v]
        stale_wrong = [int(v.get("pred_is_stale_initial", 0)) for v in vals if task == "Entity" and int(v.get("correct", 1)) == 0]
        out.append({
            "arm": arm,
            "checkpoint": ck,
            "task": task,
            "role": role,
            "group": group,
            "n": len(vals),
            "mean_nll_per_token": mean([float(v[nll_key]) for v in vals]),
            "sd_nll_per_token": pstdev([float(v[nll_key]) for v in vals]),
            "mean_margin_logprob": mean(margins),
            "accuracy_from_raw_margin": 100.0 * sum(correct) / len(correct) if correct else float("nan"),
            "stale_when_wrong_pct": 100.0 * sum(stale_wrong) / len(stale_wrong) if stale_wrong else float("nan"),
        })
    return out


def compute_contrasts(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = {(r["arm"], r["checkpoint"], r["task"], r["role"], r["group"]): r for r in summary}
    out: list[dict[str, Any]] = []
    for seed in [43022, 43122]:
        arms = {"V": f"D_V_{seed}", "C": f"D_C_{seed}", "R": f"D_R_{seed}"}
        for ck in CKS:
            groups = sorted({(r["task"], r["role"], r["group"]) for r in summary if r["checkpoint"] == ck and r["arm"].endswith(str(seed))})
            for task, role, group in groups:
                for contrast, a, b in [("VminusC", "V", "C"), ("VminusR", "V", "R"), ("CminusR", "C", "R")]:
                    ka = (arms[a], ck, task, role, group)
                    kb = (arms[b], ck, task, role, group)
                    if ka not in idx or kb not in idx:
                        continue
                    ra, rb = idx[ka], idx[kb]
                    out.append({
                        "seed": seed,
                        "checkpoint": ck,
                        "task": task,
                        "role": role,
                        "group": group,
                        "contrast": contrast,
                        "delta_nll_a_minus_b": float(ra["mean_nll_per_token"]) - float(rb["mean_nll_per_token"]),
                        "delta_margin_a_minus_b": float(ra["mean_margin_logprob"]) - float(rb["mean_margin_logprob"]),
                        "delta_accuracy_a_minus_b": float(ra["accuracy_from_raw_margin"]) - float(rb["accuracy_from_raw_margin"]),
                        "delta_stale_when_wrong_pct_a_minus_b": float(ra["stale_when_wrong_pct"]) - float(rb["stale_when_wrong_pct"]),
                        "n": int(ra["n"]),
                    })
    return out


def late_summary(contrast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in contrast_rows:
        if r["checkpoint"] in CKS:
            groups[(r["seed"], r["task"], r["role"], r["group"], r["contrast"])].append(r)
    out: list[dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        seed, task, role, group, contrast = key
        out.append({
            "seed": seed,
            "task": task,
            "role": role,
            "group": group,
            "contrast": contrast,
            "n_checkpoints": len(vals),
            "late_mean_delta_nll_a_minus_b": mean([float(v["delta_nll_a_minus_b"]) for v in vals]),
            "late_mean_delta_margin_a_minus_b": mean([float(v["delta_margin_a_minus_b"]) for v in vals]),
            "late_mean_delta_accuracy_a_minus_b": mean([float(v["delta_accuracy_a_minus_b"]) for v in vals]),
            "late_mean_delta_stale_when_wrong_pct_a_minus_b": mean([float(v["delta_stale_when_wrong_pct_a_minus_b"]) for v in vals]),
        })
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    preferred = [
        "arm", "checkpoint", "seed", "task", "role", "group", "contrast", "n", "n_checkpoints",
        "numops", "entity_type", "uid", "id", "subset", "negative_sample_type", "distraction_type",
        "mean_nll_per_token", "delta_nll_a_minus_b", "late_mean_delta_nll_a_minus_b",
        "mean_margin_logprob", "delta_margin_a_minus_b", "late_mean_delta_margin_a_minus_b",
        "accuracy_from_raw_margin", "delta_accuracy_a_minus_b", "late_mean_delta_accuracy_a_minus_b",
        "stale_when_wrong_pct", "delta_stale_when_wrong_pct_a_minus_b", "late_mean_delta_stale_when_wrong_pct_a_minus_b",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS))
    ap.add_argument("--checkpoints", nargs="+", default=CKS)
    ap.add_argument("--tasks", nargs="+", default=["Entity", "COMPS", "Reading"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--max-comps", type=int, default=8000)
    ap.add_argument("--max-entity-per-depth-type", type=int, default=0, help="0 means all official-filtered Entity items")
    ap.add_argument("--max-reading", type=int, default=0, help="0 means all Reading rows")
    ap.add_argument("--out-dir", default=str(OUT), help="Output directory for this shard or full run")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    OUT = pathlib.Path(args.out_dir)
    if not OUT.is_absolute():
        OUT = ROOT / OUT
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for ck in args.checkpoints:
            mp = ARM_CONFIGS[arm] / "hf_model" / ck
            if not mp.exists():
                raise FileNotFoundError(mp)

    items_by_task: dict[str, list[dict[str, Any]]] = {}
    if "Entity" in args.tasks:
        items_by_task["Entity"] = load_entity_items(None if args.max_entity_per_depth_type <= 0 else args.max_entity_per_depth_type)
    if "COMPS" in args.tasks:
        items_by_task["COMPS"] = load_comps_items(None if args.max_comps <= 0 else args.max_comps)
    if "Reading" in args.tasks:
        items_by_task["Reading"] = load_reading_items(None if args.max_reading <= 0 else args.max_reading)
    sizes = {k: len(v) for k, v in items_by_task.items()}
    plan = {"status": "BENCHMARK_ITEM_FIT_PLAN", "arms": args.arms, "checkpoints": args.checkpoints, "tasks": args.tasks, "sizes": sizes, "scoring": "multi-mask untempered MLM role/candidate log-prob; relative contrasts only"}
    if args.plan_only:
        print(json.dumps(plan, indent=2), flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(json.dumps({**plan, "device": str(device), "started_utc": now()}, indent=2), flush=True)

    all_fit_rows: list[dict[str, Any]] = []
    all_choice_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for arm in args.arms:
        tok_root = ARM_CONFIGS[arm] / "hf_model"
        tokenizer = AutoTokenizer.from_pretrained(str(tok_root), use_fast=True)
        for ck in args.checkpoints:
            model_path = ARM_CONFIGS[arm] / "hf_model" / ck
            t0 = time.time()
            print(f"[LOAD] {arm} {ck} {model_path}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(model_path), torch_dtype=torch.float32)
            model.eval().to(device)
            arm_fit_rows: list[dict[str, Any]] = []
            arm_choice_rows: list[dict[str, Any]] = []
            for task, items in items_by_task.items():
                fit_records, cand_records = build_records(tokenizer, task, items)
                print(f"[RUN] {arm} {ck} {task}: items={len(items)} fit_records={len(fit_records)} cand_records={len(cand_records)}", flush=True)
                fit_rows = score_records(model, fit_records, device, args.batch_size)
                cand_token_rows = score_records(model, cand_records, device, args.batch_size)
                choice_rows = aggregate_candidates(cand_token_rows, items_by_task)
                for r in fit_rows:
                    r["arm"] = arm; r["checkpoint"] = ck
                for r in choice_rows:
                    r["arm"] = arm; r["checkpoint"] = ck
                arm_fit_rows.extend(fit_rows)
                arm_choice_rows.extend(choice_rows)
                meta_rows.append({"arm": arm, "checkpoint": ck, "task": task, "items": len(items), "fit_records": len(fit_records), "candidate_records": len(cand_records)})
            write_csv(OUT / f"fit_rows_{arm}_{ck}.csv", arm_fit_rows)
            write_csv(OUT / f"choice_rows_{arm}_{ck}.csv", arm_choice_rows)
            all_fit_rows.extend(arm_fit_rows)
            all_choice_rows.extend(arm_choice_rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {arm} {ck} elapsed={time.time()-t0:.1f}s", flush=True)

    summary_rows = summarize_fit(all_fit_rows, all_choice_rows)
    contrast_rows = compute_contrasts(summary_rows)
    late_rows = late_summary(contrast_rows)
    write_csv(OUT / "benchmark_item_fit_role_rows.csv", all_fit_rows)
    write_csv(OUT / "benchmark_item_fit_choice_rows.csv", all_choice_rows)
    write_csv(OUT / "benchmark_item_fit_summary.csv", summary_rows)
    write_csv(OUT / "benchmark_item_fit_contrasts.csv", contrast_rows)
    write_csv(OUT / "benchmark_item_fit_late_contrasts.csv", late_rows)
    write_csv(OUT / "benchmark_item_fit_meta.csv", meta_rows)

    result = {
        "status": "BENCHMARK_ITEM_FIT_DONE",
        "finished_utc": now(),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "tasks": args.tasks,
        "sizes": sizes,
        "files": {
            "role_rows": rel(OUT / "benchmark_item_fit_role_rows.csv"),
            "choice_rows": rel(OUT / "benchmark_item_fit_choice_rows.csv"),
            "summary": rel(OUT / "benchmark_item_fit_summary.csv"),
            "contrasts": rel(OUT / "benchmark_item_fit_contrasts.csv"),
            "late_contrasts": rel(OUT / "benchmark_item_fit_late_contrasts.csv"),
            "meta": rel(OUT / "benchmark_item_fit_meta.csv"),
        },
        "interpretation": "For delta_nll_a_minus_b, negative means arm a has lower untempered MLM loss on that benchmark text/role. A deep Entity score advantage without lower operation/query loss is evidence against simple register fit; a lower operation-sentence/query loss aligned with score advantage supports a benchmark-text fit/allocation account.",
    }
    (OUT / "benchmark_item_fit_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nKEY LATE CONTRASTS")
    for r in late_rows:
        if r["contrast"] in {"VminusC", "VminusR"} and r["task"] in {"Entity", "COMPS", "Reading"} and r["group"] in {"ALL", "numops_0", "numops_3", "numops_4", "numops_5"}:
            print(f"seed={r['seed']} {r['task']} {r['role']} {r['group']} {r['contrast']} dNLL={r['late_mean_delta_nll_a_minus_b']:+.4f} dMargin={r['late_mean_delta_margin_a_minus_b']:+.3f} dAcc={r['late_mean_delta_accuracy_a_minus_b']:+.2f}")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
