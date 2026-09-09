#!/usr/bin/env python3
"""research: benchmark-item pseudo-MLM loss for Entity, COMPS, and Reading texts.

Scientific question:
  research found VIEW improves Entity and Reading scores while worsening clean heldout
  MLM loss. That is only evidence for conversion beyond fit if VIEW is not simply a
  better language model on the benchmark items themselves. This script measures
  pseudo-MLM loss on the exact benchmark text strings used by scoring.

For Entity and COMPS, it computes the same candidate-completion pseudo-log-probability
used by the BabyLM MLM evaluator, then reports gold candidate NLL/token and a ranking
margin. For Reading, it computes pseudo-MLM NLL/token on the per-word target token in
its sentence context, matching the reading evaluator's p2_mlm orientation.

No training, no official evaluation, no upload.
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

ROOT = _public_path('experiments/archive/relation_learning/scripts/benchmark_item_loss.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
OUT = _public_path('experiments/archive/relation_learning/data/benchmark_item_loss')
DATA_ROOT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
frontier_consolidation_RUNS = _public_path('experiments/archive/frontier_consolidation/training/runs')

ARM_CONFIGS = {
    "D_V_43022": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    "D_C_43022": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    "D_R_43022": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022'),
    "D_V_43122": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    "D_C_43122": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
    "D_R_43122": _public_path('experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122'),
}
CKS = ["chck_80M", "chck_90M", "chck_100M"]

BATCH_MASKED_TOKENS = 512
MAX_ENTITY_ITEMS = None
MAX_COMPS_ITEMS_DEFAULT = 12000  # representative stratified subset, enough for quick mechanism decision
MAX_READING_ITEMS = None


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


def load_entity_items(max_items: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for fn in ["regular.jsonl", "ambiref.jsonl", "move_contents.jsonl"]:
        typ = fn[:-6]
        for obj in read_jsonl(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking') / fn):
            if any("nothing" in str(option).lower() for option in obj.get("options", [])):
                continue
            uid = f"{typ}_{int(obj['numops'])}_ops"
            completions = [str(x) for x in obj["options"]]
            items.append({
                "task": "Entity", "uid": uid, "entity_type": typ, "numops": int(obj["numops"]),
                "id": f"{typ}:{obj.get('example_id')}:{len(items)}", "prefix": obj["input_prefix"],
                "completions": completions, "label": 0,
                "gold_text": obj["input_prefix"] + completions[0],
            })
            if max_items is not None and len(items) >= max_items:
                return items
    return items


def load_comps_items(max_items: int | None = MAX_COMPS_ITEMS_DEFAULT) -> list[dict[str, Any]]:
    specs = [
        ("comps_base.jsonl", "base"),
        ("comps_wugs.jsonl", "wugs"),
        ("comps_wugs_dist-before.jsonl", "wugs_dist_before"),
        ("comps_wugs_dist-in-between.jsonl", "wugs_dist_in_between"),
    ]
    # Stratified deterministic sample to avoid a 91k-row full pass for a first mechanism readout.
    per_file_limit = None
    if max_items is not None:
        per_file_limit = max(1, max_items // len(specs))
    items: list[dict[str, Any]] = []
    for fn, subset in specs:
        rows = list(read_jsonl(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/comps') / fn))
        if per_file_limit is not None and len(rows) > per_file_limit:
            # Evenly spaced deterministic sample preserves property/negative-type spread better than prefix truncation.
            idxs = np.linspace(0, len(rows) - 1, per_file_limit).round().astype(int).tolist()
            rows = [rows[i] for i in idxs]
        for obj in rows:
            acc = " ".join([obj["prefix_acceptable"], obj["property_phrase"]])
            unacc = " ".join([obj["prefix_unacceptable"], obj["property_phrase"]])
            items.append({
                "task": "COMPS", "uid": subset, "negative_sample_type": obj.get("negative_sample_type"),
                "distraction_type": obj.get("distraction_type", "base"), "id": f"{subset}:{obj.get('id')}",
                "prefix": None, "completions": [acc, unacc], "label": 0,
                "gold_text": acc,
            })
    return items


def load_reading_items(max_items: int | None = None) -> list[dict[str, Any]]:
    p = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv')
    items: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # The reading task predicts the current word in the full sentence. Keep a local context string.
            sentence = row["sentence"]
            word = row["word"]
            context = row["item"]  # prefix before word in the CSV
            # Ensure the target word appears as a completion in the sentence; use the prefix + word span.
            completion = word
            text = (context + " " + word).strip()
            items.append({
                "task": "Reading", "uid": "reading", "id": row["item_id"] + ":" + row["sent_id"] + ":" + str(len(items)),
                "sentence": sentence, "context": context, "word": word, "prefix": context + " ",
                "completions": [completion], "label": 0, "gold_text": text,
                "length": int(row.get("length") or 0), "context_length": int(row.get("context_length") or 0),
            })
            if max_items is not None and len(items) >= max_items:
                break
    return items


def completion_token_positions(tokenizer, sentence: str, completion: str) -> tuple[list[int], list[int], list[int]]:
    enc = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    tokens = list(enc["input_ids"])
    att = list(enc["attention_mask"])
    offsets = enc["offset_mapping"]
    start_char_idx = len(sentence) - len(completion)
    indices, targets = [], []
    for i, (start, end) in enumerate(offsets):
        if end > start_char_idx:
            indices.append(i)
            targets.append(tokens[i])
    return tokens, att, indices, targets


def build_masked_examples(tokenizer, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mask_id = tokenizer.mask_token_id
    examples: list[dict[str, Any]] = []
    for item_idx, item in enumerate(items):
        for cand_idx, comp in enumerate(item["completions"]):
            if item["task"] == "Entity":
                sentence = item["prefix"] + comp
                completion = comp
            elif item["task"] == "COMPS":
                sentence = comp  # completion is already full sentence for COMPS candidate
                completion = comp
            else:  # Reading
                sentence = item["gold_text"]
                completion = item["word"]
            tokens, att, indices, targets = completion_token_positions(tokenizer, sentence, completion)
            for j, (pos, target) in enumerate(zip(indices, targets)):
                mtoks = list(tokens)
                mtoks[pos] = mask_id
                examples.append({
                    "item_idx": item_idx, "cand_idx": cand_idx, "token_j": j,
                    "input_ids": mtoks, "attention_mask": att, "index": pos, "target": target,
                })
    return examples


@torch.no_grad()
def score_items(model, tokenizer, items: list[dict[str, Any]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = build_masked_examples(tokenizer, items)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    token_logps = np.empty(len(examples), dtype=np.float64)

    for start in range(0, len(examples), BATCH_MASKED_TOKENS):
        batch = examples[start:start + BATCH_MASKED_TOKENS]
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        indices = torch.empty(len(batch), dtype=torch.long)
        targets = torch.empty(len(batch), dtype=torch.long)
        for i, ex in enumerate(batch):
            L = len(ex["input_ids"])
            input_ids[i, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
            attn[i, :L] = torch.tensor(ex["attention_mask"], dtype=torch.long)
            indices[i] = ex["index"]
            targets[i] = ex["target"]
        input_ids = input_ids.to(device); attn = attn.to(device); indices = indices.to(device); targets = targets.to(device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        rows = torch.arange(logits.shape[0], device=device)
        masked_logits = logits[rows, indices]
        logp = torch.nn.functional.log_softmax(masked_logits, dim=-1)[rows, targets]
        token_logps[start:start + len(batch)] = logp.detach().cpu().numpy()

    # Aggregate by candidate.
    cand_lp: dict[tuple[int, int], list[float]] = defaultdict(list)
    for ex, lp in zip(examples, token_logps):
        cand_lp[(ex["item_idx"], ex["cand_idx"])].append(float(lp))

    rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(items):
        cand_sums = []
        cand_lens = []
        for c in range(len(item["completions"])):
            vals = cand_lp.get((item_idx, c), [])
            cand_sums.append(sum(vals))
            cand_lens.append(len(vals))
        gold_lp = cand_sums[item["label"]]
        gold_len = max(cand_lens[item["label"]], 1)
        pred = int(np.argmax(np.array(cand_sums))) if len(cand_sums) > 1 else 0
        sorted_sums = sorted(cand_sums, reverse=True)
        margin = cand_sums[0] - (max(cand_sums[1:]) if len(cand_sums) > 1 else float("nan"))
        out = {
            "task": item["task"], "uid": item["uid"], "id": item["id"],
            "gold_nll_per_token": -gold_lp / gold_len,
            "gold_logprob_sum": gold_lp,
            "gold_completion_tokens": gold_len,
            "pred_idx": pred,
            "correct": int(pred == item["label"]),
            "margin_logprob": margin,
            "n_candidates": len(item["completions"]),
        }
        for k in ["entity_type", "numops", "negative_sample_type", "distraction_type", "length", "context_length"]:
            if k in item:
                out[k] = item[k]
        rows.append(out)

    meta = {"n_items": len(items), "n_masked_token_forwards": len(examples)}
    return rows, meta


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Weighted by item, not token, because scores are item-level accuracies; loss already token-normalized.
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["arm"], r["checkpoint"], r["task"], r.get("uid", "ALL"))
        groups[key].append(r)
        key_all = (r["arm"], r["checkpoint"], r["task"], "ALL")
        groups[key_all].append(r)
        if r["task"] == "Entity":
            groups[(r["arm"], r["checkpoint"], r["task"], f"numops_{r['numops']}")].append(r)
        if r["task"] == "COMPS":
            groups[(r["arm"], r["checkpoint"], r["task"], f"neg_{r.get('negative_sample_type')}")].append(r)
            groups[(r["arm"], r["checkpoint"], r["task"], f"dist_{r.get('distraction_type')}")].append(r)
    out = []
    for (arm, ck, task, group), vals in sorted(groups.items()):
        nlls = [float(v["gold_nll_per_token"]) for v in vals if math.isfinite(float(v["gold_nll_per_token"]))]
        margins = [float(v["margin_logprob"]) for v in vals if math.isfinite(float(v["margin_logprob"]))]
        out.append({
            "arm": arm, "checkpoint": ck, "task": task, "group": group,
            "n": len(vals), "mean_gold_nll_per_token": statistics.mean(nlls),
            "sd_gold_nll_per_token": statistics.pstdev(nlls) if len(nlls) > 1 else 0.0,
            "mean_margin_logprob": statistics.mean(margins) if margins else float("nan"),
            "accuracy_from_argmax": 100.0 * sum(int(v.get("correct", 0)) for v in vals) / len(vals) if vals and task != "Reading" else float("nan"),
        })
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    preferred = ["arm", "checkpoint", "task", "group", "uid", "id", "numops", "entity_type", "negative_sample_type", "distraction_type", "n", "mean_gold_nll_per_token", "accuracy_from_argmax", "mean_margin_logprob", "gold_nll_per_token", "correct", "margin_logprob"]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def compute_contrasts(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = {(r["arm"], r["checkpoint"], r["task"], r["group"]): r for r in summary_rows}
    out = []
    for seed in [43022, 43122]:
        for ck in CKS:
            for task in ["Entity", "COMPS", "Reading"]:
                groups = sorted({r["group"] for r in summary_rows if r["checkpoint"] == ck and r["task"] == task and r["arm"].endswith(str(seed))})
                for group in groups:
                    for name, a, b in [("VminusC", f"D_V_{seed}", f"D_C_{seed}"), ("VminusR", f"D_V_{seed}", f"D_R_{seed}"), ("CminusR", f"D_C_{seed}", f"D_R_{seed}")]:
                        ka = (a, ck, task, group); kb = (b, ck, task, group)
                        if ka not in idx or kb not in idx:
                            continue
                        ra, rb = idx[ka], idx[kb]
                        out.append({
                            "seed": seed, "checkpoint": ck, "task": task, "group": group, "contrast": name,
                            "delta_nll_a_minus_b": float(ra["mean_gold_nll_per_token"]) - float(rb["mean_gold_nll_per_token"]),
                            "delta_accuracy_a_minus_b": (float(ra.get("accuracy_from_argmax", float("nan"))) - float(rb.get("accuracy_from_argmax", float("nan")))) if task != "Reading" else float("nan"),
                            "delta_margin_a_minus_b": float(ra.get("mean_margin_logprob", float("nan"))) - float(rb.get("mean_margin_logprob", float("nan"))),
                            "score_a_nll": float(ra["mean_gold_nll_per_token"]), "score_b_nll": float(rb["mean_gold_nll_per_token"]),
                            "n": int(ra["n"]),
                        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS))
    ap.add_argument("--checkpoints", nargs="+", default=CKS)
    ap.add_argument("--tasks", nargs="+", default=["Entity", "COMPS", "Reading"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-comps-items", type=int, default=MAX_COMPS_ITEMS_DEFAULT)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    for arm in args.arms:
        for ck in args.checkpoints:
            p = ARM_CONFIGS[arm] / "hf_model" / ck
            if not p.exists():
                raise FileNotFoundError(p)

    items_by_task = {}
    if "Entity" in args.tasks:
        items_by_task["Entity"] = load_entity_items(MAX_ENTITY_ITEMS)
    if "COMPS" in args.tasks:
        items_by_task["COMPS"] = load_comps_items(args.max_comps_items)
    if "Reading" in args.tasks:
        items_by_task["Reading"] = load_reading_items(MAX_READING_ITEMS)
    task_sizes = {k: len(v) for k, v in items_by_task.items()}
    if args.plan_only:
        print(json.dumps({"status": "BENCHMARK_ITEM_LOSS_PLAN", "arms": args.arms, "checkpoints": args.checkpoints, "tasks": args.tasks, "task_sizes": task_sizes}, indent=2), flush=True)
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    all_item_rows: list[dict[str, Any]] = []
    all_meta: list[dict[str, Any]] = []

    for arm in args.arms:
        tok = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[arm] / "hf_model"), use_fast=True)
        for ck in args.checkpoints:
            model_path = ARM_CONFIGS[arm] / "hf_model" / ck
            t0 = time.time()
            print(f"[LOAD] {arm} {ck} from {model_path}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(model_path), torch_dtype=torch.float32)
            model.eval().to(device)
            for task, items in items_by_task.items():
                print(f"[RUN] {arm} {ck} {task}: {len(items)} items", flush=True)
                rows, meta = score_items(model, tok, items, device)
                for r in rows:
                    r["arm"] = arm; r["checkpoint"] = ck
                all_item_rows.extend(rows)
                meta.update({"arm": arm, "checkpoint": ck, "task": task})
                all_meta.append(meta)
                # Save arm/task shard immediately for safety.
                shard = OUT / f"item_rows_{arm}_{ck}_{task}.csv"
                write_csv(shard, rows)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {arm} {ck} elapsed={time.time()-t0:.1f}s", flush=True)

    summary_rows = summarize(all_item_rows)
    contrast_rows = compute_contrasts(summary_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/benchmark_item_loss/benchmark_item_loss_rows.csv'), all_item_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/benchmark_item_loss/benchmark_item_loss_summary.csv'), summary_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/benchmark_item_loss/benchmark_item_loss_contrasts.csv'), contrast_rows)
    summary = {
        "status": "BENCHMARK_ITEM_LOSS_DONE", "finished_utc": now(),
        "arms": args.arms, "checkpoints": args.checkpoints, "tasks": args.tasks,
        "task_sizes": task_sizes, "meta": all_meta,
        "files": {
            "item_rows": rel(_public_path('experiments/archive/relation_learning/data/benchmark_item_loss/benchmark_item_loss_rows.csv')),
            "summary": rel(_public_path('experiments/archive/relation_learning/data/benchmark_item_loss/benchmark_item_loss_summary.csv')),
            "contrasts": rel(_public_path('experiments/archive/relation_learning/data/benchmark_item_loss/benchmark_item_loss_contrasts.csv')),
        },
        "note": "For contrasts, negative delta_nll_a_minus_b means arm a has lower pseudo-MLM loss than arm b on the benchmark item text; positive benchmark gain with negative/equal item-loss is fit/register, positive benchmark gain with higher/equal item-loss supports conversion beyond item fit.",
    }
    (_public_path('experiments/archive/relation_learning/data/benchmark_item_loss/benchmark_item_loss_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    print("\nCOMPACT CONTRASTS FOR GROUP=ALL")
    for r in contrast_rows:
        if r["group"] == "ALL" and r["contrast"] in ["VminusC", "VminusR"]:
            print(f"seed={r['seed']} {r['checkpoint']} {r['task']} {r['contrast']} delta_nll={r['delta_nll_a_minus_b']:+.4f} acc_delta={r['delta_accuracy_a_minus_b']:+.2f} margin_delta={r['delta_margin_a_minus_b']:+.3f} n={r['n']}")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
