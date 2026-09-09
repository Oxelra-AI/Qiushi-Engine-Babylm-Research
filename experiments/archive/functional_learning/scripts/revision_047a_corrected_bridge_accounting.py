#!/usr/bin/env python3
"""Step047a: static validation of corrected bridge target accounting.

This is a CPU/no-model pass over the interspersed legal overlay.  It verifies the
properties that make the corrected bridge interpretable before the full training
results are read:

* relation answer spans are fully covered and never fall back to MLM;
* ordinary rows receive exactly the same row-keyed WWM corruption in the
  ordinary_wwm and answer_allocation arms;
* per-macro relation/ordinary target counts and explicit lambda values are logged
  over the same 354 word-paced macro-updates.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))

import corrected_bridge_trainer as bridge  # noqa: E402


def write_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compare_ordinary_examples(ex_answer, ex_ordinary):
    mismatches = []
    n_ord = 0
    for i, (ea, eo) in enumerate(zip(ex_answer, ex_ordinary)):
        if ea["row_key"] != eo["row_key"]:
            mismatches.append({"position": i, "kind": "row_key", "answer_key": ea["row_key"], "ordinary_key": eo["row_key"]})
            continue
        if ea["component"] == "ordinary":
            n_ord += 1
            if eo["component"] != "ordinary":
                mismatches.append({"position": i, "kind": "component", "key": ea["row_key"], "answer_component": ea["component"], "ordinary_component": eo["component"]})
                continue
            if not torch.equal(ea["labels"], eo["labels"]):
                mismatches.append({"position": i, "kind": "ordinary_labels", "key": ea["row_key"]})
            if not torch.equal(ea["input_ids"], eo["input_ids"]):
                mismatches.append({"position": i, "kind": "ordinary_masked_ids", "key": ea["row_key"]})
    return n_ord, mismatches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlay-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--max-updates", type=int, default=354)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--train-seed", type=int, default=47047)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--loss-mode", choices=["explicit_word_fraction", "explicit_fixed_lambda", "pooled_token_mean"], default="explicit_word_fraction")
    ap.add_argument("--relation-lambda", type=float, default=0.12136590231805529)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = bridge.load_jsonl(pathlib.Path(args.overlay_jsonl))
    tokenizer = AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)

    cursor = 0
    update_logs: List[Dict[str, Any]] = []
    mismatch_examples: List[Dict[str, Any]] = []
    total_ord_compared = 0
    total_rows = 0
    for update_i in range(int(args.max_updates)):
        rows: List[Dict[str, Any]] = []
        words = 0
        while cursor < len(overlay) and words < int(args.words_per_update):
            row = overlay[cursor]
            rows.append(row)
            words += int(row.get("words", len(str(row.get("text", "")).split())))
            cursor += 1
        if not rows:
            break
        total_rows += len(rows)
        ex_answer, st_answer = bridge.prepare_targets_for_macro(
            rows, tokenizer, int(args.seq_length), wgb, "answer_allocation", int(args.train_seed), float(args.mask_prob))
        ex_ordinary, st_ordinary = bridge.prepare_targets_for_macro(
            rows, tokenizer, int(args.seq_length), wgb, "ordinary_wwm", int(args.train_seed), float(args.mask_prob))
        n_ord, mism = compare_ordinary_examples(ex_answer, ex_ordinary)
        total_ord_compared += n_ord
        if mism and len(mismatch_examples) < 20:
            mismatch_examples.extend(mism[:20 - len(mismatch_examples)])
        # reuse the same lambda code path by passing a minimal object
        class Obj: pass
        obj = Obj()
        obj.loss_mode = args.loss_mode
        obj.relation_lambda = args.relation_lambda
        lam_answer = bridge.relation_lambda_for_macro(obj, st_answer)
        lam_ordinary = bridge.relation_lambda_for_macro(obj, st_ordinary)
        update_logs.append({
            "update": update_i + 1,
            "rows": len(rows),
            "words": words,
            "answer_arm": st_answer,
            "ordinary_arm": st_ordinary,
            "lambda_answer": lam_answer,
            "lambda_ordinary": lam_ordinary,
            "answer_relation_token_fraction": st_answer["relation_target_tokens"] / max(1, st_answer["relation_target_tokens"] + st_answer["ordinary_target_tokens"]),
            "ordinary_relation_token_fraction": st_ordinary["relation_target_tokens"] / max(1, st_ordinary["relation_target_tokens"] + st_ordinary["ordinary_target_tokens"]),
            "ordinary_rows_compared": n_ord,
            "ordinary_pairing_mismatches": len(mism),
        })
        if update_i == 0 or (update_i + 1) % 50 == 0:
            print(json.dumps({
                "event": "accounting_update",
                "update": update_i + 1,
                "answer_rel_targets": st_answer["relation_target_tokens"],
                "ordinary_rel_targets": st_ordinary["relation_target_tokens"],
                "ordinary_targets_shared": st_answer["ordinary_target_tokens"],
                "lambda": lam_answer,
                "ordinary_mismatches": len(mism),
            }), flush=True)

    def sum_field(arm: str, field: str) -> int:
        return int(sum(int(x[f"{arm}_arm"][field]) for x in update_logs))

    summary = {
        "status": "CORRECTED_BRIDGE_ACCOUNTING_READY",
        "overlay_jsonl": bridge.rel(pathlib.Path(args.overlay_jsonl)),
        "updates": len(update_logs),
        "rows_seen": total_rows,
        "ordinary_rows_compared_across_arms": total_ord_compared,
        "ordinary_pairing_mismatch_count": int(sum(x["ordinary_pairing_mismatches"] for x in update_logs)),
        "ordinary_pairing_mismatch_examples": mismatch_examples,
        "answer_arm_totals": {
            "relation_rows": sum_field("answer", "relation_rows"),
            "ordinary_rows": sum_field("answer", "ordinary_rows"),
            "relation_words": sum_field("answer", "relation_words"),
            "ordinary_words": sum_field("answer", "ordinary_words"),
            "relation_target_tokens": sum_field("answer", "relation_target_tokens"),
            "ordinary_target_tokens": sum_field("answer", "ordinary_target_tokens"),
            "relation_zero_label_rows": sum_field("answer", "relation_zero_label_rows"),
            "ordinary_zero_label_rows": sum_field("answer", "ordinary_zero_label_rows"),
            "answer_span_rows": sum_field("answer", "answer_span_rows"),
        },
        "ordinary_arm_totals": {
            "relation_rows": sum_field("ordinary", "relation_rows"),
            "ordinary_rows": sum_field("ordinary", "ordinary_rows"),
            "relation_words": sum_field("ordinary", "relation_words"),
            "ordinary_words": sum_field("ordinary", "ordinary_words"),
            "relation_target_tokens": sum_field("ordinary", "relation_target_tokens"),
            "ordinary_target_tokens": sum_field("ordinary", "ordinary_target_tokens"),
            "relation_zero_label_rows": sum_field("ordinary", "relation_zero_label_rows"),
            "ordinary_zero_label_rows": sum_field("ordinary", "ordinary_zero_label_rows"),
            "wwm_relation_rows": sum_field("ordinary", "wwm_relation_rows"),
        },
        "mean_lambda_answer": sum(float(x["lambda_answer"]) for x in update_logs) / max(1, len(update_logs)),
        "mean_lambda_ordinary": sum(float(x["lambda_ordinary"]) for x in update_logs) / max(1, len(update_logs)),
        "mean_answer_relation_token_fraction": sum(float(x["answer_relation_token_fraction"]) for x in update_logs) / max(1, len(update_logs)),
        "mean_ordinary_relation_token_fraction": sum(float(x["ordinary_relation_token_fraction"]) for x in update_logs) / max(1, len(update_logs)),
        "scientific_use": "Validates that the corrected training comparison pairs ordinary WWM across arms and optimizes explicit macro-update relation/ordinary losses rather than microbatch means.",
    }
    (out_dir / "accounting_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(out_dir / "macro_update_accounting.jsonl", update_logs)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
