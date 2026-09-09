#!/usr/bin/env python3
"""research: operation-preserving expression-transfer probe.

The bridge tells us whether relation-sensitive experience can coexist with ordinary
BabyLM continuation.  This script checks a separate boundary: whether the acquired
relation-first selector is tied to the exact research update/use wording or
survives low-cost natural-language re-expression while preserving the same
recipient-only state-selection problem.

For each relation-first held pair, the source facts and state map are fixed:
entity_a -> value_a, entity_b -> value_b, and an update to one recipient assigns
the same shared_new_value.  For a fixed query and fixed candidate set
(value_a,value_b,shared_new_value), changing only the update recipient must flip
the correct answer between the source value and shared_new_value.  Candidates are
all same relation type (mostly place names), so simple candidate-type matching is
not enough.

The scorer uses simultaneous full-candidate-span masking, not single-token
pseudo-likelihood.  It records separate source-vs-wrong-source and
source-vs-replacement margins.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import gc
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPTS))

import bridge_eval as bridge_eval  # noqa: E402

PARENT_PATH = _public_path('models/frontier')
SPECIALIST_PATH = _public_path('experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity/seed_40040/hf_model/final')
FIXED_PAIRS = _public_path('experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity/fixed_repaired_pairs.jsonl')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


# The variants intentionally keep the state map fixed.  They only rephrase the
# correction statement and/or the final query frame.  `orig` is retained as a
# continuity anchor rather than expression-transfer evidence.
VARIANT_SPECS: Dict[str, Dict[str, Dict[str, str]]] = {
    "orig": {},
    "query_short": {
        "death_place": {
            "use": "Using the passage above, {QUERY}'s death place is {STATE}.",
        },
        "birthplace": {
            "use": "Using the passage above, {QUERY}'s birthplace is {STATE}.",
        },
        "located_in": {
            "use": "Using the passage above, {QUERY}'s location is {STATE}.",
        },
        "founded_year": {
            "use": "Using the passage above, the founding year of {QUERY} is {STATE}.",
        },
    },
    "query_sentence": {
        "death_place": {
            "use": "From this information, {QUERY} died in {STATE}.",
        },
        "birthplace": {
            "use": "From this information, {QUERY} was born in {STATE}.",
        },
        "located_in": {
            "use": "From this information, {QUERY} is in {STATE}.",
        },
        "founded_year": {
            "use": "From this information, {QUERY} was founded in {STATE}.",
        },
    },
    "update_record": {
        "death_place": {
            "update": "Later records revise the death-place entry for {ENTITY} to {NEW_VALUE}.",
        },
        "birthplace": {
            "update": "Later records revise the birthplace entry for {ENTITY} to {NEW_VALUE}.",
        },
        "located_in": {
            "update": "Later records revise the location entry for {ENTITY} to {NEW_VALUE}.",
        },
        "founded_year": {
            "update": "Later records revise the founding year for {ENTITY} to {NEW_VALUE}.",
        },
    },
    "both_record_short": {
        "death_place": {
            "update": "Later records revise the death-place entry for {ENTITY} to {NEW_VALUE}.",
            "use": "Using the passage above, {QUERY}'s death place is {STATE}.",
        },
        "birthplace": {
            "update": "Later records revise the birthplace entry for {ENTITY} to {NEW_VALUE}.",
            "use": "Using the passage above, {QUERY}'s birthplace is {STATE}.",
        },
        "located_in": {
            "update": "Later records revise the location entry for {ENTITY} to {NEW_VALUE}.",
            "use": "Using the passage above, {QUERY}'s location is {STATE}.",
        },
        "founded_year": {
            "update": "Later records revise the founding year for {ENTITY} to {NEW_VALUE}.",
            "use": "Using the passage above, the founding year of {QUERY} is {STATE}.",
        },
    },
    "both_correction_sentence": {
        "death_place": {
            "update": "A correction now says that {ENTITY} died in {NEW_VALUE}.",
            "use": "From this information, {QUERY} died in {STATE}.",
        },
        "birthplace": {
            "update": "A correction now says that {ENTITY} was born in {NEW_VALUE}.",
            "use": "From this information, {QUERY} was born in {STATE}.",
        },
        "located_in": {
            "update": "A correction now says that {ENTITY} belongs in {NEW_VALUE}.",
            "use": "From this information, {QUERY} is in {STATE}.",
        },
        "founded_year": {
            "update": "A correction now says that {ENTITY} began in {NEW_VALUE}.",
            "use": "From this information, {QUERY} was founded in {STATE}.",
        },
    },
}


def default_update(pair: Dict[str, Any], which: str) -> str:
    return str(pair[f"update_{which}_sentence"])


def default_use(pair: Dict[str, Any], which: str) -> str:
    return str(pair[f"use_frame_{which}"])


def render_variant(pair: Dict[str, Any], variant_name: str) -> Dict[str, Any] | None:
    relation = str(pair.get("relation"))
    spec = VARIANT_SPECS.get(variant_name)
    if spec is None:
        raise KeyError(f"unknown variant {variant_name}")
    rel_spec = spec.get(relation, {})
    if variant_name != "orig" and not rel_spec:
        return None

    source = str(pair["source_context"])
    newv = str(pair["shared_new_value"])
    ea, eb = str(pair["entity_a"]), str(pair["entity_b"])
    if "update" in rel_spec:
        update_a = rel_spec["update"].format(ENTITY=ea, NEW_VALUE=newv)
        update_b = rel_spec["update"].format(ENTITY=eb, NEW_VALUE=newv)
    else:
        update_a = default_update(pair, "a")
        update_b = default_update(pair, "b")
    if "use" in rel_spec:
        use_a = rel_spec["use"].format(QUERY=ea, STATE="{STATE}")
        use_b = rel_spec["use"].format(QUERY=eb, STATE="{STATE}")
    else:
        use_a = default_use(pair, "a")
        use_b = default_use(pair, "b")

    return {
        "pair_id": pair["pair_id"],
        "split": pair.get("split"),
        "relation": relation,
        "variant": variant_name,
        "entity_a": ea,
        "entity_b": eb,
        "value_a": str(pair["value_a"]),
        "value_b": str(pair["value_b"]),
        "shared_new_value": newv,
        "source_context": source,
        "update_a_sentence": update_a,
        "update_b_sentence": update_b,
        "use_frame_a": use_a,
        "use_frame_b": use_b,
        "context_neutral": source,
        "context_update_a": source.rstrip() + " " + update_a,
        "context_update_b": source.rstrip() + " " + update_b,
        "variant_changes": {
            "update_changed": update_a != default_update(pair, "a") or update_b != default_update(pair, "b"),
            "query_frame_changed": use_a != default_use(pair, "a") or use_b != default_use(pair, "b"),
        },
    }


def locate_span_positions(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    pos: List[int] = []
    for i, (s, e) in enumerate(offsets):
        s, e = int(s), int(e)
        if e <= s:
            continue
        if s < end and e > start:
            pos.append(i)
    return pos


def make_scoring_item(tokenizer, context_frame: str, candidate: str, seq_length: int, device: torch.device) -> Dict[str, Any]:
    if context_frame.count("{STATE}") != 1:
        raise ValueError(f"bad frame with {context_frame.count('{STATE}')} slots: {context_frame[:120]!r}")
    before, after = context_frame.split("{STATE}")
    full = before + candidate + after
    start, end = len(before), len(before) + len(candidate)
    enc = tokenizer(
        full,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        max_length=seq_length,
        truncation=True,
        padding="max_length",
    )
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
    pos = locate_span_positions(offsets, start, end)
    if not pos:
        raise ValueError(f"candidate span unavailable: {candidate!r} start={start} end={end}")
    covered_start = min(offsets[p][0] for p in pos)
    covered_end = max(offsets[p][1] for p in pos)
    if covered_start > start or covered_end < end:
        raise ValueError(f"candidate span partial: {candidate!r} span={(start,end)} covered={(covered_start,covered_end)}")
    input_ids = enc["input_ids"].squeeze(0)
    masked_ids = input_ids.clone()
    target_ids = input_ids[pos].clone()
    for p in pos:
        masked_ids[p] = int(tokenizer.mask_token_id)
    return {
        "input_ids": masked_ids.to(device),
        "attention_mask": enc["attention_mask"].squeeze(0).to(device),
        "positions": torch.tensor(pos, dtype=torch.long, device=device),
        "target_ids": target_ids.to(device),
        "n_tokens": len(pos),
    }


def score_variant(model, tokenizer, item: Dict[str, Any], device: torch.device, seq_length: int) -> Dict[str, Any]:
    va, vb, vn = item["value_a"], item["value_b"], item["shared_new_value"]
    candidates = [("value_a", va), ("value_b", vb), ("shared_new", vn)]
    contexts = {
        "update_a_query_a": item["context_update_a"] + " " + item["use_frame_a"],
        "update_b_query_a": item["context_update_b"] + " " + item["use_frame_a"],
        "update_a_query_b": item["context_update_a"] + " " + item["use_frame_b"],
        "update_b_query_b": item["context_update_b"] + " " + item["use_frame_b"],
        "neutral_query_a": item["context_neutral"] + " " + item["use_frame_a"],
        "neutral_query_b": item["context_neutral"] + " " + item["use_frame_b"],
    }
    requests: List[Dict[str, Any]] = []
    tensors: List[Dict[str, Any]] = []
    for ctx_name, ctx_frame in contexts.items():
        for lab, cand in candidates:
            tensors.append(make_scoring_item(tokenizer, ctx_frame, cand, seq_length, device))
            requests.append({"context": ctx_name, "candidate_label": lab, "candidate": cand})
    input_ids = torch.stack([t["input_ids"] for t in tensors], dim=0)
    attention_mask = torch.stack([t["attention_mask"] for t in tensors], dim=0)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    results: Dict[str, Dict[str, Any]] = {k: {} for k in contexts}
    for bi, (req, t) in enumerate(zip(requests, tensors)):
        pos = t["positions"]
        target = t["target_ids"]
        lp = F.log_softmax(logits[bi, pos], dim=-1)
        tok_lp = lp[torch.arange(len(pos), device=device), target].detach().cpu().tolist()
        results[req["context"]][req["candidate_label"]] = {
            "candidate": req["candidate"],
            "n_tokens": int(t["n_tokens"]),
            "sum_logp": float(sum(tok_lp)),
            "mean_logp": float(sum(tok_lp) / len(tok_lp)),
        }
    return results


def mean_logp(scored: Dict[str, Dict[str, Any]], ctx: str, lab: str) -> float:
    return float(scored[ctx][lab]["mean_logp"])


def analyze_record(item: Dict[str, Any], scored: Dict[str, Any], model_label: str) -> Dict[str, Any]:
    # Neutral source retrieval.
    n_qa_c = mean_logp(scored, "neutral_query_a", "value_a")
    n_qa_w = mean_logp(scored, "neutral_query_a", "value_b")
    n_qa_n = mean_logp(scored, "neutral_query_a", "shared_new")
    n_qb_c = mean_logp(scored, "neutral_query_b", "value_b")
    n_qb_w = mean_logp(scored, "neutral_query_b", "value_a")
    n_qb_n = mean_logp(scored, "neutral_query_b", "shared_new")
    # Distractor update retention.
    r_qa_c = mean_logp(scored, "update_b_query_a", "value_a")
    r_qa_w = mean_logp(scored, "update_b_query_a", "value_b")
    r_qa_n = mean_logp(scored, "update_b_query_a", "shared_new")
    r_qb_c = mean_logp(scored, "update_a_query_b", "value_b")
    r_qb_w = mean_logp(scored, "update_a_query_b", "value_a")
    r_qb_n = mean_logp(scored, "update_a_query_b", "shared_new")
    # Self update: shared_new should dominate both source values.
    u_qa_n = mean_logp(scored, "update_a_query_a", "shared_new")
    u_qa_src = mean_logp(scored, "update_a_query_a", "value_a")
    u_qa_wrong = mean_logp(scored, "update_a_query_a", "value_b")
    u_qb_n = mean_logp(scored, "update_b_query_b", "shared_new")
    u_qb_src = mean_logp(scored, "update_b_query_b", "value_b")
    u_qb_wrong = mean_logp(scored, "update_b_query_b", "value_a")

    rec = {
        "model": model_label,
        "pair_id": item["pair_id"],
        "split": item.get("split"),
        "relation": item["relation"],
        "variant": item["variant"],
        "entity_a": item["entity_a"],
        "entity_b": item["entity_b"],
        "value_a": item["value_a"],
        "value_b": item["value_b"],
        "shared_new_value": item["shared_new_value"],
        "variant_changes": item["variant_changes"],
        "neutral_qa_cross_source": n_qa_c - n_qa_w,
        "neutral_qb_cross_source": n_qb_c - n_qb_w,
        "neutral_qa_correct_over_new": n_qa_c - n_qa_n,
        "neutral_qb_correct_over_new": n_qb_c - n_qb_n,
        "retain_qa_cross_source": r_qa_c - r_qa_w,
        "retain_qb_cross_source": r_qb_c - r_qb_w,
        "retain_qa_correct_over_new": r_qa_c - r_qa_n,
        "retain_qb_correct_over_new": r_qb_c - r_qb_n,
        "self_update_qa_new_over_source": u_qa_n - u_qa_src,
        "self_update_qb_new_over_source": u_qb_n - u_qb_src,
        "self_update_qa_new_over_wrong": u_qa_n - u_qa_wrong,
        "self_update_qb_new_over_wrong": u_qb_n - u_qb_wrong,
    }
    rec["neutral_qa_full_source"] = rec["neutral_qa_cross_source"] > 0 and rec["neutral_qa_correct_over_new"] > 0
    rec["neutral_qb_full_source"] = rec["neutral_qb_cross_source"] > 0 and rec["neutral_qb_correct_over_new"] > 0
    rec["retain_qa_full_source"] = rec["retain_qa_cross_source"] > 0 and rec["retain_qa_correct_over_new"] > 0
    rec["retain_qb_full_source"] = rec["retain_qb_cross_source"] > 0 and rec["retain_qb_correct_over_new"] > 0
    rec["self_update_qa_full_new"] = rec["self_update_qa_new_over_source"] > 0 and rec["self_update_qa_new_over_wrong"] > 0
    rec["self_update_qb_full_new"] = rec["self_update_qb_new_over_source"] > 0 and rec["self_update_qb_new_over_wrong"] > 0
    rec["neutral_both_full_source"] = rec["neutral_qa_full_source"] and rec["neutral_qb_full_source"]
    rec["retain_both_full_source"] = rec["retain_qa_full_source"] and rec["retain_qb_full_source"]
    rec["self_update_both_full_new"] = rec["self_update_qa_full_new"] and rec["self_update_qb_full_new"]
    rec["recipient_only_flip_both_queries"] = rec["retain_both_full_source"] and rec["self_update_both_full_new"]
    return rec


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        groups[(r["model"], r["variant"])].append(r)
    by_model_variant = {}
    for (model, variant), rs in sorted(groups.items()):
        key = f"{model}::{variant}"
        by_model_variant[key] = {
            "model": model,
            "variant": variant,
            "n_pairs": len(rs),
            "relations": dict(collections.Counter(r["relation"] for r in rs)),
            "neutral_both_full_source": sum(1 for r in rs if r["neutral_both_full_source"]),
            "retain_both_full_source": sum(1 for r in rs if r["retain_both_full_source"]),
            "self_update_both_full_new": sum(1 for r in rs if r["self_update_both_full_new"]),
            "recipient_only_flip_both_queries": sum(1 for r in rs if r["recipient_only_flip_both_queries"]),
            "mean_neutral_correct_vs_wrong": finite_mean([r["neutral_qa_cross_source"] for r in rs] + [r["neutral_qb_cross_source"] for r in rs]),
            "mean_neutral_correct_vs_replacement": finite_mean([r["neutral_qa_correct_over_new"] for r in rs] + [r["neutral_qb_correct_over_new"] for r in rs]),
            "mean_retain_correct_vs_wrong": finite_mean([r["retain_qa_cross_source"] for r in rs] + [r["retain_qb_cross_source"] for r in rs]),
            "mean_retain_correct_vs_replacement": finite_mean([r["retain_qa_correct_over_new"] for r in rs] + [r["retain_qb_correct_over_new"] for r in rs]),
            "mean_self_update_new_vs_source": finite_mean([r["self_update_qa_new_over_source"] for r in rs] + [r["self_update_qb_new_over_source"] for r in rs]),
            "mean_self_update_new_vs_wrong": finite_mean([r["self_update_qa_new_over_wrong"] for r in rs] + [r["self_update_qb_new_over_wrong"] for r in rs]),
        }
    model_totals = {}
    for model in sorted(set(r["model"] for r in records)):
        rs = [r for r in records if r["model"] == model and r["variant"] != "orig"]
        if rs:
            model_totals[model] = {
                "n_expression_records": len(rs),
                "n_unique_pairs": len(set(r["pair_id"] for r in rs)),
                "expression_retain_both_full_source": sum(1 for r in rs if r["retain_both_full_source"]),
                "expression_self_update_both_full_new": sum(1 for r in rs if r["self_update_both_full_new"]),
                "expression_recipient_only_flip_both_queries": sum(1 for r in rs if r["recipient_only_flip_both_queries"]),
                "mean_expression_retain_correct_vs_wrong": finite_mean([r["retain_qa_cross_source"] for r in rs] + [r["retain_qb_cross_source"] for r in rs]),
                "mean_expression_retain_correct_vs_replacement": finite_mean([r["retain_qa_correct_over_new"] for r in rs] + [r["retain_qb_correct_over_new"] for r in rs]),
            }
    return {"by_model_variant": by_model_variant, "model_expression_totals_excluding_orig": model_totals}


def load_model_for_label(label: str, device: torch.device, private_scale: float):
    if label == "parent":
        model, ident = bridge_eval.load_parent(device, private_scale)
        return model, ident
    if label == "specialist_seed40040":
        model, ident = bridge_eval.strict_load_checkpoint(SPECIALIST_PATH, device, private_scale)
        return model, ident
    if label.startswith("checkpoint:"):
        path = pathlib.Path(label.split(":", 1)[1])
        if not path.is_absolute():
            path = ROOT / path
        model, ident = bridge_eval.strict_load_checkpoint(path, device, private_scale)
        return model, ident
    raise ValueError(f"unknown model label {label}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=str(FIXED_PAIRS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=["parent", "specialist_seed40040"], help="parent, specialist_seed40040, or checkpoint:PATH")
    ap.add_argument("--variants", nargs="+", default=["orig", "query_short", "query_sentence", "update_record", "both_record_short", "both_correction_sentence"])
    ap.add_argument("--relations", nargs="+", default=["death_place", "birthplace"], help="relations to include")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, cuda:1")
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--torch-threads", type=int, default=8)
    args = ap.parse_args()

    if args.device == "cpu" and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device if args.device != "cuda" else "cuda:0")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [p for p in read_jsonl(pathlib.Path(args.pairs)) if p.get("split") == "held" and p.get("relation") in set(args.relations)]
    # Stable order: include both-novel examples if present, but no manual cherry-picking.
    pairs = sorted(pairs, key=lambda p: (p.get("relation", ""), p.get("pair_id", "")))
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[:int(args.max_pairs)]

    suite: List[Dict[str, Any]] = []
    skipped = []
    for p in pairs:
        for v in args.variants:
            item = render_variant(p, v)
            if item is None:
                skipped.append({"pair_id": p.get("pair_id"), "relation": p.get("relation"), "variant": v, "reason": "no_template"})
                continue
            suite.append(item)
    write_jsonl(out_dir / "expression_transfer_suite.jsonl", suite)

    tokenizer = AutoTokenizer.from_pretrained(str(PARENT_PATH), local_files_only=True, use_fast=True)
    records: List[Dict[str, Any]] = []
    identities: Dict[str, Any] = {}
    errors: List[Dict[str, Any]] = []
    start_time = time.time()
    for model_label in args.models:
        print(json.dumps({"event": "model_start", "model": model_label, "device": str(device)}), flush=True)
        model, ident = load_model_for_label(model_label, device, args.private_scale)
        model.eval()
        identities[model_label] = ident
        for i, item in enumerate(suite):
            try:
                scored = score_variant(model, tokenizer, item, device, args.seq_length)
                rec = analyze_record(item, scored, model_label)
                records.append(rec)
            except Exception as e:
                errors.append({"model": model_label, "pair_id": item.get("pair_id"), "variant": item.get("variant"), "error": repr(e)})
            if (i + 1) % 20 == 0:
                print(json.dumps({"event": "score_progress", "model": model_label, "n": i + 1, "total": len(suite)}), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_jsonl(out_dir / "expression_transfer_records.jsonl", records)
    summary = {
        "status": "EXPRESSION_TRANSFER_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_purpose": "Operation-preserving expression transfer: same relation-first state maps and simultaneous-span scoring, but rephrased update and/or query frames. This tests wording dependence of the acquired selector, not broad portability by itself.",
        "pairs_path": rel(args.pairs),
        "n_pairs": len(pairs),
        "n_suite_items": len(suite),
        "models": args.models,
        "variants": args.variants,
        "relations": dict(collections.Counter(p["relation"] for p in pairs)),
        "skipped": skipped,
        "identities": identities,
        "scoring_contract": "For a fixed query and fixed candidate set (value_a,value_b,shared_new), self-update should choose shared_new and distractor-update should retain the queried source value; all candidate spans are masked simultaneously.",
        "same_type_boundary": "Most examples are death_place or birthplace; value_a, value_b, and shared_new are place-like candidates, reducing but not eliminating non-contextual priors.",
        "summary": summarize(records),
        "n_errors": len(errors),
        "errors": errors[:20],
        "outputs": {
            "suite": rel(out_dir / "expression_transfer_suite.jsonl"),
            "records": rel(out_dir / "expression_transfer_records.jsonl"),
            "summary": rel(out_dir / "expression_transfer_summary.json"),
        },
        "elapsed_sec": round(time.time() - start_time, 3),
    }
    (out_dir / "expression_transfer_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
