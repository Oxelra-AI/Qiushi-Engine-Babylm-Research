#!/usr/bin/env python3
"""research: saved-state replication with neutral retrieval and 3-entity probe.

This script repairs research's transient-state limitation.  It fixes the relation-first
construction seed so all training seeds answer the same held questions, explicitly
sets the training RNG, saves every scored trained model, emits a model-identity
preamble for the saved checkpoint, and evaluates:

1. held two-entity three-way behavior including the neutral contexts omitted in
   research's analysis;
2. source reassignment for contextual reading rather than memorization;
3. a small three-entity held probe where one entity is updated and the query
   switches between the two unaffected entities, so query identity cannot be
   inferred from the update identity.

The scientific object is behavioral compatibility and replication of the acquired
state-selection computation, not an official BabyLM result.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import copy
import gc
import hashlib
import json
import math
import pathlib
import random
import shutil
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
sys.path.insert(0, str(_public_path('experiments/archive/functional_learning/scripts')))

import threeway_and_reassignment_probe as probe  # noqa: E402


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def finite_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def reset_training_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # These flags make the intended separation explicit.  The exact CUDA kernels
    # used by DeBERTa may still choose nondeterministic algorithms if unavailable,
    # so the saved identity and replicated behavioral readouts remain the evidence.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Fixed construction and standard row set
# ---------------------------------------------------------------------------


def fixed_repaired_pairs(construction_seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    v1_pairs = probe.read_jsonl(probe.V1_PAIRS)
    pairs = probe.construct_repaired_pairs(v1_pairs, construction_seed)
    train_pairs = [p for p in pairs if p.get("split") == "train"]
    held_pairs = [p for p in pairs if p.get("split") == "held"]
    rows = probe.build_scoring_rows(pairs)
    train_rows = [r for r in rows if r.get("split") == "train" and r.get("role") != "NEUTRAL"]
    held_rows = [r for r in rows if r.get("split") == "held"]
    return pairs, train_pairs, held_pairs, train_rows, held_rows


# ---------------------------------------------------------------------------
# Extended neutral-vs-distractor analysis
# ---------------------------------------------------------------------------


def _mean_logp(tw_context: Dict[str, Dict[str, Any]], label: str) -> float:
    return float(tw_context[label]["mean_logp"])


def analyze_threeway_extended(pair_results: Sequence[Dict[str, Any]], pairs: Sequence[Dict[str, Any]], label: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Analyze three-way records including neutral source retrieval.

    For query_a: correct source is value_a and wrong source is value_b.
    For query_b: correct source is value_b and wrong source is value_a.
    RETAIN contexts are exactly the same queries after an irrelevant update to
    the other entity.  Comparing RETAIN to NEUTRAL distinguishes basic source
    binding from resistance to correction-induced new-phrase preference.
    """
    records: List[Dict[str, Any]] = []
    for pair, tw in zip(pairs, pair_results):
        # Neutral contexts
        n_qa = tw["neutral_query_a"]
        n_qb = tw["neutral_query_b"]
        n_qa_correct = _mean_logp(n_qa, "value_a")
        n_qa_wrong = _mean_logp(n_qa, "value_b")
        n_qa_new = _mean_logp(n_qa, "shared_new")
        n_qb_correct = _mean_logp(n_qb, "value_b")
        n_qb_wrong = _mean_logp(n_qb, "value_a")
        n_qb_new = _mean_logp(n_qb, "shared_new")

        # Distractor-update RETAIN contexts
        r_qa = tw["update_b_query_a"]  # B updated, query A should keep value_a
        r_qb = tw["update_a_query_b"]  # A updated, query B should keep value_b
        r_qa_correct = _mean_logp(r_qa, "value_a")
        r_qa_wrong = _mean_logp(r_qa, "value_b")
        r_qa_new = _mean_logp(r_qa, "shared_new")
        r_qb_correct = _mean_logp(r_qb, "value_b")
        r_qb_wrong = _mean_logp(r_qb, "value_a")
        r_qb_new = _mean_logp(r_qb, "shared_new")

        # Self-update contexts: shared_new should dominate the old source value
        u_qa = tw["update_a_query_a"]
        u_qb = tw["update_b_query_b"]
        u_qa_new = _mean_logp(u_qa, "shared_new")
        u_qa_source = _mean_logp(u_qa, "value_a")
        u_qb_new = _mean_logp(u_qb, "shared_new")
        u_qb_source = _mean_logp(u_qb, "value_b")

        rec = {
            "pair_id": pair["pair_id"],
            "split": pair.get("split"),
            "relation": pair.get("relation"),
            "entity_a": pair.get("entity_a"),
            "entity_b": pair.get("entity_b"),
            "value_a": pair.get("value_a"),
            "value_b": pair.get("value_b"),
            "shared_new_value": pair.get("shared_new_value"),
            # Neutral source selection
            "neutral_qa_cross_source": n_qa_correct - n_qa_wrong,
            "neutral_qb_cross_source": n_qb_correct - n_qb_wrong,
            "neutral_qa_correct_over_new": n_qa_correct - n_qa_new,
            "neutral_qb_correct_over_new": n_qb_correct - n_qb_new,
            "neutral_qa_entity_retrieval": n_qa_correct > n_qa_wrong,
            "neutral_qb_entity_retrieval": n_qb_correct > n_qb_wrong,
            "neutral_qa_full_source": n_qa_correct > n_qa_wrong and n_qa_correct > n_qa_new,
            "neutral_qb_full_source": n_qb_correct > n_qb_wrong and n_qb_correct > n_qb_new,
            # Retain after irrelevant/distractor update
            "retain_qa_cross_source": r_qa_correct - r_qa_wrong,
            "retain_qb_cross_source": r_qb_correct - r_qb_wrong,
            "retain_qa_correct_over_new": r_qa_correct - r_qa_new,
            "retain_qb_correct_over_new": r_qb_correct - r_qb_new,
            "retain_qa_entity_retrieval": r_qa_correct > r_qa_wrong,
            "retain_qb_entity_retrieval": r_qb_correct > r_qb_wrong,
            "retain_qa_full_source": r_qa_correct > r_qa_wrong and r_qa_correct > r_qa_new,
            "retain_qb_full_source": r_qb_correct > r_qb_wrong and r_qb_correct > r_qb_new,
            # Update behavior
            "update_qa_new_over_source": u_qa_new - u_qa_source,
            "update_qb_new_over_source": u_qb_new - u_qb_source,
            "update_qa_correct": u_qa_new > u_qa_source,
            "update_qb_correct": u_qb_new > u_qb_source,
            # Delta caused by the irrelevant update (negative correct-over-new = correction-induced loss)
            "distractor_delta_qa_cross_source": (r_qa_correct - r_qa_wrong) - (n_qa_correct - n_qa_wrong),
            "distractor_delta_qb_cross_source": (r_qb_correct - r_qb_wrong) - (n_qb_correct - n_qb_wrong),
            "distractor_delta_qa_correct_over_new": (r_qa_correct - r_qa_new) - (n_qa_correct - n_qa_new),
            "distractor_delta_qb_correct_over_new": (r_qb_correct - r_qb_new) - (n_qb_correct - n_qb_new),
            "neutral_scores": {
                "query_a": {k: _mean_logp(n_qa, k) for k in ["value_a", "value_b", "shared_new"]},
                "query_b": {k: _mean_logp(n_qb, k) for k in ["value_a", "value_b", "shared_new"]},
            },
            "retain_scores": {
                "query_a_after_update_b": {k: _mean_logp(r_qa, k) for k in ["value_a", "value_b", "shared_new"]},
                "query_b_after_update_a": {k: _mean_logp(r_qb, k) for k in ["value_a", "value_b", "shared_new"]},
            },
        }
        records.append(rec)

    n = len(records)
    summary = {
        "label": label,
        "n_pairs": n,
        "neutral_both_entity_retrieval": sum(1 for r in records if r["neutral_qa_entity_retrieval"] and r["neutral_qb_entity_retrieval"]),
        "neutral_both_full_source": sum(1 for r in records if r["neutral_qa_full_source"] and r["neutral_qb_full_source"]),
        "retain_both_entity_retrieval": sum(1 for r in records if r["retain_qa_entity_retrieval"] and r["retain_qb_entity_retrieval"]),
        "retain_both_full_source": sum(1 for r in records if r["retain_qa_full_source"] and r["retain_qb_full_source"]),
        "update_both_correct": sum(1 for r in records if r["update_qa_correct"] and r["update_qb_correct"]),
        "mean_neutral_cross_source": finite_mean([r["neutral_qa_cross_source"] for r in records] + [r["neutral_qb_cross_source"] for r in records]),
        "mean_retain_cross_source": finite_mean([r["retain_qa_cross_source"] for r in records] + [r["retain_qb_cross_source"] for r in records]),
        "mean_neutral_correct_over_new": finite_mean([r["neutral_qa_correct_over_new"] for r in records] + [r["neutral_qb_correct_over_new"] for r in records]),
        "mean_retain_correct_over_new": finite_mean([r["retain_qa_correct_over_new"] for r in records] + [r["retain_qb_correct_over_new"] for r in records]),
        "mean_distractor_delta_cross_source": finite_mean([r["distractor_delta_qa_cross_source"] for r in records] + [r["distractor_delta_qb_cross_source"] for r in records]),
        "mean_distractor_delta_correct_over_new": finite_mean([r["distractor_delta_qa_correct_over_new"] for r in records] + [r["distractor_delta_qb_correct_over_new"] for r in records]),
        "mean_update_new_over_source": finite_mean([r["update_qa_new_over_source"] for r in records] + [r["update_qb_new_over_source"] for r in records]),
    }
    return summary, records


# ---------------------------------------------------------------------------
# Source reassignment and entity familiarity helpers
# ---------------------------------------------------------------------------


def score_reassignment_held(model, tokenizer, pairs: Sequence[Dict[str, Any]], held_tw: Sequence[Dict[str, Any]], device, seq_length: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int]:
    valid_pairs: List[Dict[str, Any]] = []
    valid_ctxs: List[Dict[str, Any]] = []
    valid_tw: List[Dict[str, Any]] = []
    for p, tw in zip(pairs, held_tw):
        ctx, valid, _info = probe.construct_reassignment_contexts(p)
        if valid:
            valid_pairs.append(p)
            valid_ctxs.append(ctx)
            valid_tw.append(tw)
    results = []
    for p, ctx in zip(valid_pairs, valid_ctxs):
        results.append(probe.reassignment_score_pair(model, tokenizer, p, ctx, device, seq_length))
    summary, records = probe.analyze_reassignment(results, valid_pairs, valid_tw, "held_reassignment")
    return summary, records, len(valid_pairs)


def entity_familiarity_records(train_pairs: Sequence[Dict[str, Any]], held_pairs: Sequence[Dict[str, Any]], extended_records: Sequence[Dict[str, Any]], reassignment_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    train_entities = {p["entity_a"] for p in train_pairs} | {p["entity_b"] for p in train_pairs}
    fam_by_pid: Dict[str, str] = {}
    for p in held_pairs:
        ea, eb = p["entity_a"], p["entity_b"]
        if ea not in train_entities and eb not in train_entities:
            fam = "both_novel"
        elif ea in train_entities and eb in train_entities:
            fam = "both_familiar"
        else:
            fam = "one_familiar"
        fam_by_pid[p["pair_id"]] = fam

    def summarize_ext(sub: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n": len(sub),
            "neutral_both_full_source": sum(1 for r in sub if r["neutral_qa_full_source"] and r["neutral_qb_full_source"]),
            "retain_both_full_source": sum(1 for r in sub if r["retain_qa_full_source"] and r["retain_qb_full_source"]),
            "retain_both_entity_retrieval": sum(1 for r in sub if r["retain_qa_entity_retrieval"] and r["retain_qb_entity_retrieval"]),
            "mean_retain_cross_source": finite_mean([r["retain_qa_cross_source"] for r in sub] + [r["retain_qb_cross_source"] for r in sub]),
            "mean_retain_correct_over_new": finite_mean([r["retain_qa_correct_over_new"] for r in sub] + [r["retain_qb_correct_over_new"] for r in sub]),
        }

    def summarize_reassign(sub: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n": len(sub),
            "follows_both": sum(1 for r in sub if r["swap_qa_follows_reassignment"] and r["swap_qb_follows_reassignment"]),
            "correct_source_both": sum(1 for r in sub if r["swap_qa_correct_source"] and r["swap_qb_correct_source"]),
            "mean_swap_margin": finite_mean([r["swap_qa_margin"] for r in sub] + [r["swap_qb_margin"] for r in sub]),
        }

    out = {
        "train_entity_count": len(train_entities),
        "held_counts": collections.Counter(fam_by_pid.values()),
        "by_familiarity_threeway": {},
        "by_familiarity_reassignment": {},
    }
    out["held_counts"] = dict(out["held_counts"])
    for fam in ["both_novel", "one_familiar", "both_familiar"]:
        out["by_familiarity_threeway"][fam] = summarize_ext([r for r in extended_records if fam_by_pid.get(r["pair_id"]) == fam])
        out["by_familiarity_reassignment"][fam] = summarize_reassign([r for r in reassignment_records if fam_by_pid.get(r["pair_id"]) == fam])
    return out


# ---------------------------------------------------------------------------
# Three-entity held probe
# ---------------------------------------------------------------------------


def triple_records_from_pairs(pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for p in pairs:
        records.append({
            "pair_id": p["pair_id"], "split": p.get("split"), "side": "a", "relation": p["relation"],
            "entity": p["entity_a"], "value": p["value_a"], "sentence": p["source_a_sentence"],
        })
        records.append({
            "pair_id": p["pair_id"], "split": p.get("split"), "side": "b", "relation": p["relation"],
            "entity": p["entity_b"], "value": p["value_b"], "sentence": p["source_b_sentence"],
        })
    return records


def build_three_entity_cases(pairs: Sequence[Dict[str, Any]], train_pairs: Sequence[Dict[str, Any]], held_pairs: Sequence[Dict[str, Any]], construction_seed: int, max_cases: int = 12) -> List[Dict[str, Any]]:
    rng = random.Random(construction_seed + 4242)
    train_entities = {p["entity_a"] for p in train_pairs} | {p["entity_b"] for p in train_pairs}
    triples = triple_records_from_pairs(pairs)
    updater_pool = [t for t in triples if t["split"] == "train"]

    # Prefer held pairs whose two queried entities were unseen during training.
    candidate_pairs = [p for p in held_pairs if p["entity_a"] not in train_entities and p["entity_b"] not in train_entities]
    if len(candidate_pairs) < max_cases:
        rest = [p for p in held_pairs if p not in candidate_pairs]
        rng.shuffle(rest)
        candidate_pairs.extend(rest)
    cases: List[Dict[str, Any]] = []
    for p in candidate_pairs:
        if len(cases) >= max_cases:
            break
        rel = p["relation"]
        tmpl = probe.RELATION_TEMPLATES.get(rel)
        if tmpl is None:
            continue
        forbidden_entities = {p["entity_a"].lower(), p["entity_b"].lower()}
        forbidden_values = {p["value_a"].lower(), p["value_b"].lower(), p["shared_new_value"].lower()}
        pool = [t for t in updater_pool
                if t["relation"] == rel
                and t["entity"].lower() not in forbidden_entities
                and t["value"].lower() not in forbidden_values
                and p["entity_a"].lower() not in t["sentence"].lower()
                and p["entity_b"].lower() not in t["sentence"].lower()
                and p["value_a"].lower() not in t["sentence"].lower()
                and p["value_b"].lower() not in t["sentence"].lower()]
        if not pool:
            continue
        c = rng.choice(pool)
        source_context = p["source_a_sentence"].rstrip() + " " + p["source_b_sentence"].rstrip() + " " + c["sentence"].rstrip()
        # Skip overly long cases that risk truncating the final answer span.
        update_c = tmpl["update"].format(ENTITY=c["entity"], NEW_VALUE=p["shared_new_value"])
        context_update_c = source_context + " " + update_c
        cases.append({
            "case_id": f"three_entity_{len(cases):03d}_{p['pair_id']}",
            "base_pair_id": p["pair_id"],
            "relation": rel,
            "split": "held_three_entity",
            "unaffected_entity_a": p["entity_a"],
            "unaffected_entity_b": p["entity_b"],
            "updated_entity_c": c["entity"],
            "value_a": p["value_a"],
            "value_b": p["value_b"],
            "value_c": c["value"],
            "shared_new_value": p["shared_new_value"],
            "source_context": source_context,
            "update_c_sentence": update_c,
            "context_update_c": context_update_c,
            "context_neutral": source_context,
            "use_frame_a": p["use_frame_a"],
            "use_frame_b": p["use_frame_b"],
            "query_entities_both_unseen_in_train": (p["entity_a"] not in train_entities and p["entity_b"] not in train_entities),
            "updated_entity_source_pair": c["pair_id"],
        })
    return cases


def score_three_entity_case(model, tokenizer, case: Dict[str, Any], device, seq_length: int) -> Dict[str, Any]:
    candidates = [case["value_a"], case["value_b"], case["shared_new_value"]]
    labels = ["value_a", "value_b", "shared_new"]
    contexts = {
        "neutral_query_a": case["context_neutral"] + " " + case["use_frame_a"],
        "neutral_query_b": case["context_neutral"] + " " + case["use_frame_b"],
        "update_c_query_a": case["context_update_c"] + " " + case["use_frame_a"],
        "update_c_query_b": case["context_update_c"] + " " + case["use_frame_b"],
    }
    out: Dict[str, Any] = {}
    for ctx_name, frame in contexts.items():
        scores = {}
        for lab, cand in zip(labels, candidates):
            s = probe.score_candidate_in_context(model, tokenizer, frame, cand, device, seq_length)
            scores[lab] = s
        out[ctx_name] = scores
    return out


def analyze_three_entity(scored: Sequence[Dict[str, Any]], cases: Sequence[Dict[str, Any]], label: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for case, tw in zip(cases, scored):
        def ml(ctx: str, lab: str) -> float:
            return float(tw[ctx][lab]["mean_logp"])
        # Query A: correct value_a, competing unaffected value_b, shared_new.
        ua_a = ml("update_c_query_a", "value_a")
        ua_b = ml("update_c_query_a", "value_b")
        ua_n = ml("update_c_query_a", "shared_new")
        ub_a = ml("update_c_query_b", "value_a")
        ub_b = ml("update_c_query_b", "value_b")
        ub_n = ml("update_c_query_b", "shared_new")
        na_a = ml("neutral_query_a", "value_a")
        na_b = ml("neutral_query_a", "value_b")
        na_n = ml("neutral_query_a", "shared_new")
        nb_a = ml("neutral_query_b", "value_a")
        nb_b = ml("neutral_query_b", "value_b")
        nb_n = ml("neutral_query_b", "shared_new")
        rec = {
            "case_id": case["case_id"],
            "base_pair_id": case["base_pair_id"],
            "relation": case["relation"],
            "query_entities_both_unseen_in_train": bool(case["query_entities_both_unseen_in_train"]),
            "update_c_query_a_cross_source": ua_a - ua_b,
            "update_c_query_b_cross_source": ub_b - ub_a,
            "update_c_query_a_correct_over_new": ua_a - ua_n,
            "update_c_query_b_correct_over_new": ub_b - ub_n,
            "update_c_query_a_correct": ua_a > ua_b and ua_a > ua_n,
            "update_c_query_b_correct": ub_b > ub_a and ub_b > ub_n,
            "neutral_query_a_correct": na_a > na_b and na_a > na_n,
            "neutral_query_b_correct": nb_b > nb_a and nb_b > nb_n,
            "neutral_query_a_cross_source": na_a - na_b,
            "neutral_query_b_cross_source": nb_b - nb_a,
            "neutral_query_a_correct_over_new": na_a - na_n,
            "neutral_query_b_correct_over_new": nb_b - nb_n,
            "scores": {
                "update_c_query_a": {"value_a": ua_a, "value_b": ua_b, "shared_new": ua_n},
                "update_c_query_b": {"value_a": ub_a, "value_b": ub_b, "shared_new": ub_n},
                "neutral_query_a": {"value_a": na_a, "value_b": na_b, "shared_new": na_n},
                "neutral_query_b": {"value_a": nb_a, "value_b": nb_b, "shared_new": nb_n},
            },
        }
        rec["update_c_both_query_indexed"] = rec["update_c_query_a_correct"] and rec["update_c_query_b_correct"]
        rec["neutral_both_query_indexed"] = rec["neutral_query_a_correct"] and rec["neutral_query_b_correct"]
        records.append(rec)
    summary = {
        "label": label,
        "n_cases": len(records),
        "n_both_unseen_query_cases": sum(1 for r in records if r["query_entities_both_unseen_in_train"]),
        "neutral_both_query_indexed": sum(1 for r in records if r["neutral_both_query_indexed"]),
        "update_c_both_query_indexed": sum(1 for r in records if r["update_c_both_query_indexed"]),
        "neutral_mean_cross_source": finite_mean([r["neutral_query_a_cross_source"] for r in records] + [r["neutral_query_b_cross_source"] for r in records]),
        "update_c_mean_cross_source": finite_mean([r["update_c_query_a_cross_source"] for r in records] + [r["update_c_query_b_cross_source"] for r in records]),
        "neutral_mean_correct_over_new": finite_mean([r["neutral_query_a_correct_over_new"] for r in records] + [r["neutral_query_b_correct_over_new"] for r in records]),
        "update_c_mean_correct_over_new": finite_mean([r["update_c_query_a_correct_over_new"] for r in records] + [r["update_c_query_b_correct_over_new"] for r in records]),
    }
    return summary, records


# ---------------------------------------------------------------------------
# Saved checkpoint identity
# ---------------------------------------------------------------------------


def inspect_saved_checkpoint(model_dir: pathlib.Path) -> Dict[str, Any]:
    cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    dynamic_file = model_dir / "frozen82_private_modeling.py"
    out: Dict[str, Any] = {
        "checkpoint_path": rel(model_dir),
        "config_architectures": list(getattr(cfg, "architectures", []) or []),
        "config_model_type": getattr(cfg, "model_type", None),
        "config_auto_map": getattr(cfg, "auto_map", None),
        "dynamic_modeling_file": rel(dynamic_file) if dynamic_file.exists() else None,
        "model_safetensors_sha256": sha256_file(model_dir / "model.safetensors") if (model_dir / "model.safetensors").exists() else None,
    }
    try:
        loaded = AutoModelForMaskedLM.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
        named = list(loaded.named_parameters())
        private = [(n, p) for n, p in named if ".private_adapter." in n]
        scales = []
        try:
            scales = [float(layer.private_adapter.scale) for layer in loaded.deberta.encoder.layer]
        except Exception:
            pass
        out.update({
            "trust_remote_code_loaded_class": type(loaded).__name__,
            "trust_remote_code_loaded_module": type(loaded).__module__,
            "total_params": int(sum(p.numel() for _, p in named)),
            "private_adapter_params": int(sum(p.numel() for _, p in private)),
            "private_adapter_tensor_count": len(private),
            "executed_private_scales": scales,
        })
        del loaded
    except Exception as exc:
        out["trust_remote_code_load_error"] = repr(exc)
    try:
        plain, info = AutoModelForMaskedLM.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False, output_loading_info=True
        )
        state_unexpected = list(info.get("unexpected_keys") or [])
        state_missing = list(info.get("missing_keys") or [])
        out["plain_loader"] = {
            "class": type(plain).__name__,
            "module": type(plain).__module__,
            "total_params": int(sum(p.numel() for p in plain.parameters())),
            "private_adapter_params": int(sum(p.numel() for n, p in plain.named_parameters() if ".private_adapter." in n)),
            "unexpected_private_key_count": sum(1 for k in state_unexpected if "private_adapter" in k),
            "unexpected_key_count": len(state_unexpected),
            "missing_key_count": len(state_missing),
        }
        del plain
    except Exception as exc:
        out["plain_loader"] = {"error": repr(exc)}
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_seed(args: argparse.Namespace, train_seed: int, fixed: Dict[str, Any], tokenizer, parent_cache: Optional[Dict[str, Any]], device) -> Dict[str, Any]:
    out_dir = pathlib.Path(args.out_dir)
    seed_dir = out_dir / f"seed_{train_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pairs = fixed["pairs"]
    train_pairs = fixed["train_pairs"]
    held_pairs = fixed["held_pairs"]
    train_rows = fixed["train_rows"]
    three_cases = fixed["three_entity_cases"]

    reset_training_rng(train_seed)
    model, load_info = probe.load_trusted_model(device, private_scale=args.private_scale)
    parent_identity = probe.model_identity(model, load_info)
    print(f"seed {train_seed}: loaded parent {parent_identity['class']} private={parent_identity['private_params']} scales={parent_identity['executed_private_scales']}", flush=True)

    # Parent held scoring is invariant across train seeds; run only when not provided.
    if parent_cache is None:
        print(f"seed {train_seed}: scoring parent held three-way", flush=True)
        model.eval()
        parent_held_tw = []
        for i, p in enumerate(held_pairs):
            parent_held_tw.append(probe.threeway_score_pair(model, tokenizer, p, device, args.seq_length))
            if (i + 1) % 10 == 0:
                print(f"  parent held {i+1}/{len(held_pairs)}", flush=True)
        parent_ext_summary, parent_ext_records = analyze_threeway_extended(parent_held_tw, held_pairs, "parent_held")
        parent_reassign_summary, parent_reassign_records, parent_reassign_n = score_reassignment_held(model, tokenizer, held_pairs, parent_held_tw, device, args.seq_length)
        parent_three_scored = [score_three_entity_case(model, tokenizer, c, device, args.seq_length) for c in three_cases]
        parent_three_summary, parent_three_records = analyze_three_entity(parent_three_scored, three_cases, "parent_three_entity")
        parent_cache = {
            "held_tw": parent_held_tw,
            "extended_summary": parent_ext_summary,
            "extended_records": parent_ext_records,
            "reassignment_summary": parent_reassign_summary,
            "reassignment_records": parent_reassign_records,
            "reassignment_valid_n": parent_reassign_n,
            "three_entity_summary": parent_three_summary,
            "three_entity_records": parent_three_records,
        }
        write_jsonl(out_dir / "parent_held_extended_records.jsonl", parent_ext_records)
        write_jsonl(out_dir / "parent_reassignment_records.jsonl", parent_reassign_records)
        write_jsonl(out_dir / "parent_three_entity_records.jsonl", parent_three_records)

    # Train the parent model in-place into the acquired state.
    print(f"seed {train_seed}: training answer-only private adapter for {args.epochs} epochs with fixed construction seed {args.construction_seed}", flush=True)
    trainable_info, elapsed = probe.train_model(
        model, tokenizer, train_rows, args.epochs, args.batch_size,
        args.lr, args.weight_decay, args.max_grad_norm, train_seed, device, args.seq_length
    )

    # Save exact state before scoring/cleanup.
    model_dir = seed_dir / "hf_model" / "final"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(str(model_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(model_dir))
    checkpoint_identity = inspect_saved_checkpoint(model_dir)
    (seed_dir / "checkpoint_identity.json").write_text(json.dumps(checkpoint_identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Score held behavior on the same in-memory state.
    print(f"seed {train_seed}: scoring trained held three-way", flush=True)
    trained_held_tw = []
    for i, p in enumerate(held_pairs):
        trained_held_tw.append(probe.threeway_score_pair(model, tokenizer, p, device, args.seq_length))
        if (i + 1) % 10 == 0:
            print(f"  trained held {i+1}/{len(held_pairs)}", flush=True)
    trained_ext_summary, trained_ext_records = analyze_threeway_extended(trained_held_tw, held_pairs, f"trained_held_seed{train_seed}")
    trained_reassign_summary, trained_reassign_records, trained_reassign_n = score_reassignment_held(model, tokenizer, held_pairs, trained_held_tw, device, args.seq_length)
    trained_three_scored = [score_three_entity_case(model, tokenizer, c, device, args.seq_length) for c in three_cases]
    trained_three_summary, trained_three_records = analyze_three_entity(trained_three_scored, three_cases, f"trained_three_entity_seed{train_seed}")
    fam = entity_familiarity_records(train_pairs, held_pairs, trained_ext_records, trained_reassign_records)

    write_jsonl(seed_dir / "trained_held_extended_records.jsonl", trained_ext_records)
    write_jsonl(seed_dir / "trained_reassignment_records.jsonl", trained_reassign_records)
    write_jsonl(seed_dir / "trained_three_entity_records.jsonl", trained_three_records)

    seed_result = {
        "seed": train_seed,
        "construction_seed": args.construction_seed,
        "training_rng_seed": train_seed,
        "epochs": args.epochs,
        "model_dir": rel(model_dir),
        "parent_identity_in_memory": parent_identity,
        "checkpoint_identity": checkpoint_identity,
        "trainable_info": trainable_info,
        "training_elapsed_sec": elapsed,
        "trained_extended_held": trained_ext_summary,
        "trained_reassignment_held": trained_reassign_summary,
        "trained_reassignment_valid_n": trained_reassign_n,
        "trained_three_entity": trained_three_summary,
        "entity_familiarity": fam,
    }
    (seed_dir / "seed_summary.json").write_text(json.dumps(seed_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "seed_done",
        "seed": train_seed,
        "model_dir": rel(model_dir),
        "neutral_full": trained_ext_summary["neutral_both_full_source"],
        "retain_full": trained_ext_summary["retain_both_full_source"],
        "reassign_both": trained_reassign_summary.get("n_swap_both_follow"),
        "three_entity": trained_three_summary.get("update_c_both_query_indexed"),
    }, ensure_ascii=False), flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return seed_result, parent_cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--train-seeds", default="40040")
    ap.add_argument("--construction-seed", type=int, default=40040)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--max-three-entity-cases", type=int, default=12)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    train_seeds = [int(x) for x in args.train_seeds.split(",") if x.strip()]

    pairs, train_pairs, held_pairs, train_rows, held_rows = fixed_repaired_pairs(args.construction_seed)
    three_cases = build_three_entity_cases(pairs, train_pairs, held_pairs, args.construction_seed, max_cases=args.max_three_entity_cases)
    fixed = {
        "pairs": pairs,
        "train_pairs": train_pairs,
        "held_pairs": held_pairs,
        "train_rows": train_rows,
        "held_rows": held_rows,
        "three_entity_cases": three_cases,
    }
    write_jsonl(out_dir / "fixed_repaired_pairs.jsonl", pairs)
    write_jsonl(out_dir / "fixed_train_rows.jsonl", train_rows)
    write_jsonl(out_dir / "three_entity_cases.jsonl", three_cases)
    construction_summary = {
        "status": "FIXED_CONSTRUCTION_READY",
        "construction_seed": args.construction_seed,
        "n_pairs": len(pairs),
        "n_train_pairs": len(train_pairs),
        "n_held_pairs": len(held_pairs),
        "n_train_answer_rows": len(train_rows),
        "n_three_entity_cases": len(three_cases),
        "n_three_entity_both_unseen_queries": sum(1 for c in three_cases if c["query_entities_both_unseen_in_train"]),
        "train_seeds": train_seeds,
    }
    (out_dir / "construction_summary.json").write_text(json.dumps(construction_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(construction_summary, ensure_ascii=False), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(probe.MODEL_PATH), local_files_only=True, use_fast=True)
    parent_cache: Optional[Dict[str, Any]] = None
    results = []
    for seed in train_seeds:
        # run_seed signature keeps tokenizer before the mutable parent_cache;
        # the failed research first launch accidentally passed these in reverse order.
        result, parent_cache = run_seed(args, seed, fixed, tokenizer, parent_cache, device)
        results.append(result)

    final = {
        "status": "SAVED_STATE_REPLICATE_NEUTRAL_THREEENTITY_DONE",
        "construction_summary": construction_summary,
        "parent_extended_held": parent_cache["extended_summary"] if parent_cache else None,
        "parent_reassignment_held": parent_cache["reassignment_summary"] if parent_cache else None,
        "parent_three_entity": parent_cache["three_entity_summary"] if parent_cache else None,
        "per_seed": results,
        "aggregate": {
            "seeds": train_seeds,
            "mean_retain_both_full_source": finite_mean([r["trained_extended_held"]["retain_both_full_source"] for r in results]),
            "mean_neutral_both_full_source": finite_mean([r["trained_extended_held"]["neutral_both_full_source"] for r in results]),
            "mean_reassignment_both_follow": finite_mean([r["trained_reassignment_held"].get("n_swap_both_follow", float("nan")) for r in results]),
            "mean_three_entity_update_c_both_query_indexed": finite_mean([r["trained_three_entity"].get("update_c_both_query_indexed", float("nan")) for r in results]),
        },
    }
    (out_dir / "saved_state_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + "=" * 60, flush=True)
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
