#!/usr/bin/env python3
"""research repaired relation-first harness.

Preserves the research relation-first extraction but restores the experimental
contract needed for recipient-sensitive binding evidence:

1. Trusted coherent86 loading: use the established FrozenSlowPrivateDebertaV2
   private-adapter loader, not generic AutoModelForMaskedLM.
2. Recipient-only contrasts: for a fixed query and candidate pair, changing the
   update recipient changes only the recipient identity, not the competing new
   value. Each entity pair therefore has one shared replacement value.
3. One final-span measurement: answer and foil scores mask the explicit final
   {STATE} span simultaneously. No token-search fallback is allowed.
4. Correct success statistics: pair success is the minimum over four individual
   signed margins, not gamma after averaging.

The script can construct repaired rows, score the trusted parent, and run a
bounded private-adapter-only answer supervision pilot.
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
import json
import math
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
A01_SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(A02_SCRIPTS))
sys.path.insert(0, str(A01_SCRIPTS))

import coherent86_continuation_trainer as trusted_loader  # noqa: E402

MODEL_PATH = _public_path('models/frontier')
V1_PAIRS = _public_path('experiments/archive/functional_learning/data/relation_first_packets/relation_first_pairs.jsonl')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/relation_first_repaired')

REPLACEMENT_CITIES = [
    "Amsterdam", "Barcelona", "Dublin", "Milan", "Tokyo", "Sydney",
    "Vienna", "Copenhagen", "Stockholm", "Athens", "Lisbon", "Prague",
    "Budapest", "Helsinki", "Cairo", "Montreal", "Zurich", "Brisbane",
    "Edinburgh", "Florence", "Geneva", "Hamburg", "Kyoto", "Munich",
    "Osaka", "Salzburg", "Venice", "Brussels", "Oslo", "Marseille",
    "Ankara", "Lima", "Bogota", "Manila", "Jakarta", "Havana",
    "Nairobi", "Doha", "Riyadh", "Bangalore",
]
REPLACEMENT_YEARS = [str(y) for y in range(1880, 2010, 5)]

RELATION_TEMPLATES = {
    "birthplace": {
        "update": "However, recent records show that {ENTITY} was actually born in {NEW_VALUE}.",
        "use": "According to this information, the birthplace of {QUERY} is {STATE}.",
    },
    "death_place": {
        "update": "However, updated records confirm that {ENTITY} actually died in {NEW_VALUE}.",
        "use": "According to this information, the place where {QUERY} died is {STATE}.",
    },
    "located_in": {
        "update": "However, after a boundary change, {ENTITY} is now part of {NEW_VALUE}.",
        "use": "According to current records, {QUERY} is located in {STATE}.",
    },
    "founded_year": {
        "update": "However, newly discovered documents show that {ENTITY} was actually founded in {NEW_VALUE}.",
        "use": "According to the latest records, {QUERY} was founded in {STATE}.",
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
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


def finite_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def quantiles(xs: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return {"n": 0}
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = p * (len(vals) - 1)
        lo, hi = math.floor(idx), math.ceil(idx)
        if lo == hi:
            return vals[int(lo)]
        return vals[int(lo)] * (hi - idx) + vals[int(hi)] * (idx - lo)
    return {"n": len(vals), "mean": finite_mean(vals), "min": vals[0], "p10": q(0.10), "median": q(0.50), "p90": q(0.90), "max": vals[-1]}


# ---------------------------------------------------------------------------
# Trusted model loading
# ---------------------------------------------------------------------------

def load_trusted_model(device: torch.device, private_scale: float = 0.75):
    model, missing, unexpected = trusted_loader.load_model(MODEL_PATH, device, 128, private_scale)
    if unexpected:
        raise RuntimeError(f"Trusted loader unexpected keys: {unexpected[:10]}")
    bad_missing = [k for k in missing if "private_adapter" in k]
    if bad_missing:
        raise RuntimeError(f"Trusted loader missing private keys: {bad_missing[:10]}")
    if hasattr(model, "set_private_enabled"):
        model.set_private_enabled(True)
    return model, {"missing": list(missing), "unexpected": list(unexpected)}


def executed_scales(model) -> List[float]:
    vals = []
    for layer in model.deberta.encoder.layer:
        vals.append(float(layer.private_adapter.scale))
    return vals


def model_identity(model, load_info: Dict[str, Any]) -> Dict[str, Any]:
    named = list(model.named_parameters())
    private = [(n, p) for n, p in named if ".private_adapter." in n]
    return {
        "class": type(model).__name__,
        "module": type(model).__module__,
        "total_tensors": len(named),
        "total_params": int(sum(p.numel() for _, p in named)),
        "private_tensors": len(private),
        "private_params": int(sum(p.numel() for _, p in private)),
        "executed_private_scales": executed_scales(model),
        "load_missing": load_info.get("missing", []),
        "load_unexpected": load_info.get("unexpected", []),
        "has_set_private_enabled": bool(hasattr(model, "set_private_enabled")),
        "has_private_adapter_rms": bool(hasattr(model, "private_adapter_rms")),
    }


def freeze_to_private_optimizer(model, lr: float, weight_decay: float):
    for _, p in model.named_parameters():
        p.requires_grad_(False)
    private_names = {n for n, _ in model.named_parameters() if ".private_adapter." in n}
    if not private_names:
        raise RuntimeError("No private_adapter parameters found")
    for n, p in model.named_parameters():
        p.requires_grad_(n in private_names)
    private_up_ids = {id(p) for n, p in model.named_parameters() if ".private_adapter.up." in n}
    private_normal = [p for n, p in model.named_parameters() if n in private_names and id(p) not in private_up_ids]
    private_zero_wd = [p for n, p in model.named_parameters() if n in private_names and id(p) in private_up_ids]
    opt = torch.optim.AdamW([
        {"params": private_normal, "weight_decay": float(weight_decay)},
        {"params": private_zero_wd, "weight_decay": 0.0},
    ], lr=float(lr), betas=(0.9, 0.98), eps=1e-6)
    return opt, {
        "trainable_tensor_count": sum(1 for _, p in model.named_parameters() if p.requires_grad),
        "trainable_param_count": int(sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)),
        "nonprivate_trainable_tensors": sum(1 for n, p in model.named_parameters() if p.requires_grad and n not in private_names),
        "private_names_first10": sorted(private_names)[:10],
    }


# ---------------------------------------------------------------------------
# Repaired construction
# ---------------------------------------------------------------------------

def replacement_pool(relation: str) -> List[str]:
    return REPLACEMENT_YEARS if relation == "founded_year" else REPLACEMENT_CITIES


def choose_shared_new(pair: Dict[str, Any], rng: random.Random) -> str:
    relation = pair["relation"]
    current = {pair["value_a"].lower(), pair["value_b"].lower()}
    source = str(pair["source_context"]).lower()
    # Reuse a research value if it is valid; this preserves some continuity while
    # still enforcing one shared replacement across recipient variants.
    preferred = [pair.get("new_value_a"), pair.get("new_value_b")]
    pool = [x for x in preferred + replacement_pool(relation) if x]
    candidates = []
    for v in pool:
        vl = str(v).lower()
        if vl in current:
            continue
        if vl in source:
            continue
        if v not in candidates:
            candidates.append(str(v))
    if not candidates:
        raise ValueError(f"No valid shared replacement for {pair['pair_id']}")
    # Deterministic but not always the first city in the global list.
    return rng.choice(candidates[: min(12, len(candidates))])


def construct_repaired_pairs(v1_pairs: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for p in v1_pairs:
        tmpl = RELATION_TEMPLATES.get(p["relation"])
        if tmpl is None:
            continue
        q = copy.deepcopy(p)
        q["contract_version"] = "shared_new_recipient_only_final_span"
        q["shared_new_value"] = choose_shared_new(q, rng)
        q["new_value_a_old_step039"] = q.get("new_value_a")
        q["new_value_b_old_step039"] = q.get("new_value_b")
        q["new_value_a"] = q["shared_new_value"]
        q["new_value_b"] = q["shared_new_value"]
        q["update_a_sentence"] = tmpl["update"].format(ENTITY=q["entity_a"], NEW_VALUE=q["shared_new_value"])
        q["update_b_sentence"] = tmpl["update"].format(ENTITY=q["entity_b"], NEW_VALUE=q["shared_new_value"])
        q["use_frame_a"] = tmpl["use"].format(QUERY=q["entity_a"], STATE="{STATE}")
        q["use_frame_b"] = tmpl["use"].format(QUERY=q["entity_b"], STATE="{STATE}")
        q["context_update_a"] = q["source_context"] + " " + q["update_a_sentence"]
        q["context_update_b"] = q["source_context"] + " " + q["update_b_sentence"]
        q["context_neutral"] = q["source_context"]
        q["scientific_contrast"] = "For each fixed query, update_a vs update_b changes only the update recipient while the source answer and shared new-value candidate stay fixed."
        out.append(q)
    return out


def build_rows(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    p = pair
    rows = []
    def row(row_type: str, role: str, orientation: str, updated_entity: str, query_entity: str,
            source_answer: str, answer: str, foil: str, context_prefix: str, final_frame: str) -> Dict[str, Any]:
        return {
            "pair_id": p["pair_id"],
            "split": p.get("split"),
            "relation": p["relation"],
            "contract_version": p["contract_version"],
            "row_type": row_type,
            "role": role,
            "query_orientation": orientation,
            "updated_entity": updated_entity,
            "query_entity": query_entity,
            "entity_a": p["entity_a"],
            "entity_b": p["entity_b"],
            "value_a": p["value_a"],
            "value_b": p["value_b"],
            "source_answer": source_answer,
            "shared_new_value": p["shared_new_value"],
            "answer_text": answer,
            "foil_text": foil,
            "context_prefix": context_prefix,
            "final_frame": final_frame,
            "use_sentence_frame": context_prefix + " " + final_frame,
            "answer_kind": "shared_new" if answer == p["shared_new_value"] else "source_value",
            "foil_kind": "shared_new" if foil == p["shared_new_value"] else "source_value",
        }
    # Query A: candidates are shared_new vs value_a. Only update recipient changes.
    rows.append(row("query_a_update_a", "UPDATE", "query_a", p["entity_a"], p["entity_a"], p["value_a"], p["shared_new_value"], p["value_a"], p["context_update_a"], p["use_frame_a"]))
    rows.append(row("query_a_update_b", "RETAIN", "query_a", p["entity_b"], p["entity_a"], p["value_a"], p["value_a"], p["shared_new_value"], p["context_update_b"], p["use_frame_a"]))
    # Query B: candidates are shared_new vs value_b. Only update recipient changes.
    rows.append(row("query_b_update_a", "RETAIN", "query_b", p["entity_a"], p["entity_b"], p["value_b"], p["value_b"], p["shared_new_value"], p["context_update_a"], p["use_frame_b"]))
    rows.append(row("query_b_update_b", "UPDATE", "query_b", p["entity_b"], p["entity_b"], p["value_b"], p["shared_new_value"], p["value_b"], p["context_update_b"], p["use_frame_b"]))
    rows.append(row("neutral_query_a", "NEUTRAL", "query_a", "none", p["entity_a"], p["value_a"], p["value_a"], p["shared_new_value"], p["context_neutral"], p["use_frame_a"]))
    rows.append(row("neutral_query_b", "NEUTRAL", "query_b", "none", p["entity_b"], p["value_b"], p["value_b"], p["shared_new_value"], p["context_neutral"], p["use_frame_b"]))
    return rows


def construct_outputs(seed: int, out_dir: pathlib.Path) -> Dict[str, Any]:
    v1_pairs = read_jsonl(V1_PAIRS)
    repaired = construct_repaired_pairs(v1_pairs, seed)
    rows = [r for p in repaired for r in build_rows(p)]
    write_jsonl(out_dir / "repaired_pairs.jsonl", repaired)
    write_jsonl(out_dir / "repaired_scoring_rows.jsonl", rows)
    summary = {
        "status": "REPAIRED_CONSTRUCTION",
        "v1_pairs_path": rel(V1_PAIRS),
        "n_pairs": len(repaired),
        "n_train": sum(1 for p in repaired if p.get("split") == "train"),
        "n_held": sum(1 for p in repaired if p.get("split") == "held"),
        "n_rows": len(rows),
        "by_relation": dict(collections.Counter(p["relation"] for p in repaired)),
        "shared_new_distinct_from_both_source_values": all(p["shared_new_value"].lower() not in {p["value_a"].lower(), p["value_b"].lower()} for p in repaired),
        "rows_per_role": dict(collections.Counter(r["role"] for r in rows)),
        "recipient_only_contract": "Within query_a rows, query_entity=value_a candidates fixed and update recipient A/B changes; within query_b rows, query_entity=value_b candidates fixed and update recipient A/B changes.",
        "old_step039_different_new_values_count": sum(1 for p in repaired if p.get("new_value_a_old_step039") != p.get("new_value_b_old_step039")),
    }
    (out_dir / "repaired_construction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Explicit final-span scorer
# ---------------------------------------------------------------------------

def locate_span_positions(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    pos = []
    for i, (s, e) in enumerate(offsets):
        s, e = int(s), int(e)
        if e <= s:
            continue
        if s < end and e > start:
            pos.append(i)
    return pos


def full_text_and_span(row: Dict[str, Any], candidate: str) -> Tuple[str, int, int]:
    frame = row["use_sentence_frame"]
    if frame.count("{STATE}") != 1:
        raise ValueError(f"Frame must have one {{STATE}} for {row.get('pair_id')} {row.get('row_type')}: {frame!r}")
    before, after = frame.split("{STATE}")
    full = before + candidate + after
    start = len(before)
    end = start + len(candidate)
    # This is the explicit final contract. Do not search elsewhere.
    if full[start:end] != candidate:
        raise RuntimeError("Internal span construction error")
    return full, start, end


def score_candidate(model, tokenizer, row: Dict[str, Any], candidate: str, device: torch.device, seq_length: int) -> Dict[str, Any]:
    full, start, end = full_text_and_span(row, candidate)
    enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True,
                    return_tensors="pt", max_length=seq_length, truncation=True)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
    positions = locate_span_positions(offsets, start, end)
    if not positions:
        raise ValueError(f"Final candidate span unavailable/truncated: pair={row.get('pair_id')} row={row.get('row_type')} candidate={candidate!r} start={start} end={end} len={len(full)}")
    # Require the whole character span to be covered by located tokens, so partial
    # final fragments cannot silently score.
    covered_start = min(offsets[p][0] for p in positions)
    covered_end = max(offsets[p][1] for p in positions)
    if covered_start > start or covered_end < end:
        raise ValueError(f"Final span only partially covered: pair={row.get('pair_id')} row={row.get('row_type')} candidate={candidate!r} span={start,end} covered={covered_start,covered_end} tokens={[tokenizer.convert_ids_to_tokens(int(enc['input_ids'][0,p])) for p in positions]}")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    target_ids = input_ids[0, positions].detach().clone()
    masked_ids = input_ids.clone()
    masked_ids[0, positions] = int(tokenizer.mask_token_id)
    with torch.no_grad():
        logits = model(input_ids=masked_ids, attention_mask=attention_mask).logits[0, positions]
        lp = F.log_softmax(logits, dim=-1)
        token_logps = lp[torch.arange(len(positions), device=device), target_ids].detach().cpu().tolist()
        top_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    token_ids = [int(x) for x in target_ids.detach().cpu().tolist()]
    return {
        "candidate": candidate,
        "char_start": start,
        "char_end": end,
        "token_positions": [int(p) for p in positions],
        "token_ids": token_ids,
        "tokens": [tokenizer.convert_ids_to_tokens(t) for t in token_ids],
        "n_tokens": len(positions),
        "sum_logp": float(sum(token_logps)),
        "mean_logp": float(sum(token_logps) / len(token_logps)),
        "token_logps": [float(x) for x in token_logps],
        "top_tokens": [tokenizer.convert_ids_to_tokens(int(t)) for t in top_ids],
        "seq_len": int(input_ids.shape[1]),
    }


def score_rows(model, tokenizer, rows: Sequence[Dict[str, Any]], device: torch.device, seq_length: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    scored, errors = [], []
    for i, row in enumerate(rows):
        try:
            ans = score_candidate(model, tokenizer, row, str(row["answer_text"]), device, seq_length)
            foil = score_candidate(model, tokenizer, row, str(row["foil_text"]), device, seq_length)
            out = dict(row)
            out.update({
                "answer_score": ans,
                "foil_score": foil,
                "margin_mean": ans["mean_logp"] - foil["mean_logp"],
                "margin_sum": ans["sum_logp"] - foil["sum_logp"],
                "same_token_length": ans["n_tokens"] == foil["n_tokens"],
                "correct_mean": ans["mean_logp"] > foil["mean_logp"],
                "correct_sum": ans["sum_logp"] > foil["sum_logp"],
            })
            scored.append(out)
        except Exception as e:
            errors.append({"index": i, "pair_id": row.get("pair_id"), "row_type": row.get("row_type"), "error": repr(e)})
        if (i + 1) % 100 == 0:
            print(f"scored {i+1}/{len(rows)} rows; errors={len(errors)}", flush=True)
    return scored, errors


def pair_metrics(scored_rows: Sequence[Dict[str, Any]], use_key: str = "margin_mean") -> List[Dict[str, Any]]:
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    for r in scored_rows:
        by_pair[r["pair_id"]][r["row_type"]] = r
    metrics = []
    for pid, d in sorted(by_pair.items()):
        required_names = ["query_a_update_a", "query_a_update_b", "query_b_update_a", "query_b_update_b"]
        if any(k not in d for k in required_names):
            metrics.append({"pair_id": pid, "has_error": True, "error": "missing_required_rows", "present": sorted(d)})
            continue
        u_a = float(d["query_a_update_a"][use_key])
        r_a = float(d["query_a_update_b"][use_key])
        r_b = float(d["query_b_update_a"][use_key])
        u_b = float(d["query_b_update_b"][use_key])
        n_a = float(d["neutral_query_a"][use_key]) if "neutral_query_a" in d else float("nan")
        n_b = float(d["neutral_query_b"][use_key]) if "neutral_query_b" in d else float("nan")
        beta_a = (u_a + r_a) / 2.0
        alpha_a = (u_a - r_a) / 2.0
        beta_b = (u_b + r_b) / 2.0
        alpha_b = (u_b - r_b) / 2.0
        m = {
            "pair_id": pid,
            "split": d["query_a_update_a"].get("split"),
            "relation": d["query_a_update_a"].get("relation"),
            "entity_a": d["query_a_update_a"].get("entity_a"),
            "entity_b": d["query_a_update_a"].get("entity_b"),
            "value_a": d["query_a_update_a"].get("value_a"),
            "value_b": d["query_a_update_a"].get("value_b"),
            "shared_new_value": d["query_a_update_a"].get("shared_new_value"),
            "has_error": False,
            "u_a": u_a,
            "r_a": r_a,
            "u_b": u_b,
            "r_b": r_b,
            "n_a_source_over_new": n_a,
            "n_b_source_over_new": n_b,
            "beta_a": beta_a,
            "alpha_a": alpha_a,
            "gamma_a_min_signed_margin": min(u_a, r_a),
            "beta_b": beta_b,
            "alpha_b": alpha_b,
            "gamma_b_min_signed_margin": min(u_b, r_b),
            "mean_U": (u_a + u_b) / 2.0,
            "mean_R": (r_a + r_b) / 2.0,
            "mean_beta": (beta_a + beta_b) / 2.0,
            "mean_abs_alpha": (abs(alpha_a) + abs(alpha_b)) / 2.0,
            "min_four_signed_margin": min(u_a, r_a, u_b, r_b),
            "query_a_success": (u_a > 0.0 and r_a > 0.0),
            "query_b_success": (u_b > 0.0 and r_b > 0.0),
            "update_success_both": (u_a > 0.0 and u_b > 0.0),
            "retain_success_both": (r_a > 0.0 and r_b > 0.0),
            "four_condition_success": (u_a > 0.0 and r_a > 0.0 and u_b > 0.0 and r_b > 0.0),
        }
        metrics.append(m)
    return metrics


def summarize_metrics(metrics: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    good = [m for m in metrics if not m.get("has_error")]
    n = len(good)
    if n == 0:
        return {"label": label, "n": 0}
    # Orientation records make query-specific success visible.
    orient = []
    for m in good:
        orient.append({"query": "a", "beta": m["beta_a"], "alpha": m["alpha_a"], "gamma": m["gamma_a_min_signed_margin"], "success": m["query_a_success"]})
        orient.append({"query": "b", "beta": m["beta_b"], "alpha": m["alpha_b"], "gamma": m["gamma_b_min_signed_margin"], "success": m["query_b_success"]})
    return {
        "label": label,
        "n_pairs": n,
        "n_pair_errors": len(metrics) - n,
        "n_update_success_both": sum(1 for m in good if m["update_success_both"]),
        "n_retain_success_both": sum(1 for m in good if m["retain_success_both"]),
        "n_four_condition_success": sum(1 for m in good if m["four_condition_success"]),
        "n_query_orientation_success": sum(1 for o in orient if o["success"]),
        "n_query_orientations": len(orient),
        "mean_U": finite_mean([m["mean_U"] for m in good]),
        "mean_R": finite_mean([m["mean_R"] for m in good]),
        "mean_beta_pair_average": finite_mean([m["mean_beta"] for m in good]),
        "mean_abs_alpha_pair_average": finite_mean([m["mean_abs_alpha"] for m in good]),
        "mean_min_four_signed_margin": finite_mean([m["min_four_signed_margin"] for m in good]),
        "min_four_signed_margin_quantiles": quantiles([m["min_four_signed_margin"] for m in good]),
        "orientation_beta_quantiles": quantiles([o["beta"] for o in orient]),
        "orientation_gamma_quantiles": quantiles([o["gamma"] for o in orient]),
        "n_neutral_a_source_over_new_positive": sum(1 for m in good if m["n_a_source_over_new"] > 0),
        "n_neutral_b_source_over_new_positive": sum(1 for m in good if m["n_b_source_over_new"] > 0),
    }


def run_score(args, out_dir: pathlib.Path) -> Dict[str, Any]:
    if not (out_dir / "repaired_scoring_rows.jsonl").exists():
        construct_outputs(args.seed, out_dir)
    rows = read_jsonl(out_dir / "repaired_scoring_rows.jsonl")
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, use_fast=True)
    model, load_info = load_trusted_model(device, private_scale=args.private_scale)
    ident = model_identity(model, load_info)
    print(f"trusted model {ident['class']} on {device}; private_params={ident['private_params']}; scales={ident['executed_private_scales']}", flush=True)
    scored, errors = score_rows(model, tokenizer, rows, device, args.seq_length)
    write_jsonl(out_dir / "trusted_parent_scored_rows.jsonl", scored)
    (out_dir / "trusted_parent_scoring_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = pair_metrics(scored, use_key="margin_mean")
    write_jsonl(out_dir / "trusted_parent_pair_metrics.jsonl", metrics)
    train = [m for m in metrics if m.get("split") == "train"]
    held = [m for m in metrics if m.get("split") == "held"]
    by_rel = collections.defaultdict(list)
    for m in metrics:
        by_rel[m.get("relation")].append(m)
    summary = {
        "status": "REPAIRED_TRUSTED_PARENT_SCORE",
        "model_identity": ident,
        "scoring_contract": "candidate span is the explicit final {STATE} span; all candidate tokens are masked simultaneously; no token-search fallback; primary margin is mean token log-prob answer minus foil",
        "n_rows": len(rows),
        "n_scored_rows": len(scored),
        "n_errors": len(errors),
        "all": summarize_metrics(metrics, "all"),
        "train": summarize_metrics(train, "train"),
        "held": summarize_metrics(held, "held"),
        "by_relation": {str(k): summarize_metrics(v, str(k)) for k, v in sorted(by_rel.items())},
    }
    (out_dir / "trusted_parent_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# Bounded answer-only private-adapter acquisition pilot
# ---------------------------------------------------------------------------

class RowDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], tokenizer, seq_length: int):
        self.items = []
        self.tokenizer = tokenizer
        for row in rows:
            if row["role"] == "NEUTRAL":
                continue
            full, start, end = full_text_and_span(row, str(row["answer_text"]))
            enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True,
                            return_tensors="pt", max_length=seq_length, truncation=True,
                            padding="max_length")
            offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
            pos = locate_span_positions(offsets, start, end)
            if not pos:
                raise ValueError(f"training answer span unavailable: {row['pair_id']} {row['row_type']}")
            covered_start = min(offsets[p][0] for p in pos)
            covered_end = max(offsets[p][1] for p in pos)
            if covered_start > start or covered_end < end:
                raise ValueError(f"training answer span partial: {row['pair_id']} {row['row_type']} span={start,end} covered={covered_start,covered_end}")
            input_ids = enc["input_ids"].squeeze(0)
            labels = torch.full_like(input_ids, -100)
            masked = input_ids.clone()
            for p in pos:
                labels[p] = input_ids[p]
                masked[p] = int(tokenizer.mask_token_id)
            self.items.append({
                "input_ids": masked,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels,
                "n_answer_tokens": len(pos),
                "pair_id": row["pair_id"],
                "row_type": row["row_type"],
            })
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx: int):
        return self.items[idx]


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "n_answer_tokens": torch.tensor([int(b["n_answer_tokens"]) for b in batch], dtype=torch.long),
    }


def compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    keep = ["n_pairs", "n_four_condition_success", "n_query_orientation_success", "n_query_orientations", "mean_U", "mean_R", "mean_beta_pair_average", "mean_abs_alpha_pair_average", "mean_min_four_signed_margin"]
    return {k: summary.get(k) for k in keep}


def run_train(args, out_dir: pathlib.Path) -> Dict[str, Any]:
    if not (out_dir / "repaired_scoring_rows.jsonl").exists():
        construct_outputs(args.seed, out_dir)
    rows = read_jsonl(out_dir / "repaired_scoring_rows.jsonl")
    train_rows = [r for r in rows if r.get("split") == "train" and r.get("role") != "NEUTRAL"]
    eval_train_rows = [r for r in rows if r.get("split") == "train"]
    eval_held_rows = [r for r in rows if r.get("split") == "held"]
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, use_fast=True)
    model, load_info = load_trusted_model(device, private_scale=args.private_scale)
    ident = model_identity(model, load_info)
    optimizer, trainable_info = freeze_to_private_optimizer(model, args.lr, args.weight_decay)
    if trainable_info["nonprivate_trainable_tensors"] != 0:
        raise RuntimeError(f"Non-private tensors trainable: {trainable_info}")
    ds = RowDataset(train_rows, tokenizer, args.seq_length)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator)
    print(f"train rows={len(train_rows)} examples={len(ds)} trusted_class={ident['class']} private_trainable_params={trainable_info['trainable_param_count']}", flush=True)
    trajectory = []

    def evaluate_epoch(epoch: int, loss: float | None):
        scored_train, err_train = score_rows(model, tokenizer, eval_train_rows, device, args.seq_length)
        scored_held, err_held = score_rows(model, tokenizer, eval_held_rows, device, args.seq_length)
        if err_train or err_held:
            raise RuntimeError(f"Scoring errors during training eval: train={err_train[:3]} held={err_held[:3]}")
        mt = pair_metrics(scored_train)
        mh = pair_metrics(scored_held)
        st = summarize_metrics(mt, f"train_e{epoch}")
        sh = summarize_metrics(mh, f"held_e{epoch}")
        entry = {"epoch": epoch, "loss": loss, "train": st, "held": sh}
        trajectory.append(entry)
        print(f"[e{epoch:04d}]" + (f" loss={loss:.4f}" if loss is not None else "") +
              f" train four={st['n_four_condition_success']}/{st['n_pairs']} orient={st['n_query_orientation_success']}/{st['n_query_orientations']} min4={st['mean_min_four_signed_margin']:+.3f}" +
              f" | held four={sh['n_four_condition_success']}/{sh['n_pairs']} orient={sh['n_query_orientation_success']}/{sh['n_query_orientations']} min4={sh['mean_min_four_signed_margin']:+.3f}", flush=True)
        return st, sh

    evaluate_epoch(0, None)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_num, tok_den = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ntok = int((labels != -100).sum().item())
            loss_num += float(loss.detach().cpu()) * ntok
            tok_den += ntok
        mean_loss = loss_num / max(1, tok_den)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluate_epoch(epoch, mean_loss)
    result = {
        "status": "REPAIRED_ANSWER_ONLY_PRIVATE_ADAPTER_PILOT",
        "model_identity": ident,
        "trainable_info": trainable_info,
        "training_contract": "private_adapter_only, explicit final answer span masked simultaneously, answer-only labels, repaired shared-new recipient-only rows",
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_train_examples": len(ds),
        "trajectory": trajectory,
        "final_train": trajectory[-1]["train"],
        "final_held": trajectory[-1]["held"],
        "elapsed_sec": time.time() - t0,
    }
    run_dir = out_dir / f"answer_only_private_e{args.epochs}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    # Save endpoint only for inspection, not as a BabyLM candidate.
    if args.save_model:
        dst = run_dir / "hf_model"
        model.config.private_adapter_scale = float(args.private_scale)
        for layer in model.deberta.encoder.layer:
            layer.private_adapter.scale = float(args.private_scale)
        model.save_pretrained(str(dst), safe_serialization=True)
        tokenizer.save_pretrained(str(dst))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_markdown(out_dir: pathlib.Path, construction: Dict[str, Any], parent_summary: Dict[str, Any] | None, train_summary: Dict[str, Any] | None) -> None:
    lines = ["# research repaired relation-first harness\n\n"]
    lines.append("## Why this repair was necessary\n\n")
    lines.append("The research relation-first extraction is useful, but its first scorer/trainer did not preserve the previous experimental substrate. Generic `AutoModelForMaskedLM` loading without trusted custom loading selects stock `DebertaV2ForMaskedLM`; the answer-only trainer also optimized `model.parameters()`. The research construction selected separate new values for the two update recipients, so changing recipient also changed the competing answer phrase. Step039b masked candidate tokens one at a time and allowed a token-search fallback that could drop the first candidate token. This script repairs those problems before interpreting acquisition.\n\n")
    lines.append("## Repaired construction\n\n")
    lines.append(f"- Pairs: {construction.get('n_pairs')} ({construction.get('n_train')} train, {construction.get('n_held')} held); by relation: {construction.get('by_relation')}\n")
    lines.append(f"- Rows per role: {construction.get('rows_per_role')}\n")
    lines.append(f"- Original research pairs with different new values now repaired: {construction.get('old_step039_different_new_values_count')}\n")
    lines.append("- For query A, `query_a_update_a` and `query_a_update_b` have the same query entity, source answer, and shared new-value candidate; only the update recipient changes. Query B is analyzed the same way.\n")
    lines.append("- Pair success is `min(u_a, r_a, u_b, r_b) > 0`, not gamma after averaging.\n\n")
    if parent_summary:
        lines.append("## Trusted parent score\n\n")
        mi = parent_summary["model_identity"]
        lines.append(f"- Loaded class: `{mi['module']}.{mi['class']}`; private params {mi['private_params']}; executed scales {mi['executed_private_scales']}\n")
        for key in ["all", "train", "held"]:
            s = parent_summary[key]
            lines.append(f"- {key}: four-condition {s['n_four_condition_success']}/{s['n_pairs']}; orientation success {s['n_query_orientation_success']}/{s['n_query_orientations']}; mean U={s['mean_U']:+.3f}, mean R={s['mean_R']:+.3f}, mean beta={s['mean_beta_pair_average']:+.3f}, mean |alpha|={s['mean_abs_alpha_pair_average']:+.3f}, mean min4={s['mean_min_four_signed_margin']:+.3f}\n")
        lines.append("\n")
    if train_summary:
        lines.append("## Bounded answer-only private-adapter pilot\n\n")
        ti = train_summary["trainable_info"]
        lines.append(f"- Trainable params: {ti['trainable_param_count']} in {ti['trainable_tensor_count']} tensors; non-private trainable tensors {ti['nonprivate_trainable_tensors']}\n")
        lines.append(f"- Final train: {compact_summary(train_summary['final_train'])}\n")
        lines.append(f"- Final held: {compact_summary(train_summary['final_held'])}\n")
        lines.append("\n")
    lines.append("## Files\n\n")
    for fn in ["repaired_pairs.jsonl", "repaired_scoring_rows.jsonl", "trusted_parent_summary.json", "trusted_parent_pair_metrics.jsonl"]:
        p = out_dir / fn
        if p.exists():
            lines.append(f"- `{rel(p)}`\n")
    (out_dir / "repaired_harness_summary.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=40040)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--construct-only", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    construction = construct_outputs(args.seed, out_dir)
    parent_summary = None
    train_summary = None
    if args.score:
        parent_summary = run_score(args, out_dir)
    if args.train:
        train_summary = run_train(args, out_dir)
    write_markdown(out_dir, construction, parent_summary, train_summary)
    print(json.dumps({
        "status": "REPAIRED_HARNESS_DONE",
        "out_dir": rel(out_dir),
        "construction": construction,
        "parent_all": compact_summary(parent_summary["all"]) if parent_summary else None,
        "parent_held": compact_summary(parent_summary["held"]) if parent_summary else None,
        "train_final_held": compact_summary(train_summary["final_held"]) if train_summary else None,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
