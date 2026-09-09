#!/usr/bin/env python3
"""research: symmetric multi-token scorer for strict contrastive rows.

This script adapts the recipient-change scorer to the natural-language
rows, where answer/foil strings are often multi-token under the coherent86
compliant16k tokenizer. It performs zero-training scoring on the exact coherent86
private-adapter model and writes pair-level evidence.

Scoring convention
------------------
For each row, build
  source_sentence + update_sentence + use_sentence_frame.replace({STATE}, candidate)
then locate only the candidate span inside the final use sentence, replace those
candidate token positions by <mask>, and sum/average log p(candidate_token) at the
masked positions. Both answer and foil are scored symmetrically by the same method.

For a pair:
  UPDATE row: U = score(new answer) - score(source foil)
  RETAIN row: R = score(source answer) - score(new foil)
The signed recipient-change contrast is U + R. Joint correctness requires U>0 and
R>0. Mean-token log likelihood is the primary comparison when candidate lengths
differ; raw summed scores and equal-token-length subsets are also recorded.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

SCRIPT_DIR = _public_path('experiments/archive/functional_learning/scripts')
sys.path.insert(0, str(SCRIPT_DIR))

from corruption_vs_loss_factorial import (  # noqa: E402
    locate_span_token_positions,
    load_private_model,
)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def quantiles(xs: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(x) for x in xs)
    if not vals:
        return {"n": 0}
    def q(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = p * (len(vals) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return {
        "n": len(vals), "mean": safe_mean(vals), "median": q(0.5),
        "p10": q(0.1), "p90": q(0.9), "min": vals[0], "max": vals[-1],
    }


def build_candidate_full_text(row: Dict[str, Any], candidate: str) -> Tuple[str, int, int]:
    frame = row.get("use_sentence_frame") or ""
    if frame.count("{STATE}") != 1:
        raise ValueError(f"use_sentence_frame must contain exactly one {{STATE}}: {frame!r}")
    source = row["source_sentence"].rstrip()
    update = row["update_sentence"].strip()
    before_state, after_state = frame.split("{STATE}")
    use = before_state + candidate + after_state
    prefix = source + " " + update + " "
    full = prefix + use
    cand_start = len(prefix) + len(before_state)
    cand_end = cand_start + len(candidate)
    return full, cand_start, cand_end


def score_candidate(model, tokenizer, row: Dict[str, Any], candidate: str,
                    device: torch.device, seq_length: int) -> Dict[str, Any]:
    full, cand_start, cand_end = build_candidate_full_text(row, candidate)
    enc = tokenizer(
        full, add_special_tokens=True, return_offsets_mapping=True,
        return_tensors="pt", max_length=seq_length, truncation=True,
    )
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"].squeeze(0).tolist()]
    positions = locate_span_token_positions(offsets, cand_start, cand_end)
    if not positions:
        raise ValueError(f"candidate span not tokenized or truncated: candidate={candidate!r} row={row.get('pair_id')}")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    target_ids = input_ids[0, positions].detach().clone()
    masked_ids = input_ids.clone()
    masked_ids[0, positions] = int(tokenizer.mask_token_id)
    with torch.no_grad():
        logits = model(input_ids=masked_ids, attention_mask=attention_mask).logits[0]
    log_probs = F.log_softmax(logits[positions], dim=-1)
    token_logps = log_probs[torch.arange(len(positions), device=device), target_ids].detach().cpu().tolist()
    token_ids = [int(x) for x in target_ids.detach().cpu().tolist()]
    tokens = [tokenizer.convert_ids_to_tokens(x) for x in token_ids]
    top_ids = logits[positions].argmax(dim=-1).detach().cpu().tolist()
    return {
        "candidate": candidate,
        "n_tokens": len(positions),
        "token_positions": [int(p) for p in positions],
        "token_ids": token_ids,
        "tokens": tokens,
        "token_logps": [float(x) for x in token_logps],
        "sum_logp": float(sum(token_logps)),
        "mean_logp": float(sum(token_logps) / max(1, len(token_logps))),
        "top_tokens": [tokenizer.convert_ids_to_tokens(int(x)) for x in top_ids],
        "char_start": cand_start,
        "char_end": cand_end,
        "truncated_length": int(input_ids.shape[1]),
    }


def score_row(model, tokenizer, row: Dict[str, Any], device: torch.device, seq_length: int) -> Dict[str, Any]:
    answer = str(row["answer_text"]).strip()
    foil = str(row["foil_text"]).strip()
    ans = score_candidate(model, tokenizer, row, answer, device, seq_length)
    foi = score_candidate(model, tokenizer, row, foil, device, seq_length)
    return {
        "pair_id": row["pair_id"],
        "packet_type": row["packet_type"],
        "role_position_relation": row.get("role_position_relation"),
        "entity_name": row.get("entity_name"),
        "distractor_entity": row.get("distractor_entity"),
        "update_entity": row.get("update_entity"),
        "answer_text": answer,
        "foil_text": foil,
        "answer_state_kind": row.get("answer_state_kind"),
        "answer_also_in_source_or_update": bool(row.get("answer_also_in_source_or_update", False)),
        "answer_score": ans,
        "foil_score": foi,
        "margin_sum_answer_minus_foil": ans["sum_logp"] - foi["sum_logp"],
        "margin_mean_answer_minus_foil": ans["mean_logp"] - foi["mean_logp"],
        "answer_n_tokens": ans["n_tokens"],
        "foil_n_tokens": foi["n_tokens"],
        "same_token_length": ans["n_tokens"] == foi["n_tokens"],
        "word_count": row.get("word_count"),
    }


def validate_pairs(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, Any]]]]:
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    issues = []
    for r in rows:
        by_pair[str(r["pair_id"])][str(r["packet_type"]).upper()] = r
    for pid, d in by_pair.items():
        if set(d) != {"UPDATE", "RETAIN"}:
            issues.append({"pair_id": pid, "issue": "missing_update_or_retain", "types": sorted(d)})
            continue
        u, r = d["UPDATE"], d["RETAIN"]
        for key in ["source_sentence", "use_sentence_frame", "entity_name", "distractor_entity", "role_position_relation"]:
            if u.get(key) != r.get(key):
                issues.append({"pair_id": pid, "issue": f"mismatch_{key}", "update": u.get(key), "retain": r.get(key)})
        if str(u.get("answer_text", "")).strip() != str(r.get("foil_text", "")).strip():
            issues.append({"pair_id": pid, "issue": "update_answer_not_retain_foil",
                           "update_answer": u.get("answer_text"), "retain_foil": r.get("foil_text")})
        if str(u.get("foil_text", "")).strip() != str(r.get("answer_text", "")).strip():
            issues.append({"pair_id": pid, "issue": "update_foil_not_retain_answer",
                           "update_foil": u.get("foil_text"), "retain_answer": r.get("answer_text")})
        if u.get("update_entity") == r.get("update_entity"):
            issues.append({"pair_id": pid, "issue": "update_entity_not_changed", "entity": u.get("update_entity")})
    meta = {
        "n_rows": len(rows),
        "n_pairs": len(by_pair),
        "n_pair_issues": len(issues),
        "pair_issues_first20": issues[:20],
        "packet_type_counts": dict(Counter(str(r.get("packet_type")).upper() for r in rows)),
        "role_position_distribution_rows": dict(Counter(str(r.get("role_position_relation")) for r in rows)),
        "source_distribution_rows": dict(Counter(str(r.get("source", "unknown")) for r in rows)),
    }
    return meta, by_pair


def entity_position_meta(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    examples = []
    seen_pair = set()
    for r in rows:
        pid = r.get("pair_id")
        if pid in seen_pair:
            continue
        seen_pair.add(pid)
        src = r.get("source_sentence", "")
        target = str(r.get("entity_name", ""))
        dist = str(r.get("distractor_entity", ""))
        it = src.find(target) if target else -1
        idst = src.find(dist) if dist else -1
        if it >= 0 and idst >= 0:
            rel = "target_earlier" if it < idst else "target_later"
        elif it >= 0:
            rel = "target_found_distractor_missing"
        elif idst >= 0:
            rel = "distractor_found_target_missing"
        else:
            rel = "both_missing"
        counts[rel] += 1
        if len(examples) < 10 and rel != r.get("role_position_relation"):
            examples.append({"pair_id": pid, "metadata_role": r.get("role_position_relation"),
                             "literal_role": rel, "target": target, "distractor": dist,
                             "source_prefix": src[:220]})
    return {"literal_target_position_counts": dict(counts), "metadata_literal_mismatch_examples": examples}


def summarize_pairs(row_scores: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_pair = defaultdict(dict)
    for s in row_scores:
        by_pair[s["pair_id"]][s["packet_type"].upper()] = s
    pair_scores = []
    for pid, d in sorted(by_pair.items()):
        if set(d) != {"UPDATE", "RETAIN"}:
            continue
        u = d["UPDATE"]
        r = d["RETAIN"]
        U_mean = u["margin_mean_answer_minus_foil"]
        R_mean = r["margin_mean_answer_minus_foil"]
        U_sum = u["margin_sum_answer_minus_foil"]
        R_sum = r["margin_sum_answer_minus_foil"]
        pair_scores.append({
            "pair_id": pid,
            "role_position_relation": u.get("role_position_relation"),
            "entity_name": u.get("entity_name"),
            "distractor_entity": u.get("distractor_entity"),
            "update_answer": u.get("answer_text"),
            "retain_answer": r.get("answer_text"),
            "U_mean_new_minus_source": U_mean,
            "R_mean_source_minus_new": R_mean,
            "UR_mean_signed_recipient_contrast": U_mean + R_mean,
            "U_sum_new_minus_source": U_sum,
            "R_sum_source_minus_new": R_sum,
            "UR_sum_signed_recipient_contrast": U_sum + R_sum,
            "update_correct_mean": U_mean > 0,
            "retain_correct_mean": R_mean > 0,
            "joint_correct_mean": U_mean > 0 and R_mean > 0,
            "update_correct_sum": U_sum > 0,
            "retain_correct_sum": R_sum > 0,
            "joint_correct_sum": U_sum > 0 and R_sum > 0,
            "same_token_length_both_rows": bool(u["same_token_length"] and r["same_token_length"]),
            "update_answer_n_tokens": u["answer_n_tokens"],
            "update_foil_n_tokens": u["foil_n_tokens"],
            "retain_answer_n_tokens": r["answer_n_tokens"],
            "retain_foil_n_tokens": r["foil_n_tokens"],
            "answer_also_elsewhere_update": u.get("answer_also_in_source_or_update"),
            "answer_also_elsewhere_retain": r.get("answer_also_in_source_or_update"),
        })

    def agg_for(sub: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not sub:
            return {"n_pairs": 0}
        return {
            "n_pairs": len(sub),
            "mean_U_mean": safe_mean([p["U_mean_new_minus_source"] for p in sub]),
            "mean_R_mean": safe_mean([p["R_mean_source_minus_new"] for p in sub]),
            "mean_UR_mean": safe_mean([p["UR_mean_signed_recipient_contrast"] for p in sub]),
            "n_update_correct_mean": sum(1 for p in sub if p["update_correct_mean"]),
            "n_retain_correct_mean": sum(1 for p in sub if p["retain_correct_mean"]),
            "n_joint_correct_mean": sum(1 for p in sub if p["joint_correct_mean"]),
            "mean_U_sum": safe_mean([p["U_sum_new_minus_source"] for p in sub]),
            "mean_R_sum": safe_mean([p["R_sum_source_minus_new"] for p in sub]),
            "mean_UR_sum": safe_mean([p["UR_sum_signed_recipient_contrast"] for p in sub]),
            "n_joint_correct_sum": sum(1 for p in sub if p["joint_correct_sum"]),
            "UR_mean_quantiles": quantiles([p["UR_mean_signed_recipient_contrast"] for p in sub]),
        }

    role_stats = {}
    for role in sorted(set(str(p.get("role_position_relation")) for p in pair_scores)):
        role_stats[role] = agg_for([p for p in pair_scores if str(p.get("role_position_relation")) == role])
    same_len = [p for p in pair_scores if p["same_token_length_both_rows"]]
    summary = {
        "all_pairs": agg_for(pair_scores),
        "same_token_length_both_rows": agg_for(same_len),
        "by_role_position_relation": role_stats,
        "token_length_distribution": {
            "update_answer": dict(Counter(p["update_answer_n_tokens"] for p in pair_scores)),
            "update_foil": dict(Counter(p["update_foil_n_tokens"] for p in pair_scores)),
            "retain_answer": dict(Counter(p["retain_answer_n_tokens"] for p in pair_scores)),
            "retain_foil": dict(Counter(p["retain_foil_n_tokens"] for p in pair_scores)),
            "same_token_length_both_rows_n": len(same_len),
        },
    }
    return pair_scores, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="experiments/archive/relation_learning/data/contrastive_binding_validated_strict/accepted_contrastive_binding_training_rows_strict_pilot512.jsonl")
    ap.add_argument("--model-path", default="models/frontier")
    ap.add_argument("--out-dir", default="experiments/archive/functional_learning/data/multitoken_scorer_pilot512")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seq-length", type=int, default=256)
    ap.add_argument("--private-bottleneck", type=int, default=128)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    rows_path = pathlib.Path(args.rows)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(rows_path)
    validation, by_pair = validate_pairs(rows)
    if args.max_pairs and args.max_pairs > 0:
        keep = set(sorted(by_pair)[:args.max_pairs])
        rows = [r for r in rows if r["pair_id"] in keep]
        validation, by_pair = validate_pairs(rows)

    from transformers import AutoTokenizer
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    model, load_info = load_private_model(pathlib.Path(args.model_path), args.private_bottleneck,
                                          args.private_scale, device)
    model.eval()
    print(f"Loaded model on {device}; rows={len(rows)} pairs={len(by_pair)}", flush=True)

    row_scores: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        try:
            row_scores.append(score_row(model, tokenizer, row, device, args.seq_length))
        except Exception as e:
            errors.append({"row_index": i, "pair_id": row.get("pair_id"),
                           "packet_type": row.get("packet_type"), "error": repr(e)})
        if (i + 1) % 20 == 0:
            print(f"scored {i+1}/{len(rows)} rows; errors={len(errors)}", flush=True)

    pair_scores, pair_summary = summarize_pairs(row_scores)
    validation.update(entity_position_meta(rows))
    validation["n_scoring_errors"] = len(errors)
    validation["scoring_errors_first20"] = errors[:20]

    summary = {
        "status": "A02_MULTITOKEN_BASELINE_SCORER",
        "rows_path": str(rows_path),
        "model_path": args.model_path,
        "device": str(device),
        "seq_length": args.seq_length,
        "private_scale": args.private_scale,
        "scoring": {
            "primary_margin": "mean per candidate token log probability difference",
            "secondary_margin": "summed candidate token log probability difference, most interpretable on equal-token-length rows",
            "masked_positions": "candidate span inside final use_sentence_frame only; source/update occurrences are not masked",
            "pair_contrast": "U + R under update_new_minus_source and retain_source_minus_new signed margins",
        },
        "validation": validation,
        "pair_summary": pair_summary,
        "load_info": {"missing": list(load_info.get("missing", []))[:20],
                       "unexpected": list(load_info.get("unexpected", []))[:20]},
    }
    write_jsonl(out_dir / "row_scores.jsonl", row_scores)
    write_jsonl(out_dir / "pair_scores.jsonl", pair_scores)
    (out_dir / "multitoken_scorer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# research multi-token scorer baseline\n\n"]
    md.append("## Purpose\n\n")
    md.append("research rows usually contain multi-token answers. This scorer symmetrically masks the candidate span in the final use frame and compares answer versus foil by mean token log probability, while also recording summed scores and equal-token-length subsets.\n\n")
    md.append("## Validation\n\n")
    md.append(f"- Rows: {validation['n_rows']}; pairs: {validation['n_pairs']}; pair issues: {validation['n_pair_issues']}; scoring errors: {validation['n_scoring_errors']}\n")
    md.append(f"- Metadata role distribution: {validation.get('role_position_distribution_rows')}\n")
    md.append(f"- Literal target-position counts: {validation.get('literal_target_position_counts')}\n")
    md.append(f"- Token length distribution: {pair_summary['token_length_distribution']}\n\n")
    md.append("## Baseline pair scores\n\n")
    allp = pair_summary["all_pairs"]
    md.append(f"All scored pairs: n={allp['n_pairs']}, mean U={allp['mean_U_mean']:+.4f}, mean R={allp['mean_R_mean']:+.4f}, mean U+R={allp['mean_UR_mean']:+.4f}, joint correct={allp['n_joint_correct_mean']}/{allp['n_pairs']} by mean-token margin.\n\n")
    same = pair_summary["same_token_length_both_rows"]
    md.append(f"Equal-token-length subset: n={same['n_pairs']}, mean U+R (mean-margin)={same.get('mean_UR_mean', float('nan')):+.4f}, joint correct by summed score={same.get('n_joint_correct_sum', 0)}/{same['n_pairs']}.\n\n")
    md.append("By target source position:\n\n")
    for role, st in pair_summary["by_role_position_relation"].items():
        md.append(f"- {role}: n={st['n_pairs']}, mean U={st['mean_U_mean']:+.4f}, mean R={st['mean_R_mean']:+.4f}, mean U+R={st['mean_UR_mean']:+.4f}, joint={st['n_joint_correct_mean']}/{st['n_pairs']}\n")
    md.append("\n## Interpretation\n\n")
    md.append("This is a zero-training readout of accepted pilot rows, not evidence that the rows are semantically clean or suitable for training. Its value is to validate the scorer, expose token-length and position effects, and define the pair-level contrast for the bounded ALN-preserving pilot once cleaner accepted packets are available.\n")
    (out_dir / "multitoken_scorer_summary.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(out_dir),
        "n_rows": validation["n_rows"],
        "n_pairs": validation["n_pairs"],
        "n_errors": validation["n_scoring_errors"],
        "all_pairs_joint_mean": f"{allp['n_joint_correct_mean']}/{allp['n_pairs']}",
        "mean_U": allp["mean_U_mean"],
        "mean_R": allp["mean_R_mean"],
        "mean_U_plus_R": allp["mean_UR_mean"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
