#!/usr/bin/env python3
"""research: official-compatible EWoK candidate-margin exporter.

The saved official EWoK predictions contain only the selected sentence.  This
script reconstructs the masked-LM candidate scores used by the official
`sentence_zero_shot` scorer, but saves the two candidate log-probability sums
before argmax so relation-instability can be separated into small-margin flips
versus confident alternative choices.

It deliberately does not modify the pristine official checkout.  The scoring
logic mirrors the current official MLM path:
  * decode_ewok: candidates are Context1+Target1 and Context2+Target1;
    completion span is " " + Target1; label is candidate 0.
  * process_mlm_sentences: tokenize full sentence with special tokens and
    offsets, mask each token whose offset overlaps the completion span.
  * compute_mlm_results: sum target log-probs across completion tokens at
    temperature 1.0; choose the candidate with the larger sum.

Designed to run either on CPU for small targeted subsets while both H100s are
busy, or on a GPU later for the full 7,618-row EWoK coordinate.
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
from pathlib import Path
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoProcessor, PreTrainedTokenizerFast


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
A01 = USER_ROOT / "experiments/archive/representation_and_objectives"
A02 = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = A01
EWOK_DIR = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
DEFAULT_OUT = WORKSPACE / "data/official_ewok_margin_subset"

MODEL_PATHS: dict[str, Path] = {
    # inherited-tokenizer four cells used in research/46 relation-instability work
    "clean430": USER_ROOT / "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_100M",
    "clean431": USER_ROOT / "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43122/hf_model/chck_100M",
    "reinv430": USER_ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M",
    "reinv431": USER_ROOT / "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model/chck_100M",
    # corrected-tokenizer endpoints will exist after research retrains finish
    "strictsmalltok430": USER_ROOT / "experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model/chck_100M",
    "strictsmalltok431": USER_ROOT / "experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122/hf_model/chck_100M",
}

OFFICIAL_PRED_PATHS: dict[str, Path] = {
    "clean430": A02 / "data/official_ewok_clean_qwen/official_outputs/clean_qwen_seed43022/EWoK/chck_100M/official_ewok_clean_qwen_seed43022/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "clean431": A02 / "data/official_ewok_clean_qwen/official_outputs/clean_qwen_seed43122/EWoK/chck_100M/official_ewok_clean_qwen_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "reinv430": A01 / "data/official_ewok_reeval/official_outputs/EWoK/chck_100M/official_ewok_reinvest/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "reinv431": A01 / "data/official_ewok_reeval_seed43122/official_outputs/EWoK/chck_100M/official_ewok_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_all_ewok_rows(ewok_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(ewok_dir.glob("*.jsonl")):
        domain_from_file = p.stem
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                raw = json.loads(line)
                domain = raw.get("Domain", domain_from_file)
                context1 = raw["Context1"]
                context2 = raw["Context2"]
                target1 = raw["Target1"]
                sentence0 = " ".join([context1, target1]).strip()
                sentence1 = " ".join([context2, target1]).strip()
                rows.append({
                    "domain": domain,
                    "file_domain": domain_from_file,
                    "idx": idx,
                    "uid": f"{domain_from_file}_{idx}",
                    "ConceptA": raw.get("ConceptA"),
                    "ConceptB": raw.get("ConceptB"),
                    "ContextType": raw.get("ContextType"),
                    "ContextDiff": raw.get("ContextDiff"),
                    "TargetDiff": raw.get("TargetDiff"),
                    "Context1": context1,
                    "Context2": context2,
                    "Target1": target1,
                    "Target2": raw.get("Target2"),
                    "sentences": [sentence0, sentence1],
                    "completion": " " + target1,
                    "label": 0,
                })
    return rows


def load_selection(selection_csv: Path | None, rows: list[dict[str, Any]], max_items: int | None,
                   domain_filter: set[str] | None) -> list[dict[str, Any]]:
    by_key = {(r["file_domain"], int(r["idx"])): r for r in rows}
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    if selection_csv is not None:
        with selection_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for rec in csv.DictReader(f):
                dom = rec.get("domain") or rec.get("file_domain")
                idx_s = rec.get("idx")
                if dom is None or idx_s is None:
                    raise RuntimeError(f"selection CSV must contain domain and idx columns: {selection_csv}")
                key = (dom, int(idx_s))
                if key in seen:
                    continue
                if key not in by_key:
                    raise KeyError({"missing_selection_key": key, "selection_csv": str(selection_csv)})
                r = dict(by_key[key])
                # preserve research interaction metadata when present
                for k, v in rec.items():
                    if k not in r:
                        r[f"selection_{k}"] = v
                selected.append(r)
                seen.add(key)
                if max_items is not None and len(selected) >= max_items:
                    break
    else:
        for r0 in rows:
            if domain_filter is not None and r0["file_domain"] not in domain_filter and r0["domain"] not in domain_filter:
                continue
            selected.append(dict(r0))
            if max_items is not None and len(selected) >= max_items:
                break
    return selected


def load_tokenizer_like_official(model_path: Path):
    try:
        processor = AutoProcessor.from_pretrained(str(model_path), padding_side="right", trust_remote_code=True, local_files_only=True)
    except Exception:
        processor = PreTrainedTokenizerFast.from_pretrained(str(model_path), padding_side="right", local_files_only=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "cls_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.cls_token_id
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def candidate_score(model, tokenizer, sentence: str, completion: str, device: torch.device,
                    non_causal_batch_size: int, temperature: float = 1.0) -> tuple[float, list[int], list[int], int]:
    """Return official MLM completion log-prob sum for one candidate sentence.

    Mirrors CompletionRankingDataset.process_mlm_sentences and compute_mlm_results
    for a single sentence/candidate at temperature 1.0.
    """
    tok = tokenizer(text=sentence, return_offsets_mapping=True)
    tokens = list(tok["input_ids"])
    attn = list(tok["attention_mask"])
    offsets = list(tok["offset_mapping"])
    start_char_idx = len(sentence) - len(completion)
    phrase_indices: list[int] = []
    target_tokens: list[int] = []
    for i, (start, end) in enumerate(offsets):
        if end > start_char_idx:
            phrase_indices.append(i)
            target_tokens.append(tokens[i])
    if not phrase_indices:
        return float("-inf"), [], [], len(tokens)
    base_tokens = torch.LongTensor(tokens)
    base_attn = torch.LongTensor(attn)
    target_tensor = torch.LongTensor(target_tokens)
    mask_index = tokenizer.mask_token_id
    if mask_index is None:
        raise RuntimeError("tokenizer has no mask_token_id")

    all_target_log_probs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(phrase_indices), non_causal_batch_size):
            end = min(start + non_causal_batch_size, len(phrase_indices))
            bsz = end - start
            batch_tokens = base_tokens.unsqueeze(0).repeat(bsz, 1)
            batch_attn = base_attn.unsqueeze(0).repeat(bsz, 1)
            sub_positions = phrase_indices[start:end]
            sub_targets = target_tensor[start:end]
            for j, pos in enumerate(sub_positions):
                batch_tokens[j, pos] = mask_index
            logits = model(input_ids=batch_tokens.to(device), attention_mask=batch_attn.to(device)).logits
            idx = torch.arange(logits.shape[0], device=device)
            masked_logits = logits[idx, torch.LongTensor(sub_positions).to(device)]
            log_probs = F.log_softmax(masked_logits / temperature, dim=-1)
            target_lp = torch.gather(log_probs, -1, sub_targets.to(device).unsqueeze(-1)).squeeze(-1)
            all_target_log_probs.append(target_lp.detach().cpu())
    score = torch.cat(all_target_log_probs, dim=0).sum().item()
    return float(score), phrase_indices, target_tokens, len(tokens)


def score_rows_for_model(model_key: str, model_path: Path, rows: list[dict[str, Any]], device: torch.device,
                         non_causal_batch_size: int, official_predictions: dict[str, Any] | None) -> dict[str, Any]:
    t0 = time.time()
    tokenizer = load_tokenizer_like_official(model_path)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.eval()

    records: list[dict[str, Any]] = []
    n_correct = 0
    n_match = 0
    n_with_official = 0
    tie_count = 0
    for i, row in enumerate(rows):
        cand_records = []
        scores = []
        for cidx, sentence in enumerate(row["sentences"]):
            score, phrase_indices, target_tokens, ntok = candidate_score(
                model, tokenizer, sentence, row["completion"], device,
                non_causal_batch_size=non_causal_batch_size,
                temperature=1.0,
            )
            scores.append(score)
            cand_records.append({
                "candidate_index": cidx,
                "sentence": sentence,
                "score": score,
                "num_completion_tokens": len(target_tokens),
                "phrase_indices": phrase_indices,
                "target_tokens": target_tokens,
                "target_token_strings": tokenizer.convert_ids_to_tokens(target_tokens),
                "num_sentence_tokens": ntok,
            })
        if scores[0] == scores[1]:
            tie_count += 1
            pred_idx = 0
        else:
            pred_idx = int(scores[1] > scores[0])
        margin = scores[0] - scores[1]
        correct = int(pred_idx == 0)
        n_correct += correct
        pred_sentence = row["sentences"][pred_idx]
        official_pred = None
        official_match = None
        if official_predictions is not None:
            dom = row["domain"]
            idx = int(row["idx"])
            try:
                official_pred = official_predictions[dom]["predictions"][idx]["pred"].strip()
                official_match = int(official_pred == pred_sentence.strip())
                n_with_official += 1
                n_match += official_match
            except Exception as exc:
                official_pred = f"__OFFICIAL_PRED_LOOKUP_FAILED__:{type(exc).__name__}:{exc}"
                official_match = 0
        rec = {k: row.get(k) for k in [
            "uid", "domain", "file_domain", "idx", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff",
            "Context1", "Context2", "Target1", "Target2",
        ]}
        for k, v in row.items():
            if k.startswith("selection_"):
                rec[k] = v
        rec.update({
            "model_key": model_key,
            "score_candidate0_context1_target1": scores[0],
            "score_candidate1_context2_target1": scores[1],
            "margin_c0_minus_c1": margin,
            "pred_candidate_index": pred_idx,
            "correct": correct,
            "tie": scores[0] == scores[1],
            "official_pred": official_pred,
            "official_prediction_match": official_match,
            "candidates": cand_records,
        })
        records.append(rec)
        if (i + 1) % 25 == 0:
            print(json.dumps({"event": "progress", "model_key": model_key, "rows": i + 1, "acc": n_correct / (i + 1) * 100.0, "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    # Macro over domains on the selected subset, for comparison with official EWoK convention.
    dom_tot: dict[str, int] = {}
    dom_cor: dict[str, int] = {}
    for r in records:
        dom = r["domain"]
        dom_tot[dom] = dom_tot.get(dom, 0) + 1
        dom_cor[dom] = dom_cor.get(dom, 0) + int(r["correct"])
    by_domain = {d: {"n": dom_tot[d], "accuracy": dom_cor[d] / dom_tot[d] * 100.0} for d in sorted(dom_tot)}
    macro = statistics.mean([v["accuracy"] for v in by_domain.values()]) if by_domain else float("nan")
    margins = [r["margin_c0_minus_c1"] for r in records]
    abs_margins = [abs(x) for x in margins]
    summary = {
        "model_key": model_key,
        "model_path": str(model_path),
        "n_rows": len(records),
        "micro_accuracy": n_correct / len(records) * 100.0 if records else float("nan"),
        "domain_macro_accuracy_on_selected_domains": macro,
        "by_domain": by_domain,
        "tie_count": tie_count,
        "margin_mean": statistics.mean(margins) if margins else float("nan"),
        "margin_median": statistics.median(margins) if margins else float("nan"),
        "abs_margin_mean": statistics.mean(abs_margins) if abs_margins else float("nan"),
        "abs_margin_median": statistics.median(abs_margins) if abs_margins else float("nan"),
        "near_zero_frac_abs_lt_0p5": sum(1 for x in abs_margins if x < 0.5) / len(abs_margins) if abs_margins else float("nan"),
        "near_zero_frac_abs_lt_1": sum(1 for x in abs_margins if x < 1.0) / len(abs_margins) if abs_margins else float("nan"),
        "near_zero_frac_abs_lt_2": sum(1 for x in abs_margins if x < 2.0) / len(abs_margins) if abs_margins else float("nan"),
        "official_prediction_matches": n_match,
        "official_prediction_compared": n_with_official,
        "official_prediction_match_rate": n_match / n_with_official if n_with_official else None,
        "elapsed_sec": time.time() - t0,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"summary": summary, "records": records}


def combined_fourcell_analysis(results_by_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    needed = ["clean430", "reinv430", "clean431", "reinv431"]
    if any(k not in results_by_model for k in needed):
        return {"status": "not_fourcell", "available_model_keys": sorted(results_by_model)}
    n = len(results_by_model[needed[0]]["records"])
    rows = []
    for idx in range(n):
        base = {k: results_by_model[needed[0]]["records"][idx].get(k) for k in [
            "uid", "domain", "file_domain", "idx", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff",
            "selection_pattern", "selection_DiD_item", "selection_TE430", "selection_TE431",
        ]}
        bits = []
        margins = {}
        for mk in needed:
            rec = results_by_model[mk]["records"][idx]
            bits.append(str(int(rec["correct"])))
            margins[mk] = rec["margin_c0_minus_c1"]
        pattern = "".join(bits)
        te430 = int(bits[1]) - int(bits[0])
        te431 = int(bits[3]) - int(bits[2])
        did = te431 - te430
        base.update({
            "computed_pattern": pattern,
            "computed_TE430": te430,
            "computed_TE431": te431,
            "computed_DiD": did,
            "margins": margins,
            "abs_margins": {k: abs(v) for k, v in margins.items()},
        })
        rows.append(base)

    def summarize(arr: list[float]) -> dict[str, float]:
        if not arr:
            return {}
        aa = [abs(x) for x in arr]
        return {
            "mean": statistics.mean(arr),
            "median": statistics.median(arr),
            "abs_mean": statistics.mean(aa),
            "abs_median": statistics.median(aa),
            "near_zero_frac_abs_lt_0p5": sum(1 for x in aa if x < 0.5) / len(aa),
            "near_zero_frac_abs_lt_1": sum(1 for x in aa if x < 1.0) / len(aa),
            "near_zero_frac_abs_lt_2": sum(1 for x in aa if x < 2.0) / len(aa),
        }

    pattern_summary: dict[str, Any] = {}
    for r in rows:
        pat = r["computed_pattern"]
        if pat not in pattern_summary:
            pattern_summary[pat] = {"n": 0, "margins": {mk: [] for mk in needed}, "did_sum": 0}
        pattern_summary[pat]["n"] += 1
        pattern_summary[pat]["did_sum"] += r["computed_DiD"]
        for mk in needed:
            pattern_summary[pat]["margins"][mk].append(r["margins"][mk])
    for pat, d in pattern_summary.items():
        d["mean_DiD"] = d["did_sum"] / d["n"]
        d["margin_stats"] = {mk: summarize(vals) for mk, vals in d["margins"].items()}
        del d["margins"]
    pattern_summary = dict(sorted(pattern_summary.items(), key=lambda kv: (kv[1]["mean_DiD"], -kv[1]["n"], kv[0])))

    neg_rows = [r for r in rows if r["computed_DiD"] < 0]
    neg_abs = {mk: [abs(r["margins"][mk]) for r in neg_rows] for mk in needed}
    negative_summary = {
        "n_negative_did_rows": len(neg_rows),
        "fraction_negative_rows_with_all_abs_margins_lt_1": sum(1 for r in neg_rows if all(abs(r["margins"][mk]) < 1.0 for mk in needed)) / len(neg_rows) if neg_rows else None,
        "fraction_negative_rows_with_reinv430_abs_lt_1": sum(1 for r in neg_rows if abs(r["margins"]["reinv430"]) < 1.0) / len(neg_rows) if neg_rows else None,
        "fraction_negative_rows_with_reinv431_abs_lt_1": sum(1 for r in neg_rows if abs(r["margins"]["reinv431"]) < 1.0) / len(neg_rows) if neg_rows else None,
        "abs_margin_stats_on_negative_rows": {mk: summarize([r["margins"][mk] for r in neg_rows]) for mk in needed},
    }
    return {
        "status": "fourcell_analysis",
        "n_rows": n,
        "rows": rows,
        "pattern_summary": pattern_summary,
        "negative_summary": negative_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", nargs="+", required=True, choices=sorted(MODEL_PATHS))
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default="subset")
    ap.add_argument("--selection-csv", type=Path, default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--domain", action="append", default=None, help="Optional domain filter when no selection CSV is used")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, etc. Use cpu while H100 retrains are running")
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--non-causal-batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not EWOK_DIR.exists():
        raise FileNotFoundError(EWOK_DIR)
    all_rows = load_all_ewok_rows(EWOK_DIR)
    domain_filter = set(args.domain) if args.domain else None
    selected = load_selection(args.selection_csv.resolve() if args.selection_csv else None, all_rows, args.max_items, domain_filter)
    domain_counts: dict[str, int] = {}
    for r in selected:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
    preflight = {
        "created_utc": now_utc(),
        "ewok_dir": str(EWOK_DIR),
        "all_ewok_rows": len(all_rows),
        "selected_rows": len(selected),
        "selected_domain_counts": dict(sorted(domain_counts.items())),
        "selection_csv": str(args.selection_csv) if args.selection_csv else None,
        "max_items": args.max_items,
        "domain_filter": sorted(domain_filter) if domain_filter else None,
        "model_keys": args.model_key,
        "model_paths": {k: str(MODEL_PATHS[k]) for k in args.model_key},
        "official_prediction_paths": {k: str(OFFICIAL_PRED_PATHS[k]) for k in args.model_key if k in OFFICIAL_PRED_PATHS},
        "method": "MLM candidate scores mirror official sentence_zero_shot: tokenize candidate sentence, mask each token overlapping the completion span, sum target log-probs at temperature 1.0, compare Context1+Target1 vs Context2+Target1.",
    }
    preflight_path = out_root / f"official_ewok_margin_{args.tag}_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "EWOK_MARGIN_DRYRUN", "preflight": str(preflight_path), "selected_rows": len(selected), "domain_counts": preflight["selected_domain_counts"]}, indent=2), flush=True)
        return
    if not selected:
        raise RuntimeError("No EWoK rows selected")

    device = torch.device(args.device)
    results_by_model: dict[str, dict[str, Any]] = {}
    for mk in args.model_key:
        mp = MODEL_PATHS[mk]
        if not mp.exists():
            raise FileNotFoundError({"model_key": mk, "model_path": str(mp)})
        official_preds = None
        if mk in OFFICIAL_PRED_PATHS and OFFICIAL_PRED_PATHS[mk].exists():
            official_preds = json.loads(OFFICIAL_PRED_PATHS[mk].read_text(encoding="utf-8"))
        print(json.dumps({"event": "score_model_start", "model_key": mk, "model_path": str(mp), "device": str(device), "rows": len(selected)}), flush=True)
        results_by_model[mk] = score_rows_for_model(
            mk, mp, selected, device=device,
            non_causal_batch_size=args.non_causal_batch_size,
            official_predictions=official_preds,
        )
        print(json.dumps({"event": "score_model_done", "model_key": mk, **results_by_model[mk]["summary"]}, default=str), flush=True)

    combined = combined_fourcell_analysis(results_by_model)
    payload = {
        "status": "OFFICIAL_COMPATIBLE_EWOK_MARGINS",
        "created_utc": now_utc(),
        "elapsed_sec": time.time() - t0,
        "preflight": preflight,
        "results_by_model": results_by_model,
        "combined_fourcell_analysis": combined,
        "interpretation_limits": [
            "This is exact-coordinate EWoK candidate scoring for the selected rows, not a new model endpoint.",
            "CPU and GPU arithmetic can differ only on extremely small margins; official prediction comparison is included to validate scoring logic.",
            "Inherited-tokenizer model keys remain mechanism evidence after the Strict-Small-tokenizer correction; corrected-tokenizer model keys should be scored after research retrains finish.",
        ],
    }
    out_json = out_root / f"official_ewok_margin_{args.tag}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Compact CSV for browsing.
    out_csv = out_root / f"official_ewok_margin_{args.tag}_records.csv"
    flat_fields = [
        "model_key", "uid", "domain", "idx", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff",
        "selection_pattern", "selection_DiD_item", "score_candidate0_context1_target1", "score_candidate1_context2_target1",
        "margin_c0_minus_c1", "pred_candidate_index", "correct", "official_prediction_match",
        "num_cand0_completion_tokens", "num_cand1_completion_tokens",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for mk, res in results_by_model.items():
            for rec in res["records"]:
                w.writerow({
                    "model_key": mk,
                    "uid": rec.get("uid"),
                    "domain": rec.get("domain"),
                    "idx": rec.get("idx"),
                    "ConceptA": rec.get("ConceptA"),
                    "ConceptB": rec.get("ConceptB"),
                    "ContextType": rec.get("ContextType"),
                    "ContextDiff": rec.get("ContextDiff"),
                    "TargetDiff": rec.get("TargetDiff"),
                    "selection_pattern": rec.get("selection_pattern"),
                    "selection_DiD_item": rec.get("selection_DiD_item"),
                    "score_candidate0_context1_target1": rec.get("score_candidate0_context1_target1"),
                    "score_candidate1_context2_target1": rec.get("score_candidate1_context2_target1"),
                    "margin_c0_minus_c1": rec.get("margin_c0_minus_c1"),
                    "pred_candidate_index": rec.get("pred_candidate_index"),
                    "correct": rec.get("correct"),
                    "official_prediction_match": rec.get("official_prediction_match"),
                    "num_cand0_completion_tokens": rec.get("candidates", [{}, {}])[0].get("num_completion_tokens"),
                    "num_cand1_completion_tokens": rec.get("candidates", [{}, {}])[1].get("num_completion_tokens"),
                })

    summaries = {mk: res["summary"] for mk, res in results_by_model.items()}
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "elapsed_sec": payload["elapsed_sec"],
        "model_summaries": summaries,
        "fourcell_negative_summary": combined.get("negative_summary") if isinstance(combined, dict) else None,
        "pattern_summary": combined.get("pattern_summary") if isinstance(combined, dict) else None,
    }, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
