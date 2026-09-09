#!/usr/bin/env python3
"""research: four-way EWoK context-target masked-LM margin readout.

Scientific purpose
------------------
Measure whether the crossed context-target interaction in EWoK examples tracks
real differences among existing BabyLM Strict-Small endpoints. This is a
mechanistic lens only. It is not a training objective and does not use EWoK
identities to construct training data.

For each EWoK row, with Context1/Context2 and Target1/Target2, score four
completion pseudo-log-likelihoods under an MLM using the official BabyLM
sentence-zero-shot convention: append completion with a leading space, mask each
completion token one at a time, sum log p(original token | masked sentence).

Scores:
  s11 = PLL(Context1, Target1)
  s21 = PLL(Context2, Target1)  # official competitor for Target1
  s12 = PLL(Context1, Target2)
  s22 = PLL(Context2, Target2)

Official two-candidate margin: s11 - s21.
Crossed interaction: (s11 + s22) - (s12 + s21). This removes absolute target
preference and asks whether each context specifically supports its matching
target rather than the crossed alternatives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
EVAL_EWOK = ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
OUT_DIR = STUDY / "data/ewok_fourway_margin_readout"

DEFAULT_MODELS = {
    # Legal research same-pool tokenizer compact-view reinvest, final legal endpoint.
    "a02_legal16_reinvest_100M": {
        "ckpt": "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M",
        "pred": "experiments/archive/frontier_consolidation/data/compliant_full_eval/official_outputs/complianttok_reinvest_seed43022/EWoK/chck_100M/full_complianttok_reinvest_seed43022_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
        "broad": {"Overall": 41.257770896404615, "EWoK": 50.39},
        "note": "best complete fully legal A02 endpoint",
    },
    # Same tokenizer/corpus family, clean control at 80M for treatment comparison.
    "a02_legal16_clean_80M": {
        "ckpt": "experiments/archive/frontier_consolidation/training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M",
        "pred": "",
        "broad": {"cheap7_mean80M": 41.6022},
        "note": "legal16 clean-Qwen 80M control; no saved EWoK prediction path in this script",
    },
    # Same tokenizer/corpus family, reinvest at matched 80M.
    "a02_legal16_reinvest_80M": {
        "ckpt": "experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
        "pred": "experiments/archive/frontier_consolidation/data/legal_mature_treatment_effect_eval/official_outputs/complianttok_reinvest_seed43022_80M/EWoK/chck_80M/full_complianttok_reinvest_seed43022_80M_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
        "broad": {"cheap7_mean80M": 42.9486},
        "note": "legal16 compact-view reinvest at 80M reference",
    },
    # Negative conditional-innovation probability intervention.
    "a02_step075_strict_innov_80M": {
        "ckpt": "experiments/archive/frontier_consolidation/training/runs/strict_content_innovation_wwm_reinvest_seed43022_80M/hf_model/chck_80M",
        "pred": "experiments/archive/frontier_consolidation/data/strict_innovation_70_80M_eval/official_outputs/strict_content_innovation_seed43022_80M/EWoK/chck_80M/full_strict_content_innovation_seed43022_80M_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
        "broad": {"cheap7_mean80M": 41.8957},
        "note": "closed research innovation-probability route",
    },
    # Non-submittable old-tokenizer endpoint, retained only as mechanism evidence.
    "a02_oldtok_reinvest_100M": {
        "ckpt": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M",
        "pred": "experiments/archive/representation_and_objectives/data/official_ewok_reeval/official_outputs/EWoK/chck_100M/official_ewok_reinvest/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
        "broad": {"Overall_non_submittable": 42.0331347900748, "EWoK": 53.54},
        "note": "old inherited tokenizer; not legal but strong mechanism reference",
    },
    # Legal40k representation baseline and depth variant.
    "a01_legal40k_8x480_100M": {
        "ckpt": "experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M",
        "pred": "experiments/archive/representation_and_objectives/data/legal40k_accum_seed43022_official_ewok/official_outputs/EWoK/chck_100M/official_ewok_legal40k_seed43022/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
        "broad": {"Overall": 41.1406},
        "note": "A01 legal40k 8x480 matched baseline",
    },
    "a01_legal40k_depth12_100M": {
        "ckpt": "experiments/archive/representation_and_objectives/training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model/chck_100M",
        "pred": "experiments/archive/representation_and_objectives/data/legal40k_12x384_depth_seed43022_official_ewok/official_outputs/EWoK/chck_100M/official_ewok_legal40k_12x384_depth_seed43022/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
        "broad": {"Overall": 41.0276},
        "note": "A01 depth-alone route, complete and negative",
    },
}


def jdump(obj: Any, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: pathlib.Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def norm_text(s: Any) -> str:
    return " ".join(str(s).strip().split())


@dataclass(frozen=True)
class EwokRow:
    uid: str
    domain: str
    index: int
    global_index: int
    row: dict[str, Any]

    @property
    def c1(self) -> str:
        return str(self.row["Context1"])

    @property
    def c2(self) -> str:
        return str(self.row["Context2"])

    @property
    def t1(self) -> str:
        return str(self.row["Target1"])

    @property
    def t2(self) -> str:
        return str(self.row["Target2"])

    def sent(self, ci: int, tj: int) -> str:
        c = self.c1 if ci == 1 else self.c2
        t = self.t1 if tj == 1 else self.t2
        return " ".join([c, t])


def load_ewok_rows(data_dir: pathlib.Path) -> list[EwokRow]:
    rows: list[EwokRow] = []
    g = 0
    for path in sorted(data_dir.glob("*.jsonl")):
        domain = path.stem
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(EwokRow(uid=f"{domain}_{i}", domain=domain, index=i, global_index=g, row=row))
                g += 1
    if not rows:
        raise RuntimeError(f"No EWoK rows found under {data_dir}")
    return rows


def choose_rows(rows: list[EwokRow], rows_per_domain: int, seed: int, domains: set[str] | None, hard_domains: bool) -> list[EwokRow]:
    by_domain: dict[str, list[EwokRow]] = {}
    for r in rows:
        if domains is not None and r.domain not in domains:
            continue
        by_domain.setdefault(r.domain, []).append(r)
    if hard_domains:
        wanted = {"physical-dynamics", "spatial-relations", "physical-relations", "social-relations", "material-dynamics"}
        by_domain = {k: v for k, v in by_domain.items() if k in wanted}
    rng = random.Random(seed)
    out: list[EwokRow] = []
    for d, rs in sorted(by_domain.items()):
        if rows_per_domain <= 0 or rows_per_domain >= len(rs):
            out.extend(rs)
        else:
            out.extend(sorted(rng.sample(rs, rows_per_domain), key=lambda x: x.index))
    return out


def load_saved_preds(pred_path: pathlib.Path | None) -> dict[str, list[str]]:
    if pred_path is None or not str(pred_path):
        return {}
    if not pred_path.exists():
        return {}
    raw = json.loads(pred_path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for domain, block in raw.items():
        preds = block.get("predictions", block) if isinstance(block, dict) else block
        out[domain] = [norm_text(p.get("pred")) for p in preds]
    return out


def load_tokenizer(model_path: pathlib.Path):
    try:
        tok = AutoTokenizer.from_pretrained(str(model_path), padding_side="right", trust_remote_code=True)
    except Exception:
        tok = PreTrainedTokenizerFast.from_pretrained(str(model_path), padding_side="right")
    if tok.pad_token_id is None:
        if tok.cls_token_id is not None:
            tok.pad_token_id = tok.cls_token_id
        elif tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        else:
            tok.add_special_tokens({"pad_token": "<pad>"})
    if tok.mask_token_id is None:
        raise RuntimeError(f"Tokenizer at {model_path} has no mask_token_id")
    return tok


def combo_encoding(tokenizer, context: str, target: str, max_length: int) -> dict[str, Any]:
    completion = " " + target
    sentence = " ".join([context, target])
    enc = tokenizer(sentence, return_offsets_mapping=True, truncation=True, max_length=max_length)
    tokens = enc["input_ids"]
    attn = enc["attention_mask"]
    offsets = enc["offset_mapping"]
    start_char_idx = len(sentence) - len(completion)
    phrase_indices: list[int] = []
    target_tokens: list[int] = []
    for i, (start, end) in enumerate(offsets):
        # Match official process_mlm_sentences: include any token whose end offset
        # falls after the completion span begins.
        if end > start_char_idx:
            phrase_indices.append(i)
            target_tokens.append(tokens[i])
    if not phrase_indices:
        return {
            "valid": False,
            "sentence": sentence,
            "tokens": [],
            "attn": [],
            "indices": [],
            "targets": [],
            "reason": "no_completion_tokens",
        }
    return {
        "valid": True,
        "sentence": sentence,
        "tokens": tokens,
        "attn": attn,
        "indices": phrase_indices,
        "targets": target_tokens,
        "reason": "ok",
    }


def score_encoded_batch(model, tokenizer, encs: list[dict[str, Any]], device: torch.device, non_causal_batch_size: int) -> list[float]:
    # Build one masked sequence per completion token, then sum logprobs back to each sentence.
    records: list[tuple[int, list[int], list[int], int, int]] = []
    invalid = [False for _ in encs]
    for ex_i, enc in enumerate(encs):
        if not enc["valid"]:
            invalid[ex_i] = True
            continue
        toks = enc["tokens"]
        attn = enc["attn"]
        for idx, tgt in zip(enc["indices"], enc["targets"]):
            masked = list(toks)
            masked[idx] = tokenizer.mask_token_id
            records.append((ex_i, masked, list(attn), idx, int(tgt)))
    sums = [0.0 for _ in encs]
    if not records:
        return [float("nan") for _ in encs]
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(records), non_causal_batch_size):
        batch = records[start:start + non_causal_batch_size]
        max_len = max(len(r[1]) for r in batch)
        input_ids = []
        attn_mask = []
        idxs = []
        tgts = []
        exs = []
        for ex_i, toks, attn, idx, tgt in batch:
            pad_n = max_len - len(toks)
            input_ids.append(toks + [pad_id] * pad_n)
            attn_mask.append(attn + [0] * pad_n)
            idxs.append(idx)
            tgts.append(tgt)
            exs.append(ex_i)
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
                attention_mask=torch.tensor(attn_mask, dtype=torch.long, device=device),
            )
            logits = out["logits"] if isinstance(out, dict) or hasattr(out, "__getitem__") else out.logits
            if logits.size(1) != max_len:
                logits = logits[:, -max_len:]
            batch_idx = torch.arange(logits.shape[0], device=device)
            masked_logits = logits[batch_idx, torch.tensor(idxs, dtype=torch.long, device=device)]
            lp = F.log_softmax(masked_logits, dim=-1)
            vals = torch.gather(lp, -1, torch.tensor(tgts, dtype=torch.long, device=device).unsqueeze(-1)).squeeze(-1).detach().cpu().tolist()
        for ex_i, val in zip(exs, vals):
            sums[ex_i] += float(val)
    for i, bad in enumerate(invalid):
        if bad:
            sums[i] = float("nan")
    return sums


def official_pred_from_scores(r: EwokRow, s11: float, s21: float) -> str:
    # Official EWoK decode ranks Context1+Target1 against Context2+Target1. Ties are random
    # upstream; here mark deterministic C1 on equality for reproducibility.
    return norm_text(r.sent(1, 1) if s11 >= s21 else r.sent(2, 1))


def official_correct_sentence(r: EwokRow) -> str:
    return norm_text(r.sent(1, 1))


def safe_mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.mean(vals) if vals else None


def safe_median(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return statistics.median(vals) if vals else None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {"n": 0}
    official_correct = [r["official_margin"] > 0 for r in records]
    interaction_pos = [r["interaction"] > 0 for r in records]
    near_tie = [abs(r["official_margin"]) < 1.0 for r in records]
    context_insens = [abs(r["context_effect_t1_C1_minus_C2"]) < 1.0 and abs(r["context_effect_t2_C1_minus_C2"]) < 1.0 for r in records]
    target_prior_abs = [abs(r["target_prior_c1_T1_minus_T2"] + r["target_prior_c2_T1_minus_T2"]) / 2.0 for r in records]
    interaction_vals = [r["interaction"] for r in records]
    off_vals = [r["official_margin"] for r in records]
    return {
        "n": n,
        "official_accuracy_from_scores": 100.0 * sum(official_correct) / n,
        "interaction_positive_rate": 100.0 * sum(interaction_pos) / n,
        "mean_official_margin": safe_mean(off_vals),
        "median_official_margin": safe_median(off_vals),
        "mean_interaction": safe_mean(interaction_vals),
        "median_interaction": safe_median(interaction_vals),
        "mean_abs_target_prior": safe_mean(target_prior_abs),
        "frac_near_tie_abs_official_margin_lt_1": sum(near_tie) / n,
        "frac_context_insensitive_abs_both_context_effects_lt_1": sum(context_insens) / n,
        "wrong_count": sum(not x for x in official_correct),
        "wrong_mean_interaction": safe_mean([r["interaction"] for r, c in zip(records, official_correct) if not c]),
        "correct_mean_interaction": safe_mean([r["interaction"] for r, c in zip(records, official_correct) if c]),
    }


def summarize_by(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(str(r.get(field)), []).append(r)
    out = []
    for k, rs in sorted(groups.items()):
        rec = {field: k}
        rec.update(summarize_records(rs))
        out.append(rec)
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    mx = statistics.mean(x for x, _ in pairs)
    my = statistics.mean(y for _, y in pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    # Simple average-rank Spearman; avoids scipy dependency at interpretation time.
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and vals[order[j]] == vals[order[i]]:
                j += 1
            rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                r[order[k]] = rank
            i = j
        return r
    return pearson(ranks([x for x, _ in pairs]), ranks([y for _, y in pairs]))


def score_model(model_name: str, cfg: dict[str, Any], rows: list[EwokRow], args) -> dict[str, Any]:
    model_path = ROOT / cfg["ckpt"]
    if not model_path.exists():
        raise FileNotFoundError(f"model checkpoint not found for {model_name}: {model_path}")
    pred_path = ROOT / cfg["pred"] if cfg.get("pred") else None
    saved_preds = load_saved_preds(pred_path)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = load_tokenizer(model_path)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.to(device)
    model.eval()
    if args.bf16 and device.type == "cuda":
        model.to(dtype=torch.bfloat16)
    records: list[dict[str, Any]] = []
    validation_total = 0
    validation_match = 0
    validation_mismatches: list[dict[str, Any]] = []
    start_time = time.time()
    for row_start in range(0, len(rows), args.row_batch_size):
        batch_rows = rows[row_start:row_start + args.row_batch_size]
        encs = []
        combo_keys = []
        for r in batch_rows:
            for ci, tj in [(1, 1), (2, 1), (1, 2), (2, 2)]:
                encs.append(combo_encoding(tok, r.c1 if ci == 1 else r.c2, r.t1 if tj == 1 else r.t2, args.max_length))
                combo_keys.append((r.uid, ci, tj))
        vals = score_encoded_batch(model, tok, encs, device, args.non_causal_batch_size)
        by_uid: dict[str, dict[tuple[int, int], float]] = {}
        for (uid, ci, tj), val in zip(combo_keys, vals):
            by_uid.setdefault(uid, {})[(ci, tj)] = val
        for r in batch_rows:
            d = by_uid[r.uid]
            s11 = d[(1, 1)]
            s21 = d[(2, 1)]
            s12 = d[(1, 2)]
            s22 = d[(2, 2)]
            official_margin = s11 - s21
            target2_official_margin = s22 - s12  # C2 supports T2 over C1 if positive.
            interaction = (s11 + s22) - (s12 + s21)
            target_prior_c1 = s11 - s12
            target_prior_c2 = s21 - s22
            context_effect_t1 = s11 - s21
            context_effect_t2 = s12 - s22
            pred = official_pred_from_scores(r, s11, s21)
            correct = pred == official_correct_sentence(r)
            saved_pred = None
            saved_match = None
            if r.domain in saved_preds and r.index < len(saved_preds[r.domain]):
                saved_pred = saved_preds[r.domain][r.index]
                saved_match = pred == saved_pred
                validation_total += 1
                validation_match += int(saved_match)
                if not saved_match and len(validation_mismatches) < 20:
                    validation_mismatches.append({
                        "uid": r.uid,
                        "domain": r.domain,
                        "index": r.index,
                        "computed_pred": pred,
                        "saved_pred": saved_pred,
                        "s11": s11,
                        "s21": s21,
                        "official_margin": official_margin,
                    })
            records.append({
                "uid": r.uid,
                "domain": r.domain,
                "index": r.index,
                "global_index": r.global_index,
                "ContextType": r.row.get("ContextType"),
                "ContextDiff": r.row.get("ContextDiff"),
                "TargetDiff": r.row.get("TargetDiff"),
                "ConceptA": r.row.get("ConceptA"),
                "ConceptB": r.row.get("ConceptB"),
                "s11_C1T1": s11,
                "s21_C2T1": s21,
                "s12_C1T2": s12,
                "s22_C2T2": s22,
                "official_margin": official_margin,
                "target2_context_margin": target2_official_margin,
                "interaction": interaction,
                "target_prior_c1_T1_minus_T2": target_prior_c1,
                "target_prior_c2_T1_minus_T2": target_prior_c2,
                "context_effect_t1_C1_minus_C2": context_effect_t1,
                "context_effect_t2_C1_minus_C2": context_effect_t2,
                "computed_official_pred_correct": correct,
                "saved_pred_match": saved_match,
            })
    # Free GPU memory before next model.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    validation_rate = validation_match / validation_total if validation_total else None
    summary = {
        "model_name": model_name,
        "checkpoint": str(model_path),
        "prediction_path": str(pred_path) if pred_path else None,
        "broad_reference": cfg.get("broad", {}),
        "note": cfg.get("note"),
        "tokenizer_vocab_size": len(tok),
        "tokenizer_mask_token_id": tok.mask_token_id,
        "tokenizer_sha256": sha256_file(model_path / "tokenizer.json") if (model_path / "tokenizer.json").exists() else None,
        "n_rows_scored": len(records),
        "elapsed_sec": time.time() - start_time,
        "validation_against_saved_predictions": {
            "available": validation_total > 0,
            "total": validation_total,
            "matches": validation_match,
            "match_rate": validation_rate,
            "mismatches_first20": validation_mismatches,
            "interpretation": "Must be near 1.0 before treating margins as aligned with official MLM EWoK scoring.",
        },
        "overall": summarize_records(records),
        "by_domain": summarize_by(records, "domain"),
        "by_context_type": summarize_by(records, "ContextType"),
        "records": records,
    }
    return summary


def compare_models(model_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for name, summ in model_summaries.items():
        overall = summ["overall"]
        broad = summ.get("broad_reference", {}) or {}
        broad_scalar = None
        for k in ["Overall", "Overall_non_submittable", "cheap7_mean80M", "EWoK"]:
            if k in broad:
                broad_scalar = float(broad[k])
                break
        rows.append({
            "model": name,
            "n": overall.get("n"),
            "official_accuracy_from_scores": overall.get("official_accuracy_from_scores"),
            "mean_official_margin": overall.get("mean_official_margin"),
            "mean_interaction": overall.get("mean_interaction"),
            "median_interaction": overall.get("median_interaction"),
            "interaction_positive_rate": overall.get("interaction_positive_rate"),
            "frac_context_insensitive": overall.get("frac_context_insensitive_abs_both_context_effects_lt_1"),
            "broad_scalar_for_correlations": broad_scalar,
            "broad_reference": broad,
        })
    # Use score-derived official accuracy on the same sampled rows and broad scalar when available.
    mean_inter = [r["mean_interaction"] for r in rows]
    sampled_acc = [r["official_accuracy_from_scores"] for r in rows]
    broad_scalar = [r["broad_scalar_for_correlations"] if r["broad_scalar_for_correlations"] is not None else float("nan") for r in rows]
    return {
        "model_table": sorted(rows, key=lambda r: (r["mean_interaction"] if r["mean_interaction"] is not None else -1e9), reverse=True),
        "correlations_across_models": {
            "pearson_mean_interaction_vs_sampled_official_accuracy": pearson(mean_inter, sampled_acc),
            "spearman_mean_interaction_vs_sampled_official_accuracy": spearman(mean_inter, sampled_acc),
            "pearson_mean_interaction_vs_broad_reference_scalar": pearson(mean_inter, broad_scalar),
            "spearman_mean_interaction_vs_broad_reference_scalar": spearman(mean_inter, broad_scalar),
            "note": "Small-N descriptive correlations only; used to decide whether the interaction is worth testing on legal corpus-derived contrasts.",
        },
    }


def write_csv_records(path: pathlib.Path, model_summaries: dict[str, dict[str, Any]]) -> None:
    fields = [
        "model", "uid", "domain", "index", "global_index", "ContextType", "ContextDiff", "TargetDiff", "ConceptA", "ConceptB",
        "s11_C1T1", "s21_C2T1", "s12_C1T2", "s22_C2T2", "official_margin", "target2_context_margin", "interaction",
        "target_prior_c1_T1_minus_T2", "target_prior_c2_T1_minus_T2", "context_effect_t1_C1_minus_C2", "context_effect_t2_C1_minus_C2",
        "computed_official_pred_correct", "saved_pred_match",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model_name, summ in model_summaries.items():
            for r in summ["records"]:
                row = {k: r.get(k) for k in fields}
                row["model"] = model_name
                w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-domain", type=int, default=2, help="Rows sampled per EWoK domain; <=0 means all rows.")
    ap.add_argument("--seed", type=int, default=80080)
    ap.add_argument("--models", nargs="+", default=["a02_legal16_reinvest_100M", "a02_oldtok_reinvest_100M"], choices=sorted(DEFAULT_MODELS))
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--hard-domains", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--row-batch-size", type=int, default=8)
    ap.add_argument("--non-causal-batch-size", type=int, default=64)
    ap.add_argument("--out-subdir", default="pilot")
    args = ap.parse_args()

    out_dir = OUT_DIR / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_ewok_rows(EVAL_EWOK)
    domains = set(args.domains) if args.domains else None
    sample_rows = choose_rows(all_rows, args.rows_per_domain, args.seed, domains, args.hard_domains)
    if not sample_rows:
        raise RuntimeError("No sampled rows selected")

    manifest = {
        "status": "EWOK_FOURWAY_READOUT_START",
        "purpose": "Mechanistic four-way EWoK context-target scoring on existing checkpoints; not a training objective.",
        "official_decoder_reference": {
            "read_files_py": "evaluation_pipeline/sentence_zero_shot/read_files.py: decode_ewok uses Context1+Target1 and Context2+Target1 only, label 0.",
            "compute_results_py": "evaluation_pipeline/sentence_zero_shot/compute_results.py: MLM sums log probs over completion tokens masked one-at-a-time.",
            "calculate_results_py": "calculate_results_from_pred.py: correct EWoK sentence is Context1 + Target1.",
        },
        "data_dir": str(EVAL_EWOK),
        "all_rows": len(all_rows),
        "sample_rows": len(sample_rows),
        "rows_per_domain": args.rows_per_domain,
        "seed": args.seed,
        "models": args.models,
        "args": vars(args),
        "sample_uids": [r.uid for r in sample_rows],
    }
    jdump(manifest, out_dir / "manifest.json")

    model_summaries: dict[str, dict[str, Any]] = {}
    for model_name in args.models:
        print(json.dumps({"event": "model_start", "model": model_name, "rows": len(sample_rows)}, ensure_ascii=False), flush=True)
        model_summaries[model_name] = score_model(model_name, DEFAULT_MODELS[model_name], sample_rows, args)
        jdump(model_summaries[model_name], out_dir / f"{model_name}.json")
        vr = model_summaries[model_name]["validation_against_saved_predictions"]
        print(json.dumps({"event": "model_done", "model": model_name, "validation_match_rate": vr["match_rate"], "overall": model_summaries[model_name]["overall"]}, ensure_ascii=False), flush=True)

    comparison = compare_models(model_summaries)
    result = {
        "status": "EWOK_FOURWAY_READOUT_DONE",
        "manifest": manifest,
        "comparison": comparison,
        "model_summaries_without_records": {
            k: {kk: vv for kk, vv in v.items() if kk != "records"}
            for k, v in model_summaries.items()
        },
        "validation_requirement": "Interpret crossed interactions only for models whose saved-prediction validation match rate is near 1.0 or whose prediction path is intentionally absent.",
    }
    jdump(result, out_dir / "ewok_fourway_readout_summary.json")
    write_csv_records(out_dir / "ewok_fourway_records.csv", model_summaries)

    lines = ["# research EWoK four-way context-target readout\n\n"]
    lines.append("This is a mechanistic measurement over existing checkpoints, not a pretraining target. Official EWoK uses only C1T1 vs C2T1; this readout additionally scores C1T2 and C2T2 and reports interaction `(s11+s22)-(s12+s21)`.\n\n")
    lines.append(f"Rows: {len(sample_rows)} from `{EVAL_EWOK}`; rows/domain={args.rows_per_domain}; seed={args.seed}.\n\n")
    lines.append("## Model table\n")
    for r in comparison["model_table"]:
        lines.append(f"- {r['model']}: acc_from_scores={r['official_accuracy_from_scores']:.3f}, mean_official_margin={r['mean_official_margin']:.4f}, mean_interaction={r['mean_interaction']:.4f}, interaction_positive_rate={r['interaction_positive_rate']:.2f}, context_insensitive_frac={r['frac_context_insensitive']:.3f}, broad={r['broad_reference']}\n")
    lines.append("\n## Validation against saved official predictions\n")
    for name, summ in model_summaries.items():
        vr = summ["validation_against_saved_predictions"]
        lines.append(f"- {name}: available={vr['available']}, match_rate={vr['match_rate']}, matches={vr['matches']}/{vr['total']}\n")
        if vr["mismatches_first20"]:
            lines.append(f"  first mismatch: {vr['mismatches_first20'][0]}\n")
    lines.append("\n## Correlations across scored models\n")
    for k, v in comparison["correlations_across_models"].items():
        lines.append(f"- {k}: {v}\n")
    lines.append(f"\nCSV records: `{out_dir / 'ewok_fourway_records.csv'}`\n")
    lines.append(f"Full JSON: `{out_dir / 'ewok_fourway_readout_summary.json'}`\n")
    (out_dir / "ewok_fourway_readout_summary.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "out_dir": str(out_dir),
        "model_table": comparison["model_table"],
        "correlations": comparison["correlations_across_models"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
