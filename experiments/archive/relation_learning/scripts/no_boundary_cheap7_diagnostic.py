#!/usr/bin/env python3
"""research: no-boundary cheap7 diagnostic for chck82 and coherent86.

This is not a submission evaluator. It reproduces the BabyLM masked-LM
candidate-ranking surface except for one explicit input perturbation:
`input_form=no_special` tokenizes candidate inputs with add_special_tokens=False,
removing the boundary tokens that official evaluation always supplies.  It also
supports `with_special` so the same scorer can be checked against existing
accepted cheap7 values.

The scientific question is whether the frozen v4 trunk/coherent86 score profile,
especially Supplement and EWoK, improves or deteriorates when boundary tokens are
removed.  If removing boundary tokens raises Supplement/EWoK, then the coherent-
special and half-format losses are branch redistribution rather than the price of
using boundary tokens.  If removing boundary tokens lowers Supplement/EWoK, then
part of the v4 score profile depends on not using the boundary channel.
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
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

import pandas as pd
import statsmodels.formula.api as smf
import torch
import torch.nn.functional as F
from tqdm import tqdm
# Transformers dynamic modules consult their cache path at import time.  In the
# execution environment the default shared model cache is read-only, so set a
# local cache before importing Transformers; setup_cache() below then
# redirects later file artifacts to the requested output directory.
USER_ROOT = _public_path('.')
def _preimport_cache_base() -> pathlib.Path:
    for _i, _arg in enumerate(sys.argv):
        if _arg == "--out-dir" and _i + 1 < len(sys.argv):
            _p0 = pathlib.Path(sys.argv[_i + 1])
            return _p0 if _p0.is_absolute() else USER_ROOT / _p0
        if _arg.startswith("--out-dir="):
            _p0 = pathlib.Path(_arg.split("=", 1)[1])
            return _p0 if _p0.is_absolute() else USER_ROOT / _p0
    return _public_path('experiments/archive/relation_learning/data/no_boundary_cheap7_diagnostic')


_PREIMPORT_CACHE = _preimport_cache_base() / "runtime_cache_preimport_global"
for _k, _p in {
    "HF_HOME": _PREIMPORT_CACHE / "hf_home",
    "HF_HUB_CACHE": _PREIMPORT_CACHE / "hf_home" / "hub",
    "HUGGINGFACE_HUB_CACHE": _PREIMPORT_CACHE / "hf_home" / "hub",
    "TRANSFORMERS_CACHE": _PREIMPORT_CACHE / "transformers",
    "HF_MODULES_CACHE": _PREIMPORT_CACHE / "modules",
    "HF_DATASETS_CACHE": _PREIMPORT_CACHE / "datasets",
    "TMPDIR": _PREIMPORT_CACHE / "tmp",
}.items():
    _p.mkdir(parents=True, exist_ok=True)
    os.environ[_k] = str(_p.resolve())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast
INITIAL_MODEL_STUDIES_REPO = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
PRISTINE_FULL = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval')
GLOBALPIQA_FULL = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/no_boundary_cheap7_diagnostic')

ENDPOINTS = {
    "chck82_slow_scale1p75": {
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
        "known_official_scores": {
            "BLiMP": 69.49128440366973,
            "Supplement": 62.86,
            "EWoK": 50.73545339288396,
            "Entity": 28.314041631085145,
            "COMPS": 52.10117449664429,
            "GlobalPIQA": 37.577777777777776,
            "Reading": 6.63625,
            "cheap7": 43.95944987645173,
        },
    },
    "coherent86_alpha075": {
        "model_path": _public_path('models/frontier'),
        "known_official_scores": {
            "BLiMP": 69.51,
            "Supplement": 63.562188577999236,
            "EWoK": 50.69999505220938,
            "Entity": 28.320012915807294,
            "COMPS": 51.96,
            "GlobalPIQA": 38.56504854368932,
            "Reading": 6.6525,
            "cheap7": 44.18139272710075,
        },
    },
}

ZERO_SHOT_TASKS = [
    {"column": "BLiMP", "task": "blimp", "data_path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/blimp_filtered'), "batch_size": 64},
    {"column": "Supplement", "task": "blimp", "data_path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered'), "batch_size": 64},
    {"column": "EWoK", "task": "ewok", "data_path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered'), "batch_size": 64},
    {"column": "Entity", "task": "entity_tracking", "data_path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking'), "batch_size": 64},
    {"column": "COMPS", "task": "comps", "data_path": _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/comps'), "batch_size": 64},
    {"column": "GlobalPIQA_parallel", "task": "global_piqa_parallel", "data_path": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_parallel'), "batch_size": 64},
    {"column": "GlobalPIQA_nonparallel", "task": "global_piqa_nonparallel", "data_path": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_nonparallel'), "batch_size": 64},
]
CHEAP7_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
LENGTH_NORMALIZED_TASKS = {"global_piqa_parallel", "global_piqa_nonparallel"}


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def setup_cache(out_dir: pathlib.Path) -> None:
    cache = out_dir / "runtime_cache_preimport"
    for k, p in {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_jsonl(path: pathlib.Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def iter_jsonl_files(data_path: pathlib.Path) -> list[pathlib.Path]:
    if data_path.is_file():
        return [data_path]
    return sorted([p for p in data_path.glob("*.jsonl") if p.is_file()], key=lambda p: p.name)


@dataclass
class CandidateExample:
    uid: str
    metadata: dict[str, str]
    sentences: list[str]
    completions: list[str]
    label: int
    source_file: str
    raw_index: int
    raw: dict[str, Any]


def decode_blimp(raw: dict[str, Any], file_path: pathlib.Path, raw_index: int) -> CandidateExample:
    field = raw.get("field", "supplement")
    if field == "syntax_semantics":
        field = "syntax/semantics"
    uid = str(raw.get("UID", file_path.stem))
    term = str(raw.get("linguistics_term", "supplement"))
    return CandidateExample(uid=uid, metadata={"UID": uid, "field": field, "linguistics_term": term},
                            sentences=[raw["sentence_good"], raw["sentence_bad"]],
                            completions=[raw["sentence_good"], raw["sentence_bad"]], label=0,
                            source_file=file_path.stem, raw_index=raw_index, raw=raw)


def decode_ewok(raw: dict[str, Any], file_path: pathlib.Path, raw_index: int) -> CandidateExample:
    uid = str(raw["Domain"])
    return CandidateExample(uid=uid,
                            metadata={"UID": uid, "context_type": str(raw.get("ContextType", "")),
                                      "context_contrast": str(raw.get("ContextDiff", "")),
                                      "target_contrast": str(raw.get("TargetDiff", ""))},
                            sentences=[" ".join([raw["Context1"], raw["Target1"]]),
                                       " ".join([raw["Context2"], raw["Target1"]])],
                            completions=[" " + raw["Target1"], " " + raw["Target1"]], label=0,
                            source_file=file_path.stem, raw_index=raw_index, raw=raw)


def decode_entity(raw: dict[str, Any], file_path: pathlib.Path, raw_index: int) -> CandidateExample | None:
    if any("nothing" in opt for opt in raw["options"]):
        return None
    uid = f'{file_path.stem}_{raw["numops"]}_ops'
    return CandidateExample(uid=uid, metadata={"UID": uid},
                            sentences=[raw["input_prefix"] + opt for opt in raw["options"]],
                            completions=[opt for opt in raw["options"]], label=0,
                            source_file=file_path.stem, raw_index=raw_index, raw=raw)


def decode_comps(raw: dict[str, Any], file_path: pathlib.Path, raw_index: int) -> CandidateExample:
    acc = " ".join([raw["prefix_acceptable"], raw["property_phrase"]])
    bad = " ".join([raw["prefix_unacceptable"], raw["property_phrase"]])
    if file_path.stem == "comps_base":
        subset = "base"
    elif file_path.stem == "comps_wugs":
        subset = "wugs"
    elif file_path.stem == "comps_wugs_dist-before":
        subset = "wugs_dist_before"
    else:
        subset = "wugs_dist_in_between"
    return CandidateExample(uid=subset, metadata={"UID": subset},
                            sentences=[acc, bad], completions=[raw["property_phrase"], raw["property_phrase"]],
                            label=0, source_file=file_path.stem, raw_index=raw_index, raw=raw)


def decode_global(raw: dict[str, Any], file_path: pathlib.Path, raw_index: int, n_solutions: int) -> CandidateExample:
    sols = [raw[f"solution{i}"] for i in range(n_solutions)]
    uid = str(raw.get("example_id", raw_index))
    return CandidateExample(uid=uid, metadata={"UID": uid},
                            sentences=[" ".join([raw["prompt"], s]) for s in sols],
                            completions=[" " + s for s in sols], label=int(raw["label"]),
                            source_file=file_path.stem, raw_index=raw_index, raw=raw)


def load_examples(task: str, data_path: pathlib.Path, limit_per_file: int = 0) -> list[CandidateExample]:
    examples: list[CandidateExample] = []
    for fp in iter_jsonl_files(data_path):
        rows = read_jsonl(fp, limit=limit_per_file)
        for i, raw in enumerate(rows):
            ex: CandidateExample | None
            if task == "blimp":
                ex = decode_blimp(raw, fp, i)
            elif task == "ewok":
                ex = decode_ewok(raw, fp, i)
            elif task == "entity_tracking":
                ex = decode_entity(raw, fp, i)
            elif task == "comps":
                ex = decode_comps(raw, fp, i)
            elif task == "global_piqa_parallel":
                ex = decode_global(raw, fp, i, 4)
            elif task == "global_piqa_nonparallel":
                ex = decode_global(raw, fp, i, 2)
            else:
                raise NotImplementedError(task)
            if ex is not None:
                examples.append(ex)
    return examples


def load_tokenizer(model_path: pathlib.Path):
    try:
        proc = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        return proc.tokenizer if hasattr(proc, "tokenizer") else proc
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        except Exception:
            return PreTrainedTokenizerFast.from_pretrained(str(model_path))


def ensure_pad(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "cls_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.cls_token_id
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def encode_candidate(tokenizer, sentence: str, completion: str, add_special_tokens: bool) -> tuple[list[int], list[int], list[int]]:
    enc = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    tokens = list(enc["input_ids"])
    attn = list(enc["attention_mask"])
    offsets = list(enc["offset_mapping"])
    start_char = len(sentence) - len(completion)
    phrase_indices: list[int] = []
    targets: list[int] = []
    for i, (start, end) in enumerate(offsets):
        # Special tokens usually have (0,0), so they are not scored when
        # completion starts at char 0.  This matches the official script.
        if end > start_char:
            phrase_indices.append(i)
            targets.append(int(tokens[i]))
    return tokens, attn, phrase_indices, targets


def score_candidate(model, tokenizer, sentence: str, completion: str, add_special_tokens: bool,
                    device: torch.device, non_causal_batch_size: int) -> tuple[float, int]:
    tokens, attn, idxs, targets = encode_candidate(tokenizer, sentence, completion, add_special_tokens)
    if not idxs:
        return float("nan"), 0
    mask_id = int(tokenizer.mask_token_id)
    rows = []
    masks = []
    for idx in idxs:
        row = list(tokens)
        row[idx] = mask_id
        rows.append(torch.tensor(row, dtype=torch.long))
        masks.append(torch.tensor(attn, dtype=torch.long))
    total = 0.0
    with torch.no_grad():
        for s in range(0, len(rows), non_causal_batch_size):
            batch_rows = rows[s:s + non_causal_batch_size]
            batch_masks = masks[s:s + non_causal_batch_size]
            # All rows for one sentence have identical length, so stack directly.
            input_ids = torch.stack(batch_rows, dim=0).to(device)
            attention_mask = torch.stack(batch_masks, dim=0).to(device)
            indices = torch.tensor(idxs[s:s + non_causal_batch_size], dtype=torch.long, device=device)
            target = torch.tensor(targets[s:s + non_causal_batch_size], dtype=torch.long, device=device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out[0] if isinstance(out, tuple) else out["logits"]
            mb = torch.arange(logits.shape[0], device=device)
            masked_logits = logits[mb, indices]
            lp = F.log_softmax(masked_logits, dim=-1)
            total += float(lp.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum().item())
    return total, len(idxs)


def process_result(task: str, examples: list[CandidateExample], chosen: list[int]) -> tuple[dict[str, dict[str, float]], float]:
    # Mirrors sentence_zero_shot.run.process_results: subdomain accuracies by
    # metadata key, average over UID for all tasks except entity, where regular,
    # ambiref, move_contents splits are averaged.
    total: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    corr: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex, c in zip(examples, chosen):
        ok = int(c == ex.label)
        for key, val in ex.metadata.items():
            sval = str(val)
            total[key][sval] += 1
            corr[key][sval] += ok
    accs: dict[str, dict[str, float]] = {}
    for key in total:
        accs[key] = {v: 100.0 * corr[key][v] / total[key][v] for v in total[key]}
    if task != "entity_tracking":
        score = sum(accs["UID"].values()) / len(accs["UID"])
    else:
        split_accs = []
        split_dict = dict(accs.get("UID", {}))
        for split in ["regular", "ambiref", "move_contents"]:
            vals = [v for k, v in split_dict.items() if k.startswith(split)]
            if vals:
                split_val = sum(vals) / len(vals)
                split_dict[split] = split_val
                split_accs.append(split_val)
        accs["UID"] = split_dict
        score = sum(split_accs) / len(split_accs)
    return accs, float(score)


def score_task(model, tokenizer, spec: dict[str, Any], add_special_tokens: bool, device: torch.device,
               out_dir: pathlib.Path, max_items_per_file: int = 0, batch_size: int = 64,
               non_causal_batch_size: int = 64, save_predictions: bool = True) -> dict[str, Any]:
    task = spec["task"]
    column = spec["column"]
    t0 = time.time()
    examples = load_examples(task, pathlib.Path(spec["data_path"]), limit_per_file=max_items_per_file)
    if not examples:
        raise RuntimeError(f"no examples loaded for {column}")
    chosen: list[int] = []
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    token_counts: list[int] = []
    score_rows: list[dict[str, Any]] = []
    iterator = tqdm(range(0, len(examples), batch_size), desc=f"{column}", leave=False)
    for start in iterator:
        for ex in examples[start:start + batch_size]:
            cand_scores = []
            cand_token_counts = []
            for sent, comp in zip(ex.sentences, ex.completions):
                lp_sum, n_tok = score_candidate(model, tokenizer, sent, comp, add_special_tokens, device, non_causal_batch_size)
                if task in LENGTH_NORMALIZED_TASKS and n_tok > 0:
                    score = lp_sum / n_tok
                else:
                    score = lp_sum
                cand_scores.append(score)
                cand_token_counts.append(n_tok)
            # If a candidate somehow gets no scored tokens, push it to -inf.
            clean_scores = [(-float("inf") if math.isnan(x) else x) for x in cand_scores]
            c = int(max(range(len(clean_scores)), key=lambda i: clean_scores[i]))
            chosen.append(c)
            token_counts.extend(cand_token_counts)
            pred_text = ex.sentences[c] if task in ("comps", "ewok") else ex.completions[c]
            num_id_matches = len(predictions[ex.uid])
            predictions[ex.uid].append({"id": f"{ex.uid}_{num_id_matches}", "pred": pred_text})
            if save_predictions:
                score_rows.append({
                    "column": column, "task": task, "uid": ex.uid, "source_file": ex.source_file,
                    "raw_index": ex.raw_index, "label": ex.label, "chosen": c,
                    "correct": int(c == ex.label), "candidate_scores": cand_scores,
                    "candidate_token_counts": cand_token_counts,
                })
    accs, score = process_result(task, examples, chosen)
    task_out = out_dir / column
    task_out.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        pred_obj = {uid: {"predictions": vals} for uid, vals in predictions.items()}
        (task_out / "predictions.json").write_text(json.dumps(pred_obj, ensure_ascii=False) + "\n", encoding="utf-8")
        with (task_out / "candidate_scores.jsonl").open("w", encoding="utf-8") as f:
            for row in score_rows:
                print(json.dumps(row, ensure_ascii=False), file=f)
    report_lines = ["TEMPERATURE: 1.00", ""]
    for key, vals in accs.items():
        report_lines.append(f"### {key.upper()} ACCURACY")
        for sub, val in vals.items():
            report_lines.append(f"{sub}: {val:.2f}")
        report_lines.append("")
    report_lines += ["### AVERAGE ACCURACY", f"{score:.2f}", ""]
    (task_out / "best_temperature_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "column": column, "task": task, "data_path": rel(spec["data_path"]), "score": score,
        "n_examples": len(examples), "mean_scored_tokens_per_candidate": float(mean(token_counts)) if token_counts else None,
        "add_special_tokens": add_special_tokens, "elapsed_sec": round(time.time() - t0, 3),
        "output_dir": rel(task_out), "predictions": rel(task_out / "predictions.json") if save_predictions else None,
        "report": rel(task_out / "best_temperature_report.txt"),
    }


def get_logits(outputs):
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs["logits"]


def p_mlm_next(sentence: str, word: str, model, tokenizer, device: torch.device,
               add_special_tokens: bool, num_mask_tokens: int = 3) -> tuple[float, int]:
    text = "".join([sentence, "".join([tokenizer.mask_token for _ in range(num_mask_tokens)])])
    inpts = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
    if int(inpts.input_ids[:, -1].item()) == int(tokenizer.mask_token_id):
        position = -num_mask_tokens
    else:
        position = -(num_mask_tokens + 1)
    with torch.no_grad():
        outputs = model(**inpts)
        logits = get_logits(outputs)[:, position, :].cpu()
    target_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
    if not target_ids:
        return float("nan"), 0
    if len(target_ids) == 1:
        p = torch.softmax(logits[0], dim=-1)[int(target_ids[0])].item()
        return p, 1
    out_p = []
    target_id = int(target_ids[0])
    out_p.append(torch.softmax(logits[0], dim=-1)[target_id].item())
    next_sentence = sentence + tokenizer.decode(target_id)
    for tok in target_ids[1:]:
        t = tokenizer.decode(int(tok))
        p, _ = p_mlm_next(next_sentence, t, model, tokenizer, device, add_special_tokens, num_mask_tokens)
        out_p.append(p)
        next_sentence = next_sentence + t
    # Match official get_p2_mlm: product of subword probabilities.
    prod = 1.0
    for p in out_p:
        prod *= p
    return prod, 1


def parse_reading_scores_from_df(df: pd.DataFrame, out_dir: pathlib.Path) -> dict[str, float]:
    variables = ['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound', 'self_paced_reading_time', 'ELAN', 'LAN', 'N400', 'P600', 'EPNP', 'PNP']
    correlations = df[["pred"] + variables].corr()["pred"]
    with (out_dir / "correlations.txt").open("w", encoding="utf-8") as f:
        for index, values in correlations.items():
            if index != "pred":
                print(f"{index}\t{values:.4f}", file=f)
    results = []
    report_values = []
    for dv in variables:
        temp = df[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        aic_baseline = float(OLS_baseline.aic)
        temp = df[["pred", dv, "Subtlex_log10", "length", "context_length"]].dropna()
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred', data=temp).fit()
        results.append({
            "Predicted variable": dv, "Coefficient": float(OLS_model.params["pred"]),
            "Number of standard deviations": float(OLS_model.tvalues["pred"]),
            "P-value": float(OLS_model.pvalues["pred"]), "R2": float(OLS_model.rsquared),
            "Change in R2 from baseline": float(OLS_model.rsquared - R2_baseline),
            "AIC": float(OLS_model.aic), "Change in AIC from baseline": float(OLS_model.aic - aic_baseline),
        })
        if "RT" in dv:
            report_values.append(((float(OLS_model.rsquared)-R2_baseline)/(1-R2_baseline)) * 100)
    eye_score = float(sum(report_values) / len(report_values))
    with (out_dir / "predictive_power.jsonl").open("w", encoding="utf-8") as f:
        for res in results:
            print(json.dumps(res), file=f)
    results2 = []
    self_score = float("nan")
    for dv in variables:
        temp = df[[dv, "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        aic_baseline = float(OLS_baseline.aic)
        temp = df[["pred", dv, "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred + pred', data=temp).fit()
        results2.append({
            "Predicted variable": dv, "Coefficient": float(OLS_model.params["pred"]),
            "Number of standard deviations": float(OLS_model.tvalues["pred"]),
            "P-value": float(OLS_model.pvalues["pred"]), "R2": float(OLS_model.rsquared),
            "Change in R2 from baseline": float(OLS_model.rsquared - R2_baseline),
            "AIC": float(OLS_model.aic), "Change in AIC from baseline": float(OLS_model.aic - aic_baseline),
        })
        if "self" in dv:
            self_score = ((float(OLS_model.rsquared)-R2_baseline)/(1-R2_baseline)) * 100
    with (out_dir / "predictive_power_spillover.jsonl").open("w", encoding="utf-8") as f:
        for res in results2:
            print(json.dumps(res), file=f)
    with (out_dir / "report.txt").open("w", encoding="utf-8") as f:
        print(f"EYE TRACKING SCORE: {eye_score:.2f}", file=f)
        print(f"SELF-PACED READING SCORE: {self_score:.2f}", file=f)
    return {"Reading_eye": eye_score, "Reading_self_paced": float(self_score), "Reading": float((eye_score + self_score) / 2.0)}


def score_reading(model, tokenizer, add_special_tokens: bool, device: torch.device,
                  out_dir: pathlib.Path, max_rows: int = 0) -> dict[str, Any]:
    t0 = time.time()
    data_path = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/reading/reading_data.csv')
    df = pd.read_csv(data_path, dtype={"item": str})
    df["item"] = df["item"].fillna("None")
    if max_rows:
        df = df.head(max_rows).copy()
    out = []
    prev = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading", leave=False):
        p, _ = p_mlm_next(row["item"], row["word"], model, tokenizer, device, add_special_tokens, num_mask_tokens=3)
        out.append(p)
        if isinstance(row["prev_item"], str):
            pp, _ = p_mlm_next(row["prev_item"], row["prev_word"], model, tokenizer, device, add_special_tokens, num_mask_tokens=3)
            prev.append(-math.log(pp))
        else:
            prev.append(float("nan"))
    df["pred"] = [-math.log(max(p, 1e-45)) for p in out]
    df["prev_pred"] = prev
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "prediction.jsonl").open("w", encoding="utf-8") as f:
        for index, row in df.iterrows():
            print(json.dumps({"Index": int(index), "Sentence": row["item"], "Word": row["word"], "Logprob": row["pred"], "Prev_Logprob": row["prev_pred"]}), file=f)
    with (out_dir / "predictions.json").open("w", encoding="utf-8") as f:
        preds_dict = {"reading": {"predictions": []}}
        for index, row in df.iterrows():
            preds_dict["reading"]["predictions"].append({"id": int(index), "pred": row["pred"], "prev_pred": row["prev_pred"]})
        json.dump(preds_dict, f)
    scores = parse_reading_scores_from_df(df, out_dir)
    return {"column": "Reading", "data_path": rel(data_path), "scores": scores, "score": scores["Reading"],
            "n_rows": int(len(df)), "add_special_tokens": add_special_tokens,
            "elapsed_sec": round(time.time() - t0, 3), "output_dir": rel(out_dir),
            "predictions": rel(out_dir / "predictions.json"), "report": rel(out_dir / "report.txt")}


def model_identity(model, model_path: pathlib.Path) -> dict[str, Any]:
    params = list(model.named_parameters())
    names = [n for n, _ in params]
    return {
        "model_path": rel(model_path),
        "loaded_class": model.__class__.__name__,
        "total_params": int(sum(p.numel() for _, p in params)),
        "slow_adapter_params": int(sum(p.numel() for n, p in params if "slow" in n.lower() or "adapter" in n.lower())),
        "private_params": int(sum(p.numel() for n, p in params if "private" in n.lower())),
        "sample_special_or_adapter_names": [n for n in names if any(x in n.lower() for x in ["private", "adapter", "slow"] )][:20],
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def evaluate_endpoint(endpoint: str, input_forms: list[str], args: argparse.Namespace) -> dict[str, Any]:
    spec = ENDPOINTS[endpoint]
    model_path = pathlib.Path(spec["model_path"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = load_tokenizer(model_path)
    ensure_pad(tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.to(device)
    model.eval()
    ident = model_identity(model, model_path)
    endpoint_out = pathlib.Path(args.out_dir) / endpoint
    endpoint_out.mkdir(parents=True, exist_ok=True)
    with (endpoint_out / "model_identity.json").open("w", encoding="utf-8") as f:
        json.dump(ident, f, indent=2)
    results_by_form: dict[str, Any] = {}
    for form in input_forms:
        add_special = form == "with_special"
        form_out = endpoint_out / form
        form_out.mkdir(parents=True, exist_ok=True)
        records: dict[str, Any] = {}
        for zspec in ZERO_SHOT_TASKS:
            rec = score_task(model, tokenizer, zspec, add_special, device, form_out,
                             max_items_per_file=args.max_items_per_file,
                             batch_size=args.batch_size,
                             non_causal_batch_size=args.non_causal_batch_size,
                             save_predictions=not args.no_predictions)
            records[zspec["column"]] = rec
            print(json.dumps({"event": "column_done", "endpoint": endpoint, "form": form, "column": zspec["column"], "score": rec["score"], "elapsed_sec": rec["elapsed_sec"]}), flush=True)
        if not args.skip_reading:
            rrec = score_reading(model, tokenizer, add_special, device, form_out / "Reading", max_rows=args.max_reading_rows)
            records["Reading"] = rrec
            print(json.dumps({"event": "column_done", "endpoint": endpoint, "form": form, "column": "Reading", "score": rrec["score"], "elapsed_sec": rrec["elapsed_sec"]}), flush=True)
        scores: dict[str, float | None] = {}
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
            scores[col] = float(records[col]["score"])
        gp = mean([float(records["GlobalPIQA_parallel"]["score"]), float(records["GlobalPIQA_nonparallel"]["score"])])
        scores["GlobalPIQA"] = float(gp)
        if "Reading" in records:
            scores["Reading"] = float(records["Reading"]["score"])
        else:
            scores["Reading"] = None
        cheap_vals = [scores[c] for c in CHEAP7_COLUMNS if scores.get(c) is not None]
        c7 = float(mean(cheap_vals)) if len(cheap_vals) == len(CHEAP7_COLUMNS) else None
        form_summary = {"input_form": form, "add_special_tokens": add_special, "scores": scores, "cheap7": c7, "records": records}
        results_by_form[form] = form_summary
        (form_out / "form_summary.json").write_text(json.dumps(form_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"endpoint": endpoint, "model_identity": ident, "known_official_scores": spec.get("known_official_scores", {}), "forms": results_by_form}


def build_overall_summary(all_results: dict[str, Any], out_dir: pathlib.Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    contrasts: dict[str, Any] = {}
    for endpoint, edata in all_results.items():
        known = edata.get("known_official_scores", {})
        for form, fdata in edata["forms"].items():
            scores = fdata["scores"]
            row = {"endpoint": endpoint, "input_form": form, "cheap7": fdata["cheap7"]}
            for c in CHEAP7_COLUMNS:
                row[c] = scores.get(c)
                if known.get(c) is not None:
                    row[f"{c}_minus_known_official"] = None if scores.get(c) is None else float(scores[c] - known[c])
            if known.get("cheap7") is not None and fdata.get("cheap7") is not None:
                row["cheap7_minus_known_official"] = float(fdata["cheap7"] - known["cheap7"])
            rows.append(row)
        if "with_special" in edata["forms"] and "no_special" in edata["forms"]:
            ws = edata["forms"]["with_special"]["scores"]
            ns = edata["forms"]["no_special"]["scores"]
            delta = {c: (None if ns.get(c) is None or ws.get(c) is None else float(ns[c] - ws[c])) for c in CHEAP7_COLUMNS}
            delta["cheap7"] = float(edata["forms"]["no_special"]["cheap7"] - edata["forms"]["with_special"]["cheap7"]) if edata["forms"]["no_special"].get("cheap7") is not None and edata["forms"]["with_special"].get("cheap7") is not None else None
            contrasts[f"{endpoint}_no_special_minus_with_special"] = delta
    write_csv(out_dir / "score_rows.csv", rows)
    summary = {"status": "NO_BOUNDARY_CHEAP7_DIAGNOSTIC_DONE", "created_utc": now(),
               "scientific_question": "Does removing official boundary tokens improve or harm the v4/coherent86 cheap7 surface, especially Supplement and EWoK?",
               "note": "Diagnostic only; official BabyLM evaluation always uses boundary tokens.",
               "rows": rows, "contrasts": contrasts, "results": all_results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research no-boundary cheap7 diagnostic", "", "This is not a submission path.  It evaluates the same local masked-LM candidate-ranking tasks with candidate inputs tokenized either as the official pipeline does (`with_special`) or with boundary tokens stripped (`no_special`).", "", "## Scores", "", "| endpoint | input form | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(x: Any) -> str:
        return "" if x is None else f"{float(x):.4f}"
    for r in rows:
        lines.append(f"| {r['endpoint']} | {r['input_form']} | {fmt(r.get('BLiMP'))} | {fmt(r.get('Supplement'))} | {fmt(r.get('EWoK'))} | {fmt(r.get('Entity'))} | {fmt(r.get('COMPS'))} | {fmt(r.get('GlobalPIQA'))} | {fmt(r.get('Reading'))} | {fmt(r.get('cheap7'))} |")
    lines += ["", "## No-boundary minus with-boundary", "", "| endpoint | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | cheap7 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, d in contrasts.items():
        endpoint = name.replace("_no_special_minus_with_special", "")
        lines.append(f"| {endpoint} | {fmt(d.get('BLiMP'))} | {fmt(d.get('Supplement'))} | {fmt(d.get('EWoK'))} | {fmt(d.get('Entity'))} | {fmt(d.get('COMPS'))} | {fmt(d.get('GlobalPIQA'))} | {fmt(d.get('Reading'))} | {fmt(d.get('cheap7'))} |")
    lines += ["", "Full JSON: `" + rel(out_dir / "summary.json") + "`"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--endpoints", nargs="*", default=["chck82_slow_scale1p75", "coherent86_alpha075"], choices=sorted(ENDPOINTS))
    ap.add_argument("--input-forms", nargs="*", default=["with_special", "no_special"], choices=["with_special", "no_special"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--non-causal-batch-size", type=int, default=64)
    ap.add_argument("--max-items-per-file", type=int, default=0, help="Debug only: limit JSONL rows per subtask file")
    ap.add_argument("--max-reading-rows", type=int, default=0, help="Debug only: limit Reading rows")
    ap.add_argument("--skip-reading", action="store_true")
    ap.add_argument("--no-predictions", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_cache(out_dir)
    torch.manual_seed(0)
    all_results = {}
    for endpoint in args.endpoints:
        print(json.dumps({"event": "endpoint_start", "endpoint": endpoint, "input_forms": args.input_forms, "utc": now()}), flush=True)
        all_results[endpoint] = evaluate_endpoint(endpoint, list(args.input_forms), args)
    summary = build_overall_summary(all_results, out_dir)
    print(json.dumps({"status": summary["status"], "summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"), "contrasts": summary["contrasts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
