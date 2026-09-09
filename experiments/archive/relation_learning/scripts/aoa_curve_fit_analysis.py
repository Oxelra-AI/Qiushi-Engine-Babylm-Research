#!/usr/bin/env python3
"""research: analyze the official AoA curve-fit zero for coherent86 alpha0.75.

This script reads the already-computed official-compatible AoA surprisal ladder
and the official AoAEvaluator implementation. It does not alter scoring. It
reports why the final AoA score is exactly zero: word eligibility, child and
model fit outcomes, the unclipped correlation/p-value, and the broader local
score landscape across previous strict-small measurements.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = _public_path('.')
UTILS_DIR = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline')
sys.path.insert(0, str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')))
from evaluation_pipeline.utils import AoAEvaluator, sigmoid_function  # noqa: E402

research = _public_path('experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0')
SUMMARY_PATH = _public_path('experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0/alpha075_aoa_minctx0_summary.json')
SURPRISAL_PATH = _public_path('experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0/collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/surprisal.json')
SCORE_PATH = _public_path('experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0/collate_fast/results/hf_model/main/zero_shot/mlm/AoA_word/aoa_score.json')
CDI_HUMAN = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/aoa_curve_fit_analysis')


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def extract_step_number(step_name: str | int | float) -> float | None:
    return AoAEvaluator.extract_step_number(AoAEvaluator.__new__(AoAEvaluator), step_name)  # avoid constructing twice


def subword_len_fn(tokenizer):
    prefix_ids = tokenizer("The", add_special_tokens=False)["input_ids"]

    def subword_len(word: str) -> int:
        ids = tokenizer("The " + word, add_special_tokens=False)["input_ids"]
        return max(1, len(ids) - len(prefix_ids))

    return subword_len


def child_aoa_status(evaluator: AoAEvaluator, word_idx: int) -> tuple[float | None, str]:
    # Use the official method for the value, then add transparent reason bins.
    val = evaluator.compute_child_aoa(word_idx)
    if val is not None:
        return float(val), "ok"
    props = np.asarray(evaluator.age_data[word_idx], dtype=float)
    valid = ~np.isnan(props)
    if not np.any(valid):
        return None, "child_no_age_values"
    vals = props[valid]
    if len(vals) < 3:
        return None, "child_lt3_age_values"
    if np.nanmax(vals) < 0.5:
        return None, "child_never_reaches_50pct"
    if np.nanmin(vals) > 0.5:
        return None, "child_above_50pct_from_first_age"
    return None, "child_sigmoid_or_bounds_failed"


def model_aoa_debug(surprisals_in: list[float], steps_in: list[float], vocab_size: int, n_subword_tokens: int) -> dict[str, Any]:
    if len(surprisals_in) != len(steps_in):
        return {"status": "length_mismatch"}
    if len(surprisals_in) < 3:
        return {"status": "lt3_points"}
    steps = np.asarray(steps_in, dtype=float)
    surprisals = np.asarray(surprisals_in, dtype=float)
    valid_mask = ~np.isnan(surprisals)
    if not np.any(valid_mask):
        return {"status": "all_nan"}
    valid_steps = steps[valid_mask]
    valid_surprisals = surprisals[valid_mask]
    random_chance = float(n_subword_tokens * math.log(vocab_size))
    min_surprisal = float(np.min(valid_surprisals))
    threshold = float(random_chance - 0.5 * (random_chance - min_surprisal))
    neg_surprisals = -valid_surprisals
    log_steps = np.log10(valid_steps + 1)
    rng = float(np.max(neg_surprisals) - np.min(neg_surprisals))
    if not math.isfinite(rng):
        return {"status": "nonfinite_range", "random_chance": random_chance, "min_surprisal": min_surprisal, "threshold": threshold}
    initial_guess = [rng, 1.0, float(np.mean(log_steps)), float(np.min(neg_surprisals))]
    lower = [0.0, 0.0, float(np.min(log_steps) - 1), float(np.min(neg_surprisals) - 2 * rng - 1)]
    upper = [10 * rng + 1, 100.0, float(np.max(log_steps) + 1), float(np.max(neg_surprisals) + 1)]
    try:
        popt, _ = curve_fit(sigmoid_function, log_steps, neg_surprisals, p0=initial_guess, bounds=(lower, upper), maxfev=20000)
    except Exception as e:
        return {"status": "model_fit_exception", "exception": repr(e), "random_chance": random_chance, "min_surprisal": min_surprisal, "threshold": threshold, "range_neg": rng}
    a, b, c, d = [float(x) for x in popt]
    neg_threshold = -threshold
    out: dict[str, Any] = {
        "a": a, "b": b, "c": c, "d": d,
        "random_chance": random_chance,
        "min_surprisal": min_surprisal,
        "threshold": threshold,
        "neg_threshold": float(neg_threshold),
        "first_step": float(valid_steps[0]),
        "last_step": float(valid_steps[-1]),
        "first_surprisal": float(valid_surprisals[0]),
        "last_surprisal": float(valid_surprisals[-1]),
        "mean_improvement_first_minus_last": float(valid_surprisals[0] - valid_surprisals[-1]),
        "best_step": float(valid_steps[int(np.argmin(valid_surprisals))]),
        "best_surprisal": min_surprisal,
        "range_neg": rng,
    }
    if b <= 1e-6 or a <= 1e-6:
        out["status"] = "flat_or_zero_amplitude"
        return out
    if neg_threshold <= d:
        out["status"] = "threshold_below_lower_asymptote"
        return out
    if neg_threshold >= a + d:
        out["status"] = "threshold_above_upper_asymptote"
        return out
    try:
        log_aoa_step = float(c - math.log((a / (neg_threshold - d)) - 1) / b)
        aoa_step = float(10**log_aoa_step - 1)
    except Exception as e:
        out["status"] = "solve_exception"
        out["exception"] = repr(e)
        return out
    out["log_aoa_step"] = log_aoa_step
    out["aoa_step"] = aoa_step
    if aoa_step < valid_steps[0]:
        out["status"] = "aoa_before_first_checkpoint"
        return out
    if aoa_step > valid_steps[-1]:
        out["status"] = "aoa_after_last_checkpoint"
        return out
    out["status"] = "ok"
    return out


def summarize_aoa_score_files(limit_rows: int = 200) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(ROOT.glob("Sessions/**/aoa_score.json")):
        try:
            obj = read_json(p)
        except Exception as e:
            rows.append({"path": rel(p), "read_error": repr(e)})
            continue
        aoa = obj.get("aoa")
        cfr = obj.get("curve_fitness_record") if isinstance(obj, dict) else None
        if isinstance(cfr, dict) and aoa is None:
            aoa = cfr.get("curve_fitness")
        row = {"path": rel(p), "aoa": aoa}
        if isinstance(cfr, dict):
            for k in ["curve_fitness", "p_value", "n_words", "mean_monthly_score"]:
                if k in cfr:
                    row[k] = cfr[k]
        rows.append(row)
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get("aoa")))
        except Exception:
            pass
    nonzero = [v for v in vals if math.isfinite(v) and abs(v) > 1e-12]
    positive = [v for v in vals if math.isfinite(v) and v > 0]
    summary = {
        "n_files": len(rows),
        "n_numeric": len(vals),
        "n_nonzero": len(nonzero),
        "n_positive": len(positive),
        "max_aoa": max(vals) if vals else None,
        "min_aoa": min(vals) if vals else None,
        "examples_nonzero": [r for r in rows if isinstance(r.get("aoa"), (int, float)) and abs(float(r.get("aoa"))) > 1e-12][:limit_rows],
    }
    return rows, summary


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(SUMMARY_PATH)
    score = read_json(SCORE_PATH)
    surprisal = read_json(SURPRISAL_PATH)
    results = surprisal.get("results", [])
    tokenizer_path = ROOT / summary.get("score_tokenizer_path", summary.get("model_root", ""))
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True, local_files_only=True)
    except Exception:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    evaluator = AoAEvaluator(CDI_HUMAN)
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    subword_len = subword_len_fn(tokenizer)

    grouped: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"steps": [], "surprisals": [], "contexts": [], "word_counts": []})
    step_counts = Counter()
    for r in results:
        step = r.get("step")
        step_val = evaluator.extract_step_number(step)
        try:
            surp = float(r.get("surprisal"))
        except Exception:
            surp = float("nan")
        w = str(r.get("target_word"))
        grouped[w]["steps"].append(step_val)
        grouped[w]["surprisals"].append(surp)
        grouped[w]["contexts"].append(str(r.get("context", "")))
        grouped[w]["word_counts"].append(r.get("word_count"))
        step_counts[str(step)] += 1

    cdi_words = list(map(str, evaluator.cdi_data["word"].tolist()))
    cdi_index = {w: i for i, w in enumerate(cdi_words)}
    word_rows: list[dict[str, Any]] = []
    valid_model = []
    valid_child = []
    valid_words = []
    for word, data in sorted(grouped.items()):
        steps_raw = np.asarray(data["steps"], dtype=float)
        surp_raw = np.asarray(data["surprisals"], dtype=float)
        finite_mask = np.isfinite(steps_raw) & np.isfinite(surp_raw)
        row: dict[str, Any] = {
            "word": word,
            "n_rows": int(len(surp_raw)),
            "n_finite_rows": int(finite_mask.sum()),
            "n_unique_steps": int(len(np.unique(steps_raw[finite_mask])) if finite_mask.any() else 0),
            "in_cdi_human": int(word in cdi_index),
            "mean_context_words": float(np.mean([len(str(c).split()) for c in data["contexts"]])) if data["contexts"] else None,
            "min_context_words": int(min([len(str(c).split()) for c in data["contexts"]])) if data["contexts"] else None,
            "max_context_words": int(max([len(str(c).split()) for c in data["contexts"]])) if data["contexts"] else None,
        }
        if word not in cdi_index:
            row.update({"child_status": "not_in_cdi_human", "model_status": "not_attempted_no_child"})
            word_rows.append(row)
            continue
        idx = cdi_index[word]
        child_val, child_status = child_aoa_status(evaluator, idx)
        row["child_status"] = child_status
        row["child_aoa"] = child_val
        if child_val is None:
            row["model_status"] = "not_attempted_no_child_aoa"
            word_rows.append(row)
            continue
        if not finite_mask.any():
            row["model_status"] = "no_finite_surprisal"
            word_rows.append(row)
            continue
        uniq_steps = np.unique(steps_raw[finite_mask])
        mean_surps = [float(surp_raw[finite_mask & (steps_raw == s)].mean()) for s in uniq_steps]
        swl = subword_len(word)
        m = model_aoa_debug(mean_surps, [float(x) for x in uniq_steps], vocab_size, swl)
        row.update({f"model_{k}": v for k, v in m.items() if k not in {"a", "b", "c", "d"}})
        row["model_status"] = m.get("status")
        row["subword_len"] = int(swl)
        row["curve_first_1M"] = mean_surps[0] if mean_surps else None
        row["curve_last_100M"] = mean_surps[-1] if mean_surps else None
        row["curve_best"] = min(mean_surps) if mean_surps else None
        row["curve_first_minus_last"] = float(mean_surps[0] - mean_surps[-1]) if mean_surps else None
        if m.get("status") == "ok":
            model_aoa = float(m["log_aoa_step"])
            row["model_aoa"] = model_aoa
            valid_model.append(model_aoa)
            valid_child.append(float(child_val))
            valid_words.append(word)
        word_rows.append(row)

    if len(valid_model) >= 3:
        pear = pearsonr(valid_model, valid_child)
        spear = spearmanr(valid_model, valid_child)
        correlation_summary = {
            "n_words": len(valid_model),
            "pearson_r": float(pear.statistic),
            "pearson_p": float(pear.pvalue),
            "spearman_r": float(spear.statistic),
            "spearman_p": float(spear.pvalue),
            "official_clipped_curve_fitness": 0.0 if pear.pvalue > 0.1 else float(pear.statistic),
        }
    else:
        correlation_summary = {"n_words": len(valid_model), "official_clipped_curve_fitness": 0.0}

    status_counts = Counter(str(r.get("model_status")) for r in word_rows)
    child_counts = Counter(str(r.get("child_status")) for r in word_rows)
    by_subword: list[dict[str, Any]] = []
    for swl, rows in sorted(defaultdict(list, {k: [r for r in word_rows if r.get("subword_len") == k] for k in sorted(set(r.get("subword_len") for r in word_rows if r.get("subword_len") is not None))}).items()):
        by_subword.append({
            "subword_len": int(swl),
            "n_words_after_child": len(rows),
            "model_ok": sum(1 for r in rows if r.get("model_status") == "ok"),
            "before_first": sum(1 for r in rows if r.get("model_status") == "aoa_before_first_checkpoint"),
            "after_last": sum(1 for r in rows if r.get("model_status") == "aoa_after_last_checkpoint"),
            "threshold_above_upper": sum(1 for r in rows if r.get("model_status") == "threshold_above_upper_asymptote"),
            "threshold_below_lower": sum(1 for r in rows if r.get("model_status") == "threshold_below_lower_asymptote"),
        })

    aoa_rows, aoa_landscape = summarize_aoa_score_files()
    write_csv(out_dir / "word_curve_fit_records.csv", word_rows)
    with (out_dir / "word_curve_fit_records.jsonl").open("w", encoding="utf-8") as f:
        for r in word_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_csv(out_dir / "local_aoa_score_files.csv", aoa_rows)
    write_csv(out_dir / "model_status_by_subword_len.csv", by_subword)

    official_step_means = summary.get("step_mean_surprisal", {})
    out = {
        "status": "AOA_CURVE_FIT_ANALYSIS_DONE",
        "created_utc": now(),
        "inputs": {
            "summary_path": rel(SUMMARY_PATH),
            "score_path": rel(SCORE_PATH),
            "surprisal_path": rel(SURPRISAL_PATH),
            "cdi_human": rel(CDI_HUMAN),
            "tokenizer_path": rel(tokenizer_path),
            "official_utils": rel(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/utils.py')),
        },
        "official_score_object": score,
        "surprisal_rows": len(results),
        "n_target_words_in_surprisal": len(grouped),
        "step_counts_unique": sorted(set(step_counts.values())),
        "step_mean_surprisal_first_last": {
            "chck_1M": official_step_means.get("chck_1M"),
            "chck_100M": official_step_means.get("chck_100M"),
        },
        "child_status_counts": dict(child_counts),
        "model_status_counts": dict(status_counts),
        "correlation_summary": correlation_summary,
        "valid_words_head": valid_words[:50],
        "valid_words_tail": valid_words[-50:],
        "subword_len_summary": by_subword,
        "local_aoa_score_landscape": aoa_landscape,
        "outputs": {
            "word_csv": rel(out_dir / "word_curve_fit_records.csv"),
            "word_jsonl": rel(out_dir / "word_curve_fit_records.jsonl"),
            "local_scores_csv": rel(out_dir / "local_aoa_score_files.csv"),
            "subword_csv": rel(out_dir / "model_status_by_subword_len.csv"),
            "summary_json": rel(out_dir / "summary.json"),
            "summary_md": rel(out_dir / "summary.md"),
        },
        "reading": "The official AoA column uses Pearson correlation between fitted model log-step AoA and child 50pct AoA. If the p-value is above 0.1, the score object stores curve_fitness 0 even when many word curves fit; this is a measurement-side reading only and does not modify the submitted score.",
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research AoA curve-fit analysis",
        "",
        out["reading"],
        "",
        "## Official score object",
        "",
        f"`aoa_score.json`: `{json.dumps(score, ensure_ascii=False)}`",
        "",
        "## Core counts",
        "",
        f"- Surprisal rows: {len(results)}; target words: {len(grouped)}; rows per step values: {sorted(set(step_counts.values()))}.",
        f"- Official step mean surprisal: chck_1M {official_step_means.get('chck_1M'):.4f}, chck_100M {official_step_means.get('chck_100M'):.4f}.",
        f"- Child statuses: {dict(child_counts)}.",
        f"- Model statuses after child AoA: {dict(status_counts)}.",
        "",
        "## Unclipped correlation on words that fit both sides",
        "",
        f"- n = {correlation_summary.get('n_words')}",
        f"- Pearson r = {correlation_summary.get('pearson_r')}, p = {correlation_summary.get('pearson_p')}",
        f"- Spearman r = {correlation_summary.get('spearman_r')}, p = {correlation_summary.get('spearman_p')}",
        f"- Official stored value after p>0.1 rule = {correlation_summary.get('official_clipped_curve_fitness')}",
        "",
        "## Local score landscape",
        "",
        f"- AoA score files found: {aoa_landscape.get('n_files')}; numeric: {aoa_landscape.get('n_numeric')}; nonzero: {aoa_landscape.get('n_nonzero')}; positive: {aoa_landscape.get('n_positive')}; max: {aoa_landscape.get('max_aoa')}; min: {aoa_landscape.get('min_aoa')}.",
        "- Nonzero examples are written to the JSON and CSV outputs.",
        "",
        f"Full JSON: `{rel(out_dir / 'summary.json')}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "summary_json": rel(out_dir / "summary.json"), "summary_md": rel(out_dir / "summary.md"), "correlation_summary": correlation_summary, "model_status_counts": dict(status_counts), "local_aoa_score_landscape": aoa_landscape}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
