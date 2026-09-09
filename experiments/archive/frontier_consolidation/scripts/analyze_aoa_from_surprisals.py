#!/usr/bin/env python3
"""research acquisition-timing localization from existing AoA surprisal files.

This CPU script reproduces the unthresholded BabyLM AoA curve fitting used in
research, but extends it to the research/017 density models.  It reads existing
surprisal.json files only.  It is analysis of trained checkpoint ladders, not a
training signal and not a corpus-selection procedure.
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
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
STUDY = _public_path('experiments/archive/frontier_consolidation')
USER_ROOT = _public_path('.')
STRICT_ROOT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
sys.path.insert(0, str(_public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')))
from evaluation_pipeline.utils import AoAEvaluator, sigmoid_function  # noqa: E402

OUT_DIR_DEFAULT = _public_path('experiments/archive/frontier_consolidation/data/aoa_mechanism_analysis')
CDI_HUMAN = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')

MODEL_SOURCES: Dict[str, Dict[str, str]] = {
    "clean_qwen_seed43022": {
        "surprisal": "experiments/archive/compact_experience/data/full_eval/aoa_outputs/qwen_clean_aligned/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json",
        "tokenizer": "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model",
        "role": "inherited clean-Qwen aligned baseline, official AoA thresholded to 0",
    },
    "clean_qwen_seed43122": {
        "surprisal": "experiments/archive/compact_experience/data/full_eval/aoa_outputs/qwen_clean_aligned_seed43122/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json",
        "tokenizer": "experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43122/hf_model",
        "role": "inherited clean-Qwen aligned second seed, official AoA thresholded to 0",
    },
    "density_compact_view_core": {
        "surprisal": "experiments/archive/frontier_consolidation/data/density_full_eval/aoa_outputs/compact_view_core/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json",
        "tokenizer": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_core_neutral_16k_seed43022/hf_model",
        "role": "compact semantic views on shared FineWeb core; full official AoA = -12.687",
    },
    "density_near_repeat": {
        "surprisal": "experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_near_repeat/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json",
        "tokenizer": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_repeat_near_core_16k_seed43022/hf_model",
        "role": "FineWeb near-source repetition on clean-Qwen row-holdout base",
    },
    "density_near_view": {
        "surprisal": "experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_near_view/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json",
        "tokenizer": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_near_view_core_16k_seed43022/hf_model",
        "role": "near-length generated views against near-repeat",
    },
    "density_compact_repeat_core": {
        "surprisal": "experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_compact_repeat_core/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json",
        "tokenizer": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_repeat_compact_core_neutral_16k_seed43022/hf_model",
        "role": "compact-core source repetition matched to compact-view core",
    },
    "density_compact_view_reinvest": {
        "surprisal": "experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_compact_view_reinvest/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json",
        "tokenizer": "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model",
        "role": "compact semantic views with saved words reinvested into additional source-view packets",
    },
}

BROAD_SCORES = {
    "clean_qwen_seed43022": {"Overall": 41.34429066479573, "AoA": 0.0, "SuperGLUE": 70.30861598316157, "NLP_average": 52.04837325789551},
    "density_compact_view_core": {"Overall": 39.987651867522125, "AoA": -12.687286024554457, "SuperGLUE": 68.90115283225359, "NLP_average": 52.04659326175051},
}


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(_public_path('.')))
    except Exception:
        return str(path)


def load_child_aoas() -> Dict[str, float]:
    evaluator = AoAEvaluator(CDI_HUMAN)
    out: Dict[str, float] = {}
    for idx, row in evaluator.cdi_data.iterrows():
        val = evaluator.compute_child_aoa(idx)
        if val is not None:
            out[str(row["word"]).lower()] = float(val)
    return out


def subword_length(tokenizer: Any, word: str) -> int:
    prefix_ids = tokenizer("The", add_special_tokens=False)["input_ids"]
    ids = tokenizer("The " + word, add_special_tokens=False)["input_ids"]
    return max(1, len(ids) - len(prefix_ids))


def step_value(step_name: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", str(step_name), re.IGNORECASE)
    if not m:
        return None
    number = float(m.group(1))
    unit = m.group(2).upper() if m.group(2) else ""
    return number * {"K": 1000, "M": 1000000, "B": 1000000000}.get(unit, 1)


def compute_model_aoa(
    surprisals: List[float],
    training_steps: List[float],
    vocab_size: int,
    n_subword_tokens: int,
    threshold_percentile: float = 0.5,
) -> Optional[float]:
    steps = np.array(training_steps, dtype=float)
    surp = np.array(surprisals, dtype=float)
    valid_mask = ~np.isnan(surp)
    if not np.any(valid_mask):
        return None
    valid_steps = steps[valid_mask]
    valid_surprisals = surp[valid_mask]
    random_chance_surprisal = n_subword_tokens * np.log(vocab_size)
    min_surprisal = float(np.min(valid_surprisals))
    threshold_surprisal = random_chance_surprisal - threshold_percentile * (random_chance_surprisal - min_surprisal)
    try:
        neg = -valid_surprisals
        log_steps = np.log10(valid_steps + 1)
        rng = float(np.max(neg) - np.min(neg))
        initial_guess = [rng, 1.0, float(np.mean(log_steps)), float(np.min(neg))]
        lower = [0.0, 0.0, float(np.min(log_steps) - 1), float(np.min(neg) - 2 * rng - 1)]
        upper = [10 * rng + 1, 100.0, float(np.max(log_steps) + 1), float(np.max(neg) + 1)]
        popt, _ = curve_fit(sigmoid_function, log_steps, neg, p0=initial_guess, bounds=(lower, upper), maxfev=20000)
        a, b, c, d = [float(x) for x in popt]
        neg_threshold = -threshold_surprisal
        if b <= 1e-6 or a <= 1e-6:
            return None
        if neg_threshold <= d or neg_threshold >= a + d:
            return None
        log_aoa_step = c - np.log((a / (neg_threshold - d)) - 1) / b
        aoa_step = 10 ** log_aoa_step - 1
        if aoa_step < valid_steps[0] or aoa_step > valid_steps[-1]:
            return None
        return float(log_aoa_step)
    except Exception:
        return None


def basic_stats(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": sum(xs) / len(xs), "median": q(0.5), "p95": q(0.95), "max": xs[-1]}


def age_bin(month: float) -> str:
    if month <= 20.0:
        return "early_le20"
    if month <= 25.0:
        return "middle_20to25"
    return "late_gt25"


def fit_model(name: str, child_aoa: Dict[str, float], keep_per_word: bool = True) -> Dict[str, Any]:
    src = MODEL_SOURCES[name]
    surprisal_path = USER_ROOT / src["surprisal"]
    tokenizer_path = USER_ROOT / src["tokenizer"]
    if not surprisal_path.exists():
        return {"status": "missing_surprisal", "name": name, "path": rel(surprisal_path), "role": src["role"]}
    if not tokenizer_path.exists():
        return {"status": "missing_tokenizer", "name": name, "path": rel(surprisal_path), "tokenizer_path": rel(tokenizer_path), "role": src["role"]}
    d = json.loads(surprisal_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    word_data: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in d.get("results", []):
        w = str(r.get("target_word", "")).lower()
        if w not in child_aoa:
            continue
        sv = step_value(str(r.get("step")))
        if sv is None:
            continue
        try:
            word_data[w][sv].append(float(r["surprisal"]))
        except Exception:
            pass
    model_aoas: List[float] = []
    child_vals: List[float] = []
    valid_words: List[str] = []
    invalid_words: List[str] = []
    per_word: Dict[str, Any] = {}
    for w in sorted(word_data):
        steps = sorted(word_data[w])
        mean_surps = [float(np.mean(word_data[w][s])) for s in steps]
        sw = subword_length(tokenizer, w)
        ma = compute_model_aoa(mean_surps, steps, tokenizer.vocab_size, sw)
        if ma is None:
            invalid_words.append(w)
            continue
        valid_words.append(w)
        model_aoas.append(ma)
        child_vals.append(child_aoa[w])
        per_word[w] = {
            "model_aoa_log10_words": ma,
            "child_aoa_month": child_aoa[w],
            "subword_len": sw,
            "mean_surprisal_by_step": {str(int(s)): float(np.mean(word_data[w][s])) for s in steps},
        }
    if len(valid_words) >= 3:
        pr, pp = pearsonr(model_aoas, child_vals)
        sr, sp = spearmanr(model_aoas, child_vals)
    else:
        pr = pp = sr = sp = float("nan")
    official_thresholded = 0.0 if (not math.isfinite(pp) or pp > 0.1) else float(pr)
    bins: Dict[str, List[float]] = defaultdict(list)
    for ma, ca in zip(model_aoas, child_vals):
        bins[age_bin(ca)].append(ma)
    bin_stats = {k: basic_stats(v) for k, v in bins.items()}
    early_mean = bin_stats.get("early_le20", {}).get("mean")
    late_mean = bin_stats.get("late_gt25", {}).get("mean")
    out: Dict[str, Any] = {
        "status": "fit_done",
        "name": name,
        "role": src["role"],
        "surprisal_path": rel(surprisal_path),
        "tokenizer_path": rel(tokenizer_path),
        "rows": len(d.get("results", [])),
        "candidate_words_before_fit": len(word_data),
        "n_valid_words": len(valid_words),
        "n_invalid_words": len(invalid_words),
        "invalid_words_sample": invalid_words[:60],
        "pearson_model_aoa_vs_child_aoa": float(pr) if math.isfinite(pr) else None,
        "pearson_p_value": float(pp) if math.isfinite(pp) else None,
        "spearman_model_aoa_vs_child_aoa": float(sr) if math.isfinite(sr) else None,
        "spearman_p_value": float(sp) if math.isfinite(sp) else None,
        "official_thresholded_curve_fitness_raw": official_thresholded,
        "official_thresholded_leaderboard_score": 100.0 * official_thresholded,
        "model_aoa_log10_stats": basic_stats(model_aoas),
        "model_aoa_by_child_age_bin": bin_stats,
        "late_minus_early_model_aoa_log10": (float(late_mean - early_mean) if early_mean is not None and late_mean is not None else None),
        "valid_words": valid_words,
    }
    if name in BROAD_SCORES:
        out["broad_scores"] = BROAD_SCORES[name]
    if keep_per_word:
        out["per_word"] = per_word
    return out


def compare_to_anchor(anchor: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    if anchor.get("status") != "fit_done" or other.get("status") != "fit_done":
        return {"status": "not_available", "anchor_status": anchor.get("status"), "other_status": other.get("status")}
    apw = anchor["per_word"]; opw = other["per_word"]
    common = sorted(set(apw) & set(opw))
    rows: List[Tuple[str, float, float, float, float]] = []
    for w in common:
        ca = float(apw[w]["child_aoa_month"])
        a = float(apw[w]["model_aoa_log10_words"])
        o = float(opw[w]["model_aoa_log10_words"])
        rows.append((w, ca, a, o, o - a))
    child = [r[1] for r in rows]
    delta = [r[4] for r in rows]
    if len(rows) >= 3:
        pr, pp = pearsonr(child, delta)
        sr, sp = spearmanr(child, delta)
    else:
        pr = pp = sr = sp = float("nan")
    by_bin: Dict[str, List[float]] = defaultdict(list)
    for _, ca, _, _, de in rows:
        by_bin[age_bin(ca)].append(de)
    def pack(row: Tuple[str, float, float, float, float]) -> Dict[str, Any]:
        w, ca, a, o, de = row
        return {"word": w, "child_aoa_month": ca, "anchor_model_aoa": a, "other_model_aoa": o, "delta_other_minus_anchor": de}
    return {
        "status": "compared",
        "anchor": anchor["name"],
        "other": other["name"],
        "common_n": len(rows),
        "mean_delta_log10_words_other_minus_anchor": float(np.mean(delta)) if delta else None,
        "delta_stats": basic_stats(delta),
        "pearson_child_aoa_vs_delta": float(pr) if math.isfinite(pr) else None,
        "pearson_child_aoa_vs_delta_p": float(pp) if math.isfinite(pp) else None,
        "spearman_child_aoa_vs_delta": float(sr) if math.isfinite(sr) else None,
        "spearman_child_aoa_vs_delta_p": float(sp) if math.isfinite(sp) else None,
        "delta_by_child_age_bin": {k: basic_stats(v) for k, v in by_bin.items()},
        "largest_advances_other_vs_anchor": [pack(r) for r in sorted(rows, key=lambda x: x[4])[:30]],
        "largest_delays_other_vs_anchor": [pack(r) for r in sorted(rows, key=lambda x: x[4], reverse=True)[:30]],
    }


def make_note(payload: Dict[str, Any]) -> str:
    lines = ["# research AoA mechanism analysis from existing surprisals", ""]
    lines.append("This analysis recomputes unthresholded fitted model-AoA from existing official AoA surprisal files. It is passive analysis of already-trained checkpoint ladders; it does not use AoA words to build or train any model.")
    lines.append("")
    lines.append("## Available model fits")
    for name, rec in payload["models"].items():
        if rec.get("status") != "fit_done":
            lines.append(f"- {name}: {rec.get('status')} ({rec.get('path') or rec.get('surprisal_path')})")
            continue
        r = rec.get("pearson_model_aoa_vs_child_aoa")
        p = rec.get("pearson_p_value")
        lb = rec.get("official_thresholded_leaderboard_score")
        late_early = rec.get("late_minus_early_model_aoa_log10")
        lines.append(f"- {name}: n={rec['n_valid_words']}/{rec['candidate_words_before_fit']}, r={r:.4f}, p={p:.4g}, leaderboard_AoA={lb:.3f}, late-minus-early model AoA={late_early}")
    lines.append("")
    if "density_compact_view_core_minus_clean_qwen_seed43022" in payload["anchor_comparisons"]:
        cmp = payload["anchor_comparisons"]["density_compact_view_core_minus_clean_qwen_seed43022"]
        if cmp.get("status") == "compared":
            lines.append("## Compact-view core relative to inherited clean-Qwen")
            lines.append(f"Common fitted words: {cmp['common_n']}. Mean delta in fitted log10 word exposure (compact minus clean): {cmp['mean_delta_log10_words_other_minus_anchor']:.4f}.")
            lines.append(f"Correlation of child AoA with compact-minus-clean delta: r={cmp['pearson_child_aoa_vs_delta']:.4f}, p={cmp['pearson_child_aoa_vs_delta_p']:.4g}. Negative here means compact views advance human-late words relative to human-early words more than clean-Qwen.")
            for b, st in cmp.get("delta_by_child_age_bin", {}).items():
                lines.append(f"- {b}: n={st.get('n')}, mean_delta={st.get('mean')}")
            adv = cmp.get("largest_advances_other_vs_anchor", [])[:8]
            delay = cmp.get("largest_delays_other_vs_anchor", [])[:8]
            lines.append("Largest advances of compact vs clean: " + ", ".join(f"{x['word']}({x['delta_other_minus_anchor']:.3f})" for x in adv))
            lines.append("Largest delays of compact vs clean: " + ", ".join(f"{x['word']}({x['delta_other_minus_anchor']:.3f})" for x in delay))
            lines.append("")
    lines.append("Machine-readable JSON: `" + payload["out_json"] + "`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--models", nargs="+", default=list(MODEL_SOURCES), choices=sorted(MODEL_SOURCES))
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    child = load_child_aoas()
    models: Dict[str, Any] = {}
    for name in args.models:
        print(json.dumps({"event": "fit_start", "model": name}), flush=True)
        models[name] = fit_model(name, child, keep_per_word=True)
        print(json.dumps({"event": "fit_done", "model": name, "status": models[name].get("status"), "r": models[name].get("pearson_model_aoa_vs_child_aoa"), "p": models[name].get("pearson_p_value")}), flush=True)
    anchor_comparisons: Dict[str, Any] = {}
    anchor_name = "clean_qwen_seed43022"
    if anchor_name in models:
        for name, rec in models.items():
            if name == anchor_name:
                continue
            anchor_comparisons[f"{name}_minus_{anchor_name}"] = compare_to_anchor(models[anchor_name], rec)
    # Drop per-word mean-surprisal traces from JSON? Keep fitted per-word AoAs but not per-step traces in compact payload.
    slim_models = {}
    for name, rec in models.items():
        rec2 = dict(rec)
        if rec2.get("status") == "fit_done" and "per_word" in rec2:
            rec2["per_word"] = {w: {k: v for k, v in d.items() if k != "mean_surprisal_by_step"} for w, d in rec2["per_word"].items()}
        slim_models[name] = rec2
    payload = {
        "status": "AOA_MECHANISM_ANALYSIS",
        "purpose": "Localize fitted acquisition-timing changes from existing AoA surprisal files before any new 100M training.",
        "cdi_human": rel(CDI_HUMAN),
        "models_requested": args.models,
        "models": slim_models,
        "anchor_comparisons": anchor_comparisons,
        "out_json": rel(out_dir / "aoa_mechanism_analysis.json"),
        "out_note": rel(out_dir / "aoa_mechanism_analysis.md"),
    }
    # Also slim comparison examples are already compact.
    (out_dir / "aoa_mechanism_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "aoa_mechanism_analysis.md").write_text(make_note(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": payload["out_json"], "out_note": payload["out_note"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
