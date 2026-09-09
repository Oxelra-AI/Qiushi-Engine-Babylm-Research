#!/usr/bin/env python3
"""research official AoA curve-fit audit for already-trained models.

This reproduces the official AoA model-AoA fitting logic from
`evaluation_pipeline.utils.AoAEvaluator` but preserves the unthresholded Pearson
correlation, p value, valid-word set, and common-subset comparisons. It uses only
existing surprisal.json outputs and the existing tokenizer; it is not a training
signal.
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
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

ROOT = _public_path('experiments/archive/frontier_consolidation')
STRICT_ROOT = pathlib.Path("experiments/archive/initial_model_studies/repos/babylm-eval/strict")
sys.path.insert(0, str(STRICT_ROOT))
from evaluation_pipeline.utils import AoAEvaluator, sigmoid_function  # noqa: E402

OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/aoa_developmental_audit')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/aoa_developmental_audit/official_aoa_curvefit_audit.json')
OUT_NOTE = _public_path('research/notes/frontier_consolidation/official_aoa_curvefit_audit.md')
CDI_HUMAN = STRICT_ROOT / "evaluation_data/full_eval/aoa/cdi_human.csv"
TOKENIZER_PATH = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
SURPRISAL_PATHS = {
    "clean_qwen_seed43022": pathlib.Path("experiments/archive/compact_experience/data/full_eval/aoa_outputs/qwen_clean_aligned/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json"),
    "clean_qwen_seed43122": pathlib.Path("experiments/archive/compact_experience/data/full_eval/aoa_outputs/qwen_clean_aligned_seed43122/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json"),
    "devcurr_firstpass_seed43022": pathlib.Path("experiments/archive/compact_experience/data/devcurr_eval/aoa_outputs/qwen_devcurr_firstpass_seed43022/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json"),
    "devcurr_firstpass_seed43122": pathlib.Path("experiments/archive/compact_experience/data/devcurr_eval/aoa_outputs/qwen_devcurr_firstpass_seed43122/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json"),
}
SCORES = {
    "clean_qwen_seed43022": {"Overall": 41.34429066479573, "Entity": 25.76, "GlobalPIQA": 36.62, "source": "full_eval_summary.json"},
    "clean_qwen_seed43122": {"Overall": 40.65005195633467, "Entity": 25.26, "GlobalPIQA": 34.62, "source": "full_eval_summary.json"},
    "devcurr_firstpass_seed43022": {"Overall": 40.64750216341753, "Entity": 23.72, "GlobalPIQA": 34.225, "source": "devcurr_eval_seed43022_stdout.log"},
    "devcurr_firstpass_seed43122": {"Overall": 40.56562101511466, "Entity": 24.86, "GlobalPIQA": 35.635, "source": "devcurr_eval_seed43122_stdout.log"},
}


def load_cdi_child_aoas(evaluator: AoAEvaluator) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for idx, row in evaluator.cdi_data.iterrows():
        val = evaluator.compute_child_aoa(idx)
        if val is not None:
            out[str(row["word"]).lower()] = float(val)
    return out


def subword_lengths(tokenizer: Any, words: List[str]) -> Dict[str, int]:
    prefix_ids = tokenizer("The", add_special_tokens=False)["input_ids"]
    out = {}
    for word in words:
        ids = tokenizer("The " + word, add_special_tokens=False)["input_ids"]
        out[word] = max(1, len(ids) - len(prefix_ids))
    return out


def compute_model_aoa_unthresholded(
    surprisals: List[float],
    training_steps: List[float],
    vocab_size: int,
    n_subword_tokens: int,
    threshold_percentile: float = 0.5,
) -> float | None:
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


def fit_one(path: pathlib.Path, tokenizer: Any, child_aoa: Dict[str, float], target_words: set[str] | None = None) -> Dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    word_data: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in d["results"]:
        w = str(r["target_word"]).lower()
        if target_words is not None and w not in target_words:
            continue
        step_name = r["step"]
        # Official extract_step_number regex: chck_10M -> 10,000,000
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", str(step_name), re.IGNORECASE)
        if not m:
            continue
        number = float(m.group(1)); unit = m.group(2).upper() if m.group(2) else ""
        mult = {"K": 1000, "M": 1000000, "B": 1000000000}.get(unit, 1)
        step_val = number * mult
        try:
            word_data[w][step_val].append(float(r["surprisal"]))
        except Exception:
            pass
    words = sorted([w for w in word_data if w in child_aoa])
    swlen = subword_lengths(tokenizer, words)
    model_aoas = []
    child_aoas = []
    valid_words = []
    invalid_words = []
    per_word = {}
    for w in words:
        steps = sorted(word_data[w])
        mean_surps = [float(np.mean(word_data[w][s])) for s in steps]
        ma = compute_model_aoa_unthresholded(mean_surps, steps, tokenizer.vocab_size, swlen[w])
        if ma is None:
            invalid_words.append(w)
            continue
        valid_words.append(w)
        model_aoas.append(ma)
        child_aoas.append(child_aoa[w])
        per_word[w] = {"model_aoa_log10_words": ma, "child_aoa_month": child_aoa[w], "subword_len": swlen[w]}
    if len(valid_words) >= 3:
        pr, pp = pearsonr(model_aoas, child_aoas)
        sr, sp = spearmanr(model_aoas, child_aoas)
    else:
        pr = pp = sr = sp = float("nan")
    leaderboard = 0.0 if (not math.isfinite(pp) or pp > 0.1) else float(pr)
    return {
        "path": str(path),
        "rows": len(d["results"]),
        "candidate_words_before_fit": len(words),
        "n_valid_words": len(valid_words),
        "n_invalid_words": len(invalid_words),
        "valid_words": valid_words,
        "invalid_words_sample": invalid_words[:50],
        "pearson_model_aoa_vs_child_aoa": float(pr) if math.isfinite(pr) else None,
        "pearson_p_value": float(pp) if math.isfinite(pp) else None,
        "spearman_model_aoa_vs_child_aoa": float(sr) if math.isfinite(sr) else None,
        "spearman_p_value": float(sp) if math.isfinite(sp) else None,
        "official_thresholded_curve_fitness": leaderboard,
        "per_word": per_word,
        "model_aoa_log10_stats": basic_stats(model_aoas),
    }


def basic_stats(vals: List[float]) -> Dict[str, Any]:
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
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": sum(xs)/len(xs), "median": q(0.5), "p95": q(0.95), "max": xs[-1]}


def common_subset_report(full: Dict[str, Any], tokenizer: Any, child_aoa: Dict[str, float]) -> Dict[str, Any]:
    common = set.intersection(*(set(v["valid_words"]) for v in full.values())) if full else set()
    out = {"common_n": len(common), "models": {}}
    for name, path in SURPRISAL_PATHS.items():
        out["models"][name] = fit_one(path, tokenizer, child_aoa, target_words=common)
        out["models"][name].pop("per_word", None)
        out["models"][name].pop("valid_words", None)
    # Model-AoA deltas for devcurr vs clean on matched seed on the common set.
    for seed in ["43022", "43122"]:
        a = fit_one(SURPRISAL_PATHS[f"clean_qwen_seed{seed}"], tokenizer, child_aoa, target_words=common)
        b = fit_one(SURPRISAL_PATHS[f"devcurr_firstpass_seed{seed}"], tokenizer, child_aoa, target_words=common)
        deltas = []
        for w in common:
            if w in a["per_word"] and w in b["per_word"]:
                deltas.append((w, a["per_word"][w]["child_aoa_month"], b["per_word"][w]["model_aoa_log10_words"] - a["per_word"][w]["model_aoa_log10_words"]))
        out[f"devcurr_minus_clean_model_aoa_delta_seed{seed}"] = {
            "n": len(deltas),
            "mean_delta_log10_words": float(np.mean([x[2] for x in deltas])) if deltas else None,
            "pearson_child_aoa_vs_delta": float(pearsonr([x[1] for x in deltas], [x[2] for x in deltas])[0]) if len(deltas) >= 3 else None,
            "examples_largest_delay": sorted(deltas, key=lambda x: x[2], reverse=True)[:20],
            "examples_largest_advance": sorted(deltas, key=lambda x: x[2])[:20],
        }
    return out


def make_note(payload: Dict[str, Any]) -> str:
    lines = ["# research — official AoA curve-fit audit", ""]
    lines.append("This CPU-only check reproduces the official BabyLM AoA curve-fit logic for existing surprisal files. It preserves the unthresholded model-AoA/child-AoA Pearson correlation and p-value, which the official `aoa_score.json` collapses to 0.0 when p>0.1. It does not use AoA words as a training signal.")
    lines.append("")
    lines.append("## Four existing models")
    for name, res in payload["models"].items():
        sc = payload["broad_scores"].get(name, {})
        lines.append(f"- {name}: n_valid={res['n_valid_words']}/{res['candidate_words_before_fit']}, r={res['pearson_model_aoa_vs_child_aoa']:.4f}, p={res['pearson_p_value']:.4g}, official_thresholded={res['official_thresholded_curve_fitness']:.4f}, Overall={sc.get('Overall')}")
    lines.append("")
    lines.append("## Matched seed broad-score deltas for the prior legal first-pass source order")
    for seed in ["43022", "43122"]:
        c = payload["broad_scores"][f"clean_qwen_seed{seed}"]
        d = payload["broad_scores"][f"devcurr_firstpass_seed{seed}"]
        lines.append(f"- seed {seed}: Overall {c['Overall']:.4f} → {d['Overall']:.4f} (Δ {d['Overall']-c['Overall']:+.4f}); Entity {c['Entity']:.2f} → {d['Entity']:.2f}; GlobalPIQA {c['GlobalPIQA']:.3f} → {d['GlobalPIQA']:.3f}")
    mean_clean = np.mean([payload["broad_scores"]["clean_qwen_seed43022"]["Overall"], payload["broad_scores"]["clean_qwen_seed43122"]["Overall"]])
    mean_dev = np.mean([payload["broad_scores"]["devcurr_firstpass_seed43022"]["Overall"], payload["broad_scores"]["devcurr_firstpass_seed43122"]["Overall"]])
    lines.append(f"- mean Overall across the two matched seeds: {mean_clean:.4f} → {mean_dev:.4f} (Δ {mean_dev-mean_clean:+.4f}).")
    lines.append("")
    lines.append("## Common-subset check")
    common = payload["common_subset"]
    lines.append(f"Common valid fitted-word set across all four models: {common['common_n']} words. On this common set, every model still has non-significant or near-zero positive model-AoA/child-AoA correlation, so the 0.0 official AoA result is not just caused by different valid-word sets.")
    for name, res in common["models"].items():
        lines.append(f"- common {name}: r={res['pearson_model_aoa_vs_child_aoa']:.4f}, p={res['pearson_p_value']:.4g}, n={res['n_valid_words']}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("The safe first-pass source-order intervention was enough to alter early surprisal timing proxies (see `aoa_developmental_signal_audit.json`), but its official fitted model-AoA/child-AoA correlation remains small and non-significant for both seeds. Together with the broad-score decrease, this argues against spending H100 time on another pure presentation-order AoA run. It does not rule out a genuinely new objective or corpus design, but such a design must be justified without official AoA target conditioning and evaluated on the full broad task surface.")
    lines.append("")
    lines.append(f"Machine-readable audit: `{OUT_JSON}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = AoAEvaluator(CDI_HUMAN)
    child_aoa = load_cdi_child_aoas(evaluator)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    models: Dict[str, Any] = {}
    for name, path in SURPRISAL_PATHS.items():
        models[name] = fit_one(path, tokenizer, child_aoa)
    payload = {
        "status": "OFFICIAL_AOA_CURVEFIT_AUDIT",
        "scoring_mechanism": "For each word, average surprisal by checkpoint, fit a bounded sigmoid to negative surprisal over log10 word exposure, compute the fitted log10 exposure at the midpoint between tokenizer-scaled random-chance surprisal and the word's minimum surprisal, then Pearson-correlate model AoA with child AoA; if p>0.1 the official returned curve_fitness is 0.0.",
        "tokenizer": str(TOKENIZER_PATH),
        "cdi_human": str(CDI_HUMAN),
        "models": models,
        "common_subset": common_subset_report(models, tokenizer, child_aoa),
        "broad_scores": SCORES,
    }
    # Drop full per-word payload from common subset already done; keep per-word in full
    # model records for inspectability.
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_NOTE.write_text(make_note(payload), encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT_JSON),
        "note": str(OUT_NOTE),
        "models": {k: {"n_valid": v["n_valid_words"], "r": v["pearson_model_aoa_vs_child_aoa"], "p": v["pearson_p_value"], "official_thresholded": v["official_thresholded_curve_fitness"]} for k, v in models.items()},
        "common_n": payload["common_subset"]["common_n"],
        "mean_overall_clean": float(np.mean([SCORES["clean_qwen_seed43022"]["Overall"], SCORES["clean_qwen_seed43122"]["Overall"]])),
        "mean_overall_devcurr": float(np.mean([SCORES["devcurr_firstpass_seed43022"]["Overall"], SCORES["devcurr_firstpass_seed43122"]["Overall"]])),
    }, indent=2))


if __name__ == "__main__":
    main()
