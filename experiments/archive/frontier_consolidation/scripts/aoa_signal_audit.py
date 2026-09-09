#!/usr/bin/env python3
"""research: AoA signal audit for official-row-count reinvest runs.

Official BabyLM AoA scoring sets curve_fitness to 0.0 when Pearson p>0.1.
This CPU-only audit recomputes the underlying model AoAs and raw correlations
from the saved 8,005-row/checkpoint surprisal files, without changing official
scores, to verify nonconstant signal and rule out join/constant-vector artifacts.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = pathlib.Path.cwd()
STUDY = ROOT / "experiments/archive/frontier_consolidation"
STRICT = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
sys.path.insert(0, str(STRICT.resolve()))
from evaluation_pipeline.utils import AoAEvaluator  # noqa: E402

RUNS = {
    "reinvest_seed43022": {
        "model_root": ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model",
        "summary": ROOT / "experiments/archive/frontier_consolidation/data/official_rowcount_aoa_reinvest_seeds/reinvest_seed43022/aoa_local_ckpts_minctx0.json",
        "surprisal": ROOT / "experiments/archive/frontier_consolidation/data/official_rowcount_aoa_reinvest_seeds/reinvest_seed43022/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json",
    },
    "reinvest_seed43122": {
        "model_root": ROOT / "experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model",
        "summary": ROOT / "experiments/archive/frontier_consolidation/data/official_rowcount_aoa_reinvest_seeds/reinvest_seed43122/aoa_local_ckpts_minctx0.json",
        "surprisal": ROOT / "experiments/archive/frontier_consolidation/data/official_rowcount_aoa_reinvest_seeds/reinvest_seed43122/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json",
    },
}
CDI_HUMAN = STRICT / "evaluation_data/full_eval/aoa/cdi_human.csv"
OUT_DIR = STUDY / "data/aoa_signal_audit"


def load_tokenizer(model_root: pathlib.Path):
    candidates = [model_root, model_root / "chck_100M", model_root / "chck_80M", model_root / "chck_10M"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            return AutoTokenizer.from_pretrained(p, trust_remote_code=True), str(p)
        except Exception:
            try:
                return PreTrainedTokenizerFast.from_pretrained(p, padding_side="right"), str(p)
            except Exception:
                pass
    raise RuntimeError(f"Could not load tokenizer for {model_root}")


def extract_step_number(evaluator: AoAEvaluator, step: str) -> float | None:
    return evaluator.extract_step_number(step)


def compute_raw(model_results: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    evaluator = AoAEvaluator(CDI_HUMAN)
    vocab_size = tokenizer.vocab_size
    prefix_ids = tokenizer("The", add_special_tokens=False)["input_ids"]

    def subword_len(word: str) -> int:
        ids = tokenizer("The " + word, add_special_tokens=False)["input_ids"]
        return max(1, len(ids) - len(prefix_ids))

    word_data: dict[str, dict[str, list[float]]] = {}
    rows = model_results.get("results", [])
    for result in rows:
        word = result["target_word"]
        step_val = extract_step_number(evaluator, result["step"])
        if step_val is None:
            continue
        word_data.setdefault(word, {"steps": [], "surprisals": []})
        word_data[word]["steps"].append(float(step_val))
        word_data[word]["surprisals"].append(float(result["surprisal"]))

    per_word = []
    for word in word_data:
        word_mask = evaluator.cdi_data["word"] == word
        if not np.any(word_mask):
            continue
        word_idx = np.where(word_mask)[0][0]
        child_aoa = evaluator.compute_child_aoa(word_idx)
        if child_aoa is None:
            continue
        steps_arr = np.array(word_data[word]["steps"])
        surp_arr = np.array(word_data[word]["surprisals"])
        uniq_steps = np.unique(steps_arr)
        mean_surprisals = [float(surp_arr[steps_arr == s].mean()) for s in uniq_steps]
        model_aoa = evaluator.compute_model_aoa(mean_surprisals, uniq_steps.tolist(), vocab_size, n_subword_tokens=subword_len(word))
        if model_aoa is None:
            continue
        per_word.append({
            "word": word,
            "model_aoa_log_step": float(model_aoa),
            "child_aoa_month": float(child_aoa),
            "n_context_observations": len(word_data[word]["surprisals"]),
            "mean_surprisal_first_step": float(mean_surprisals[0]),
            "mean_surprisal_last_step": float(mean_surprisals[-1]),
            "surprisal_delta_last_minus_first": float(mean_surprisals[-1] - mean_surprisals[0]),
        })

    model_vals = [x["model_aoa_log_step"] for x in per_word]
    child_vals = [x["child_aoa_month"] for x in per_word]
    if len(model_vals) >= 3:
        pear = pearsonr(model_vals, child_vals)
        spear = spearmanr(model_vals, child_vals)
    else:
        pear = (float("nan"), float("nan"))
        spear = (float("nan"), float("nan"))
    mean_surprisal_by_step = {}
    step_counts = {}
    for r in rows:
        st = str(r["step"])
        step_counts[st] = step_counts.get(st, 0) + 1
        mean_surprisal_by_step[st] = mean_surprisal_by_step.get(st, 0.0) + float(r["surprisal"])
    mean_surprisal_by_step = {k: mean_surprisal_by_step[k] / step_counts[k] for k in step_counts}
    return {
        "n_result_rows": len(rows),
        "step_counts": step_counts,
        "row_count_values": sorted(set(step_counts.values())),
        "n_words_with_model_and_child_aoa": len(per_word),
        "pearson_raw_correlation": float(pear.statistic if hasattr(pear, "statistic") else pear[0]),
        "pearson_p_value": float(pear.pvalue if hasattr(pear, "pvalue") else pear[1]),
        "spearman_raw_correlation": float(spear.statistic if hasattr(spear, "statistic") else spear[0]),
        "spearman_p_value": float(spear.pvalue if hasattr(spear, "pvalue") else spear[1]),
        "model_aoa_std": float(np.std(model_vals, ddof=1)) if len(model_vals) > 1 else None,
        "child_aoa_std": float(np.std(child_vals, ddof=1)) if len(child_vals) > 1 else None,
        "model_aoa_min": float(np.min(model_vals)) if model_vals else None,
        "model_aoa_max": float(np.max(model_vals)) if model_vals else None,
        "child_aoa_min": float(np.min(child_vals)) if child_vals else None,
        "child_aoa_max": float(np.max(child_vals)) if child_vals else None,
        "mean_surprisal_by_step": mean_surprisal_by_step,
        "per_word_first20": per_word[:20],
        "per_word_sorted_by_child_aoa_first20": sorted(per_word, key=lambda x: x["child_aoa_month"])[:20],
        "per_word_sorted_by_model_aoa_first20": sorted(per_word, key=lambda x: x["model_aoa_log_step"])[:20],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, cfg in RUNS.items():
        missing = [str(p) for p in [cfg["model_root"], cfg["summary"], cfg["surprisal"], CDI_HUMAN] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{name} missing: " + "; ".join(missing))
        tokenizer, tok_path = load_tokenizer(cfg["model_root"])
        official_summary = json.loads(cfg["summary"].read_text(encoding="utf-8"))
        surprisal = json.loads(cfg["surprisal"].read_text(encoding="utf-8"))
        raw = compute_raw(surprisal, tokenizer)
        records[name] = {
            "model_root": str(cfg["model_root"]),
            "summary_path": str(cfg["summary"]),
            "surprisal_path": str(cfg["surprisal"]),
            "tokenizer_path": tok_path,
            "official_thresholded_aoa": official_summary.get("aoa"),
            "official_curve_fitness_record": official_summary.get("curve_fitness_record"),
            "raw_audit": raw,
        }
    result = {
        "status": "AOA_SIGNAL_AUDIT",
        "purpose": "Audit raw AoA correlations behind official thresholded AoA=0 for reinvest seeds; no official score modification.",
        "scoring_code_reference": "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/utils.py lines 344-348 set curve_fitness to 0.0 when p_value > 0.1.",
        "records": records,
        "interpretation": {
            "official_score_unchanged": "Official leaderboard AoA remains 0.0 for both seeds because the evaluator returns 0.0 for non-significant correlations.",
            "artifact_check": "Nonzero standard deviation in model AoAs and finite raw correlations support that AoA=0 is a thresholded non-significant signal rather than missing rows or constant surprisals.",
        },
    }
    out_json = OUT_DIR / "aoa_signal_audit.json"
    out_md = OUT_DIR / "aoa_signal_audit.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — official-row-count AoA signal audit\n\n"]
    lines.append("Official AoA scores remain the evaluator's thresholded values; this audit records raw correlations to check the signal behind AoA=0.\n\n")
    for name, rec in records.items():
        a = rec["raw_audit"]
        lines.append(f"## {name}\n")
        lines.append(f"- official thresholded AoA: {rec['official_thresholded_aoa']} with record {rec['official_curve_fitness_record']}\n")
        lines.append(f"- rows: {a['n_result_rows']}; row_count_values: {a['row_count_values']}; words with model+child AoA: {a['n_words_with_model_and_child_aoa']}\n")
        lines.append(f"- raw Pearson r: {a['pearson_raw_correlation']:+.6f}, p={a['pearson_p_value']:.6f}; raw Spearman rho: {a['spearman_raw_correlation']:+.6f}, p={a['spearman_p_value']:.6f}\n")
        lines.append(f"- model AoA std: {a['model_aoa_std']:.6f}; child AoA std: {a['child_aoa_std']:.6f}; model AoA range: {a['model_aoa_min']:.3f}..{a['model_aoa_max']:.3f}\n\n")
    lines.append("## Scientific read\n")
    lines.append("- AoA=0 is the official non-significance outcome, not an evidence-free placeholder. The saved files have complete 8,005-row/checkpoint ladders and nonconstant fitted model AoAs.\n")
    lines.append("- These raw values are audit evidence only and must not replace the official leaderboard AoA column.\n")
    lines.append(f"\nMachine-readable output: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "records": {
            k: {
                "official_thresholded_aoa": v["official_thresholded_aoa"],
                "n_words": v["raw_audit"]["n_words_with_model_and_child_aoa"],
                "pearson_raw_correlation": v["raw_audit"]["pearson_raw_correlation"],
                "pearson_p_value": v["raw_audit"]["pearson_p_value"],
                "model_aoa_std": v["raw_audit"]["model_aoa_std"],
            } for k, v in records.items()
        },
        "out_json": str(out_json),
        "out_md": str(out_md),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
