#!/usr/bin/env python3
"""Calibrate dense-MLM target density against research mixed-objective arms.

This is a pre-separator design utility, not an evaluation-score selector.  It reads
only training metrics/logs for completed research mixed-objective runs and estimates
the WWM mask probability that would produce a comparable number of supervised MLM
target tokens over the same clean-Qwen corpus, architecture, update count, and word
exposure.

The resulting probabilities are intended for matched-from-scratch dense-MLM control
arms if, and only if, the full mixed-objective package shows useful score signal.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import json
import pathlib
from statistics import mean

ROOT = _public_path('experiments/archive/compact_experience')
RUNS = {
    "causal50": _public_path('experiments/archive/compact_experience/training/runs/mixed_causal50_qwen_seed43022'),
    "causal15": _public_path('experiments/archive/compact_experience/training/runs/mixed_causal15_qwen_seed43022'),
}
OUT = _public_path('experiments/archive/compact_experience/data/mixed_objective_mechanism/dense_mlm_target_density_calibration.json')
NOTE = _public_path('research/notes/compact_experience/dense_mlm_target_density_calibration.md')
TOKENIZER = pathlib.Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
TRAIN_FILE = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl')
META = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json')
BASELINE_RUN = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022')


def load_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def summarize_training_log(run: pathlib.Path) -> dict:
    p = run / "training_log.jsonl"
    if not p.exists():
        return {"training_log_missing": str(p)}
    mlm_rates = []
    causal_rates = []
    total_targets = 0
    total_words = 0
    rows = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows += 1
            targets = int(r.get("targets") or 0)
            words = int(r.get("batch_words") or 0)
            total_targets += targets
            total_words += words
            if r.get("objective") == "mlm":
                mlm_rates.append(targets / max(1, words))
            elif r.get("objective") == "causal":
                causal_rates.append(targets / max(1, words))
    return {
        "log_rows": rows,
        "log_total_targets": total_targets,
        "log_total_words": total_words,
        "mean_mlm_targets_per_word": mean(mlm_rates) if mlm_rates else None,
        "mean_causal_targets_per_word": mean(causal_rates) if causal_rates else None,
        "realized_total_targets_per_word": total_targets / max(1, total_words),
    }


def calibrate_one(arm: str, run: pathlib.Path, nominal_clean_mlm_target_rate: float = 0.15) -> dict:
    metrics_path = run / "scientific_metrics.json"
    if not metrics_path.exists():
        return {"arm": arm, "run_dir": str(run), "ready": False, "missing": str(metrics_path)}
    m = load_json(metrics_path)
    total_targets = int(m.get("mlm_targets", 0) or 0) + int(m.get("causal_targets", 0) or 0)
    words = int(m.get("actual_word_exposure", 0) or 0)
    if total_targets <= 0 or words <= 0:
        ready = False
        target_rate = None
    else:
        ready = True
        target_rate = total_targets / words
    # WWM expansion means nominal mask probability is not exactly target tokens/word.
    # We therefore produce two estimates: naive token-equivalent and WWM-scaled using
    # the arm's own realized MLM target/word rate divided by its nominal 0.15.
    realized_mlm_rate = (float(m.get("mlm_targets", 0)) / max(1.0, float(words)))
    mlm_batch_fraction = float(m.get("mlm_batches", 0)) / max(1.0, float(m.get("actual_steps", 0) or 0))
    mlm_rate_per_all_words = realized_mlm_rate
    # The realized MLM rate per total words includes only MLM batches. Convert to per-MLM-batch word rate.
    per_mlm_batch_word_rate = realized_mlm_rate / max(1e-12, mlm_batch_fraction)
    wwm_expansion_factor = per_mlm_batch_word_rate / nominal_clean_mlm_target_rate if per_mlm_batch_word_rate else None
    naive_mask_prob = target_rate if target_rate is not None else None
    wwm_scaled_mask_prob = (target_rate / wwm_expansion_factor) if (target_rate is not None and wwm_expansion_factor) else None
    rec = {
        "arm": arm,
        "run_dir": str(run),
        "ready": ready,
        "causal_fraction": m.get("causal_fraction"),
        "realized_causal_batch_fraction": m.get("realized_causal_batch_fraction"),
        "actual_word_exposure": words,
        "actual_steps": m.get("actual_steps"),
        "mlm_batches": m.get("mlm_batches"),
        "causal_batches": m.get("causal_batches"),
        "mlm_targets": m.get("mlm_targets"),
        "causal_targets": m.get("causal_targets"),
        "total_targets": total_targets,
        "realized_total_targets_per_word": target_rate,
        "realized_mlm_targets_per_total_word": realized_mlm_rate,
        "realized_mlm_targets_per_mlm_batch_word": per_mlm_batch_word_rate,
        "estimated_wwm_expansion_factor_from_mixed_mlm_batches": wwm_expansion_factor,
        "dense_mlm_mask_prob_token_equivalent_naive": naive_mask_prob,
        "dense_mlm_mask_prob_wwm_scaled": wwm_scaled_mask_prob,
        "suggested_dense_mlm_mask_prob_clamped": None if wwm_scaled_mask_prob is None else max(0.15, min(0.80, wwm_scaled_mask_prob)),
        "training_log_summary": summarize_training_log(run),
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    rows = [calibrate_one(arm, run) for arm, run in RUNS.items()]
    missing = [r for r in rows if not r.get("ready")]
    if missing and not args.allow_incomplete:
        raise SystemExit("research metrics are not ready: " + json.dumps(missing, indent=2))
    payload = {
        "status": "DENSE_MLM_TARGET_DENSITY_CALIBRATION",
        "source": "training metrics/logs only; no downstream evaluation scores or AoA item data used",
        "baseline_reference": {"run_dir": str(BASELINE_RUN), "nominal_mask_prob": 0.15, "overall": 41.34429066479573, "equal7": 43.112857142857145},
        "tokenizer": str(TOKENIZER),
        "train_file": str(TRAIN_FILE),
        "metadata": str(META),
        "candidate_dense_mlm_runs_should_use": "masking_curriculum_trainer.py wwm_fixed, same clean-Qwen corpus/init/optimizer/exposure, mask_prob_start=end set from suggested_dense_mlm_mask_prob_clamped",
        "rows": rows,
    }
    _public_path('experiments/archive/compact_experience/data/mixed_objective_mechanism').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    lines = [
        "# research dense-MLM target-density calibration", "",
        "This file uses only research training metrics/logs. It does not inspect evaluation predictions, AoA words, AoA curves, or downstream scores.", "",
        "| arm | ready | causal frac | total targets/word | estimated WWM expansion | suggested dense WWM mask_prob | mlm batches | causal batches | total targets |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r.get('arm')} | {r.get('ready')} | {r.get('causal_fraction')} | {r.get('realized_total_targets_per_word')} | {r.get('estimated_wwm_expansion_factor_from_mixed_mlm_batches')} | {r.get('suggested_dense_mlm_mask_prob_clamped')} | {r.get('mlm_batches')} | {r.get('causal_batches')} | {r.get('total_targets')} |")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "note": str(NOTE), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
