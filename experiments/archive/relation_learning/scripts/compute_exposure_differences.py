#!/usr/bin/env python3
"""research: compute per-word credited-exposure differences for both calibration arms.

For each CDI/AoA target word, computes:
  - Arm 1 (row schedule): cumulative word occurrence at each checkpoint under the
    reordered stream versus v4 order → Δlog_exposure_w,c
  - Arm 2 (enrichment masking): per-word masking probability ratio → Δlog_credit_w
    (constant across checkpoints because the enrichment-to-uniform taper is position-independent)

These exposure/credit differences are the regressors for the β measurement.

CPU only; run before or after training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse, importlib.util, json, math, pathlib, re, sys, time
from collections import Counter
from typing import Any

import numpy as np

ROOT = _public_path('.')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
SCHEDULE_STREAM = _public_path('experiments/archive/relation_learning/data/aoa_calibration_prep/schedule_reordered_30M.jsonl')
V4_STREAM = _public_path('experiments/archive/relation_learning/data/aoa_calibration_prep/v4_order_30M.jsonl')
ENRICHMENT_PATH = _public_path('experiments/archive/relation_learning/data/aoa_calibration_prep/token_enrichment_weights.json')
TOKENIZER_DIR = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_1M')
OUT = _public_path('experiments/archive/relation_learning/data/exposure_differences')
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p):
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_cdi_words():
    """Load CDI target words from the official evaluation data."""
    cdi_path = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/cdi_childes/cdi_childes.json')
    data = json.loads(cdi_path.read_text(encoding="utf-8"))
    words = set()
    if isinstance(data, dict):
        for w in data:
            words.add(w.lower().strip())
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "word" in entry:
                words.add(str(entry["word"]).lower().strip())
            elif isinstance(entry, str):
                words.add(entry.lower().strip())
    return words


def load_step106_aoa_words():
    """Load fitted AoA words from research analysis."""
    path = _public_path('experiments/archive/relation_learning/data/aoa_curve_fit_analysis/word_curve_fit_records.csv')
    if not path.exists():
        return {}
    import csv
    records = {}
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row.get("word", "").lower().strip()
            if not w:
                continue
            try:
                child_aoa = float(row.get("child_aoa", "")) if row.get("child_aoa") else None
                model_aoa = float(row.get("model_aoa", "")) if row.get("model_aoa") else None
            except (ValueError, TypeError):
                child_aoa = model_aoa = None
            records[w] = {"child_aoa": child_aoa, "model_aoa": model_aoa}
    return records


def scan_stream_exposures(stream_path, target_words, checkpoints):
    """Scan a JSONL stream and compute per-word cumulative occurrences at each checkpoint."""
    targets = set(target_words)
    cp = sorted(checkpoints)
    cp_i = 0
    cum: Counter[str] = Counter()
    exposures: dict[int, dict[str, int]] = {c: {} for c in cp}
    consumed = 0
    with stream_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            words = int(obj.get("words", len(text.split())))
            counts: Counter[str] = Counter()
            for m in WORD_RE.finditer(text):
                w = m.group(0).lower()
                if w in targets:
                    counts[w] += 1
            before = consumed
            after = before + words
            while cp_i < len(cp) and cp[cp_i] <= after:
                b = cp[cp_i]
                frac = max(0.0, min(1.0, (b - before) / max(words, 1)))
                snapshot = dict(cum)
                for w, n in counts.items():
                    snapshot[w] = snapshot.get(w, 0) + frac * n
                exposures[b] = snapshot
                cp_i += 1
            consumed = after
            for w, n in counts.items():
                cum[w] += n
    for j in range(cp_i, len(cp)):
        exposures[cp[j]] = dict(cum)
    return exposures, consumed


def compute_enrichment_credit_ratios(target_words, alpha=1.5, base_prob=0.15):
    """Compute per-word credit ratio: enrichment-weighted mask prob / uniform mask prob."""
    enr_data = json.loads(_public_path('experiments/archive/relation_learning/data/aoa_calibration_prep/token_enrichment_weights.json').read_text(encoding="utf-8"))
    z = np.array(enr_data["z_enrichment"], dtype=np.float64)
    
    # Build per-token probs (same as in the enrichment trainer)
    log_weights = alpha * z
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights_mean = weights.mean()
    probs = base_prob * weights / weights_mean
    probs = np.clip(probs, 0.02, 0.60)
    
    # Map target words to token IDs and get their enrichment probs
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), trust_remote_code=True)
    
    credit_ratios = {}
    for w in target_words:
        ids = tokenizer.encode(f" {w}", add_special_tokens=False)
        if not ids:
            continue
        word_probs = [float(probs[tid]) for tid in ids if 0 <= tid < len(probs)]
        if word_probs:
            mean_prob = np.mean(word_probs)
            credit_ratios[w] = {
                "enrichment_mask_prob": float(mean_prob),
                "uniform_mask_prob": base_prob,
                "credit_ratio": float(mean_prob / base_prob),
                "log_credit_ratio": float(math.log(mean_prob / base_prob)),
            }
    return credit_ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load target words
    cdi_words = load_cdi_words()
    aoa_records = load_step106_aoa_words()
    target_words = cdi_words | set(aoa_records.keys())
    print(json.dumps({"event": "targets_loaded", "cdi": len(cdi_words),
                      "aoa_records": len(aoa_records), "union": len(target_words)}), flush=True)

    # Matched checkpoints (same as official ladder subset within 30M)
    checkpoints = list(range(1_000_000, 10_000_000, 1_000_000)) + [10_000_000, 20_000_000, 30_000_000]

    # Scan v4 stream for baseline exposures
    print(json.dumps({"event": "scanning_v4"}), flush=True)
    v4_exp, v4_words = scan_stream_exposures(V4_STREAM, target_words, checkpoints)
    print(json.dumps({"event": "v4_scanned", "words": v4_words}), flush=True)

    # Scan schedule stream for reordered exposures
    print(json.dumps({"event": "scanning_schedule"}), flush=True)
    sched_exp, sched_words = scan_stream_exposures(SCHEDULE_STREAM, target_words, checkpoints)
    print(json.dumps({"event": "schedule_scanned", "words": sched_words}), flush=True)

    # Compute row-schedule exposure differences
    schedule_diffs = {}
    for w in target_words:
        diffs = {}
        for ck in checkpoints:
            v4_e = v4_exp[ck].get(w, 0)
            sc_e = sched_exp[ck].get(w, 0)
            if v4_e > 0 and sc_e > 0:
                diffs[ck] = {"v4_exposure": v4_e, "schedule_exposure": sc_e,
                             "dlog_exposure": math.log(sc_e / v4_e)}
            elif sc_e > 0:
                diffs[ck] = {"v4_exposure": v4_e, "schedule_exposure": sc_e,
                             "dlog_exposure": math.log(sc_e + 0.5) - math.log(v4_e + 0.5)}
        if diffs:
            schedule_diffs[w] = diffs

    # Compute enrichment credit ratios
    print(json.dumps({"event": "computing_enrichment_credit"}), flush=True)
    credit_ratios = compute_enrichment_credit_ratios(target_words)
    print(json.dumps({"event": "credit_computed", "words_with_credit": len(credit_ratios)}), flush=True)

    # Summary statistics
    abs_dlogs = []
    for w, diffs in schedule_diffs.items():
        for ck, d in diffs.items():
            abs_dlogs.append(abs(d["dlog_exposure"]))

    log_ratios = [cr["log_credit_ratio"] for cr in credit_ratios.values()]

    summary = {
        "status": "EXPOSURE_DIFFERENCES_DONE",
        "created_utc": now(),
        "checkpoints": checkpoints,
        "n_target_words": len(target_words),
        "schedule_arm": {
            "words_with_diffs": len(schedule_diffs),
            "mean_abs_dlog": float(np.mean(abs_dlogs)) if abs_dlogs else None,
            "max_abs_dlog": float(np.max(abs_dlogs)) if abs_dlogs else None,
        },
        "enrichment_arm": {
            "words_with_credit": len(credit_ratios),
            "mean_log_credit_ratio": float(np.mean(log_ratios)) if log_ratios else None,
            "std_log_credit_ratio": float(np.std(log_ratios)) if log_ratios else None,
            "mean_credit_ratio": float(np.mean([cr["credit_ratio"] for cr in credit_ratios.values()])) if credit_ratios else None,
        },
        "elapsed_sec": round(time.time() - t0, 1)
    }

    # Save
    (out / "schedule_exposure_diffs.json").write_text(
        json.dumps(schedule_diffs, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "enrichment_credit_ratios.json").write_text(
        json.dumps(credit_ratios, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
