#!/usr/bin/env python3
"""research: precompute per-word credited-exposure differences for β measurement.

Scans both the v4-order and schedule-reordered 30M streams, computing cumulative 
per-CDI-word occurrences at each official-like checkpoint boundary (1M-10M, 20M, 30M).

For the schedule arm: Δcredit = 0.15 × (cumul_occ_schedule - cumul_occ_v4)
For the enrichment arm: Δcredit = cumul_enrichment_credit - 0.15 × cumul_occ_v4
  where enrichment credit = sum_step n_w(step) × p_w(alpha(step))
  with alpha tapering linearly from alpha_start to 0 by taper_frac × lr_total_steps.

Outputs a per-word-per-checkpoint Δlog_credit table for β regression.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, sys, time
from pathlib import Path
from collections import defaultdict

HERE = _public_path('experiments/archive/relation_learning/scripts')
USER_ROOT = _public_path('data/external/users')

import torch
import numpy as np


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(
        _public_path('data/external/exposure_differences')))
    ap.add_argument("--alpha-start", type=float, default=1.5)
    ap.add_argument("--taper-frac", type=float, default=0.67)
    ap.add_argument("--lr-total-steps", type=int, default=2529)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Paths ────────────────────────────────────────────────────────────────
    v4_stream = _public_path('data/external/v4_order_30M.jsonl')
    sched_stream = _public_path('data/external/schedule_reordered_30M.jsonl')
    tok_path = _public_path('data/external/compliant_tokenizer')
    enrich_path = _public_path('data/external/token_enrichment_weights.json')
    corpus_freq_path = _public_path('data/external/corpus_token_freq.json')
    # CDI word list from the official evaluation  
    cdi_path = _public_path('data/external/cdi_human.csv')

    # ─── Load tokenizer ───────────────────────────────────────────────────────
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path.resolve()))

    # ─── Load enrichment data ─────────────────────────────────────────────────
    ew = json.loads(enrich_path.read_text())
    z_scores = torch.tensor(ew["z_enrichment"], dtype=torch.float32)
    cf = json.loads(corpus_freq_path.read_text())
    corpus_freq = torch.tensor(cf["token_freq"], dtype=torch.float32)
    freq_sum = corpus_freq.sum()

    # ─── Load CDI words ───────────────────────────────────────────────────────
    import csv
    cdi_words = set()
    if cdi_path.exists():
        with open(cdi_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                w = row.get("word", "").strip().lower()
                if w:
                    cdi_words.add(w)
    print(json.dumps({"event": "cdi_words_loaded", "n": len(cdi_words)}), flush=True)

    # ─── Tokenize CDI words ───────────────────────────────────────────────────
    word_to_tokens = {}
    for w in sorted(cdi_words):
        ids = tokenizer.encode(w, add_special_tokens=False)
        word_to_tokens[w] = ids
    all_target_tokens = set()
    for ids in word_to_tokens.values():
        all_target_tokens.update(ids)

    # ─── Precompute enrichment probability function ───────────────────────────
    taper_steps = int(args.taper_frac * args.lr_total_steps)

    def enrichment_prob(alpha):
        """Per-token masking prob at given alpha."""
        if alpha <= 0:
            return torch.full_like(z_scores, 0.15)
        log_w = alpha * z_scores
        exp_w = torch.exp(log_w)
        Z = (corpus_freq * exp_w).sum() / freq_sum
        return (0.15 * exp_w / Z).clamp(0.02, 0.60)

    # ─── Checkpoint boundaries ────────────────────────────────────────────────
    # Official AoA: chck_1M..chck_10M, chck_20M, chck_30M  (12 checkpoints for 30M)
    checkpoints = list(range(1_000_000, 10_000_001, 1_000_000)) + [20_000_000, 30_000_000]

    # ─── Scan streams ─────────────────────────────────────────────────────────
    def scan_stream(stream_path, label):
        """Scan a JSONL stream, returning per-CDI-word cumulative token counts at checkpoints."""
        cumul = {w: {c: 0 for c in checkpoints} for w in cdi_words}
        running_counts = defaultdict(int)  # token_id -> cumul count
        total_words = 0
        cp_idx = 0
        t0 = time.time()

        with open(stream_path) as f:
            for line in f:
                row = json.loads(line)
                row_words = len(row["text"].split())
                ids = tokenizer.encode(row["text"], add_special_tokens=False)
                for tid in ids:
                    if tid in all_target_tokens:
                        running_counts[tid] += 1
                total_words += row_words

                # Record at checkpoint boundaries
                while cp_idx < len(checkpoints) and total_words >= checkpoints[cp_idx]:
                    cp = checkpoints[cp_idx]
                    for w in cdi_words:
                        cumul[w][cp] = sum(running_counts.get(t, 0) for t in word_to_tokens[w])
                    cp_idx += 1

        # Fill remaining checkpoints
        while cp_idx < len(checkpoints):
            cp = checkpoints[cp_idx]
            for w in cdi_words:
                cumul[w][cp] = sum(running_counts.get(t, 0) for t in word_to_tokens[w])
            cp_idx += 1

        print(json.dumps({"event": "stream_scanned", "label": label,
                          "total_words": total_words,
                          "sec": round(time.time() - t0, 1)}), flush=True)
        return cumul

    v4_cumul = scan_stream(v4_stream, "v4_order")
    sched_cumul = scan_stream(sched_stream, "schedule")

    # ─── Compute enrichment credit for v4-order stream ────────────────────────
    # For each word and checkpoint, compute:
    #   enrichment_credit = cumul_occ × average_enrichment_prob_over_steps
    # The average_enrichment_prob depends on the alpha schedule.
    # Since the word may not appear uniformly, we approximate using checkpoint-level granularity:
    # For each checkpoint interval [c_{k-1}, c_k], count occurrences and multiply by
    # the average enrichment prob over that interval's steps.

    # Approximate step count per word-exposure: ~39K words/step
    words_per_step = 39500  # approximate from v4 reference

    def compute_enrichment_credit():
        """Per-CDI-word enrichment credit at each checkpoint."""
        credit = {w: {c: 0.0 for c in checkpoints} for w in cdi_words}

        prev_occ = {w: 0 for w in cdi_words}
        for ci, cp in enumerate(checkpoints):
            # Steps covered by this checkpoint
            cp_step = cp / words_per_step
            if ci == 0:
                step_lo = 0
            else:
                step_lo = checkpoints[ci - 1] / words_per_step
            step_hi = cp_step

            # Average alpha over this interval
            alpha_lo = max(0, args.alpha_start * (1 - step_lo / taper_steps))
            alpha_hi = max(0, args.alpha_start * (1 - step_hi / taper_steps))
            alpha_avg = (alpha_lo + alpha_hi) / 2

            # Per-token enrichment probabilities at average alpha
            probs = enrichment_prob(alpha_avg)

            for w in cdi_words:
                cur_occ = v4_cumul[w][cp]
                interval_occ = cur_occ - prev_occ[w]
                if interval_occ > 0:
                    # Average prob for this word's tokens
                    tids = word_to_tokens[w]
                    if tids:
                        word_prob = float(probs[tids].mean())
                    else:
                        word_prob = 0.15
                    credit[w][cp] = credit[w].get(checkpoints[ci - 1], 0.0) if ci > 0 else 0.0
                    credit[w][cp] += interval_occ * word_prob
                else:
                    credit[w][cp] = credit[w].get(checkpoints[ci - 1], 0.0) if ci > 0 else 0.0
                prev_occ[w] = cur_occ

        return credit

    enrichment_credit = compute_enrichment_credit()

    # ─── Build Δcredit tables ─────────────────────────────────────────────────
    results = []
    for w in sorted(cdi_words):
        tids = word_to_tokens[w]
        z_mean = float(z_scores[tids].mean()) if tids else 0.0
        for cp in checkpoints:
            v4_occ = v4_cumul[w][cp]
            sched_occ = sched_cumul[w][cp]
            enrich_cred = enrichment_credit[w][cp]
            control_cred = 0.15 * v4_occ

            # Schedule arm Δlog_credit
            sched_delta = 0.15 * (sched_occ - v4_occ)
            sched_dlog = (np.log(0.15 * sched_occ + 1e-8) - np.log(control_cred + 1e-8)
                          if v4_occ > 0 and sched_occ > 0 else 0.0)

            # Enrichment arm Δlog_credit
            enrich_delta = enrich_cred - control_cred
            enrich_dlog = (np.log(enrich_cred + 1e-8) - np.log(control_cred + 1e-8)
                           if v4_occ > 0 else 0.0)

            results.append({
                "word": w,
                "checkpoint_words": cp,
                "z_mean": round(z_mean, 4),
                "v4_occ": v4_occ,
                "schedule_occ": sched_occ,
                "enrichment_credit": round(enrich_cred, 4),
                "control_credit": round(control_cred, 4),
                "schedule_delta_credit": round(sched_delta, 4),
                "schedule_dlog_credit": round(sched_dlog, 6),
                "enrichment_delta_credit": round(enrich_delta, 4),
                "enrichment_dlog_credit": round(enrich_dlog, 6),
            })

    # ─── Save ─────────────────────────────────────────────────────────────────
    out_csv = out_dir / "exposure_differences.csv"
    import csv as csv_mod
    with open(out_csv, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "status": "EXPOSURE_DIFFERENCES_DONE",
        "n_words": len(cdi_words),
        "n_checkpoints": len(checkpoints),
        "n_rows": len(results),
        "alpha_start": args.alpha_start,
        "taper_frac": args.taper_frac,
        "taper_steps": taper_steps,
        "csv": str(out_csv.relative_to(USER_ROOT)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
