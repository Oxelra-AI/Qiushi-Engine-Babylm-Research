# aoa safety audit and route execution state and next work

## Active H100 work

Training comparison specified in this record:

- command: `bash experiments/archive/compact_experience/scripts/launch_devcurr_training.sh`
- purpose: train **dual-seed AoA-safe developmental first-pass clean-Qwen** arms on two H100s.
- output runs:
  - `training/runs/qwen_devcurr_firstpass_16k_seed43022`
  - `training/runs/qwen_devcurr_firstpass_16k_seed43122`
- baseline comparisons:
  - seed43022 vs `qwen_clean_aligned_16k_seed43022` (already fully evaluated)
  - seed43122 vs `qwen_clean_aligned_16k_seed43122` (already fully evaluated)

Training completion was pending in this record.

## What was materialized

`data/aoa_developmental_order/devcurr_materialization_metadata.json`:

- status: `AOA_SAFE_DEVCURR_QWEN_ORDER_MATERIALIZED`
- exact clean qwen compliance and validity clean-Qwen 10M row multiset preserved (`exact_row_multiset_preserved=true`)
- total 10M words and 100M exposure preserved
- first pass sorted by an AoA-safe developmental prior:
  - source rank: CHILDES → simple_wiki → qwen_pair_packed → open_subtitles → switchboard → bnc_spoken → gutenberg
  - within-source row order: high corpus frequency / low rare fraction / shorter forms earlier
- passes 2–10 use the original clean-Qwen order to minimize endpoint disturbance while shaping the official AoA checkpoint ladder.
- no official AoA/CDI evaluation words, child curves, AoA predictions, or AoA scores are read or used.

Important first-pass distribution shift:
- original first 1M: mixed sources (childes 283k, qwen_pair 155k, open_subtitles 191k, gutenberg 192k, etc.)
- developmental first 1M: 1,000,000 words CHILDES

Therefore the arm is a strong **source-stage developmental schedule** first; lexical difficulty is subordinate within source.

## AoA scorer semantics established

See `notes/aoa_scorer_semantics_decisive.md`.

`AoAEvaluator.compute_curve_fitness` does this:

1. per word, averages surprisal across contexts at each checkpoint;
2. fits a bounded sigmoid over negative surprisal versus `log10(checkpoint_words+1)`;
3. defines model AoA as the log checkpoint at which surprisal crosses halfway between random-chance ceiling and the word's best surprisal;
4. correlates model AoAs with child AoAs;
5. returns 0.0 if p-value > 0.1.

Thus the clean-Qwen arms' AoA=0.0 is a **valid non-significant developmental-order fit**, not a missing-checkpoint fallback. Official controls' negative AoA values are significant negative correlations.

## Prepared evaluation scripts after training completes

1. Correct full nine-column eval:
   - `scripts/full_eval_runner.py`
   - `scripts/launch_devcurr_eval.sh`
   - `scripts/summarize_devcurr_eval.py`

2. AoA-only ladder screen:
   - `scripts/screen_aoa_ladder.py`
   - `scripts/launch_aoa_screen.sh`

The full eval uses the corrected full overall eval runner and `babylm_official_scoring.py`, so AoA is in leaderboard units (`100*raw`). The summary decomposes same-seed deltas into NLP7, Reading, and AoA contributions.

## Prepared follow-up if source-block order helps or hurts

The current arm confounds developmental source order with lexical difficulty. The following source was written but had **not** been compiled or run:

- `scripts/materialize_source_balanced_devcurr.py`
- `scripts/launch_source_balanced_followup_training.sh`

These build/train source-balanced easy→hard vs hard→easy first-pass orders over the exact same clean-Qwen multiset. They are the next causal separation if aoa safety audit and route shows an AoA movement or a damaging NLP tradeoff.

## Immediate next actions after task delivery

1. After training, inspect both `scientific_metrics.json` files: word exposure, steps, loss_first/loss_last, 100 checkpoint ladder.
2. If training succeeded, run `bash scripts/launch_aoa_screen.sh` or directly full eval. Best order for speed:
   - first run AoA screen to see whether the developmental schedule creates positive raw AoA;
   - then run `bash scripts/launch_devcurr_eval.sh` for official nine-column result regardless, because NLP/Reading tradeoffs decide whether it is a SOTA-capable mechanism.
3. Interpret only same-seed deltas:
   - `qwen_devcurr_firstpass_seed43022 - qwen_clean_aligned_seed43022`
   - `qwen_devcurr_firstpass_seed43122 - qwen_clean_aligned_seed43122`
4. If AoA improves without NLP collapse, immediately run the source-balanced easy/hard follow-up to isolate source-block vs lexical difficulty.
5. If AoA is neutral or negative, inspect the AoA screen's checkpoint surprisal curves and then test a smoother source-balanced schedule rather than abandoning the clean-Qwen same-window principle.

No final claim: aoa safety audit and route is an active mechanism experiment toward SOTA, not a completed result.
