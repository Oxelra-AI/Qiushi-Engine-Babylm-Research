# masking curriculum 4m eval follow-up — regime-correct masking-curriculum 100M screen

## Background task
- Label: `masking curriculum 4m eval regime-correct 100M-exposure 4-arm masking-curriculum screen on 2xH100`
- Launch script: `experiments/archive/compact_experience/scripts/launch_100M_screen.sh`
- Status check in masking curriculum 4m eval showed the task running and both wave-1 arms launched at 06:43:33 UTC:
  - GPU0: `wwm_fixed_100M_seed43`
  - GPU1: `wwm_to_token_100M_seed43`
- Wave 2 will start automatically after wave 1:
  - GPU0: `amlm_hard_100M_seed43`
  - GPU1: `amlm_hard_switch_100M_seed43`

## Scientific design
This is not a leaderboard recipe copy. It holds the INITIAL_MODEL_STUDIES/COMPACT_EXPERIENCE DeBERTa-v2 8x480 + baseline16k + AdamW recipe fixed and tests prediction-control mechanisms in the regime where they are designed to act:

- official Strict-Small pool: 10M words
- exposure: 100M words = 10 epochs
- sequence-length schedule: `0.0:64,0.5:128,0.8:256`
- WWM→token switch fraction: `0.7`, which means 70M words = end of epoch 7
- AMLM-hard: nominal mask probability decay `0.30 -> 0.15`, with per-token difficulty weighting
- baseline: fixed WWM 0.15 for all 10 epochs

This corrects the masking curriculum 4m eval/masking curriculum training done 4M screen's main weakness: those 4M runs switched after only 2.8M words in a partial first pass and therefore did not test the live leader's 7-of-10-epoch granularity curriculum.

## Exact run dirs expected
- `experiments/archive/compact_experience/training/runs/wwm_fixed_100M_seed43`
- `experiments/archive/compact_experience/training/runs/wwm_to_token_100M_seed43`
- `experiments/archive/compact_experience/training/runs/amlm_hard_100M_seed43`
- `experiments/archive/compact_experience/training/runs/amlm_hard_switch_100M_seed43`

Each should contain:
- `train_stdout.log`, `train_stderr.log`
- `training_log.jsonl`
- `scientific_metrics.json`
- `dynamics_traces.jsonl`
- `hf_model/chck_10M`, `chck_20M`, ..., `chck_100M`

Note: this screen uses `--checkpoint_words 10000000` for epoch-level mechanism tracking. A final official run should still satisfy the official intermediate-checkpoint pattern if required by the current submission rules (1M increments through 10M, then 10M increments through 100M). This experimental screen prioritizes the epoch-level switch trajectory and disk/runtime economy.

## Immediate masking curriculum 100M eval actions after collect
1. Inspect the completed 100M screen measurements when available.
2. Verify all four runs completed and `hf_model/chck_100M` exists with `model.safetensors`, `config.json`, `tokenizer.json`, and tokenizer config files.
3. Verify `scientific_metrics.json` fields:
   - `word_exposure == 100000000`
   - `loss_first`, `loss_last`
   - `masking_curriculum`
   - `n_amlm_updates` (0 for WWM arms, large for AMLM arms)
   - checkpoints include `chck_10M` ... `chck_100M`
4. Parse `dynamics_traces.jsonl` with the correct schema keys:
   - `mask_mode_at_checkpoint`
   - `mask_prob_at_checkpoint`
   - `effective_mask_rate_mean`
   - `prediction_entropy_mean`
   - `loss_by_freq_band.{low,mid,high}.{mean,last,n}`
   - `accuracy_by_freq_band.{low,mid,high}.{mean,last,n}`
   - `amlm_weight_stats.mean` for AMLM arms
5. Evaluate the four `chck_100M` checkpoints using the repaired masking curriculum 4m eval/cs 4m residualized eval fast-eval invocation pattern:
   - cwd: `experiments/archive/initial_model_studies/repos/babylm-eval/strict`
   - BLiMP: `evaluation_data/fast_eval/blimp_fast`
   - Supplement: task `blimp`, path `evaluation_data/fast_eval/supplement_fast`
   - EWoK: `evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast`
   - Entity: `evaluation_data/fast_eval/entity_tracking_fast`
   - COMPS: `evaluation_data/full_eval/comps`
   - GlobalPIQA parallel/nonparallel fast
   - Reading fast
6. For mechanism timing, evaluate at least `chck_60M`, `chck_70M`, `chck_80M`, and `chck_100M` for `wwm_fixed`, `wwm_to_token`, and `amlm_hard_switch`. This locates whether any transfer change appears before the switch, at the switch, or after token-level prediction begins.
7. Interpret effects against the masking curriculum 4m eval 4M result and the cs 4m residualized eval random-reference spread. A single-seed 100M positive still needs replication before scaling into a SOTA route, but it can identify which prediction-control mechanism deserves multi-seed or official full-column evaluation.

## What the masking curriculum 4m eval 4M screen already showed
Valid masking curriculum 4m eval fast evaluation at 4M:
- `wwm_to_token_minus_wwm_fixed`: weighted proxy `-0.006`
- `amlm_hard_minus_wwm_fixed`: `+0.034`
- `amlm_hard_switch_minus_wwm_fixed`: `+0.122`

These deltas are small and comparable to cs 4m residualized eval random-reference spread. The 4M dynamics showed AMLM and switch mechanics working, but frequency-band top-1 accuracy was degenerate (mid/low accuracy 0.0), so the 4M screen cannot explain transfer. The 100M screen exists to test the same mechanisms in the correct epoch regime and with less degenerate learning dynamics.
