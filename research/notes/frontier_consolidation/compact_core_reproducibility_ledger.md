# Compact-view core: training and reproducibility record

## Frozen data

- Metadata: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json`
- Training JSONL: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_core_neutral_100M.jsonl`
- Training JSONL sha256: `5380686975f6126eb1d9d88dfd91dea7a511d10a8e5a0cc45f2cc394a2853604`
- Data status: exact 10M-word pool and exact 10-pass 100M training file; all inherited `qwen_pair_packed` rows preserved; 2,647 non-Qwen official rows / 423,520 words held out; common filler 9,576,480 words.
- Compact-core changed block: 10,094 shared core sources; source words 218,542; rewrite words 135,403; compact source+view pair words 353,945; neutral clean-Qwen top-up inside changed block 69,575 words; weighted rewrite/source ratio 0.619574.
- Risk-hard source selection rejected source patterns highlighted by semantic review: long parenthetical/deictic/heading-chain/title-apposition/probability-hedge-like structures.

## Training recipe

- Trainer: `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py`
- Tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model`
- Model: DeBERTa-v2 masked LM, hidden size 480, 8 layers, 8 heads, FFN multiplier 4; parameter count 34,467,424; vocab size 16,384.
- Sequence geometry: fixed seq256, batch 256, no sequence-length schedule.
- Masking: fixed whole-word masking, mask probability 0.15 throughout.
- Optimizer: AdamW lr 0.001, weight decay 0.01, cosine schedule with warmup_fraction 0.06.
- Exposure: max_word_exposure 100,000,000; checkpoint_words 1,000,000; num_workers 0.
- First seed: seed 43, extra_init_seed 43022, train_rng_seed 43023.
- First-seed run dir: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_core_neutral_16k_seed43022`.
- First-seed training result: 2,527 steps, loss_first 9.818527221679688, loss_last 2.4529898166656494, 100 saved checkpoints from `chck_1M` to `chck_100M`.

## Exact first-seed command pattern

See `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_core_neutral_16k_seed43022/train_command.json` for the recorded command and preflight hash. The training command passes the frozen metadata through `--example_jsonl_meta`.

## Independent replication preflight

Script prepared: `experiments/archive/frontier_consolidation/scripts/train_density_arm_seed.py`.

Dry-run with sha256 check succeeded for second seed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/train_density_arm_seed.py \
  --arm cleanqwen_fineweb_compact_view_core_neutral \
  --gpu 0 \
  --extra-init-seed 43122 \
  --train-rng-seed 43123 \
  --check-hash \
  --dry-run
```

Default second-seed run dir would be `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_core_neutral_16k_seed43122`.

## Evaluation evidence so far

- Fast no-AoA task-family screen: `experiments/archive/frontier_consolidation/data/density_noaoa_eval_compact_core/density_noaoa_eval_summary.json`.
- Compact view core beat compact repeat core by +2.1271 equal7 and +2.1614 equal7 with full Entity. Component deltas: Supplement +7.60, EWoK +2.91, Entity_full +2.43, COMPS +1.09, GlobalPIQA_mean +1.03, Reading +0.23, BLiMP -0.16.
- Full official-compatible evaluator wrapper: `experiments/archive/frontier_consolidation/scripts/full_eval_density_compact.py`.
- Full evaluation launched; output root `experiments/archive/frontier_consolidation/data/density_full_eval`.

## Trainer-exact token/WWM exposure check

- Evidence: `experiments/archive/frontier_consolidation/data/trainer_exact_token_exposure_measurement/trainer_exact_token_exposure_summary.json` and `research/notes/frontier_consolidation/trainer_exact_token_exposure_measurement.md`.
- At equal 10M words, compact_view_core has 14,391,511 visible candidate tokens versus compact_repeat_core 14,371,158: +20,353 (+0.142%).
- Compact_view_core has 9,785,327 visible WWM groups versus compact_repeat_core 9,785,594: -267 (-0.003%).
- Therefore the large task-family gain is not explained by a raw WWM-group exposure advantage. The small candidate-token difference should still be mentioned in interpretation.

## Current rule for next work

Do not modify the corpus or recipe before the protected compact-core candidate has (1) complete official-compatible evaluation including SuperGLUE and AoA and (2) independent seed replication. The reinvestment arm is an extension; negative reinvestment results should not erase the compact-core finding.
