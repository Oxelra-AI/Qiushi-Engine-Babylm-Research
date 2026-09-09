# tail rephase probe and experience utilization route rephase probe interpretation and original-tail replay assets

CPU-only reading. No new training or evaluation was launched here.

## Exact post-80M segment from the legal continuous run
- chck_80M was saved after training step `2024` with cumulative words `80034368`.
- Original row start after that checkpoint: `518144` (step × batch_size = 2024 × 256).
- Tail rows in frozen 100M stream: `129256`.
- Tail words in frozen 100M stream: `19965632`.
- Tail words from original training log after earlier analysis: `19965632`.
- Agreement with 100M-parent words: `True`.
- Tail pass rows: `{9: 64516, 10: 64740}`.
- Tail pass words: `{9: 9965632, 10: 10000000}`.

## tail rephase experiment design restart presentation
- `pool_rows`: `64740`
- `pool_words`: `10000000`
- `pool_tokens_no_specials`: `14664519`
- `restart_token_chunks_per_pass`: `57284`
- `restart_tail_buffer_tokens`: `71`
- `restart_has_padded_tail_chunk`: `True`
- `restart_steps_per_pass`: `224`
- `restart_two_pass_steps_nominal`: `448`
- `restart_words_per_full_chunk_step_approx`: `44689.6166468822`

## Why tail rephase experiment design is only a bounded score probe
| Axis | Original continuous run | tail rephase experiment design restart | Consequence |
|---|---|---|---|
| post_80M_text_order | fixed order from frozen 100M stream after the batch that saved chck_80M | fresh random shuffle of token chunks from the 10M pool for each continuation pass, train_rng_seed=43044 | a score movement cannot by itself be assigned to optimizer-state reset |
| training_unit | one JSONL row is one example, tokenized then truncated/padded to length 256 | all rows are concatenated into a token buffer and split into length-256 chunks | context boundaries, truncation, and source/rewrite adjacency differ from the original run |
| word_accounting | batch word exposure is the sum of row word counts | batch word exposure is approximated from selected token-chunk fraction of the 10M pool | nominal total exposure is close, but row/source composition around each checkpoint differs |
| gradient_step_update | AdamW plus gradient clipping at norm 1.0 | AdamW without gradient clipping | large-update behavior is not identical, especially for base-LR restart |
| learning_rate_schedule | single 2529-step cosine schedule, LR about 1.07e-4 at 80M and decaying to zero by 100M | matched: fresh 20M cosine tail starting near 1.08e-4 with no warmup / base: fresh 20M cosine tail at peak 1e-3 with warmup 0.06 | matched-LR is closer to the inherited COMPACT_EXPERIENCE restart; base-LR is a separate late-schedule perturbation |

## If a restart improves
Before assigning the movement to optimizer-state reset or spending more GPU, run the same restart style on the original post-80M stream segment.
- Replay trainer: `experiments/archive/frontier_consolidation/scripts/original_tail_replay_trainer.py`
- Matched-LR template:
  `PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -B experiments/archive/frontier_consolidation/scripts/original_tail_replay_trainer.py --init_checkpoint experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M --train_file experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl --run_dir experiments/archive/frontier_consolidation/training/runs/original_tail_replay_matched_lr_seed43044 --lr_mode matched --gpu 0 --train_rng_seed 43044`
- Base-LR template:
  `PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -B experiments/archive/frontier_consolidation/scripts/original_tail_replay_trainer.py --init_checkpoint experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M --train_file experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl --run_dir experiments/archive/frontier_consolidation/training/runs/original_tail_replay_base_lr_seed43044 --lr_mode base --gpu 0 --train_rng_seed 43044`

JSON: `experiments/archive/frontier_consolidation/data/rephase_probe_interpretation/rephase_probe_interpretation_and_tail_assets.json`
