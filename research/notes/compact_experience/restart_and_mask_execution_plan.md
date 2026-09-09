# restart and mask execution plan restart + masking execution record

## Why this work matters

cluster continuation noaoa result closed the natural true-cluster construction but exposed a stronger clue:
E3 anchor-repeat and E4 untouched-tail differed by only 0.0176 Overall while both
were about +0.2 Overall above clean-Qwen. Their shared intervention was not the
cluster content but loading clean-Qwen `chck_80M`, discarding the original optimizer
state, and continuing with a fresh AdamW/cosine tail phase. This makes tail
optimizer/LR rephasing a real evidence-backed mechanism candidate, not a fallback
after masking fails.

The active execution work therefore has two simultaneous purposes:

1. produce a clean-parent untouched-restart lineage with an AoA-compatible
   checkpoint ladder, so the optimization signal can receive a complete
   official-style nine-column measurement rather than provisional AoA=0;
2. run the matched evidence-visible masking continuations from the same clean
   parent, including a uniform arm, so common restart gains can be separated from
   mask-target-selection gains.

## Key invariants

- Parent model root: `experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model`.
- Parent start: `chck_80M`, actual cumulative word exposure `80,003,682` in parent metrics.
- Continuation pool: `data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl`, sha256 `e0a3cdc20e39f2715fbdb0cfbc6c4aff61d51924c480a982049878272c5690b3`.
- Continuation budget used by restart and mask execution plan launchers: `19,996,318`, so parent actual + requested continuation target = `100,000,000` without intentional overexposure.
- Training signal remains WWM MLM on the clean Qwen-aligned pool; no BabyLM downstream labels/items, no official AoA/CDI words, no child curves, and no leaderboard scores are used for training.

## Scripts created/repaired in restart and mask execution plan

- `scripts/untouched_restart_ladder_trainer.py`: clean-parent untouched restart trainer. It symlinks parent checkpoints `chck_1M..chck_80M` into the output `hf_model` and trains new `chck_85M`, `chck_90M`, `chck_95M`, and final target `chck_100M`, recording actual exposure.
- `scripts/evidence_visible_continuation_trainer.py`: repaired to accept `--start_word_exposure` and to write target-labeled `chck_100M` while recording actual continuation and total exposure. Masked subword-token budget is still controlled.
- `scripts/launch_parallel_tail_restart_and_mask_training.sh`: launches the clean-tail ladder on GPU0 and matched mask arms across GPU0/GPU1.
- `scripts/launch_mask_eval.sh` and `scripts/summarize_mask_eval.py`: no-AoA trajectory measurement and summary for uniform, evidence-visible, random-priority, and inverse-priority masking arms.
- `scripts/tail_restart_full_eval.py`, `scripts/score_tail_restart_full_eval.py`, and `scripts/launch_tail_restart_full_eval.sh`: full official-style evaluation wrapper/scorer for the clean-parent tail restart.

## Interpretation targets

- If `mask_uniform_control` rises near the cluster continuation noaoa result E3/E4 provisional region, the common restart/rephasing mechanism is confirmed independently of cluster/masking content.
- If `mask_evidence_visible` beats uniform plus both random/inverse controls on no-AoA equal7 and on the EWoK+COMPS+GlobalPIQA recovery profile without losing preserve columns, mask-target selection is additive to restart.
- If the clean-tail full ladder has official AoA=0 but no-AoA/SuperGLUE gains survive, the mechanism is still useful but below SOTA unless deeper optimization pushes another ~0.25 Overall. If AoA becomes positive, it can materially change the official coordinate.
