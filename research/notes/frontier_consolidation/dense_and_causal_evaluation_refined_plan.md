# dense and causal evaluation refined plan refined evaluation plan after CPU audit and independent_review

## Current compute state

Both H100s are still occupied by the two causal GPT transfer training arms launched in earlier analysis:

- causal compact-view arm, output `training/runs/causal_gpt_compact_seed43022_100M`.
- causal repeat arm, output `training/runs/causal_gpt_repeat_seed43022_100M`.

Do not interrupt these jobs. They are the most direct cross-architecture test of whether compact semantic same-window views transfer outside the DeBERTa MLM coordinate.

## CPU/file findings from dense and causal evaluation refined plan

Artifacts:

- `data/dense_training_trajectory_audit/dense_training_trajectory_audit.{json,md}`
- `data/dense_loss_dynamics_compare/dense_loss_dynamics_compare.{json,md}`
- scripts: `scripts/dense_training_trajectory_audit.py`, `dense_loss_dynamics_compare.py`, `selected_mlm_checkpoint_eval.py`, `selected_causal_checkpoint_eval.py`, `compare_causal_selected_trajectories.py`

Corrections/findings:

1. `scale1p75_seed43122_dense` changes **both** `extra_init_seed` and `train_rng_seed` relative to the protected seed43022 run. It is a joint init+mask-stream robustness test, not pure initialization-only.
2. `scale1p25_seed43022_dense` keeps the protected seed/mask stream (`extra_init_seed=43022`, `train_rng_seed=43023`) and changes only train-time adapter scale from 1.75 to 1.25. It is the clean scale-amplitude contrast.
3. All three compared runs use the same legal 10M compact-view reinvest stream, 16k legal tokenizer, architecture size, 100M word budget, batch size, LR horizon, WWM 0.15, and 2,529 training steps.
4. Training loss still cannot select or explain the useful competence peak: in the reference, rolling MLM loss improves after 82M while cheap7 falls by 100M. Loss-based endpoint choice remains unsupported.
5. Repaired a root-path bug in `batch_trajectory_eval.py` (`parents[3]` selected the wrong root and duplicated a directory prefix) and compiled the evaluator sources with no-bytecode in-memory compilation.

## Refined DeBERTa dense trajectory scoring plan

independent_review verifier correctly pointed out that the originally proposed handpicked subset under-resolves the reference 82M peak/late region. The lowest reliable scoring set is now a **common 2M grid from 70M through 100M** for all three trajectories:

`chck_70M, chck_72M, chck_74M, chck_76M, chck_78M, chck_80M, chck_82M, chck_84M, chck_86M, chck_88M, chck_90M, chck_92M, chck_94M, chck_96M, chck_98M, chck_100M`.

Score with `scripts/selected_mlm_checkpoint_eval.py`, not the full range evaluator, so the grid is explicit and common even though the protected reference has 1M-spaced checkpoints.

Suggested commands when a GPU is free:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder \
  --gpu 0 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M \
  --label scale1p75_seed43022_reference \
  --endpoints chck_70M chck_72M chck_74M chck_76M chck_78M chck_80M chck_82M chck_84M chck_86M chck_88M chck_90M chck_92M chck_94M chck_96M chck_98M chck_100M

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_seed43122_dense100M \
  --gpu 0 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M \
  --label scale1p75_seed43122_dense \
  --endpoints chck_70M chck_72M chck_74M chck_76M chck_78M chck_80M chck_82M chck_84M chck_86M chck_88M chck_90M chck_92M chck_94M chck_96M chck_98M chck_100M

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p25_seed43022_dense100M \
  --gpu 1 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M \
  --label scale1p25_seed43022_dense \
  --endpoints chck_70M chck_72M chck_74M chck_76M chck_78M chck_80M chck_82M chck_84M chck_86M chck_88M chck_90M chck_92M chck_94M chck_96M chck_98M chck_100M
```

Interpretation after scoring:

- Compare `scale1p25_seed43022_dense` primarily to `scale1p75_seed43022_reference`, because they share seed/mask stream and differ in adapter scale.
- Compare `scale1p75_seed43122_dense` to the same reference as a joint init+mask-stream robustness test.
- Use the same trajectory summaries across runs: best exposure, contiguous near-best band using a fixed tolerance, mean 78–86M, mean 90–100M, family/column maxima, and sensitivity with volatile columns removed.
- Do not infer a broad principle from a one-column spike. The family vector matters.
- If attribution to initialization versus mask stream becomes essential, a one-factor seed control is needed; the current cross-seed run cannot separate them.

## Causal GPT transfer evaluation after training delivery

The causal GPT experiment remains orthogonal and more directly tests cross-architecture transfer of compact semantic views. After both causal training arms complete, first score a small common selected grid:

`chck_20M, chck_50M, chck_70M, chck_82M, chck_90M, chck_100M`.

Use:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_causal_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/causal_gpt_compact_seed43022_100M \
  --label causal_compact --gpu 0 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_causal_eval_compact \
  --endpoints chck_20M chck_50M chck_70M chck_82M chck_90M chck_100M

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_causal_checkpoint_eval.py \
  --run-dir experiments/archive/frontier_consolidation/training/runs/causal_gpt_repeat_seed43022_100M \
  --label causal_repeat --gpu 1 \
  --out-dir experiments/archive/frontier_consolidation/data/selected_causal_eval_repeat \
  --endpoints chck_20M chck_50M chck_70M chck_82M chck_90M chck_100M

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compare_causal_selected_trajectories.py \
  --compact experiments/archive/frontier_consolidation/data/selected_causal_eval_compact/selected_causal_trajectory.json \
  --repeat experiments/archive/frontier_consolidation/data/selected_causal_eval_repeat/selected_causal_trajectory.json \
  --out-dir experiments/archive/frontier_consolidation/data/causal_compact_repeat_contrast
```

Carry the earlier analysis exposure fact: compact has +0.2216% active token positions at equal legal words. A positive causal result needs broad compact-over-repeat movement across multiple checkpoints/families and magnitude plausibly beyond that small exposure asymmetry. A null result bounds this decoder-causal coordinate only; it does not erase the DeBERTa compact-view evidence.

## What not to do

- Do not reopen alpha-scale, anchor-confidence, retention-KL, coherence-margin, edit-state correspondence, exact-swap innovation masking, graph-packet synthetic views, or IG target-allocation routes under new names.
- Do not select checkpoints by MLM loss.
- Do not run all 0–100M dense evaluations unless the common 70–100M grid reveals a reason to expand earlier.
