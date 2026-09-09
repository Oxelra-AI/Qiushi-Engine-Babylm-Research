# earlier analysis inverse-priority full evaluation and replication state

## Current decisive measurement

The strongest no-AoA mask endpoint full eval summary profile is `mask_inverse_priority`:

- `chck_95M` (endpoint-frozen scientific measurement): equal7 = 43.86571428571428. It needs SuperGLUE + AoA leaderboard units > 69.1400 to exceed the visible 41.8 leader.
- `chck_100M` (true standard endpoint, no plateau caveat): equal7 = 43.705. It needs SuperGLUE + AoA leaderboard units > 70.2650 to exceed the visible 41.8 leader.

earlier analysis built and prefilled endpoint roots for both:

- `data/mask_endpoint_ladders/mask_inverse_priority_endpoint_95M/`
- `data/mask_endpoint_ladders/mask_inverse_priority_endpoint_100M/`
- per-target payloads under `data/mask_endpoint_full_eval/per_target/mask_inverse_priority_95M.json` and `...100M.json` have the seven no-AoA columns prefilled from the frozen endpoint trajectory.

The endpoint evaluation had started with the following configuration:

```bash
bash experiments/archive/compact_experience/scripts/launch_inverse_priority_full_eval_parallel.sh
```

It evaluates SuperGLUE and AoA in parallel:

- GPU0: `inverse_priority` `chck_95M`
- GPU1: `inverse_priority` `chck_100M`

The first attempt failed before evaluation because of inconsistent working-directory-relative paths. A corrected attempt had started, but completed measurements were pending in this record.

## Uniform mechanistic reference

The common optimizer/LR rephasing reference is byte-identical between `clean_tail_restart_ladder_seed43044` and `mask_uniform_control` at `chck_90M` and `chck_100M` (mask endpoint full eval summary). It should be kept as the mechanistic reference: inverse-priority is not compared against an unrelated run, but against the exact same restart/update schedule with unchanged uniform WWM target distribution.

## Second-seed replication assets

earlier analysis verified that the correct second clean-Qwen parent is:

- `training/runs/qwen_clean_aligned_16k_seed43122/hf_model`
- `chck_80M` actual cumulative word exposure = 80,003,682
- same continuation budget as seed43022: 19,996,318 target words, yielding conservative total target 99,999,? with actual continuation under the 100M limit by the same accounting.

The previously prepared second-seed tail launcher had the stale parent path `qwen_clean_aligned_16k_seed43122`; earlier analysis repaired it to `qwen_clean_aligned_16k_seed43122`.

Launch-ready second-seed inverse-vs-uniform replication scripts:

- Training: `scripts/launch_seed43122_inverse_uniform_training.sh`
  - run dirs: `training/runs/mask_uniform_control_seed43144` and `training/runs/mask_inverse_priority_seed43144`
  - parent: `qwen_clean_aligned_16k_seed43122/hf_model/chck_80M`
  - train file: `data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl`
  - seed: 43144
  - mask budget: token_count
- No-AoA evaluation: `scripts/launch_seed43122_inverse_uniform_noaoa_eval.sh`
- No-AoA summarizer: `scripts/summarize_seed43122_inverse_uniform_noaoa.py`
- Generic endpoint full-eval utilities for replication endpoints:
  - `scripts/make_custom_mask_endpoint_ladder.py`
  - `scripts/prefill_custom_endpoint_full_eval_from_noaoa.py`
  - `scripts/custom_endpoint_full_eval.py`
  - `scripts/score_custom_endpoint_full_eval.py`
  - `scripts/run_seed43122_inverse_endpoint_full_eval.sh`

Compilation and shell syntax checks for these new scripts had not completed in this record; their syntax validity remained unverified.

## Next action after full-eval result

If either seed43022 inverse-priority endpoint crosses or is very near the visible leader, launch the second-seed inverse-vs-uniform replication immediately rather than opening a new portfolio:

```bash
bash experiments/archive/compact_experience/scripts/launch_seed43122_inverse_uniform_training.sh
bash experiments/archive/compact_experience/scripts/launch_seed43122_inverse_uniform_noaoa_eval.sh
```

Then full-evaluate the second-seed true `chck_100M` inverse endpoint, and if the no-AoA curve again favors `chck_95M`, also run the endpoint-frozen `chck_95M` scientific measurement:

```bash
bash experiments/archive/compact_experience/scripts/run_seed43122_inverse_endpoint_full_eval.sh chck_100M 0
# optional scientific endpoint if no-AoA supports it:
bash experiments/archive/compact_experience/scripts/run_seed43122_inverse_endpoint_full_eval.sh chck_95M 1
```

If SuperGLUE collapses or AoA becomes strongly negative enough that neither seed43022 endpoint approaches the leader, use the payloads to identify the collapsed column profile before changing route. Do not infer failure from training loss; inverse-priority had low MLM loss but the decision depends on official-compatible task columns.
