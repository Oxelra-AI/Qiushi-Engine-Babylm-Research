# Restart and masking: pre-result interpretation and audits

## Mechanism interpretation

The cluster-continuation no-AoA result has the following interpretation:

- The supported intervention is the **combination** of loading clean-Qwen `chck_80M`, discarding optimizer/scheduler state, and running a fresh low-LR cosine tail. Current evidence does not yet isolate moment reset from LR schedule, data-order reshuffle, or stochastic masking.
- Clean-tail and the masking `uniform + token_count` arm should be near-equivalent under the same seed; their difference is an empirical noise/implementation reference for judging evidence-visible additions.
- Priority masking arms remain recipe comparisons, not perfect single-variable interventions, unless actual masked-token ratios and original-priority strata are measured. The training script's `high_priority_fraction` is not semantically comparable across modes because it uses mode-transformed priorities; the CPU audit below gives original-base-priority strata.
- The strict-small AoA runner uses `chck_1M..chck_9M` and `chck_10M..chck_100M` (official repo `evaluation_pipeline/AoA_word/eval_util.py` lines 44-47); 85M/95M checkpoints are useful for trajectory but ignored by AoA.

## CPU audit results

### Clean-tail vs mask-uniform first batch

`data/mask_sampler_audit/masking_equivalence_and_sampler_audit.json` verifies that the clean-tail WWM implementation and `evidence_visible_continuation_trainer.py` in `uniform + token_count` mode are exactly equivalent on the first shuffled batch under seed 43044:

- `masked_inputs_equal: true`
- `labels_equal: true`
- `label_disagreements: 0`
- `masked_input_disagreements: 0`
- both have `9673` masked tokens.

This supports interpreting `mask_uniform_control` as a repeat/noise reference for the clean-parent tail restart, not a separate masking mechanism.

### Sampler original-priority strata

The same audit sampled 2048 chunks and measured original base surface priority for every arm:

- uniform: token-budget ratio 1.0, selected high-evidence fraction 0.4422, mean selected priority 1.0236, mean wordpiece length 1.4765.
- evidence-visible: token-budget ratio 1.0, selected high-evidence fraction 0.6312, mean selected priority 1.3197, mean wordpiece length 1.6145.
- random-priority: token-budget ratio 1.0, selected high-evidence fraction 0.4374, mean selected priority 1.0170, mean wordpiece length 1.4539.
- inverse-priority: token-budget ratio 1.0, selected high-evidence fraction 0.2608, mean selected priority 0.7699, mean wordpiece length 1.2814.

Thus the route really changes which words receive loss while matching loss-bearing subword-token totals in the sampled audit. It also intentionally changes selected word count and wordpiece-length distribution, so result interpretation should say "priority-weighted WWM recipe" unless a stricter paired RNG/length-matched sampler is later built.

### Continuation word exposure

`data/exposure_audit/continuation_word_exposure_audit.json` reconstructs token chunks from the clean 10M pool and simulates the same seed/order/stop rule. With parent actual exposure `80,003,682` and requested continuation budget `19,996,318`:

- chunk-ratio estimated continuation words: `19,996,196`;
- any-token word-instance upper count: `19,996,059`;
- complete-token word-instance lower count: `19,996,041`;
- parent + any-token upper count: `99,999,741`, below the 100M exposure target.

This does not turn token chunks into exact official whitespace passes, but it is a stronger evidence carrier than a path string: the restart and mask execution plan continuation is conservative under tokenizer-offset word-instance accounting.

## Active running task

The restart/masking training comparison had started with the following configuration:

- clean-parent tail ladder: `training/runs/clean_tail_restart_ladder_seed43044`;
- masking arms: `mask_uniform_control`, `mask_evidence_visible`, `mask_random_priority`, `mask_inverse_priority`;
- parent actual exposure `80,003,682`, continuation budget `19,996,318`, seed 43044.

Do not edit `scripts/evidence_visible_continuation_trainer.py` while this task is running, because the launcher invokes it sequentially for later arms. After training finishes, verify each metrics file and run:

1. `python experiments/archive/compact_experience/scripts/verify_tail_ladder_ready.py`
2. `bash experiments/archive/compact_experience/scripts/launch_tail_restart_noaoa_eval.sh && python experiments/archive/compact_experience/scripts/summarize_tail_restart_noaoa.py`
3. `bash experiments/archive/compact_experience/scripts/launch_mask_eval.sh && python experiments/archive/compact_experience/scripts/summarize_mask_eval.py`
4. if the tail ladder is load-ready, `bash experiments/archive/compact_experience/scripts/launch_tail_restart_full_eval.sh` for SuperGLUE + AoA + all columns.
