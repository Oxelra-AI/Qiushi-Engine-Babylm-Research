# restart and mask execution plan restart/rephasing mechanism next wave

## What restart and mask execution plan can establish

The active restart and mask execution plan training compares:

- clean-tail restart with parent ladder: exact clean-Qwen `chck_80M` weights, fresh AdamW, fresh no-warmup cosine tail LR, uniform WWM, same clean pool, seed 43044;
- masking arms from the same parent and fresh-state schedule: uniform, evidence-visible, random-priority, inverse-priority.

Because `masking_curriculum_trainer.py` only saved HF model/tokenizer checkpoints (`save_hf_checkpoint` lines 566-569) and no optimizer/scheduler state, the saved checkpoints cannot supply the original 80M Adam moments or scheduler age from disk. Thus current evidence can support **fresh optimizer/LR rephasing** as a combined intervention, not a clean separation of moment reset from LR restart.

## Immediate post-training analysis

Required checks after the restart/masking comparison completes:

1. Verify all five training runs have `scientific_metrics.json` and expected `chck_85M`, `chck_90M`, `chck_95M`, `chck_100M` tail checkpoints.
2. Run `scripts/compare_tail_and_uniform_runs.py` to check whether the clean-tail and `mask_uniform_control` are exact or near-identical at `chck_100M`. Exact first-batch masking equivalence is already established in `data/mask_sampler_audit/masking_equivalence_and_sampler_audit.json`; final equality or divergence gives an implementation/noise reference.
3. Run `scripts/verify_tail_ladder_ready.py`; if ready, run no-AoA tail trajectory and full eval:
   - `bash scripts/launch_tail_restart_noaoa_eval.sh && python scripts/summarize_tail_restart_noaoa.py`
   - `python scripts/prefill_tail_full_eval_from_noaoa.py`
   - `python scripts/tail_restart_full_eval.py --target clean_tail_restart_ladder_100M --gpu 0 --columns SuperGLUE AoA --force`
   - `python scripts/score_tail_restart_full_eval.py`
4. Run mask no-AoA screen and summarize:
   - `bash scripts/launch_mask_eval.sh && python scripts/summarize_mask_eval.py`
5. If a mask arm is clearly better than uniform, attach the parent ladder and run full eval for that arm:
   - `python scripts/attach_parent_ladder_to_continuation.py --run_dir training/runs/mask_<arm>`
   - `python scripts/prefill_mask_full_eval_from_noaoa.py --arm <arm> --endpoint chck_100M`
   - `python scripts/mask_full_eval.py --arm <arm> --target mask_<arm>_100M --gpu 0 --columns SuperGLUE AoA --force`
   - `python scripts/score_mask_full_eval.py`

## Feasible next experiments from existing checkpoints

If restart survives full measurement but remains below SOTA, the next GPU wave should vary only one rephasing variable at a time from the same clean 80M parent and same clean pool:

- LR amplitude: `0.5×`, `1.0×`, `1.5×`, `2.0×` the current continuation LR (`0.00010752`). Use the same no-warmup cosine shape and save `chck_85/90/95/100M`.
- LR shape at matched peak: constant, linear decay, cosine decay, short warmup-cosine (2% continuation warmup), and pulse-cool (`80-82M` higher LR then cooling). Report `sum(lr)` and `sum(lr^2)` so displacement/noise energy are interpretable.
- Multi-cycle: restart at 80M only versus restart at 80M and a half-amplitude LR-only pulse at 90M. Do not assume repeated resets help; previous SWA and masking curricula showed small-route reversals.
- Targeted-mask timing if evidence-visible helps: release-only (`80-85M`), targeted→uniform, uniform→targeted, and all-through. The most plausible non-destructive version is targeted→uniform: use early rephasing plasticity to reinforce entity/number/relation-object prediction and then cool under broad WWM.

## What requires rebuilding a state-capturing ancestry

A true causal split of optimizer moment reset from LR rephase requires a fresh clean-Qwen lineage that saves at 80M:

- model weights;
- Adam first and second moments;
- Adam step counters/bias-correction age;
- scheduler state;
- data order and batch/mask RNG streams.

Only then can one compare preserve-moments vs reset-moments vs reset-m only vs reset-v only at the same LR and exact data/mask stream. That is scientifically valuable but expensive; it should be launched only if restart and mask execution plan confirms the restart/rephasing family remains the strongest route after full AoA/SuperGLUE measurement.
