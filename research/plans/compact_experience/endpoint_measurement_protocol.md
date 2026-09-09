# mask endpoint full eval summary endpoint measurement protocol

Scientific purpose: the cluster continuation noaoa result/045 tail-restart signal peaked at an intermediate tail checkpoint in the earlier untouched-control run, so mask endpoint full eval summary must not hard-code `chck_100M`. The proper order is:

1. Evaluate the four eligible restart endpoints `chck_85M`, `chck_90M`, `chck_95M`, `chck_100M` on the no-AoA columns only: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading.
2. Freeze the endpoint with the highest no-AoA equal7 before measuring SuperGLUE or AoA.
3. For the selected endpoint, run SuperGLUE on exactly that endpoint model path.
4. For AoA, use an endpoint-consistent model root: true parent checkpoints through 80M, true trained tail checkpoints up to the selected endpoint when those names exist, and endpoint-plateau weights for required standard names after the selected endpoint. This blocks future-trained weights from entering an earlier endpoint's acquisition trajectory.

Important scientific caveat:

- This design solves future-weight contamination, but for `<100M` endpoints the AoA is an endpoint-frozen, official-code-compatible measurement, not an ordinary complete training-ladder measurement.
- Freezing later standard names changes the sigmoid fit used by the local AoA evaluator. It is an interpretable sensitivity/selection measurement, not proof that the official leaderboard would accept the same repeated-weight ladder.
- Therefore `<100M` endpoint full scores must be reported as endpoint-frozen official-code-compatible Overall with `submit_ready=false` unless the official server accepts the artifact or an explicit rule permits plateaued checkpoints. A true `chck_100M` endpoint with all real standard checkpoints can be treated as a normal full-ladder local measurement.
- The no-AoA columns are not training signals, but they are explicitly used for endpoint selection and must be reported as post-selection evidence.

Scripts created or patched:

- `scripts/select_tail_endpoint_from_noaoa.py` selects the strongest tail endpoint after `data/tail_restart_noaoa_eval/tail_restart_noaoa_summary.json` exists.
- `scripts/make_endpoint_consistent_ladder.py` builds endpoint-frozen eval roots under `data/tail_endpoint_ladders/` and now refuses mismatched pre-existing symlinks/directories.
- `scripts/tail_endpoint_full_eval.py` registers arbitrary selected endpoints, not only `100M`.
- `scripts/prefill_tail_endpoint_full_eval_from_noaoa.py` injects zero-shot/Reading results for the selected endpoint and records that these columns are endpoint-selection evidence.
- `scripts/score_tail_endpoint_full_eval.py` recomputes corrected Overall and marks `<100M` endpoint-frozen AoA/Overall as not submit-ready.
- `scripts/aoa_local_ckpts_for_model.py` now stores `curve_fitness_record`, `p_value`, `n_words`, score tokenizer path, step counts, expected steps, and coverage/finite-surprisal checks.
- `scripts/full_overall_eval_runner.py` now carries these AoA internals into the per-target payload.
- Mask endpoint-aware analogues exist: `select_mask_endpoint_from_noaoa.py`, `make_mask_endpoint_ladder.py`, `mask_endpoint_full_eval.py`, `prefill_mask_endpoint_full_eval_from_noaoa.py`, `score_mask_endpoint_full_eval.py`, and `run_selected_mask_endpoint_full_eval.sh`.

Current running work:

- No-AoA scan of the clean-tail restart endpoints, writing to `data/tail_restart_noaoa_eval/`.
- No-AoA scan of the four mask arms into staging `staging/processing/mask_noaoa_eval/` after a first failed attempt used the wrong `staging` path.

Known evidence before downstream scoring:

- `data/run_equivalence/tail_vs_uniform_and_mask_metrics.json` showed `clean_tail_restart_ladder_seed43044` and `mask_uniform_control` are byte-identical at `chck_100M`.
- `staging/processing/tail_uniform_chck90_comparison.json` showed they are also byte-identical at `chck_90M`.
- `data/tail_restart_full_eval/tail_ladder_ready_audit.json` showed the clean-tail restart root has the strict-small AoA names and representative checkpoints load successfully.

Next interpretation:

- If the selected tail endpoint's endpoint-aware complete score is below 41.8, the restart route remains a useful learning-dynamics clue but not SOTA; use the no-AoA profile plus mask arms to understand whether the gain is common rephasing or target redistribution.
- If the selected tail endpoint's endpoint-aware complete score exceeds 41.8, immediately replicate from `qwen_clean_aligned_16k_seed43122` using `scripts/launch_tail_restart_second_seed_training.sh`, after verifying the second seed's exact `chck_80M` exposure and parent ladder.
- Do not call any `<100M` frozen-endpoint result officially submit-ready unless the repeated-weight ladder is accepted by the official evaluation route; for a leaderboard-facing artifact prefer a true full-ladder training run whose selected endpoint and AoA policy are acceptable under the official rules.
