# compliant endpoint evaluation policy and driver — compliant endpoint evaluation policy and post-delivery driver

## Current scientific state

The compliant-tokenizer `compact_view_reinvest` retrain remains the only current endpoint in this experiment that can answer the submission question.  The old-tokenizer `compact_view_reinvest` seed43022 result at Overall `42.0331347900748` remains important scientific evidence for the density-reinvestment mechanism, but it is not an end-to-end Strict-Small submission result because the tokenizer provenance exceeded the 10M-word budget.

Active retrain directories were not inspected, and no model-quality result is inferred. The following training comparisons remained pending:

- Compliant-tokenizer reinvest retraining: submission-relevant endpoint.
- Clean-Qwen fixed-tokenizer control retraining: scientific control.



## Evaluation Policy

The pending reinvest model is now the only route in this experiment that can answer the submission question.  Therefore partial columns may be used to conserve compute, but a projected partial score must not substitute for the full official result.  Remaining evaluation should stop only if the completed columns prove that even favorable unfinished columns cannot reach the decision target; otherwise finish SuperGLUE, the official 8,005-row AoA trajectory, and pristine nine-column collation.

I encoded this as a hard-upper-bound rule rather than an expected-score projection:

- Missing columns are bounded above by `100.0` each.
- Evaluation stops only when `known_sum + 100 * num_missing < 9 * decision_target`.
- A missing or non-official AoA placeholder remains missing; it is **not** counted as a real zero.
- A genuine `official_aoa_done` result with `row_count_values=[8005]` and finite surprisals may count as `0.0`, because prior valid endpoints can have official AoA exactly zero.

Default decision target in the helper/driver is the visible valid leaderboard target `41.8`, because this is the submission-SOTA question.  The helper also computes the frozen old-tokenizer reference `42.0331347900748`; pass `--decision-target 42.0331347900748` if the immediate scientific decision is whether the compliant endpoint can exceed the old non-submission reference.  In no case should an average/expected projection replace the full official result while the hard upper bound still leaves the endpoint viable.

## Files changed or added

### Continuation-policy helper

Replaced `experiments/archive/frontier_consolidation/scripts/project_compliant_eval_continuation.py`.

Important repairs:

- Removed stale duplicate `interpretation` output.
- Does not count `not_official_missing_checkpoints` AoA as zero.
- Emits `COMPLIANT_EVAL_CONTINUATION_POLICY` records.
- Computes hard upper bounds for the visible leader, frozen old-tokenizer reference, and user-specified `--decision-target`.
- Recommends continuation for the reinvest endpoint unless the hard upper bound proves target unreachable.

Preflight exercise outputs:

- `experiments/archive/frontier_consolidation/data/compliant_full_eval/complianttok_reinvest_seed43022_preflight_continuation_policy.json`
- `experiments/archive/frontier_consolidation/data/compliant_full_eval/complianttok_clean_qwen_seed43022_preflight_continuation_policy.json`

On preflight payloads, all columns are missing and the helper correctly recommends continuing the reinvest endpoint and treating the clean-Qwen arm as a scientific control.

### Post-delivery driver

Created `experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py`.

Run this only **after** the runtime delivers a terminal retrain result; it is not a wait/poll script.  It implements:

1. Run `inspect_compliant_retrain.py` for the chosen arm.
2. Stop immediately if the delivered run is not a complete compliant 100M retrain.
3. Run cheaper columns first: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA parallel/nonparallel, and Reading.
4. Run the hard-upper-bound continuation policy.
5. If still viable and not `--cheap-only`, run SuperGLUE and official AoA.
6. Run `stage_pristine_collate.py` for the authoritative current-coordinate scalar.

Dry-run records:

- `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/reinvest_postdelivery_driver.json`
- `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/clean_qwen_postdelivery_driver.json`

I fixed the driver after dry-run inspection so it now reads the real pristine-collator summary path and keys:

- summary path: `.../pristine_collate_<target>_summary.json`
- Overall key: `score_summary.official_overall.Overall`
- task-score key: `score_summary.official_overall.scores`

### Regression test for the continuation rule

Created and ran `experiments/archive/frontier_consolidation/scripts/projection_policy_regression.py`.

Output: `experiments/archive/frontier_consolidation/data/projection_policy_regression/projection_policy_regression.json`.

All checks passed:

- `placeholder_aoa_kept_missing: true`
- `placeholder_recommends_continue: true`
- `hard_impossible_stops: true`
- `complete_aoa_counted_known: true`

This directly tests the evaluation policy: non-official AoA cannot masquerade as a completed zero, and a hard stop occurs only when even missing=100 cannot reach the target.

### Path-root robustness for possible unchanged relaunch

Patched:

- `experiments/archive/frontier_consolidation/scripts/train_compliant_model.py`
- `experiments/archive/frontier_consolidation/scripts/wait_and_train_compliant.py`

The patches derive `USER_ROOT` from `__file__` and run subprocesses with `cwd=USER_ROOT`.  This is only filesystem robustness for a possible future unchanged relaunch after timeout/failure; it does **not** change data, tokenizer, seeds, architecture, objective, optimizer, batch geometry, or exposure.

AST parsing passed for both wrappers after repair.  Dry-run preflights with `--check-hash` passed for both arms:

- reinvest 100M SHA256 `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- clean-Qwen 100M SHA256 `728192f8e8c5855aaa52a6b6940ff3a4fd6aa8f87c98aca4ff018dbc04cb0345`
- same frozen recipe: DeBERTa-v2 8×480, seed 43 / init 43022 / train RNG 43023, seq256, batch256, WWM 0.15, AdamW lr 0.001, warmup 0.06, weight decay 0.01, 1M checkpoint cadence, 100M word exposure.

## Next action after runtime delivery

After the compliant-tokenizer reinvest training completes:

1. Collect the authoritative result.
2. Run `inspect_compliant_retrain.py --arms reinvest`.
3. If complete, run `compliant_postdelivery_driver.py --arm reinvest --gpu <free_gpu>`.
4. If resources are tight, the driver may be run with `--cheap-only` first, but do not treat that as a full result; use the continuation-policy output.  Unless the hard upper bound proves the decision target unreachable, complete SuperGLUE, official min-context-0 AoA, and pristine collation.
5. Use the pristine-collator summary as the authoritative current-coordinate result.

After clean-Qwen training completes, use the same inspection and driver for `--arm clean_qwen` only as a fixed-tokenizer scientific control, not as an independently valid Strict-Small submission score.

A timeout before training or a resource-collision failure is not model evidence. Any retry must retain the frozen batch size, sequence length, data, tokenizer, seeds and recipe, with separate output directories. Changing those factors creates a different scientific comparison.

## Final driver repairs after initial note

After inspecting pristine collator, I found and fixed a driver extraction bug: the collator writes `pristine_collate_<tag>_summary.json` and stores the score under `score_summary.official_overall.Overall`.  The driver now uses this path/key pair.

The driver also now accepts `--run-dir` and passes `--run-dir`, `--out-root`, and `--collate-root` explicitly to `evaluate_compliant_endpoint.py`.  This matters if a managed retrain times out before training and the identical scientific command is relaunched under a fresh suffix.  The driver derives `model_root` from the chosen run directory rather than from a hard-coded `_r2` directory.

Final dry-runs succeeded for both arms:

- Reinvest dry-run output: `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/reinvest_postdelivery_driver.json`, intended collator summary `experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_reinvest_seed43022/pristine_collate_complianttok_reinvest_seed43022_summary.json`.
- Clean-Qwen dry-run output: `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/clean_qwen_postdelivery_driver.json`, intended collator summary `experiments/archive/frontier_consolidation/data/compliant_pristine_collate/complianttok_clean_qwen_seed43022/pristine_collate_complianttok_clean_qwen_seed43022_summary.json`.

All syntax checks passed after these repairs.
