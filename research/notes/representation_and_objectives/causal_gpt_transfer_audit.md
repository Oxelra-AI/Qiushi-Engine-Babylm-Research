# Causal-GPT compact-view transfer scaffold audit
JSON: `experiments/archive/representation_and_objectives/data/causal_gpt_transfer_audit/causal_gpt_transfer_audit.json`
## What is valid
- Manifest hashes match actual files: `{'compact_pool': True, 'repeat_pool': True, 'filler_rows': True, 'tokenizer_json': True}`.
- Pools are row-aligned: 73890 compact rows / 73890 repeat rows, 10,000,000 and 10,000,000 words.
- Pair rows align one-for-one at 12155 positions with zero filler mismatches and zero word-count mismatches.
- Tokenizer is the same filler-only 16k BPE for both arms; pair-token mass differs by 32144 tokens (5.507% of repeat pair tokens) on treatment rows.

## What blocks trusted readout
- **blocking_for_cheap7_comparability** `experiments/archive/frontier_consolidation/scripts/causal_cheap7_eval.py`: reading runner passes --test_path and no --backend, but official reading/run.py requires --data_path and --backend. Consequence: Reading returns None; cheap7 becomes a 6-column average and is not comparable with prior cheap7/official readouts.
- **blocking_for_globalpiqa** `experiments/archive/frontier_consolidation/scripts/causal_cheap7_eval.py`: GlobalPIQA scoring compares prediction text to the label field instead of solution{label} text. Consequence: GlobalPIQA can be scored near-zero or otherwise incorrectly even if official predictions are valid.
- **blocking_for_entity_tracking** `experiments/archive/frontier_consolidation/scripts/causal_cheap7_eval.py`: Entity scoring does not skip gold rows whose options contain 'nothing', while official prediction generation skips them. Consequence: Prediction/gold alignment drifts; Entity score is not official-compatible.
- **blocking_for_missing_column_detection** `experiments/archive/frontier_consolidation/scripts/causal_cheap7_eval.py`: cheap7 averages over present columns instead of failing when any required column is missing. Consequence: Failed tasks can silently inflate or deflate mechanism comparisons.
- **repair_recommended** `experiments/archive/frontier_consolidation/scripts/causal_cheap7_eval.py`: Evaluation task failures are converted to None and hidden in captured stderr snippets. Consequence: A full transfer run can look finished while key official columns are absent or mis-scored.
- **operational_cwd_risk** `experiments/archive/frontier_consolidation/scripts/causal_data_and_tokenizer.py`: The data script assumes a fixed working-directory root; invoking it from a nested directory can misresolve paths. Consequence: Reproduction requires a consistent path base or explicit root discovery.
- **design_note_not_blocking** `experiments/archive/frontier_consolidation/scripts/causal_gpt_trainer.py`: Pair sides inside a treatment row have no explicit side boundary; only row/doc EOS is appended after the whole pair item. Consequence: Causal training tests directional continuation through the source/view boundary rather than a bidirectional same-row co-training signal like MLM.
- **stale_smoke_test** `experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/smoke_test/training_manifest.json`: Existing smoke manifest lacks current trainer fields such as legal_charged_words, active_tokens_per_epoch, and dropped_tail_tokens_per_epoch. Consequence: The smoke run appears to predate later exposure-counter edits; current trainer should be smoke-tested before full H100 spend.

## Causal-objective interpretation caveat
- Repeat treatment rows are exact source-prefix replay in 12155/12155 rows; compact rewrite equals the source prefix in only 1 rows.
- Therefore causal next-token training receives a copy/restart signal in repeat rows that compact semantic rewrites do not have. A compact-over-repeat result would be a strong transfer signal; a repeat-over-compact result would not by itself refute the masked compact-view mechanism.

## Bottom line
The data/tokenizer scaffold is hash-consistent and row-aligned, but the current evaluation script is not official-compatible and the causal objective changes the meaning of the compact/repeat contrast. Do not trust a earlier analysis cheap7 table until scoring is repaired; do not read repeat>=compact as falsifying the masked compact-view principle without accounting for the exact-prefix replay artifact.

## Recommended repair before companion analysis spends/interprets full H100 work
- Repair causal cheap7 evaluation to use official reading --data_path/--backend causal, official GlobalPIQA solution-text comparison, Entity 'nothing' filtering, and fail-closed all-7-column collation.
- Run a fresh current-script smoke/short run after the exposure-counter edits; the existing smoke manifest appears stale relative to the current trainer fields.
- If full causal-GPT training is launched, interpret it as architecture/objective transfer only with this caveat: causal repeat rows provide exact source-prefix replay, so repeat arm has a directional copy/restart advantage absent from compact semantic rewrites.
- Prefer additional local readouts over endpoint-only cheap7: pair-row validation perplexity separated into treatment-side vs filler-side and boundary-position loss, compact vs repeat prediction turnover on the same official tasks, and comparison to masked-triangle result.

## Verification against real evaluation data
- The eval script's `full_eval` roots (`EVAL_REPO / "full_eval" / ...`) do NOT exist at that path. The real official layout under the same repo is `strict/evaluation_data/full_eval/<task>/...` and `strict/evaluation_data/fast_eval/<task>/...`. So `BLIMP_ROOT`, `SUPP_ROOT`, `EWOK_ROOT`, `ENTITY_ROOT`, `COMPS_ROOT`, `GLOBALPIQA_FULL`, `READING_DATA` in `causal_cheap7_eval.py` point to nonexistent directories. This is an additional operational blocker: as written, the causal cheap7 evaluator cannot locate any task data, so every column would be empty and `max(n_present,1)` would divide by 1 and report a spurious cheap7.
- GlobalPIQA scoring bug confirmed concretely: real rows carry an integer `label` (e.g. `"label": 3`) and text `solution0..3`; the official scorer compares the predicted completion TEXT to `solution{label}` (`_calculate_global_piqa_results` in collate_preds.py). `items[0].get("pred") == d.get("label")` compares a text prediction to an integer index, which is always False → GlobalPIQA scores 0 for both arms even with valid predictions.
- Consequence for the mechanism question: even a completed companion analysis causal run's cheap7 table would be unreliable both because task data cannot be found at the coded roots and because GlobalPIQA/Entity/Reading are mis-wired. The scaffold DATA is clean; the EVALUATION path is not. companion analysis should reuse an already-working official-compatible causal evaluation harness (the same official `run.py`/`collate_preds.py` invocation used elsewhere with `--backend causal` and the real `evaluation_data` roots) rather than the reimplemented `collect_scores` in this script.
