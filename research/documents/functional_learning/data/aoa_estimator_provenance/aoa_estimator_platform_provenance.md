# earlier analysis AoA estimator platform provenance

Created: 2026-09-07T21:23:46Z

## Repository identity

- Local repo: `experiments/archive/initial_model_studies/repos/babylm-eval`
- Local HEAD: `6f825c291e2c4c78ad33b1935fd64d45f52642dc`
- Expected public main commit: `6f825c291e2c4c78ad33b1935fd64d45f52642dc`
- Remote main probe: `6f825c291e2c4c78ad33b1935fd64d45f52642dc	refs/heads/main`
- Knowledge archive byte-equal for `utils.py`: `True`
- `utils.py` SHA256: `b88b5bf4cf2ec1a55ee598d837faeb8b4df40fea6498b334a5fd1c22fb236b21`

## Current official-main AoA scorer features

- `model_fit_bounded`: `True`
- `model_fit_mentions_old_unbounded`: `True`
- `context_average_within_checkpoint`: `True`
- `subword_scaled_random_ceiling`: `True`
- `p_value_gates_zero`: `True`
- `extract_step_number_accepts_decimal_units`: `True`

## Space-side scoring snapshot

- `space_submission_eval_reads_numeric_aoa_object`: `True`
- `space_submission_eval_carries_aoa_surprisals`: `True`
- `space_read_evals_zeroes_if_no_aoa_surprisals`: `True`
- `space_validity_checks_raw_aoa_if_legacy_results_key`: `True`
- `space_validity_expected_aoa_size_symbol`: `True`

## Interpretation

The local working repository is at the public babylm-org/babylm-eval main commit, the Knowledge archive for that commit has byte-identical utils.py, and the current AoA run.py writes aoa_score.json using AoAEvaluator. The saved Space-side code reads the submitted numeric aoa object and requires aoa_surprisals presence for display rather than recomputing the curve server-side.

For the frontier comparison, the AoA column should be attributed to `current_official_main_6f825c2_AoAEvaluator` unless a newer platform scorer is explicitly identified. The same saved `surprisal.json` can be rescored under sensitivity estimators for scientific analysis, but those results must remain separate from the platform-matching coordinate.

Full JSON: `experiments/archive/functional_learning/data/aoa_estimator_provenance/aoa_estimator_platform_provenance.json`
