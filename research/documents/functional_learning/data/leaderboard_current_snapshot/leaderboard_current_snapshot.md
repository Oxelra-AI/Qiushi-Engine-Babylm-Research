# earlier analysis current BabyLM leaderboard Space AoA snapshot

Created: 2026-09-07T21:33:56Z

All files OK: `True`
Matches old INITIAL_MODEL_STUDIES snapshot where available: `True`

## AoA feature flags

- `submission_eval_reads_numeric_aoa_object`: `True`
- `submission_eval_preserves_aoa_surprisals`: `True`
- `read_evals_zeroes_aoa_without_surprisals`: `True`
- `validity_accepts_missing_aoa`: `True`
- `validity_legacy_aoa_results_count_check`: `True`
- `display_has_aoa_size_constant`: `True`
- `display_has_checkpoint_constants`: `True`

## Downloaded files

- `README.md` ok=`True` sha=`c326e6722277496d747f4fb770f4da40d0a0aeb738d99999570eb03b7565edf3` matches_old=`True` path=`research/documents/functional_learning/data/leaderboard_current_snapshot/README.md`
- `src/submission/eval_submission.py` ok=`True` sha=`15e6bb9317cb225919d45812b115d8d5c3c0a943b31a560afaa795b6d8d3d05c` matches_old=`True` path=`experiments/archive/functional_learning/data/leaderboard_current_snapshot/src/submission/eval_submission.py`
- `src/leaderboard/read_evals.py` ok=`True` sha=`f7e99e7c8e8ed07353ce5c7277fd226f9b88b0201aeed12fdabf4e0fd821c76c` matches_old=`True` path=`experiments/archive/functional_learning/data/leaderboard_current_snapshot/src/leaderboard/read_evals.py`
- `src/submission/check_validity.py` ok=`True` sha=`ce762ff75c0e2118c3ae57ae68e9f09a4075733c575c31dc7877736007f0b4ff` matches_old=`True` path=`experiments/archive/functional_learning/data/leaderboard_current_snapshot/src/submission/check_validity.py`
- `src/display/utils.py` ok=`True` sha=`45a7d4549e33f719e19356be8c30c202e4610ac6fc061e147b33373b169b0eb6` matches_old=`None` path=`experiments/archive/functional_learning/data/leaderboard_current_snapshot/src/display/utils.py`

## Interpretation

Current Space-side source, if all feature flags hold, accepts/reads the participant-supplied numeric `aoa` score while also requiring raw `aoa_surprisals` presence for the displayed AoA to count. It does not expose a hidden server recomputation of AoA from raw surprisals in the downloaded files.

Full JSON: `experiments/archive/functional_learning/data/leaderboard_current_snapshot/leaderboard_current_snapshot.json`
