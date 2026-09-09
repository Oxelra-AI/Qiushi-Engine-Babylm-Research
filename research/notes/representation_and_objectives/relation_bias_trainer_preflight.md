# relation bias trainer preflight relation-biased trainer preflight

CPU-only test of `relation_bias_accumulated_trainer.py`; no BabyLM model training and no official evaluation text.

## Checks

- `train_stream_exists`: `True`
- `tokenizer_exists`: `True`
- `word_group_identity_first64`: `True`
- `disabled_relation_wrapper_exactly_matches_base_masking`: `True`
- `boost2_raises_selected_relation_group_fraction`: `True`
- `boost3_raises_more_than_boost2`: `True`
- `boost2_preserves_selected_group_budget_roughly`: `True`
- `token_mode_boost2_raises_relation_token_fraction`: `True`
- `leader_shape_12x384_40k_param_count_matches_step64`: `True`

## Real 100M-stream first-256-row masking sample

- Candidate visible word groups: `38915`; relation-bearing groups: `4284` (`0.1101`).
- WWM boost 2.0 selected relation-group fraction mean: `0.2208`; selected groups multiplier vs uniform expectation: `1.0024`.
- WWM boost 3.0 selected relation-group fraction mean: `0.3301`; selected groups multiplier vs uniform expectation: `1.0020`.
- Token-mode boost 2.0 selected relation-token fraction mean: `0.1929` vs candidate `0.0966`.

## Architecture extension

- 12x384 legal40k with explicit intermediate_size=1280 has `38421952` parameters (expected `38421952`).

## Interpretation

The extension preserves the original masking path when relation bias is disabled, so it can serve as a controlled post-40k objective route. When enabled, it raises relation-cue prediction pressure inside the existing legal corpus while keeping selected word-group count near the uniform 0.15 expectation. It remains only a prepared route until the current legal40k official vectors determine whether a relation/predicate learning-signal experiment is scientifically justified.

JSON: `experiments/archive/representation_and_objectives/data/relation_bias_preflight/relation_bias_trainer_preflight.json`
