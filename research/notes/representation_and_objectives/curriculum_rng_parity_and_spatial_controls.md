# Curriculum parity and spatial counterfactual-exchange results

## Curriculum parity

The initial curriculum run was not a valid matched comparison. It used seed=43 and initialization seed=43022 but omitted `train_rng_seed`, so masking used seed 43 rather than the reference model's training seed 43023. The run also failed before producing results.

The repaired trainer (`curriculum_trainer.py`) reproduced the reference trainer at a fixed sequence length of 256: identical dataset rows, first effective batch, whole-word masks at training seed 43023, initialization, and first AdamW update. The loss, gradient, and parameter-state differences were all zero. Endpoint runs require `--min_chunk_tokens 1`; a minimum of 8 drops approximately 1058 tokens from 304 rows per 200k words.

The corrected 8x480 AdamW curriculum uses sequence lengths 64 up to 20M words, 128 up to 50M, and 256 up to 100M, with `--train_rng_seed 43023 --min_chunk_tokens 1`. Its result was pending when this note was written. The matched reference has a Cheap7 score of 43.1079.

## Spatial counterfactual exchange

The spatial counterfactual-exchange route was closed on the permitted 10M-word corpus. The stricter extraction procedure (version 5) uses exact locative or on-top-of verbs, typed frequency-matched spans, and counterfactual, inverse-equivalent, dual-query, relation-erased, and target-permuted conditions. It yields only 58 regular-expression matches and 6 clean cases among 64,183 rows, all involving "above".

Frozen-model scoring across three conditions gives a counterfactual dual-query margin (`cf_dual_m`) of approximately -0.08, close to the target-permutation null of -0.07. No item was correct under all four counterfactual views (`all_four_cf_views_correct=0`), while erasing the relation still left a dual-query margin of approximately 3.36, consistent with strong prior-driven answers. The earlier version 3 signal depended on contaminated, more permissively selected cases. These tests did not establish a sufficiently dense set of clean, nontrivial spatial-exchange examples in this corpus.

## Remaining comparison

The open question is whether the function-preserving residual-adapter path produces matched cheap7 improvement over the compact-view baseline. Combining curriculum and adapter changes would be justified only after their individual effects are isolated.
