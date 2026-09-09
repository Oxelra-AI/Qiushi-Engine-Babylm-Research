# shuf dup prestate/039: pre-result SHUF and selected-DUP seed43122 replication rules

This note preserves the shuf dup prestate pre-result replication rules and amends the stream identities after the dup identity audit construction comparison, still before reading any seed43122 SHUF or selected-DUP scores.

## Active training tasks

- SHUF seed43122: training `experiments/archive/compact_experience/data/qwen_shuffled_control/training_corpora/qwen_shuffled_100M.jsonl` with metadata `experiments/archive/compact_experience/data/qwen_shuffled_control/shuffled_control_metadata.json`, run directory `experiments/archive/relation_learning/training/runs/qwen_shuffled_control_16k_seed43122`.
- selected-DUP seed43122: training `experiments/archive/compact_experience/data/selected_original_dup_all_control/training_corpora/selected_original_dup_all_100M.jsonl` with metadata `experiments/archive/compact_experience/data/selected_original_dup_all_control/selected_original_dup_all_metadata.json`, run directory `experiments/archive/relation_learning/training/runs/selected_original_dup_all_16k_seed43122`.
- An earlier training attempt used the older `official_original_dup_100M` stream and was cancelled in dup identity audit. It must not be used as a replication of the report's DUP arm. If any partial artifacts from it remain, they are a separate older exact-recurrence construction, not the selected-DUP replication.

Baseline: OFF seed43122 is `experiments/archive/compact_experience/training/runs/official_lengthmatched_16k_seed43122`, with per-target official results at `experiments/archive/compact_experience/data/full_eval/per_target/official_lengthmatched_seed43122.json`.

## Stream identity checks already fixed

SHUF matches the seed43022 scored arm: seed43022 launch used `qwen_shuffled_100M.jsonl`, and the seed43122 task uses the same stream and metadata with only the seed pair changed. The shuffled residual-similarity comparison shows the intended correspondence break: shuffled source--rewrite mean content Jaccard 0.005672 versus aligned 0.463335, median 0, p95 0.045455, fraction content Jaccard >= 0.2 only 0.000399, and >= 0.4 zero.

DUP needed correction. The report's DUP reference values came from COMPACT_EXPERIENCE frequency concentration probe `selected_original_dup_all`, not the older COMPACT_EXPERIENCE earlier analysis `official_original_dup`. The dup identity audit construction comparison (`research/notes/relation_learning/dup_identity_audit.md` and `experiments/archive/relation_learning/data/dup_identity_audit/dup_identity_audit.json`) found:

- earlier analysis `official_original_dup`: 36,687 duplicated pair ids, 1,656,800 duplicate-pair words, 12,247 packed rows, pool SHA256 `c6ac969b46103cd342606ef8915f6e7b6610a6d3ec0e1b9db7ab97c3833ba634`, training SHA256 `77838bcbffe0e467ce339a6ffd89abe2d81ab124ed4d5c2b75835fad95d5cb2e`.
- frequency concentration probe `selected_original_dup_all`: 37,594 duplicated pair ids, 1,698,026 duplicate-pair words, 12,550 packed rows, keeps all earlier analysis selected pair ids, matches source totals to the Qwen effective target mixture, pool SHA256 `7ab08ee9db11b618b80f73e98d581d4f024f2b0f82f15405572f04f0f0e6dad7`, training SHA256 `b33c57e960ed2b0519f782d44c88d9ec8cdf7344710c0f525256d1050c3f582a`.
- Pair-id intersection is 36,687; earlier analysis omits 907 frequency concentration probe selected pair ids. Both preserve local pair boundaries and avoid truncation, but they are not the same construction.

Therefore the selected-DUP seed43122 replication must be compared to the frequency concentration probe seed43022 reference values below.

## Shared recipe identity

Both active arms should use the common COMPACT_EXPERIENCE DeBERTa recipe:

- Model: DeBERTa-v2, 8 layers, hidden size 480, 8 heads, vocabulary 16,384, 256-token window.
- Masking: fixed 0.15 whole-word masking.
- Batch/optimization: batch size 256, learning rate 0.001, weight decay 0.01, warmup fraction 0.06.
- Exposure: 100,000,000 charged words with 1,000,000-word checkpoint spacing.
- Tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model`.
- Seed pair for the new runs: `extra_init_seed=43122`, `train_rng_seed=43123`. The seed43022 references used `extra_init_seed=43022`, `train_rng_seed=43023`.

## SHUF seed43122 replication rules

SHUF supports the two-seed wrong-correspondence result if it reduces source-recurring readout relative to OFF in a way that is not explained by broad language degradation:

1. Wikipedia overlap ΔA_T = Δ(N−T) for SHUF−OFF is negative beyond pair-level standard error.
2. Compact overlap ΔA_T for SHUF−OFF is negative.
3. Ordinary held-out loss change SHUF−OFF is small, with |Δ| < 0.05 nats.
4. The unrelated-source term ΔA_U = Δ(N−U) is flat or weakly positive rather than broadly negative, especially on the near-register Wikipedia overlap readout.

If the Wikipedia overlap ΔA_T is positive or indistinguishable from zero, if ordinary held-out loss is substantially worse (>0.05 nats), or if the sign pattern reverses, SHUF remains a one-seed observation and the report should keep the one-seed wording while adding the seed43122 outcome.

Seed43022 reference values for SHUF−OFF:

- Wikipedia overlap ΔA_T: −0.9942 ± 0.0682 pair SE.
- Compact overlap ΔA_T: −0.7024.
- Compact nonoverlap ΔA_T: −0.3571.
- Ordinary held-out loss: +0.0130 ± 0.0020.
- Entity rel_eq0: −3.70 points.
- Natural-copy gain: +0.2316.

## selected-DUP seed43122 replication rules

selected-DUP supports the two-seed exact-recurrence result if it reproduces identity-oriented readout together with changed-form liability:

1. Natural-copy gain for DUP−OFF is positive and large.
2. Entity rel_eq0 for DUP−OFF is positive, indicating unchanged-state retrieval advantage.
3. Compact nonoverlap ΔA_T for DUP−OFF is negative.
4. Wikipedia nonoverlap ΔA_T for DUP−OFF is negative.
5. Ordinary held-out loss does not by itself explain the target-class crossing.

If copy gain is near zero or negative, compact nonoverlap ΔA_T is positive, or the key target-class crossing reverses, DUP remains one seed in the report and the seed43122 outcome should be stated directly.

Seed43022 reference values for selected-DUP−OFF:

- Natural-copy gain: +1.7497.
- Entity rel_eq0: +15.51 points.
- Entity rel_ge3: −2.26 points.
- Compact nonoverlap ΔA_T: −1.9263.
- Compact overlap ΔA_T: +0.8965.
- Wikipedia nonoverlap ΔA_T: −1.1840 ± 0.0659 pair SE.
- Wikipedia overlap ΔA_T: +3.3422 ± 0.1395 pair SE.
- Ordinary held-out loss: +0.0781 ± 0.0015.

## Scoring path after training completes

Use `experiments/archive/relation_learning/scripts/score_shuf_dup_seed43122_d1dab55a.py`. It is configured for the corrected OFF, SHUF, and selected-DUP run directories. The intended readouts are:

1. Compact T/U/N at overlap and nonoverlap.
2. Wikipedia T/U/N at overlap and nonoverlap.
3. Held-out natural-copy gain.
4. Entity relevant-update stratification from official Entity predictions.
5. Ordinary held-out deterministic-mask MLM loss.

No report wording should be upgraded until the terminal training results are collected, the scoring script completes, and the new numbers are compared against the rules above.
