# successor route comparison while minfreq runs successor-route comparison while minfreq50 runs

The init-matched minfreq50 80M screen and its dependent 70M/80M cheap-column evaluation were still in progress. This note adds CPU-only evidence; it does not change the corpus or launch training. A weak minfreq50 result would not, by itself, justify pair-aware sequence training merely because implementation assets exist.

## Active score state

- Best complete fully legal endpoint is still spatial repair route status same-pool-tokenizer compact-view reinvest: Overall `41.2577708964`, below the `41.8` target by `0.542229`.
- The validated learning mechanism is still compact semantic second views plus reinvested source diversity under the legal 10M-word budget. Under the same spatial repair route status tokenizer, it beats the clean fixed-tokenizer control by mean7 `+1.2921` at 70M and `+1.3464` at 80M.
- Global word-mean remains closed as a main objective: earlier analysis/68 showed 80M mean7 `-0.2457` versus token-mean reinvest, with BLiMP/Supplement/EWoK/Entity/Reading worse and gains concentrated in COMPS plus GlobalPIQA nonparallel.








## Available Sequence Evidence

The available results improved the chunk-stream implementation, not sequence efficacy:

- `experiments/archive/representation_and_objectives/data/stream_order_confound_audit/stream_order_confound_audit.json` shows each 10M block in the materialized 100M stream is an exact multiset repeat of the 10M pool but shuffled relative to canonical pool order; `first_expensive_experience_utilization_run_should_use_stream_order=true` because pool-order chunking would change data order relative to existing baselines.
- `experiments/archive/representation_and_objectives/data/dryrun_stream_order_legal40k_U256_full/dryrun_metrics.json` and `...U64_128_256_full/dryrun_metrics.json` both dry-run legal40k with exact `100000000` charged words and `2530` total steps. U256 and U64/128/256 expose the same `13942644` active tokens per epoch and about `20.916M` masked tokens over training.
- `experiments/archive/representation_and_objectives/data/mask_budget_and_pad_contract/mask_budget_and_pad_contract.json` shows pad labels are ignored and padding stays pad. Relative to legal40k row256 baseline, the stream-order U256/U64 arms have only about `1.0169x` total masked tokens at mask probability 0.15.

The implementation checks do not establish a sequence score. For legal16k, the earlier measured gain over fixed seq256 was only about `2.59%` active tokens, so sequence remains a possible controlled experiment rather than an automatic successor.

## Source-view consistency leverage from successor route comparison while minfreq runs CPU probes

### Existing pair geometry

`experiments/archive/frontier_consolidation/scripts/pair_alignment_probe.py` measured trained checkpoints on 160 sampled changed examples and 635 both-visible source/rewrite pair records from the route portfolio and intervention assets span map. Outputs:

- `experiments/archive/frontier_consolidation/data/pair_alignment_probe/pair_alignment_probe.json`
- `research/documents/frontier_consolidation/data/pair_alignment_probe/pair_alignment_probe.md`

Last-layer centered source/rewrite cosine geometry:

| checkpoint | actual cos | shuffled cos | margin | same-row-other | retrieval top1 |
|---|---:|---:|---:|---:|---:|
| tokenmean_20M | 0.7799 | -0.0150 | 0.7949 | 0.2676 | 0.9827 |
| tokenmean_80M | 0.8232 | -0.0242 | 0.8474 | 0.0318 | 1.0000 |
| tokenmean_100M | 0.8251 | -0.0239 | 0.8490 | 0.0323 | 1.0000 |
| wordmean_80M | 0.8089 | -0.0209 | 0.8298 | 0.0527 | 0.9969 |

Interpretation:

- Token-mean already learns a strong exact source/compact-view geometry by 80M, with almost no 80M→100M growth in this probe (`+0.0016` margin) and very high retrieval.
- The source/rewrite closeness is not merely same-row context: same-row-other controls fall to about `0.032` at 80M/100M.
- Word-mean slightly weakens last-layer pair geometry (`-0.0176` margin vs tokenmean_80M), so its GlobalPIQA/COMPS movement should not be attributed to stronger source-view abstraction.
- A future consistency route should not force full hidden vectors together. If used, it should extract a small invariant subspace while preserving residual syntax/entities/discourse.

### Local gradient scale

`experiments/archive/frontier_consolidation/scripts/consistency_gradient_probe.py` measured actual final-hidden gradients on the first 512 stream rows, at most 6 batches, for tokenmean_80M and wordmean_80M. Outputs:

- `experiments/archive/frontier_consolidation/data/consistency_gradient_probe/consistency_gradient_probe.json`
- `research/documents/frontier_consolidation/data/consistency_gradient_probe/consistency_gradient_probe.md`

Summary:

| checkpoint | MLM loss mean | aux cosine loss | aux/MLM hidden L2 at λ=1 | aux/MLM aux-span L2 at λ=1 | hidden grad cosine | λ for ~5% all-hidden L2 |
|---|---:|---:|---:|---:|---:|---:|
| tokenmean_80M | 2.4388 | 0.0969 | 0.093 | 0.350 | 0.0024 | 0.536 |
| wordmean_80M | 2.6077 | 0.0986 | 0.094 | 0.346 | 0.0029 | 0.532 |

Interpretation:

- A full-vector positive pair loss is not numerically huge at λ=1 across all hidden positions, but it is concentrated on aux spans and almost orthogonal to MLM at the final hidden tensor.
- If a source-view consistency trainer is built, a low-weight projected/subspace objective is more scientifically coherent than a large full-vector MSE/cosine loss. The probe points to local calibration before any training; it is not score evidence.

## Distinct optimization mechanism: fractional word-group credit

`experiments/archive/frontier_consolidation/scripts/fractional_credit_profile.py` measured intermediate group-credit exponents over the same 79 sampled actual WWM batches used in wordmean screen and substrate constraints. Outputs:

- `experiments/archive/frontier_consolidation/data/fractional_credit_profile/fractional_credit_profile.json`
- `research/documents/frontier_consolidation/data/fractional_credit_profile/fractional_credit_profile.md`

Mechanism: alpha `0` is token-mean; alpha `1` is the closed global word-mean objective. A selected word group of `k` BPE pieces receives group mass proportional to `k^(1-alpha)`. Key average-batch share shifts:

| category | tokenmean a=0 | a=0.25 Δ | a=0.5 Δ | a=0.75 Δ | wordmean a=1 Δ |
|---|---:|---:|---:|---:|---:|
| bpe_len::1 | 47.595% | +5.952% | +11.644% | +16.961% | +21.828% |
| has_upper::True | 26.692% | -2.291% | -4.379% | -6.241% | -7.865% |
| has_digit::True | 3.198% | -0.786% | -1.392% | -1.852% | -2.199% |
| source::childes | 28.401% | -0.827% | -1.669% | -2.488% | -3.254% |
| cue::mental_social_dialogue | 8.010% | +0.566% | +1.082% | +1.542% | +1.946% |
| cue::spatial_state | 4.773% | +0.328% | +0.625% | +0.890% | +1.121% |

Interpretation:

- Alpha `0.25` is materially different from global word-mean: it gives a small word-unit correction but reduces the one-piece/function-word shift by about 73% and reduces uppercase/digit downweighting by about 64% compared with alpha `1`.
- This makes fractional credit a genuinely distinct optimization hypothesis from global word-mean, but still a risky one because it inherits the same direction of credit movement.
- A fair late-scheduled test cannot simply resume from existing `chck_70M` with identical optimizer state because the saved HF checkpoint directories contain model/tokenizer files but no optimizer state. A reset-optimizer warm fork would answer a different question.

## Route comparison if minfreq50 is weak

The successor should be chosen from the minfreq50 score vector plus any new real training result, not from implementation readiness alone.

1. **If minfreq50 gives broad mature gains** in BLiMP/Supplement/EWoK with Entity and GlobalPIQA preserved, continue that representation route toward 100M/full nine-column evaluation first. It would be the first legal representation package to move the right columns at mature exposure.
2. **If minfreq50 is flat or redistributive and no strong sequence score is available**, source-view consistency is scientifically deeper than U256 as an extension of the validated compact-view mechanism, but the probes above show full hidden alignment is already high. A subspace/projection version and local gradient calibration are needed before training.
3. **If minfreq50 is weak and a strong clean U256 or short-to-long score is available**, that score should determine whether a controlled legal16k transfer comparison is justified. Such a comparison must isolate tokenizer and sequence factors from the legal40k package.
4. **If minfreq50 is weak and no route has strong external evidence**, fractional or small scheduled word-group credit is a candidate near-term optimization test, conditional on a trainer running from initialization with a controlled schedule. Its CPU profile shows a mechanism distinct from alpha=1, but expected effect size is uncertain and likely smaller than the needed full score movement.
5. **Compositional lexical embeddings and iterative depth** remain higher-construction routes. They may become more attractive if minfreq50 fails with a pattern pointing to sparse lexical estimation or if depth/sequence results are negative. Their designs remain unfinished.

Pair-aware sequence training is not the automatic post-minfreq fallback. The next expensive route depends on the minfreq50 mature score vector, actual sequence scores if available, and the distinct scientific alternatives above: source-view abstraction, faithful experience utilization/context ordering, fractional credit assignment, compositional lexical sharing, and iterative contextual computation.
