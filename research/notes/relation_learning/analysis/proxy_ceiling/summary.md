# earlier analysis legal-corpus proxy ceiling for child AoA

## Result

The designated compact ridge model reaches repeated out-of-fold Pearson **r=0.401** (Spearman **rho=0.405**, R2=0.160, RMSE=2.495 months) on all 406 valid child-AoA words. Its approximate Fisher interval, conditional on the aggregated OOF predictions, is [0.316, 0.480]. The best result selected after comparing nonlinear/linear all-feature models is **equal_ensemble_all_legal r=0.522** and is explicitly exploratory/selection-optimistic. On the 225-word official-v4 fitted subset, the exploratory maximum is **equal_ensemble_all_legal r=0.537**.

These numbers are an evaluation-only predictability ceiling, not a legal recipe: the feature values are legal, but fitting the reported coefficients consumed child ages. A deployable schedule may use the feature definitions, never these target-fitted weights.

## Feature extraction and audit

One CPU pass scanned 647,400 rows and 100,000,000 metadata-counted words. Lexical counts matched the independent earlier analysis literal scan for 501/504 CDI words. The only differences are `ant`, `mad`, and `man`, whose uppercase forms occur as CHILDES speaker codes; earlier analysis removes those tag occurrences and records the exact deltas in `summary.json`. Features cover full/source/CHILDES/spoken frequency, enrichment, actual-order timing, tokenizer geometry, word shape, row and clause context, short/bare-mention proxies, speaker role, and local syntactic cues.

CDI-derived 50% ages were joined only after this extraction. Of 504 CDI words, 406 have valid child ages; 225 are also in the faithful-v4 official fitted set.

## Interpretable feature boundary

The strongest individual coordinates are occurrences-per-row (r=-0.348), CHILDES enrichment (r=-0.324), and fraction spoken by a tagged child speaker (r=-0.321). Mean clause length is positive (r=0.267): words learned earlier by children are disproportionately repeated within rows, CHILDES-enriched, child-spoken, and found in short clauses. Whole-stream log frequency itself is weak against child age (r=-0.072), even though it strongly predicts model AoA in earlier analysis.

Actual pass-level timing contains no target differentiation: for all 406 valid words, exactly 10%, 20%, and 50% of occurrences fall in the first 10M, 20M, and 50M, and 10% in the last 10M. This follows from ten repetitions of the same 10M row pool. Only within-pass position varies, its univariate correlations are near zero, and the timing-only Ridge is wrong-signed with negative OOF R2. Acquisition-order leverage therefore requires an intentional reorder or credit change; it is not latent in the inherited pass sequence.

## Cross-validated model comparison (all 406)

| model | feature set | features | OOF Pearson | OOF Spearman | OOF R2 | RMSE months |
|---|---|---:|---:|---:|---:|---:|
| ridge_shape_token | shape_token | 12 | 0.030 | 0.053 | -0.002 | 2.726 |
| ridge_frequency_concentration | frequency_concentration | 3 | 0.375 | 0.390 | 0.140 | 2.525 |
| ridge_source_profile | source_profile | 9 | 0.388 | 0.390 | 0.150 | 2.511 |
| ridge_enrichment_compact | enrichment_compact | 10 | 0.370 | 0.383 | 0.136 | 2.531 |
| ridge_timing_only | timing_only | 10 | -0.292 | -0.351 | -0.006 | 2.731 |
| ridge_context_only | context_only | 20 | 0.462 | 0.467 | 0.214 | 2.414 |
| ridge_compact_legal | compact_legal | 30 | 0.401 | 0.405 | 0.160 | 2.495 |
| ridge_all_legal | all_legal | 60 | 0.498 | 0.496 | 0.246 | 2.364 |
| elastic_all_legal | all_legal | 60 | 0.486 | 0.495 | 0.234 | 2.383 |
| random_forest_all_legal | all_legal | 60 | 0.517 | 0.511 | 0.266 | 2.332 |
| hist_gradient_all_legal | all_legal | 60 | 0.504 | 0.504 | 0.252 | 2.356 |
| equal_ensemble_all_legal | all_legal | 60 | 0.522 | 0.515 | 0.266 | 2.333 |

## Official-v4 fitted subset (225 words)

| model | feature set | OOF Pearson | OOF Spearman | OOF R2 |
|---|---|---:|---:|---:|
| ridge_shape_token | shape_token | 0.181 | 0.214 | 0.032 |
| ridge_frequency_concentration | frequency_concentration | 0.295 | 0.340 | 0.085 |
| ridge_source_profile | source_profile | 0.383 | 0.401 | 0.142 |
| ridge_enrichment_compact | enrichment_compact | 0.396 | 0.419 | 0.156 |
| ridge_timing_only | timing_only | -0.244 | -0.421 | -0.014 |
| ridge_context_only | context_only | 0.518 | 0.524 | 0.267 |
| ridge_compact_legal | compact_legal | 0.442 | 0.449 | 0.195 |
| ridge_all_legal | all_legal | 0.509 | 0.493 | 0.258 |
| elastic_all_legal | all_legal | 0.506 | 0.493 | 0.256 |
| random_forest_all_legal | all_legal | 0.534 | 0.520 | 0.280 |
| hist_gradient_all_legal | all_legal | 0.507 | 0.499 | 0.251 |
| equal_ensemble_all_legal | all_legal | 0.537 | 0.514 | 0.283 |

## Stage-III translation

At n=225, faithful v4 starts at raw r=-0.035 and needs +0.145 to cross the positive p=0.1 boundary r=0.110. The observed 100M evidence-visible-minus-uniform movement is only +0.006, 4.4% of the requirement; the entire earlier analysis masking-family spread is 0.037, 25.4%.

As a sensitivity check, scaling that entire 0.037 spread by proxy signal relative to the existing CHILDES-enrichment correlation gives **0.046 raw-r movement** for the compact Ridge and **0.060** for the selection-optimistic ensemble: 31.6% to 41.1% of the required +0.145. This is not a causal bound; it shows that even favorable proportional extrapolation does not supply the roughly 2.4--3.2-fold additional movement still needed by the trunk route.

The earlier analysis legal schedule simulation reached raw r=0.100 (p=0.144), a predicted +0.136 from its reconstructed current value, but still clipped to zero. The explicitly illegal child-age oracle reached r=0.330 (p=3.41e-07). The gap between legal proxies and the oracle, together with weak observed 100M conversion, is the operative ceiling result.

## Decision

The corpus contains real but moderate legal information about child acquisition order. It is sufficient for a scientific proxy relation and for low-cost calibration, but current evidence does not support a materially larger trunk-level v5 investment in acquisition-order scheduling. Await the already-running matched 30M calibration; require a surprisingly large, replicated measured effect before reopening 100M spending. Otherwise treat AoA order credit as a boundary result and prioritize the practical clean-preservation candidate.

The feature most likely to raise this ceiling is an unsupervised distributional semantic/grammatical representation learned from the same corpus (for example a cross-fitted low-rank context representation). It could capture noun/verb/concreteness-like structure absent from these scalar statistics. It would still need a target-free construction and nested evaluation; target-tuned embeddings or CDI-fitted schedule weights would be illegal.

Full machine-readable result: `experiments/archive/relation_learning/analysis/proxy_ceiling/summary.json`.
