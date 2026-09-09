# wikiauto pair signal analysis — why the WikiAuto aligned-pair mechanism failed

Evidence JSON: `experiments/archive/initial_model_studies/data/wikiauto_pair_signal_analysis.json`

## Pair correspondence

- selected pairs: 212,001
- source words / target words: 5,514,089 / 4,485,907
- target/source word ratio mean 0.896, median 0.875, p10-p90 0.400–1.353
- content-token Jaccard mean 0.475, median 0.444, p10-p90 0.167–0.846
- target content-token copy fraction mean 0.711; source covered by target mean 0.589
- source has noninitial capitalized token in 81.7% of pairs; target has one in 80.6%
- when source has a capitalized token, mean source-cap preservation in target is 0.671; any overlap occurs in 89.9%
- relation-word overlap mean, conditional on any relation cue, is 0.497; both sides contain relation cues in 83.4% of such pairs

## Aggregate feature densities per 1k regex words

| feature | WikiAuto source | WikiAuto target | WikiAuto combined | official corpus |
|---|---:|---:|---:|---:|
| pronouns | 35.97 | 44.64 | 39.86 | 67.22 |
| capitalized noninitial | 191.74 | 205.44 | 197.89 | 80.37 |
| numbers | 48.61 | 50.80 | 49.59 | 11.02 |
| spatial words | 78.79 | 78.80 | 78.80 | 69.98 |
| state/transfer verbs | 6.24 | 7.67 | 6.88 | 11.44 |
| physical/material words | 3.50 | 3.67 | 3.58 | 4.60 |
| social words | 5.30 | 7.35 | 6.22 | 13.95 |
| causal/temporal words | 10.24 | 10.96 | 10.56 | 16.09 |
| any relation cue | 103.63 | 107.97 | 105.58 | 115.31 |

## Reading of the measurement

The selected WikiAuto/Turk pairs do contain many content words and some capitalized entities, but the target side is a compressed sentence-level simplification rather than an explicit entity/relation bridge. The exact-pair training signal is therefore weakly aligned with the Entity/EWoK target: many source-side entity mentions and relation cues are not preserved in the target, and high lexical overlap encourages local surface reuse rather than learning persistent state or world-relation invariants.

This makes the aligned rewrite mechanism result negative result scientifically interpretable: the current source tests passive sentence-level simplification adjacency, not a dense entity/relation correspondence principle. A better route should rebuild the experience structure around checked entity and relation invariance, or around source selection/compression that preserves factual predicates, instead of scaling this exact aligned construction.
