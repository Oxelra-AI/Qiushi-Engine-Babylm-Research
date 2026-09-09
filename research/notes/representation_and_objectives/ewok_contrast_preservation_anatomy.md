# ewok contrast preservation anatomy — EWoK contrast and broad-preservation anatomy

This CPU-only analysis used existing official-compatible prediction files and the exact compact-view-reinvest 10M corpus. It did not launch training/evaluation.

## Score-level anchor

- `legal16_8x480_43022` EWoK `49.8376` Overall `n/a`
- `legal40_8x480_43022` EWoK `51.4740` Overall `n/a`
- `legal40_12x384_depth_43022` EWoK `50.5472` Overall `n/a`
- `inherited16_non_submission_43022` EWoK `n/a` Overall `n/a`
- visible leader EWoK `56.0700`; depth remains `5.5228` points behind on this column.

## Persistent-error signal

- Legal40 8x480 and legal40 12x384 depth are both wrong on `2656` / `7618` EWoK rows (`0.349`).
- All measured legal endpoints are wrong on `1407` rows (`0.185`).
- The non-submission inherited-tokenizer seed43022 is right while legal40 and depth are both wrong on `693` rows; these are representation-sensitive but not solved by current legal/depth coordinates.
- For the legal40+depth both-wrong rows, mean minimum corpus count of context-difference words is `949.11` and mean legal40 context-token fraction below 50 pool occurrences is `0.0795`. This argues against a rare-token-only explanation for much of EWoK.

## Weakest EWoK domains under depth

| domain | n | depth acc | legal40 acc | legal16 acc | both-wrong frac | context contrast term frac |
|---|---:|---:|---:|---:|---:|---:|
| physical-dynamics | 120 | 40.83 | 50.00 | 50.00 | 0.317 | 0.250 |
| spatial-relations | 490 | 46.33 | 48.16 | 49.39 | 0.392 | 0.367 |
| physical-relations | 818 | 47.80 | 51.22 | 48.41 | 0.407 | 0.122 |
| social-relations | 1548 | 51.42 | 51.23 | 47.87 | 0.349 | 0.149 |
| quantitative-properties | 314 | 51.59 | 56.05 | 52.23 | 0.299 | 0.344 |
| material-properties | 170 | 51.76 | 59.41 | 54.71 | 0.294 | 0.294 |

## Corpus reservoir for a distinct self-contrastive route

- Exact 10M corpus SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`, words `10000000`, rows `64740`.
- Rows containing at least one side of a general contrast pair: `64668` rows / `9990206` words (`0.999` of corpus words).
- Rows with exactly one side of at least one contrast pair, suitable for deterministic real-vs-corrupted sentence preference if charged against budget: `64607` rows / `9981339` words (`0.998`).
- Rows with both sides of a pair, useful for natural contrast mining rather than corruption: `28060` rows / `4410288` words (`0.441`).

Top general contrast-pair reservoirs:

| pair | rows single-side | rows both-sides | rows any-side |
|---|---:|---:|---:|
| in/out | 41318 | 13814 | 55132 |
| on/off | 34912 | 4595 | 39507 |
| all/none | 22960 | 351 | 23311 |
| yes/no | 22923 | 5197 | 28120 |
| up/down | 21137 | 4091 | 25228 |
| like/dislike | 18424 | 36 | 18460 |
| know/ignore | 17889 | 17 | 17906 |
| can/cannot | 17726 | 400 | 18126 |
| left/right | 15010 | 797 | 15807 |
| before/after | 13710 | 1239 | 14949 |
| first/last | 12378 | 826 | 13204 |
| over/under | 12133 | 531 | 12664 |

## Mechanism opened

The supported route is **self-contrastive relational-context anchoring (SCRA)**: preserve the compact-view data mechanism and any SGCR endpoint, but add a small, exactly accounted source-grounded preference signal where an observed corpus sentence/context must outrank a minimally relation-corrupted counterfactual. This targets EWoK because EWoK scores signed context compatibility, not just rare subword support. It is not an SGCR parameter variant and not companion analysis-style innovation masking. The future training lexicon must be non-evaluation-derived, and every corrupted sequence/word exposure must be charged or placed in a replacement slice under the Strict-Small budget.

## Files

- ewok_domain_route_table: `experiments/archive/representation_and_objectives/data/ewok_contrast_preservation_anatomy/ewok_domain_route_table.csv`
- ewok_error_patterns: `experiments/archive/representation_and_objectives/data/ewok_contrast_preservation_anatomy/ewok_error_patterns.csv`
- contrast_lexicon_corpus_reservoir: `experiments/archive/representation_and_objectives/data/ewok_contrast_preservation_anatomy/contrast_lexicon_corpus_reservoir.csv`
- ewok_high_support_error_samples: `experiments/archive/representation_and_objectives/data/ewok_contrast_preservation_anatomy/ewok_high_support_error_samples.csv`
- ewok_item_correctness_and_coverage: `experiments/archive/representation_and_objectives/data/ewok_contrast_preservation_anatomy/ewok_item_correctness_and_coverage.csv`
- JSON: `experiments/archive/representation_and_objectives/data/ewok_contrast_preservation_anatomy/ewok_contrast_preservation_anatomy.json`
