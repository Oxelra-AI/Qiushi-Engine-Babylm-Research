# seed43222 entity clean integration source-token misfire mass

For each held-out compact-rewrite token-nonoverlap target, the true source is present and the masked target token is absent from the source token set. The readout measures the probability mass placed on source-span token types at the masked target position. Elevated REPEAT source mass would support an identity-misfire account of the active recurrence cost; no elevation would support displacement or another mechanism.

## Late means over 80M/90M/100M

| arch | seed | role | source mass | target prob | source/target ratio | top1 source rate | mean top10 source hits | n |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| D | 43022 | C | 0.37838 | 0.04084 | 409455.125 | 0.6218 | 1.752 | 2585 |
| D | 43022 | R | 0.48188 | 0.03268 | 2545692.958 | 0.7014 | 1.930 | 2585 |
| D | 43022 | V | 0.38160 | 0.09821 | 461590.488 | 0.5859 | 1.611 | 2585 |
| D | 43122 | C | 0.36126 | 0.03881 | 426151.515 | 0.6085 | 1.766 | 2585 |
| D | 43122 | R | 0.50309 | 0.03082 | 2446294.555 | 0.7292 | 1.956 | 2585 |
| D | 43122 | V | 0.36276 | 0.10055 | 61692.637 | 0.5636 | 1.594 | 2585 |
| D | 43222 | C | 0.37685 | 0.04093 | 320598.172 | 0.6319 | 1.794 | 2585 |
| D | 43222 | R | 0.46954 | 0.02946 | 527765.840 | 0.6983 | 2.004 | 2585 |
| D | 43222 | V | 0.36866 | 0.09470 | 332771.326 | 0.5792 | 1.601 | 2585 |

## Key contrasts

| arch | seed | contrast | source-mass Δ | target-prob Δ | ratio Δ | top1-source-rate Δ | target-rank Δ |
|---|---:|---|---:|---:|---:|---:|---:|
| D | 43022 | RminusC | +0.10351 | -0.00816 | +2136237.832 | +0.0796 | +10.8 |
| D | 43022 | VminusC | +0.00322 | +0.05737 | +52135.362 | -0.0358 | -278.8 |
| D | 43022 | VminusR | -0.10029 | +0.06553 | -2084102.470 | -0.1154 | -289.7 |
| D | 43122 | RminusC | +0.14183 | -0.00799 | +2020143.041 | +0.1207 | -22.0 |
| D | 43122 | VminusC | +0.00150 | +0.06174 | -364458.878 | -0.0449 | -361.7 |
| D | 43122 | VminusR | -0.14033 | +0.06973 | -2384601.919 | -0.1656 | -339.7 |
| D | 43222 | RminusC | +0.09269 | -0.01147 | +207167.669 | +0.0664 | -27.5 |
| D | 43222 | VminusC | -0.00819 | +0.05377 | +12173.154 | -0.0526 | -330.4 |
| D | 43222 | VminusR | -0.10088 | +0.06524 | -194994.515 | -0.1190 | -302.9 |

## Files

Output directory: `experiments/archive/relation_learning/data/source_token_misfire_mass`
Plan: `experiments/archive/relation_learning/data/source_token_misfire_mass/misfire_plan.json`
