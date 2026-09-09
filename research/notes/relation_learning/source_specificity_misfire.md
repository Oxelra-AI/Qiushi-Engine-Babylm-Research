# relation practice principle source-specificity misfire control

Tests whether REPEAT's source-token mass elevation is specific to having the TRUE source in-window (copy mechanism) or persists with an unrelated source (frequency hedging).

## Design

At each nonoverlap rewrite mask, content-token mass is measured under two conditions:

| token_set \ condition | T (true source in window) | U (unrelated source in window) |
|---|---|---|
| true_src_content | **IN-WINDOW** | OUT-OF-WINDOW |
| unrel_src_content | OUT-OF-WINDOW | **IN-WINDOW** |

Copy mechanism: R−C elevated for IN-WINDOW cells only.
Frequency hedge: R−C elevated for all cells.

## Late means (80M/90M/100M)

| arch | seed | role | cond | true_src_content_mass | unrel_src_content_mass | target_prob | n |
|---|---:|---|---|---:|---:|---:|---:|
| D | 43022 | C | T | 0.20184 | 0.00440 | 0.03755 | 2539 |
| D | 43022 | C | U | 0.02399 | 0.01480 | 0.02287 | 2539 |
| D | 43022 | R | T | 0.27450 | 0.00379 | 0.03070 | 2539 |
| D | 43022 | R | U | 0.02937 | 0.01551 | 0.02288 | 2539 |
| D | 43022 | V | T | 0.27276 | 0.00484 | 0.09497 | 2539 |
| D | 43022 | V | U | 0.03043 | 0.01782 | 0.03346 | 2539 |
| D | 43122 | C | T | 0.19077 | 0.00467 | 0.03591 | 2539 |
| D | 43122 | C | U | 0.02264 | 0.01508 | 0.02278 | 2539 |
| D | 43122 | R | T | 0.28800 | 0.00391 | 0.02910 | 2539 |
| D | 43122 | R | U | 0.03097 | 0.01787 | 0.02268 | 2539 |
| D | 43122 | V | T | 0.25779 | 0.00518 | 0.09875 | 2539 |
| D | 43122 | V | U | 0.03032 | 0.01701 | 0.03517 | 2539 |
| D | 43222 | C | T | 0.20280 | 0.00443 | 0.03852 | 2539 |
| D | 43222 | C | U | 0.02559 | 0.01935 | 0.02434 | 2539 |
| D | 43222 | R | T | 0.25784 | 0.00404 | 0.02793 | 2539 |
| D | 43222 | R | U | 0.03061 | 0.01548 | 0.02367 | 2539 |
| D | 43222 | V | T | 0.26299 | 0.00482 | 0.09225 | 2539 |
| D | 43222 | V | U | 0.03345 | 0.01899 | 0.03463 | 2539 |

## Key contrasts (R−C is decisive)

| arch | seed | contrast | cond | true_src_content_Δ | unrel_src_content_Δ | target_prob_Δ |
|---|---:|---|---|---:|---:|---:|
| D | 43022 | RminusC | T | +0.07267 | -0.00061 | -0.00685 |
| D | 43022 | VminusC | T | +0.07093 | +0.00044 | +0.05742 |
| D | 43022 | VminusR | T | -0.00174 | +0.00105 | +0.06427 |
| D | 43022 | RminusC | U | +0.00538 | +0.00071 | +0.00002 |
| D | 43022 | VminusC | U | +0.00645 | +0.00302 | +0.01059 |
| D | 43022 | VminusR | U | +0.00106 | +0.00232 | +0.01057 |
| D | 43122 | RminusC | T | +0.09723 | -0.00077 | -0.00681 |
| D | 43122 | VminusC | T | +0.06703 | +0.00051 | +0.06284 |
| D | 43122 | VminusR | T | -0.03020 | +0.00127 | +0.06965 |
| D | 43122 | RminusC | U | +0.00833 | +0.00279 | -0.00011 |
| D | 43122 | VminusC | U | +0.00768 | +0.00193 | +0.01238 |
| D | 43122 | VminusR | U | -0.00065 | -0.00086 | +0.01249 |
| D | 43222 | RminusC | T | +0.05504 | -0.00038 | -0.01059 |
| D | 43222 | VminusC | T | +0.06019 | +0.00040 | +0.05374 |
| D | 43222 | VminusR | T | +0.00515 | +0.00078 | +0.06432 |
| D | 43222 | RminusC | U | +0.00502 | -0.00387 | -0.00067 |
| D | 43222 | VminusC | U | +0.00786 | -0.00036 | +0.01029 |
| D | 43222 | VminusR | U | +0.00284 | +0.00350 | +0.01096 |

## Interpretation

If R−C `true_src_content_Δ` is positive under T but near zero under U, the copy mechanism fires specifically when the true source is in-window. If positive under both, frequency hedging or generic model uncertainty explains the elevation.

Output: `experiments/archive/relation_learning/data/source_specificity_misfire`
