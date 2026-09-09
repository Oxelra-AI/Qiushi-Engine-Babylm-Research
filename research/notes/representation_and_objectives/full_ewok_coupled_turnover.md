# full ewok coupled turnover full EWoK coupled-turnover analysis

Status: **PASS**

This is a no-training full-surface readout over the official 7,618-row EWoK data using the validated fw ewok interaction reader four-cell pseudo-likelihood reader. It asks whether the coupled sparse20 movement is real interaction improvement or row turnover/churn.

## Target full-surface summaries

| target | n | accuracy | stable failures | positive interaction | mean interaction | median interaction |
|---|---:|---:|---:|---:|---:|---:|
| mlm_only_20M | 7618 | 0.4992 | 2614 | 3736 | -0.0086 | -0.0087 |
| coupled_aligned_20M | 7618 | 0.4982 | 2366 | 3620 | -0.0232 | -0.0055 |
| coupled_shuffled_20M | 7618 | 0.5018 | 2237 | 3784 | -0.0097 | -0.0006 |

## coupled_aligned_20M_minus_mlm_only_20M

Official-row turnover: repaired baseline-wrong 1891, broken baseline-correct 1899, repair-minus-break -8, breaks/repair 1.0042, accuracy delta -0.1050 points.

All-row interaction_sum delta mean -0.0147, median -0.0030, positive-delta fraction 0.4965. Stable-failure delta -248.

Baseline-wrong rows: n 3815, interaction delta mean 0.6065, median 0.1975, candidate interaction >0.5 frac 0.0532.

Baseline-correct rows: n 3803, interaction delta mean -0.6378, median -0.1866, candidate interaction >0.5 frac 0.0665.

Baseline stable-failure rows: n 2614, stable-failure delta -1746, interaction delta mean 1.0342, median 0.4742.

Row-delta CSV: `experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/comparisons/coupled_aligned_20M_row_deltas.csv`.

## coupled_shuffled_20M_minus_mlm_only_20M

Official-row turnover: repaired baseline-wrong 1846, broken baseline-correct 1826, repair-minus-break 20, breaks/repair 0.9892, accuracy delta 0.2625 points.

All-row interaction_sum delta mean -0.0011, median 0.0111, positive-delta fraction 0.5135. Stable-failure delta -377.

Baseline-wrong rows: n 3815, interaction delta mean 0.6406, median 0.2021, candidate interaction >0.5 frac 0.0763.

Baseline-correct rows: n 3803, interaction delta mean -0.6448, median -0.1683, candidate interaction >0.5 frac 0.0755.

Baseline stable-failure rows: n 2614, stable-failure delta -1809, interaction delta mean 1.0894, median 0.4766.

Row-delta CSV: `experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/comparisons/coupled_shuffled_20M_row_deltas.csv`.

## Aligned vs shuffled on the full surface

Aligned-correct/shuffled-wrong 1649; shuffled-correct/aligned-wrong 1677; both-correct 2146; both-wrong 2146; aligned-minus-shuffled stable failures 129.

Aligned-minus-shuffled interaction_sum mean -0.0136, median -0.0038, positive fraction 0.4888.

## Scientific reading

- `coupled_aligned_20M_minus_mlm_only_20M`: {"net_ewok_accuracy_delta_points": -0.10501443948542925, "repaired_minus_broken_count": -8, "breaks_per_repair": 1.004230565838181, "all_rows_interaction_mean_delta": -0.014665624102402502, "all_rows_interaction_median_delta": -0.002997167408466339, "base_correct_interaction_mean_delta": -0.6377655739451277, "base_wrong_interaction_mean_delta": 0.6064743783227309, "stable_failure_delta_count": -248, "reading": "turnover-dominated: many repaired baseline-wrong rows are offset by broken baseline-correct rows, with weak median interaction movement"}
- `coupled_shuffled_20M_minus_mlm_only_20M`: {"net_ewok_accuracy_delta_points": 0.26253609871357314, "repaired_minus_broken_count": 20, "breaks_per_repair": 0.9891657638136512, "all_rows_interaction_mean_delta": -0.001092372493511303, "all_rows_interaction_median_delta": 0.011124301701784134, "base_correct_interaction_mean_delta": -0.6447960373609053, "base_wrong_interaction_mean_delta": 0.6405865364162395, "stable_failure_delta_count": -377, "reading": "mixed full-surface movement: inspect row/domain deltas before any training"}
- `alignment_specificity_full_surface`: {"aligned_correct_shuffled_wrong": 1649, "shuffled_correct_aligned_wrong": 1677, "aligned_minus_shuffled_stable_failure_count": 129, "interaction_mean_delta": -0.013573251608891198, "interaction_median_delta": -0.003786057233810425}

JSON: `experiments/archive/representation_and_objectives/data/full_ewok_coupled_turnover/full_ewok_coupled_turnover_summary.json`
