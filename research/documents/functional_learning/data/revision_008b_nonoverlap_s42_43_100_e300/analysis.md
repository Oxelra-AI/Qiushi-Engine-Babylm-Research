# matched support source contrast result matched-support source-specific contrast

This experiment repairs the identity shortcut bridge result v2 support mismatch before interpreting absolute degradation. Relation arms train with one matched and one unmatched final event per sequence. Matched target slot is uniformly 6/7; source slot is uniformly 0..5 and is not sorted. Probes use the same positional distribution. `repeat_full`/`repeat_masked` and `varied_full`/`varied_masked` have identical token data within each pair; the only difference is the training attention block from matched target event to its source event.

Seeds: [42, 43, 100]; epochs=300; n_seqs=500; probes=256; model d=64, heads=2, layers=3.

## Mean final metrics

| condition | copy_gain | content_gain | content_T | content_U | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|
| unique | -0.000±0.001 | -0.000±0.002 | 4.136±0.073 | 4.136±0.073 |  | 4.153±0.010 | 1.876±0.001 |
| support_control | +0.000±0.000 | +0.000±0.000 | 4.191±0.011 | 4.191±0.011 | 4.153±0.007 | 4.156±0.008 | 1.893±0.001 |
| repeat_full | +1.869±1.414 | +0.055±0.046 | 6.018±0.769 | 6.073±0.752 | 2.407±0.889 | 4.377±0.101 | 1.853±0.022 |
| repeat_masked | -0.003±0.007 | +0.029±0.022 | 4.668±0.836 | 4.697±0.849 | 4.174±0.218 | 4.152±0.011 | 1.880±0.012 |
| varied_full | +0.096±0.079 | +2.028±0.839 | 2.329±0.686 | 4.357±0.161 | 2.109±0.358 | 4.439±0.055 | 1.845±0.010 |
| varied_masked | -0.007±0.017 | +0.003±0.008 | 4.009±0.181 | 4.012±0.179 | 3.966±0.178 | 4.231±0.111 | 1.873±0.000 |
| wrong_full | +0.018±0.003 | +0.020±0.056 | 6.110±0.131 | 6.130±0.085 | 3.687±0.002 | 4.218±0.015 | 1.881±0.001 |

## Mean source-specific contrasts using U_nomatch

Positive excess true-source cost means the true source is worse relative to the unrelated-source control: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| contrast | relation | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---:|---:|---:|---:|
| repeat_full − repeat_masked | content | +1.350±1.388 | +1.376±1.354 | -0.026±0.034 | +0.026±0.034 |
| varied_full − varied_masked | content | -1.680±0.677 | +0.346±0.291 | -2.025±0.831 | +2.025±0.831 |
| repeat_full − unique | content | +1.882±0.765 | +1.937±0.746 | -0.055±0.045 | +0.055±0.045 |
| repeat_full − support_control | content | +1.827±0.779 | +1.881±0.762 | -0.054±0.045 | +0.054±0.045 |
| varied_full − unique | content | -1.807±0.670 | +0.222±0.211 | -2.029±0.838 | +2.029±0.838 |
| varied_full − support_control | content | -1.862±0.696 | +0.166±0.150 | -2.028±0.839 | +2.028±0.839 |
| repeat_full − varied_full | content | +3.689±0.147 | +1.715±0.899 | +1.974±0.832 | -1.974±0.832 |
| repeat_full − repeat_masked | copy | -1.861±1.137 | +0.011±0.374 | -1.872±1.419 | +1.872±1.419 |
| varied_full − varied_masked | copy | +0.393±0.482 | +0.496±0.481 | -0.103±0.096 | +0.103±0.096 |
| repeat_full − unique | copy | -1.910±1.032 | -0.041±0.409 | -1.869±1.414 | +1.869±1.414 |
| repeat_full − support_control | copy | -1.869±1.006 | +0.001±0.413 | -1.869±1.414 | +1.869±1.414 |
| varied_full − unique | copy | +0.785±0.450 | +0.881±0.509 | -0.096±0.079 | +0.096±0.079 |
| varied_full − support_control | copy | +0.827±0.481 | +0.923±0.547 | -0.096±0.079 | +0.096±0.079 |
| repeat_full − varied_full | copy | -2.696±1.482 | -0.922±0.255 | -1.773±1.364 | +1.773±1.364 |

## Per-seed files

- seed 42: `seed42/analysis.md`, `seed42/summary.json`
- seed 43: `seed43/analysis.md`, `seed43/summary.json`
- seed 100: `seed100/analysis.md`, `seed100/summary.json`
