# matched support source contrast result matched-support source-specific contrast

This experiment repairs the identity shortcut bridge result v2 support mismatch before interpreting absolute degradation. Relation arms train with one matched and one unmatched final event per sequence. Matched target slot is uniformly 6/7; source slot is uniformly 0..5 and is not sorted. Probes use the same positional distribution. `repeat_full`/`repeat_masked` and `varied_full`/`varied_masked` have identical token data within each pair; the only difference is the training attention block from matched target event to its source event.

Seeds: [42]; epochs=3; n_seqs=64; probes=32; model d=64, heads=2, layers=3.

## Mean final metrics

| condition | copy_gain | content_gain | content_T | content_U | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|
| unique | -0.009±0.000 | -0.010±0.000 | 4.280±0.000 | 4.270±0.000 |  | 4.474±0.000 | 3.759±0.000 |
| support_control | -0.009±0.000 | -0.010±0.000 | 4.275±0.000 | 4.265±0.000 | 4.471±0.000 | 4.281±0.000 | 3.755±0.000 |
| repeat_full | -0.008±0.000 | -0.009±0.000 | 4.295±0.000 | 4.286±0.000 | 4.295±0.000 | 4.310±0.000 | 3.747±0.000 |
| repeat_masked | -0.009±0.000 | -0.009±0.000 | 4.295±0.000 | 4.286±0.000 | 4.295±0.000 | 4.310±0.000 | 3.747±0.000 |
| varied_full | -0.008±0.000 | -0.009±0.000 | 4.290±0.000 | 4.280±0.000 | 4.350±0.000 | 4.486±0.000 | 3.757±0.000 |
| varied_masked | -0.008±0.000 | -0.010±0.000 | 4.290±0.000 | 4.280±0.000 | 4.350±0.000 | 4.486±0.000 | 3.757±0.000 |
| wrong_full | -0.008±0.000 | -0.009±0.000 | 4.269±0.000 | 4.260±0.000 | 4.368±0.000 | 4.446±0.000 | 3.761±0.000 |

## Mean source-specific contrasts using U_nomatch

Positive excess true-source cost means the true source is worse relative to the unrelated-source control: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| contrast | relation | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---:|---:|---:|---:|
| repeat_full − repeat_masked | content | +0.000±0.000 | +0.000±0.000 | -0.000±0.000 | +0.000±0.000 |
| varied_full − varied_masked | content | -0.000±0.000 | -0.000±0.000 | -0.000±0.000 | +0.000±0.000 |
| repeat_full − unique | content | +0.015±0.000 | +0.016±0.000 | -0.001±0.000 | +0.001±0.000 |
| repeat_full − support_control | content | +0.020±0.000 | +0.021±0.000 | -0.001±0.000 | +0.001±0.000 |
| varied_full − unique | content | +0.010±0.000 | +0.010±0.000 | -0.000±0.000 | +0.000±0.000 |
| varied_full − support_control | content | +0.015±0.000 | +0.015±0.000 | -0.000±0.000 | +0.000±0.000 |
| repeat_full − varied_full | content | +0.005±0.000 | +0.006±0.000 | -0.001±0.000 | +0.001±0.000 |
| repeat_full − repeat_masked | copy | -0.000±0.000 | +0.000±0.000 | -0.000±0.000 | +0.000±0.000 |
| varied_full − varied_masked | copy | -0.000±0.000 | -0.000±0.000 | -0.000±0.000 | +0.000±0.000 |
| repeat_full − unique | copy | +0.012±0.000 | +0.012±0.000 | -0.000±0.000 | +0.000±0.000 |
| repeat_full − support_control | copy | +0.017±0.000 | +0.017±0.000 | -0.000±0.000 | +0.000±0.000 |
| varied_full − unique | copy | +0.011±0.000 | +0.012±0.000 | -0.001±0.000 | +0.001±0.000 |
| varied_full − support_control | copy | +0.016±0.000 | +0.017±0.000 | -0.001±0.000 | +0.001±0.000 |
| repeat_full − varied_full | copy | +0.001±0.000 | +0.000±0.000 | +0.000±0.000 | -0.000±0.000 |

## Per-seed files

- seed 42: `seed42/analysis.md`, `seed42/summary.json`
