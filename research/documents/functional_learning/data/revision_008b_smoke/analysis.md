# matched support source contrast result matched-support source-specific contrast

This experiment repairs the identity shortcut bridge result v2 support mismatch before interpreting absolute degradation. Relation arms train with one matched and one unmatched final event per sequence. Matched target slot is uniformly 6/7; source slot is uniformly 0..5 and is not sorted. Probes use the same positional distribution. `repeat_full`/`repeat_masked` and `varied_full`/`varied_masked` have identical token data within each pair; the only difference is the training attention block from matched target event to its source event.

Seeds: [42]; epochs=3; n_seqs=64; probes=32; model d=64, heads=2, layers=3.

## Mean final metrics

| condition | copy_gain | content_gain | content_T | content_U | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|
| unique | +0.002±0.000 | +0.004±0.000 | 4.774±0.000 | 4.778±0.000 |  | 4.723±0.000 | 4.055±0.000 |
| support_control | +0.002±0.000 | +0.004±0.000 | 4.765±0.000 | 4.769±0.000 | 4.726±0.000 | 4.877±0.000 | 4.043±0.000 |
| repeat_full | +0.002±0.000 | +0.004±0.000 | 4.795±0.000 | 4.800±0.000 | 4.543±0.000 | 4.703±0.000 | 4.016±0.000 |
| repeat_masked | +0.002±0.000 | +0.004±0.000 | 4.796±0.000 | 4.800±0.000 | 4.543±0.000 | 4.703±0.000 | 4.017±0.000 |
| varied_full | +0.002±0.000 | +0.004±0.000 | 4.756±0.000 | 4.760±0.000 | 4.820±0.000 | 4.738±0.000 | 4.066±0.000 |
| varied_masked | +0.002±0.000 | +0.004±0.000 | 4.756±0.000 | 4.760±0.000 | 4.820±0.000 | 4.738±0.000 | 4.067±0.000 |
| wrong_full | +0.002±0.000 | +0.004±0.000 | 4.805±0.000 | 4.809±0.000 | 4.651±0.000 | 4.767±0.000 | 4.040±0.000 |

## Mean source-specific contrasts using U_nomatch

Positive excess true-source cost means the true source is worse relative to the unrelated-source control: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| contrast | relation | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---:|---:|---:|---:|
| repeat_full − repeat_masked | content | -0.000±0.000 | -0.000±0.000 | +0.000±0.000 | -0.000±0.000 |
| varied_full − varied_masked | content | -0.000±0.000 | -0.000±0.000 | -0.000±0.000 | +0.000±0.000 |
| repeat_full − unique | content | +0.022±0.000 | +0.022±0.000 | -0.000±0.000 | +0.000±0.000 |
| repeat_full − support_control | content | +0.030±0.000 | +0.031±0.000 | -0.001±0.000 | +0.001±0.000 |
| varied_full − unique | content | -0.018±0.000 | -0.018±0.000 | +0.000±0.000 | -0.000±0.000 |
| varied_full − support_control | content | -0.010±0.000 | -0.009±0.000 | -0.000±0.000 | +0.000±0.000 |
| repeat_full − varied_full | content | +0.040±0.000 | +0.040±0.000 | -0.000±0.000 | +0.000±0.000 |
| repeat_full − repeat_masked | copy | -0.000±0.000 | -0.000±0.000 | -0.000±0.000 | +0.000±0.000 |
| varied_full − varied_masked | copy | +0.001±0.000 | +0.000±0.000 | +0.000±0.000 | -0.000±0.000 |
| repeat_full − unique | copy | -0.051±0.000 | -0.051±0.000 | -0.000±0.000 | +0.000±0.000 |
| repeat_full − support_control | copy | -0.035±0.000 | -0.035±0.000 | -0.000±0.000 | +0.000±0.000 |
| varied_full − unique | copy | +0.009±0.000 | +0.009±0.000 | +0.000±0.000 | -0.000±0.000 |
| varied_full − support_control | copy | +0.025±0.000 | +0.025±0.000 | +0.000±0.000 | -0.000±0.000 |
| repeat_full − varied_full | copy | -0.060±0.000 | -0.060±0.000 | -0.001±0.000 | +0.001±0.000 |

## Per-seed files

- seed 42: `seed42/analysis.md`, `seed42/summary.json`
