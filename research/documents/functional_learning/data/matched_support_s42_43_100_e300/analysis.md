# matched support source contrast result matched-support source-specific contrast

This experiment repairs the identity shortcut bridge result v2 support mismatch before interpreting absolute degradation. Relation arms train with one matched and one unmatched final event per sequence. Matched target slot is uniformly 6/7; source slot is uniformly 0..5 and is not sorted. Probes use the same positional distribution. `repeat_full`/`repeat_masked` and `varied_full`/`varied_masked` have identical token data within each pair; the only difference is the training attention block from matched target event to its source event.

Seeds: [42, 43, 100]; epochs=300; n_seqs=500; probes=256; model d=64, heads=2, layers=3.

## Mean final metrics

| condition | copy_gain | content_gain | content_T | content_U | matched_train | unmatched_train | loss |
|---|---|---|---|---|---|---|---|
| unique | -0.000±0.001 | -0.000±0.001 | 3.478±0.001 | 3.478±0.001 |  | 3.463±0.004 | 1.832±0.002 |
| support_control | +0.000±0.000 | +0.000±0.000 | 3.465±0.004 | 3.466±0.004 | 3.466±0.008 | 3.462±0.006 | 1.849±0.001 |
| repeat_full | +1.712±0.104 | +0.343±0.061 | 3.506±0.065 | 3.849±0.024 | 1.958±0.066 | 3.737±0.041 | 1.818±0.000 |
| repeat_masked | +0.001±0.000 | +0.000±0.000 | 3.469±0.007 | 3.469±0.007 | 3.456±0.007 | 3.468±0.005 | 1.849±0.001 |
| varied_full | +0.343±0.012 | +1.799±0.053 | 2.062±0.021 | 3.861±0.074 | 1.945±0.048 | 3.743±0.033 | 1.818±0.001 |
| varied_masked | +0.001±0.000 | +0.000±0.000 | 3.471±0.006 | 3.471±0.007 | 3.459±0.003 | 3.463±0.004 | 1.844±0.009 |
| wrong_full | +0.000±0.000 | +0.000±0.000 | 3.474±0.005 | 3.474±0.005 | 3.460±0.009 | 3.466±0.007 | 1.849±0.002 |

## Mean source-specific contrasts using U_nomatch

Positive excess true-source cost means the true source is worse relative to the unrelated-source control: `(T_A-T_B)-(U_A-U_B) = -(gain_A-gain_B)`.

| contrast | relation | T_delta | U_delta | excess_true_cost | gain_delta |
|---|---|---:|---:|---:|---:|
| repeat_full − repeat_masked | content | +0.037±0.070 | +0.380±0.031 | -0.343±0.061 | +0.343±0.061 |
| varied_full − varied_masked | content | -1.408±0.021 | +0.391±0.073 | -1.799±0.053 | +1.799±0.053 |
| repeat_full − unique | content | +0.028±0.066 | +0.371±0.025 | -0.343±0.061 | +0.343±0.061 |
| repeat_full − support_control | content | +0.040±0.069 | +0.383±0.028 | -0.343±0.061 | +0.343±0.061 |
| varied_full − unique | content | -1.416±0.021 | +0.384±0.074 | -1.799±0.053 | +1.799±0.053 |
| varied_full − support_control | content | -1.403±0.021 | +0.396±0.073 | -1.799±0.053 | +1.799±0.053 |
| repeat_full − varied_full | content | +1.444±0.079 | -0.012±0.065 | +1.456±0.027 | -1.456±0.027 |
| repeat_full − repeat_masked | copy | -1.335±0.088 | +0.376±0.033 | -1.711±0.104 | +1.711±0.104 |
| varied_full − varied_masked | copy | +0.035±0.058 | +0.377±0.060 | -0.343±0.012 | +0.343±0.012 |
| repeat_full − unique | copy | -1.346±0.085 | +0.366±0.039 | -1.712±0.104 | +1.712±0.104 |
| repeat_full − support_control | copy | -1.333±0.084 | +0.378±0.036 | -1.711±0.103 | +1.711±0.103 |
| varied_full − unique | copy | +0.028±0.059 | +0.371±0.062 | -0.343±0.013 | +0.343±0.013 |
| varied_full − support_control | copy | +0.040±0.058 | +0.383±0.060 | -0.343±0.012 | +0.343±0.012 |
| repeat_full − varied_full | copy | -1.374±0.033 | -0.005±0.067 | -1.368±0.098 | +1.368±0.098 |

## Per-seed files

- seed 42: `seed42/analysis.md`, `seed42/summary.json`
- seed 43: `seed43/analysis.md`, `seed43/summary.json`
- seed 100: `seed100/analysis.md`, `seed100/summary.json`
