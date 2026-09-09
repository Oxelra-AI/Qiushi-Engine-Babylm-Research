# paired world stress teacher and route tiny GRU sequence baseline

## Purpose

Test a learned sequence parser against the paired-world shortcut concern. This is not BabyLM training.

## Key accuracies

### orig_only

| mode | held original | held held-template | name swap | lost-to hyp | passive hyp | retemplate |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.965 | 0.931 | 0.935 | 0.045 | 0.035 | 0.990 |
| raw | 0.547 | 0.559 | 0.472 | 0.487 | 0.480 | 0.500 |

### augmented_variants

| mode | held original | held held-template | name swap | lost-to hyp | passive hyp | retemplate |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 0.955 | 0.912 | 0.960 | 0.955 | 0.955 | 1.000 |
| raw | 0.502 | 0.505 | 0.500 | 0.500 | 0.502 | 0.500 |

## Files

- summary: `experiments/archive/representation_and_objectives/training/runs/tiny_gru_sequence_baseline_direct/tiny_gru_sequence_baseline_summary.json`
