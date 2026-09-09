# corrected endpoint interpretation — EWoK focus tokenization/margin link

CPU-only analysis on already-scored corrected 100M focused EWoK margins. It does not evaluate a model; it computes tokenizer-length features for the same items.

## Global correlations

- `n_rows`: `553`
- `sent_len_delta_summary`: `{'n': 553, 'mean': 0.659132007233273, 'std': 1.154436668054499, 'min': -1.5, 'p25': 0.0, 'p50': 0.0, 'p75': 1.0, 'max': 6.0}`
- `target_len_delta_summary`: `{'n': 553, 'mean': 0.34900542495479203, 'std': 0.6157104615678803, 'min': -1.0, 'p25': 0.0, 'p50': 0.0, 'p75': 1.0, 'max': 2.0}`
- `candidate_length_asymmetry_delta_summary`: `{'n': 553, 'mean': -0.009041591320072333, 'std': 0.3919506372843785, 'min': -2.0, 'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'max': 2.0}`
- `pearson_sent_len_delta_vs_margin430`: `-0.009840942783389017`
- `pearson_sent_len_delta_vs_margin431`: `0.012517300697807723`
- `pearson_target_len_delta_vs_margin430`: `0.02658795824321266`
- `pearson_target_len_delta_vs_margin431`: `0.028823707463520466`
- `pearson_asym_delta_vs_margin430`: `-0.005369244403627655`
- `pearson_asym_delta_vs_margin431`: `-0.0396306785459274`
- `pearson_sent_len_delta_vs_seed_margin_delta`: `-0.02429546528749745`
- `pearson_target_len_delta_vs_seed_margin_delta`: `-0.0069344084745518245`
- `pearson_asym_delta_vs_seed_margin_delta`: `0.04057118775926308`

## Domain summary

| domain | n | acc430 | acc431 | mean sent Δ | mean target Δ | asym Δ |
|---|---:|---:|---:|---:|---:|---:|
| agent-properties | 45 | 48.89 | 51.11 | 0.489 | 0.200 | 0.000 |
| material-dynamics | 120 | 32.50 | 33.33 | 0.329 | 0.117 | 0.042 |
| material-properties | 45 | 57.78 | 57.78 | 1.000 | 0.867 | 0.000 |
| physical-dynamics | 52 | 57.69 | 42.31 | 0.192 | 0.154 | -0.077 |
| physical-interactions | 69 | 65.22 | 49.28 | 0.370 | 0.290 | 0.043 |
| social-interactions | 45 | 46.67 | 64.44 | 0.511 | 0.289 | -0.133 |
| social-relations | 111 | 43.24 | 54.95 | 1.207 | 0.550 | -0.072 |
| spatial-relations | 66 | 56.06 | 28.79 | 0.992 | 0.439 | 0.076 |

## Interpretation

- Maximum absolute item-level correlation between simple tokenizer-length features and corrected 100M margins/seed-delta is 0.041.
- Simple token length/fragmentation features do not explain the focused EWoK relation-margin behavior; official EWoK movement should be read mainly as learned relation dynamics or vocabulary-target geometry, not crude item length.
