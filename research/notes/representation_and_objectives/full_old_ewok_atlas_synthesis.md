# Full old EWoK margin atlas synthesis

This is mechanism evidence for the inherited-tokenizer coordinate only. It does not decide the compliant corrected-tokenizer endpoint; the two official evaluations remain the decisive evidence.

## Full-coordinate movement
- Rows: 7618; model keys/pattern bit order: clean430 / reinv430 / clean431 / reinv431.
- Micro EWoK accuracies: clean430 50.0131, reinv430 51.7196, clean431 51.1814, reinv431 50.4988.
- Micro treatment effects: seed43022 +1.7065 pp, seed43122 -0.6826 pp; interaction TE431-TE430 -2.3891 pp.
- Margin correlations: clean seeds r=0.788, reinvest seeds r=0.694, treatment-effect margins across seeds r=0.079.
- Negative accuracy-interaction rows: 1766 (23.18%); all four margins |m|<1 in only 9.29% of those rows.
- Seed-polarized patterns: 0110 count 268 (compact helps seed43022 while seed43122 loses); 1001 count 237 (opposite direction).

## Domain table
| Domain | n | clean430 | reinv430 | clean431 | reinv431 | TE430 | TE431 | interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| material-dynamics | 770 | 48.31 | 58.18 | 54.29 | 46.62 | +9.87 | -7.66 | -17.53 |
| physical-dynamics | 120 | 52.50 | 60.00 | 62.50 | 59.17 | +7.50 | -3.33 | -10.83 |
| spatial-relations | 490 | 44.90 | 47.55 | 48.98 | 43.27 | +2.65 | -5.71 | -8.37 |
| physical-interactions | 556 | 46.94 | 51.26 | 49.64 | 48.02 | +4.32 | -1.62 | -5.94 |
| social-relations | 1548 | 49.68 | 49.74 | 51.94 | 49.87 | +0.06 | -2.07 | -2.13 |
| quantitative-properties | 314 | 54.78 | 55.41 | 57.64 | 57.32 | +0.64 | -0.32 | -0.96 |
| physical-relations | 818 | 48.78 | 51.10 | 47.92 | 49.76 | +2.32 | +1.83 | -0.49 |
| agent-properties | 2210 | 50.95 | 49.95 | 51.00 | 51.72 | -1.00 | +0.72 | +1.72 |
| social-properties | 328 | 51.52 | 55.49 | 47.87 | 54.57 | +3.96 | +6.71 | +2.74 |
| social-interactions | 294 | 59.52 | 53.74 | 55.10 | 55.78 | -5.78 | +0.68 | +6.46 |
| material-properties | 170 | 49.41 | 56.47 | 39.41 | 54.71 | +7.06 | +15.29 | +8.24 |

## Largest domain interactions
- Most negative TE431-TE430: material-dynamics -17.53 pp; physical-dynamics -10.83 pp; spatial-relations -8.37 pp; physical-interactions -5.94 pp; social-relations -2.13 pp; quantitative-properties -0.96 pp
- Most positive TE431-TE430: material-properties +8.24 pp; social-interactions +6.46 pp; social-properties +2.74 pp; agent-properties +1.72 pp

## Reusable row selections
- `old_negative_interaction_rows_selection.csv`: all old-coordinate rows where compact-view treatment was less favorable for seed43122 than seed43022.
- `old_seed_polarized_0110_rows_selection.csv`: rows with clean430 wrong, reinv430 correct, clean431 correct, reinv431 wrong.
- `stable_high_margin_control_rows_selection.csv`: up to 1000 stable high-margin rows from patterns 1111/0000, for later comparison with corrected-tokenizer margins if needed.

## Scientific reading
The full atlas strengthens the mechanism picture from ewok tokenizer eval interface audit: compact-view reinvestment can push relational preferences in a seed-specific way, especially in material dynamics, physical dynamics, spatial relations, physical interactions, and social relations. This is not reducible to a few near-zero choices. Corrected-tokenizer full official EWoK should therefore be read against these old relation patterns, while remembering that the old 42.033 endpoint itself is not a compliant submission because the tokenizer was learned outside the Strict-Small 10M budget.
