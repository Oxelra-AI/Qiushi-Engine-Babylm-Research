# fw globalpiqa relevant substrate — GlobalPIQA parallel failure anatomy

## Why this matters

The best compliant endpoint (`legal40k8x480_43022`) is already ahead of the visible leader on BLiMP, Supplement, and Reading, but remains ~0.659 Overall below 41.80. Since Overall is the mean of 9 columns, this is only ~5.93 summed column-points. GlobalPIQA alone is ~5.00 column-points behind the leader, so understanding its failure is directly SOTA-relevant.

## Core measurement

The official full-eval GlobalPIQA data used here has 103 `parallel` English four-choice items and 100 `nonparallel` English two-choice items. Existing prediction artifacts were read directly; no scorer or model evaluation was rerun.

| endpoint | parallel | nonparallel | aggregate | notes |
|---|---:|---:|---:|---|
| `legal40k8x480_43022` | 22.33 | 47.00 | 34.67 | legal40k 8x480 seed43022 (best compliant endpoint) |
| `legal40k8x480_43122` | 22.33 | 42.00 | 32.17 | legal40k 8x480 seed43122 |
| `legal16k8x480_43022` | 26.21 | 46.00 | 36.11 | legal16k 8x480 seed43022 |
| `legal16k8x480_43122` | 26.21 | 51.00 | 38.61 | legal16k 8x480 seed43122 |
| `depth12x384_43022` | 24.27 | 47.00 | 35.64 | legal40k 12x384 depth seed43022 |
| `sgcr12x384_43022` | 23.30 | 45.00 | 34.15 | exact-prefix SGCR 12x384 seed43022 |
| `inherited16k_noncompliant_43022` | 25.24 | 46.00 | 35.62 | inherited-tokenizer compact reinvest seed43022 (non-submission evidence) |
| `inherited16k_noncompliant_43122` | 24.27 | 46.00 | 35.14 | inherited-tokenizer compact reinvest seed43122 (non-submission evidence) |
| `semantic_view_treatment_100M` | 22.33 | 48.00 | 35.17 | SimpleWiki semantic-view treatment chck_100M |
| `semantic_view_control_100M` | 28.16 | 46.00 | 37.08 | SimpleWiki packet-local control chck_100M |

## What the pattern says

1. The low score is concentrated in `global_piqa_parallel`, not in the nonparallel PIQA-like set. The best compliant model gets only 22.33 on parallel but 47.00 on nonparallel.
2. This is not only a compliant-tokenizer artifact: the inherited-tokenizer 42.033 model also remains weak on parallel, so GlobalPIQA is a real knowledge/reasoning/data-substrate weakness of the lineage.
3. The parallel set is small but high-leverage: 103 four-choice English items cover physical consequences, time arithmetic/order, spatial directions, object interactions, and tool affordances. A 10–20 point movement in the parallel subset changes the aggregate GlobalPIQA by 5–10 points and Overall by 0.56–1.11.
4. Since official predictions contain only the winning option, this analysis cannot inspect probability margins. The next stronger cheap check, if needed, is a pseudo-log-likelihood/margin reader over these 103 rows for completed candidate checkpoints, not another 100M training run.

## Files

- JSON: `experiments/archive/representation_and_objectives/data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json`
- endpoint summary CSV: `experiments/archive/representation_and_objectives/data/globalpiqa_parallel_anatomy/globalpiqa_endpoint_summary.csv`
- cross-endpoint row table: `experiments/archive/representation_and_objectives/data/globalpiqa_parallel_anatomy/globalpiqa_parallel_cross_endpoint_rows.csv`
