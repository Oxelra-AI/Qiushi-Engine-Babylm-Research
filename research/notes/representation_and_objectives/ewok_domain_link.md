# ewok domain link EWoK domain link

Status: **AGGREGATED** from already-scored EWoK records; no new inference.

| target | ladder last_event | override gap | EWoK acc | EWoK stable frac | material acc | material stable frac | agent acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| legal16k_base_100M_seed43022 | 0.9875 | 1.0000 | 0.5075 | 0.3304 | 0.4935 | 0.2805 | 0.5136 |
| scale1p75_100M | 0.6125 | 1.0000 | 0.4965 | 0.3519 | 0.5039 | 0.3416 | 0.5014 |
| legal40k_8x480_100M_seed43022 | 0.3625 | 1.0000 | n/a | n/a | n/a | n/a | n/a |
| legal40k_depth12_100M_seed43022 | 0.0375 | 0.5000 | n/a | n/a | n/a | n/a | n/a |
| fw_compact_100M_seed43022 | 0.0625 | 0.8750 | 0.5043 | 0.3416 | 0.4740 | 0.3390 | 0.5149 |
| fw_rowblock_100M_seed43022 | 0.1625 | 0.8750 | 0.4957 | 0.3257 | 0.5182 | 0.2247 | 0.4986 |
| mlm_only_20M | 0.1125 | 0.1250 | 0.4992 | 0.3431 | 0.4636 | 0.3545 | 0.5113 |
| coupled_aligned_20M | 0.0000 | 0.0000 | 0.4982 | 0.3106 | 0.4286 | 0.4208 | 0.5050 |
| coupled_shuffled_20M | 0.0000 | 0.0000 | 0.5018 | 0.2936 | 0.5156 | 0.2610 | 0.5000 |

## Correlations

| metric | n | Pearson(last_event) | Pearson(action override gap) |
|---|---:|---:|---:|
| ewok_accuracy | 7 | 0.3903 | 0.1672 |
| ewok_stable_frac | 7 | 0.4109 | 0.6367 |
| material_accuracy | 7 | 0.2959 | 0.4585 |
| material_stable_frac | 7 | -0.2083 | -0.3675 |
| agent_accuracy | 7 | 0.2481 | 0.1739 |

JSON: `experiments/archive/representation_and_objectives/data/ewok_domain_link/ewok_domain_link.json`
