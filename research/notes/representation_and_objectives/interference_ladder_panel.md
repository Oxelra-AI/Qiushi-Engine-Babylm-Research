# ewok domain link interference ladder panel

Status: **SCORED** (19 targets, 800 frames)

## Crossed success by condition

| target | no_context | explicit_fin | last_event | consistent_p | distractor_e | contradict_b | contradict_t | contradict_r | explicit_ove | reported_eve |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_77M | 0.000 | 1.000 | 0.550 | 1.000 | 0.388 | 0.000 | 0.000 | 0.000 | 1.000 | 0.375 |
| scale1p75_78M | 0.000 | 1.000 | 0.512 | 1.000 | 0.362 | 0.000 | 0.000 | 0.000 | 0.750 | 0.300 |
| scale1p75_79M | 0.000 | 1.000 | 0.338 | 1.000 | 0.163 | 0.000 | 0.000 | 0.000 | 1.000 | 0.188 |
| scale1p75_80M | 0.000 | 1.000 | 0.588 | 1.000 | 0.350 | 0.000 | 0.000 | 0.000 | 0.875 | 0.375 |
| scale1p75_81M | 0.000 | 1.000 | 0.400 | 1.000 | 0.225 | 0.000 | 0.000 | 0.000 | 1.000 | 0.188 |
| scale1p75_82M | 0.000 | 1.000 | 0.500 | 1.000 | 0.250 | 0.000 | 0.000 | 0.000 | 1.000 | 0.287 |
| scale1p75_83M | 0.000 | 1.000 | 0.537 | 1.000 | 0.325 | 0.000 | 0.000 | 0.000 | 1.000 | 0.275 |
| scale1p75_100M | 0.000 | 1.000 | 0.613 | 1.000 | 0.338 | 0.000 | 0.000 | 0.000 | 1.000 | 0.425 |
| legal16k_base_80M_seed43022 | 0.000 | 1.000 | 0.887 | 1.000 | 0.487 | 0.000 | 0.000 | 0.000 | 1.000 | 0.375 |
| legal16k_base_100M_seed43022 | 0.000 | 1.000 | 0.988 | 1.000 | 0.688 | 0.000 | 0.000 | 0.000 | 1.000 | 0.625 |
| legal16k_80M_seed43122 | 0.000 | 1.000 | 0.650 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.875 | 0.425 |
| legal16k_100M_seed43122 | 0.000 | 1.000 | 0.713 | 1.000 | 0.425 | 0.000 | 0.000 | 0.000 | 0.875 | 0.500 |
| legal40k_8x480_100M_seed43022 | 0.000 | 1.000 | 0.362 | 1.000 | 0.312 | 0.000 | 0.000 | 0.000 | 1.000 | 0.263 |
| legal40k_depth12_100M_seed43022 | 0.000 | 1.000 | 0.037 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 |
| fw_compact_100M_seed43022 | 0.000 | 1.000 | 0.062 | 1.000 | 0.263 | 0.000 | 0.000 | 0.000 | 0.875 | 0.037 |
| fw_rowblock_100M_seed43022 | 0.000 | 1.000 | 0.163 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 | 0.062 |
| mlm_only_20M | 0.000 | 1.000 | 0.113 | 0.875 | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 | 0.075 |
| coupled_aligned_20M | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 | 0.000 |
| coupled_shuffled_20M | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## EWoK correlations by condition

| condition | n | EWoK acc Pearson | EWoK acc Spearman | EWoK SF Pearson | EWoK SF Spearman |
|---|---:|---:|---:|---:|---:|
| no_context | 11 | n/a | n/a | n/a | n/a |
| explicit_final | 11 | n/a | n/a | n/a | n/a |
| last_event | 11 | -0.1284 | -0.2727 | -0.0547 | 0.0545 |
| consistent_prior | 11 | 0.2209 | 0.2000 | -0.5657 | -0.5000 |
| distractor_event | 11 | 0.0281 | -0.2202 | -0.2185 | -0.0092 |
| contradict_bare | 11 | n/a | n/a | n/a | n/a |
| contradict_temporal | 11 | n/a | n/a | n/a | n/a |
| contradict_reinforced | 11 | n/a | n/a | n/a | n/a |
| explicit_override | 11 | 0.0619 | 0.0636 | -0.3928 | -0.3424 |
| reported_event | 11 | -0.2101 | -0.4384 | 0.0145 | 0.2557 |

## D-state consistency

- scale1p75_82M/last_event: D-state ref=0.41, ladder=0.500, delta=0.090
- scale1p75_82M/contradict_bare: D-state ref=0.00, ladder=0.000, delta=0.000
- scale1p75_82M/explicit_final: D-state ref=1.00, ladder=1.000, delta=0.000
- legal16k_base_100M_seed43022/last_event: D-state ref=0.89, ladder=0.988, delta=0.098
- legal16k_base_100M_seed43022/contradict_bare: D-state ref=0.00, ladder=0.000, delta=0.000
- legal16k_base_100M_seed43022/explicit_final: D-state ref=1.00, ladder=1.000, delta=0.000
- legal40k_depth12_100M_seed43022/last_event: D-state ref=0.03, ladder=0.037, delta=0.007
- legal40k_depth12_100M_seed43022/contradict_bare: D-state ref=0.00, ladder=0.000, delta=0.000
- legal40k_depth12_100M_seed43022/explicit_final: D-state ref=1.00, ladder=1.000, delta=0.000

JSON: `experiments/archive/representation_and_objectives/data/interference_ladder_scores/interference_ladder_panel_summary.json`
