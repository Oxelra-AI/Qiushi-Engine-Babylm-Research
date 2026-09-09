# official row association overwrite decomposition

Status: **NO_TRAINING_DECOMPOSITION**

This analysis addresses a confound in the official-row comparison: ewok domain link contrasted explicit final-state language with action-mediated contradiction, so it could have conflated state supersession with evidence form.

Frozen probe: 1040 frames = 80 base frames × 13 conditions.
Content SHA256: `ce45c171554c5b468e0ef6bd26eb4b8b58330a6bac35e624949a38e7707bac68`; frame SHA256: `ce45c171554c5b468e0ef6bd26eb4b8b58330a6bac35e624949a38e7707bac68`.
Target-word-free context hit count: 0 (should be 0 for literal open/closed/empty/full words).

## Synthetic decomposition by target

| target | action crossed | direct combined crossed | direct residual median | direct neg-resid frac | tf action crossed | tf combined crossed | tf residual median | tf neg-resid frac | update-sensitive EWoK acc | update-sensitive stable frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal16k_base_100M_seed43022 | 0.9875 | 0.0000 | -1.2293 | 1.0000 | 0.8250 | 0.0500 | -0.5544 | 0.6375 | 0.4806 | 0.3183 |
| scale1p75_82M | 0.5000 | 0.0000 | -0.6761 | 0.8125 | 0.3250 | 0.2500 | -0.7987 | 1.0000 | n/a | n/a |
| scale1p75_100M | 0.6125 | 0.0000 | -0.7033 | 0.7375 | 0.3750 | 0.2375 | -0.7173 | 1.0000 | 0.5043 | 0.3280 |
| legal40k_8x480_100M_seed43022 | 0.3625 | 0.0000 | -0.5857 | 0.9625 | 0.6750 | 0.0625 | -0.9618 | 0.8875 | 0.5067 | 0.4517 |
| legal40k_depth12_100M_seed43022 | 0.0375 | 0.0000 | -0.3530 | 0.6250 | 0.0000 | 0.0000 | -0.1274 | 0.1875 | 0.5135 | 0.4225 |
| fw_rowblock_100M_seed43022 | 0.1625 | 0.0000 | -0.1862 | 0.3000 | 0.2375 | 0.0000 | -0.5909 | 0.6250 | 0.5301 | 0.2258 |
| fw_compact_100M_seed43022 | 0.0625 | 0.0000 | -0.8489 | 0.6500 | 0.6250 | 0.0000 | -0.9372 | 0.7000 | 0.4667 | 0.3613 |
| mlm_only_20M | 0.1125 | 0.0000 | 0.0174 | 0.0000 | 0.0125 | 0.0125 | -0.1652 | 0.0000 | 0.4753 | 0.3409 |
| coupled_shuffled_20M | 0.0000 | 0.0000 | 0.0474 | 0.0000 | 0.0000 | 0.0000 | -0.0026 | 0.0000 | 0.5129 | 0.2742 |

## Cross-target association with update-sensitive official rows

| synthetic metric vs official surface | n | Pearson | Spearman |
|---|---:|---:|---:|
| direct_nonadditive_median__material_acc | 8 | 0.1707 | 0.3333 |
| direct_nonadditive_median__material_stable_frac | 8 | -0.1202 | -0.1190 |
| direct_nonadditive_median__update_acc | 8 | 0.4394 | 0.4762 |
| direct_nonadditive_median__update_stable_frac | 8 | -0.2538 | -0.2619 |
| last_action_crossed__material_acc | 8 | -0.0863 | -0.3095 |
| last_action_crossed__material_stable_frac | 8 | -0.0697 | 0.0476 |
| last_action_crossed__update_acc | 8 | -0.1981 | -0.2143 |
| last_action_crossed__update_stable_frac | 8 | -0.0175 | -0.0714 |
| tf_action_crossed__material_acc | 8 | -0.2509 | -0.4551 |
| tf_action_crossed__material_stable_frac | 8 | 0.1274 | 0.0599 |
| tf_action_crossed__update_acc | 8 | -0.3962 | -0.4671 |
| tf_action_crossed__update_stable_frac | 8 | 0.2233 | 0.1557 |
| tf_nonadditive_median__material_acc | 8 | 0.2109 | 0.2857 |
| tf_nonadditive_median__material_stable_frac | 8 | -0.2608 | -0.2619 |
| tf_nonadditive_median__update_acc | 8 | 0.2086 | 0.3095 |
| tf_nonadditive_median__update_stable_frac | 8 | -0.2543 | -0.3810 |

## Interpretation

Direct-state contradiction rows remain extremely nonadditive if the combined prior+action margin is far below prior-only + action-only minus no-context. A robust training route would require the same negative residual under target-word-free paraphrases and a meaningful association with update-sensitive official rows. If target-word-free residuals vanish or official association is weak/inverted, the ewok domain link universal zero should be treated as a synthetic boundary rather than a sufficient training target.

JSON: `experiments/archive/representation_and_objectives/data/overwrite_decomposition/overwrite_decomposition_summary.json`
