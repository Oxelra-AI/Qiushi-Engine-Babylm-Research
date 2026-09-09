# earlier analysis beta/alpha readout for recipient-sensitive selection

## Corrected geometry

For every recipient pair, define `U` as the new-over-source margin in the target-updated context and `R` as the source-over-new margin in the distractor-updated context. Let `beta=(U+R)/2`, `alpha=(U-R)/2`, and `gamma=beta-|alpha|`. Both answers are correct exactly when `gamma>0`.

Balanced answer CE already rewards the same contextual distinction because its binary-candidate form is `softplus(-U)+softplus(-R)`. It is therefore wrong to say that CE is indifferent to shared preference and recipient-sensitive selection. The paired loss `softplus(-2 beta)` can increase direct pressure on `beta`, but it does not reduce `|alpha|`; a run with higher mean beta can still fail RETAIN if shared preference remains large.

Figures saved:
- `figures/loss_geometry_beta_alpha.png`
- `figures/template_held_beta_alpha_trajectories.png`
- `figures/natural_held_beta_alpha_partial.png`
- `figures/natural_baseline_pair_distribution.png`

## Existing endpoint readout from available aggregate outputs

| Source | joint | U | R | beta | alpha | gamma from means |
|---|---:|---:|---:|---:|---:|---:|
| template/baseline_held | 0/12 | +0.065 | -0.056 | +0.005 | +0.061 | -0.056 |
| template/ce_only/held/e0 | 0/12 | +0.065 | -0.056 | +0.005 | +0.061 | -0.056 |
| template/ce_only/held/e500 | 3/12 | +8.698 | -4.743 | +1.977 | +6.720 | -4.743 |
| template/paired_only/held/e0 | 0/12 | +0.065 | -0.056 | +0.005 | +0.061 | -0.056 |
| template/paired_only/held/e500 | 3/12 | +4.303 | -2.647 | +0.828 | +3.475 | -2.647 |
| template/combined/held/e0 | 0/12 | +0.065 | -0.056 | +0.005 | +0.061 | -0.056 |
| template/combined/held/e500 | 3/12 | +8.713 | -4.516 | +2.099 | +6.614 | -4.516 |
| natural_partial/ce_only/held/e0 | 0/15 | +1.162 | -1.281 | -0.059 | +1.221 | -1.281 |
| natural_partial/ce_only/held/e60 | 0/15 | +2.155 | -2.069 | +0.043 | +2.112 | -2.069 |
| natural_partial/ce_only/held/e200 | 1/15 | +1.962 | -1.621 | +0.170 | +1.792 | -1.621 |
| natural_partial/combined_partial/held/e0 | 0/15 | +1.162 | -1.281 | -0.059 | +1.221 | -1.281 |
| natural_partial/combined_partial/held/e60 | 0/15 | +1.577 | -1.580 | -0.002 | +1.579 | -1.580 |

The completed decomposition and paired context template run illustrates the corrected reading: CE and combined both reached 3/12 held joint answers. Combined raised held beta only slightly relative to CE (+2.099 vs +1.977), while |alpha| stayed much larger than beta (6.614 vs 6.720 for CE); the negative gamma from means explains the persistent RETAIN failure.

## Natural earlier analysis baseline pair distribution
All 55 pilot pairs: joint 3/55, mean beta=-0.009, mean |alpha|=+3.324, mean gamma=-3.333, gamma-positive pairs=3/55.

| Subset | n | joint | mean beta | mean |alpha| | mean gamma | gamma>0 |
|---|---:|---:|---:|---:|---:|---:|
| same_token_length | 15 | 1/15 | +0.012 | +2.643 | -2.631 | 1/15 |
| unequal_token_length | 40 | 2/40 | -0.017 | +3.579 | -3.597 | 2/40 |
| target_earlier | 26 | 0/26 | -0.195 | +3.473 | -3.668 | 0/26 |
| target_later | 29 | 3/29 | +0.157 | +3.190 | -3.033 | 3/29 |
| answer_not_elsewhere_both | 1 | 0/1 | -0.025 | +0.195 | -0.220 | 0/1 |

The earlier analysis pilot rows start with a strong shared candidate preference and weak pair-level gamma. Equal token length alone does not solve this, and role-position subsets remain weak. These rows are useful for pipeline testing but are not yet strong evidence about scalable ALN-preserving training.

## Implication for pending and future runs

The running multi-seed and natural combined experiments should be read by joint correctness and pair-level gamma whenever those records are available. A higher average beta is useful only if beta overtakes |alpha| on held pairs and both UPDATE and RETAIN are correct. If the additional paired term does not accomplish that, it means this objective did not solve transferable selection on the tested substrate; it does not decide among experience support, row validity, optimization, or representation as the deeper cause.
