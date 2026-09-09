# corrected seedxdata decomposition: Corrected seed x data decomposition of row-level improvement

## Why this step was necessary

shared private fitting index produced a valuable per-row loss asset, but its interpretation was too strong.  The shared private fitting index table itself shows that, for DeBERTa heldout 60→100M, the same-data cross-seed partial correlation for VIEW was +0.370 while the same-seed cross-data partial correlation for seed43022 was +0.369.  A data substitution and a seed change therefore perturbed the row-level improvement direction by nearly the same amount.  This step evaluated the missing fourth cell, `D_C_43122`, from the existing frontier_consolidation checkpoint `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122`, then recomputed the design as 2 seeds × 2 data conditions.

No new training, official benchmark scoring, upload, or leaderboard action was performed.

## New forward-pass cell

`D_C_43122` was scored with the same deterministic masks as shared private fitting index on 6,992 heldout rows and 5,000 filler rows at 60/70/80/90/100M.

Heldout mean losses:

| checkpoint | mean loss |
|---|---:|
| 60M | 2.6803 |
| 70M | 2.6054 |
| 80M | 2.5329 |
| 90M | 2.5107 |
| 100M | 2.5037 |

This makes the DeBERTa row-level design complete:

| cell | data | seed | 60→100M mean improvement |
|---|---|---:|---:|
| D_V_43022 | VIEW | 43022 | 0.1673 |
| D_V_43122 | VIEW | 43122 | 0.1574 |
| D_C_43022 | CLEAN | 43022 | 0.1712 |
| D_C_43122 | CLEAN | 43122 | 0.1767 |

## Corrected pairwise reading

Heldout 60→100M partial correlations, controlling the two compared arms' 60M losses:

| comparison | meaning | partial r |
|---|---|---:|
| V43022 × V43122 | same data, different seed | 0.370 |
| C43022 × C43122 | same data, different seed | 0.392 |
| V43022 × C43022 | same seed, different data | 0.369 |
| V43122 × C43122 | same seed, different data | 0.368 |

Average same-data cross-seed r = 0.381.  Average same-seed cross-data r = 0.369.  Difference = -0.012.  Thus the shared private fitting index 0.37-versus-0.90 contrast should be read as a difference in trajectory determinism at these checkpoints and losses, not as evidence that DeBERTa is specifically data-responsive.

The 80→100M window is even less seed/data-stable: average same-data cross-seed r = 0.115, average same-seed cross-data r = 0.112.

## Stable row-level VIEW-minus-CLEAN effect is weak

The direct reproducibility of the row-level data effect is the correlation between `VIEW-CLEAN` contrast vectors computed independently in the two seeds.  It uses no shared arms.

| effect reproducibility | 60→100M raw r | 60→100M residualized r | 80→100M raw r | 80→100M residualized r |
|---|---:|---:|---:|---:|
| data effect `(V-C)` across seeds | 0.025 | 0.057 | 0.031 | 0.033 |
| seed effect `(43022-43122)` across data | 0.014 | 0.036 | 0.033 | 0.029 |

The stable row-level data-contrast direction is therefore at most weak in this instrument.  A VIEW/CLEAN benchmark contrast, if it replicates across seeds, is probably not carried by a simple reproducible direction in heldout-row MLM-loss improvement.

## Shared-arm correlations must be read against their algebraic null

The earlier "impressionable rows" number correlated two differences sharing `D_V_43022`: `corr(V43022-C43022, V43022-V43122)`.  A null model with independent equal-variance cell noise around a common direction predicts about 0.5 because both differences contain the same arm.  With the empirical row distributions, the permutation null is also almost exactly the observed value:

| contrast correlation | observed r | permutation null mean | null p05–p95 | observed - null mean |
|---|---:|---:|---:|---:|
| `corr(V43022-C43022, V43022-V43122)` 60→100M | 0.494 | 0.501 | 0.489–0.512 | -0.007 |
| residualized same contrast | 0.482 | 0.493 | 0.481–0.505 | -0.011 |

So shared private fitting index's "data-responsive and seed-sensitive rows overlap" interpretation is not supported; the correlation is essentially the shared-arm null.

## Decomposition magnitudes

Orthogonal 2×2 row-level decomposition of positive improvement, heldout 60→100M:

| component | mean | sd across rows | residualized sd across rows |
|---|---:|---:|---:|
| common improvement | 0.1681 | 0.0854 | 0.0773 |
| data main `(VIEW-CLEAN)/2` contribution | -0.0058 | 0.0574 | 0.0470 |
| seed main `(43022-43122)/2` contribution | 0.0011 | 0.0568 | 0.0460 |
| seed×data interaction | 0.0038 | 0.0560 | 0.0444 |

The common row-level improvement is real, but the data main component is not larger than the seed component or the interaction.  This is the opposite of a clean row-level data-steering result.

## Difficulty-controlled text-property check

Feature correlations are saved in `heldout_text_feature_partial_correlations.csv`.  In the corrected design, simple punctuation/length features do not by themselves rescue a strong seed-stable row-level data-effect story; the next useful reading is at benchmark-item or representation level, not another scalar text-property claim from these rows.

## Consequence for the data-efficient learning principle

The shared private fitting index dataset remains valuable, but its correct conclusion is a fork:

1. If VIEW/CLEAN benchmark improvements replicate across seeds while this heldout-row loss instrument shows little stable data-specific direction, then the conversion mechanism is probably representation-level: the data changes reusable coordinates or hidden-state geometry shared across many rows, not a simple list of heldout rows whose MLM loss improves.
2. If VIEW/CLEAN benchmark improvements do not replicate beyond the seed floor exposed by the register experiments, then the fixed-budget substitution ledger must be rebuilt around seed-stable sub-benchmark facts rather than single-seed half-point aggregate contrasts.

The next tests should therefore (i) link the corrected seed×data decomposition to per-item benchmark movement, and (ii) move the conversion measurement to hidden states/representational coordinates if benchmark effects are seed-stable.  RoBERTa still needs an actual model-seed pair before its VIEW/CLEAN trajectory determinism can be called data-inertness; the `n=3` entries in the earlier analysis terminal-rate ladder are row/mask replicates, not independent RoBERTa trainings.

## Files

- New `D_C_43122` per-row losses: `experiments/archive/relation_learning/data/seedxdata_loss/D_C_43122_per_row_losses.npz`
- Full corrected analysis: `experiments/archive/relation_learning/data/seedxdata_decomposition/seedxdata_decomposition.json`
- Window summary: `experiments/archive/relation_learning/data/seedxdata_decomposition/seedxdata_window_summary.csv`
- Feature partial correlations: `experiments/archive/relation_learning/data/seedxdata_decomposition/heldout_text_feature_partial_correlations.csv`
- Source means: `experiments/archive/relation_learning/data/seedxdata_decomposition/heldout_source_means.csv`
- Script: `experiments/archive/relation_learning/scripts/seedxdata_decomposition.py`
