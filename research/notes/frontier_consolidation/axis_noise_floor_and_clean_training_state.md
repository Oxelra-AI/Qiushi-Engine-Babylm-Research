# axis noise floor and clean training state axis/noise-floor correction and clean-training state

## Why this was done

A broad finite-budget conservation-vector hypothesis predicted increasing family-vector displacement with restructuring dose while the broad mean remained small. This framing was too broad: the all-family V-R RMS growth could be algebraically traced to Entity, whose high-dose effect is an operation-sensitive allocation rather than general binding. The present file-only comparison tests whether broad ex-Entity family movement has a reproducible direction or is better treated as the same-coordinate seed/basin floor.

No model loading, training, official evaluation, GlobalPIQA/SuperGLUE/AoA work, upload, leaderboard action, or new H100 task was started in this analysis.

## Main files

- Axis/noise readout script: `experiments/archive/frontier_consolidation/scripts/axis_noise_floor_readout.py`
- Axis/noise summary: `research/documents/frontier_consolidation/data/axis_noise_floor_readout/axis_noise_floor_summary.md`
- Centered-axis supplement: `experiments/archive/frontier_consolidation/scripts/centered_axis_correction.py`
- Centered-axis summary: `research/documents/frontier_consolidation/data/axis_noise_floor_readout/centered_axis_correction.md`
- Endpoint-versus-seed supplement: `experiments/archive/frontier_consolidation/scripts/endpoint_vs_seed_displacement.py`
- Endpoint-versus-seed JSON: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/endpoint_vs_seed_displacement.json`
- independent_review verification: `data/external/independent_review01_verifier1_integration.md`

## Corrected reading of the earlier analysis conservation-vector idea

The original broad form is not supported. For the seed43022 dose V-R common10_80 table:

| dose | rho | stable6 net | stable6 RMS | ex-Entity net | ex-Entity RMS | implied |Entity| | actual Entity |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x | 0.042352 | -0.0625 | 0.5356 | -0.1868 | 0.5308 | 0.5588 | +0.5587 |
| 1.82x | 0.077120 | +0.1414 | 0.6939 | -0.0436 | 0.5920 | 1.0662 | +1.0662 |
| 2.64x/MAX | 0.111872 | +0.3968 | 0.9912 | +0.0636 | 0.5729 | 2.0625 | +2.0625 |

The identity \(|Entity|=\sqrt{6RMS_6^2-5RMS_{5,exEntity}^2}\) exactly recovers the Entity component. Thus the visible all-family V-R radius growth is overwhelmingly the same Entity component already shown in two basins to be zero/nonzero-operation allocation. Outside Entity, radius stays near 0.53-0.59 and net remains near zero.

independent_review added a useful nuance: ex-Entity movement is not absent. From 1x to MAX, the ex-Entity V-R endpoint displacement is L2 1.7748, and because Entity changes by 1.5037, ex-Entity contributes about 58.2% of the squared endpoint-to-endpoint displacement. But this is tangential reorientation, not radius growth, and it is smaller than the observed same-coordinate two-seed 1x ex-Entity separation of L2 2.1463. The 1x-to-MAX ex-Entity cosine is -0.0329; the two 1x seed ex-Entity cosine over common10_80 is -0.4232. Therefore the current evidence does not establish a reproducible broad ex-Entity trade-off direction.

## Two-seed 1x direction test

For earlier analysis two 1x DeBERTa basins, compact-minus-repeat mean vectors over common10_80 are not aligned:

- stable6 cosine -0.0918, centered cosine -0.0890, ||difference||/mean||seed|| 1.4841.
- stable5_exEntity cosine -0.4232, centered cosine -0.5639, ||difference||/mean||seed|| 1.6886.
- common10_80 sign agreement: stable6 agrees in BLiMP, Supplement, Entity and disagrees in EWoK, COMPS, Reading; ex-Entity agrees in only BLiMP/Supplement and disagrees in EWoK/COMPS/Reading.

Late windows show partial alignment, but these are nested slices of the same two trajectories and are dominated by shared positive Entity/Supplement tendencies. They should not be treated as independent evidence for a broad manifold.

## scale1p75 item mechanism and cross route tradeoff PCA correction

scale1p75 item mechanism and cross route tradeoff PCA was fitted after subtracting the archive mean, so a new vector's comparable coordinate is `(delta - mean) dot PC`, not raw `delta dot PC`. The first axis noise floor and clean training state script saved both; the centered-axis supplement makes this explicit.

The stable6 common10_80 1x vectors have similar centered PC2-negative / PC3-positive coordinates, but the family-contribution table shows this similarity is mainly the positive Entity coordinate. After removing Entity, the centered PC1/PC2/PC3 projections are small and sign-unstable:

- seed43022 ex-Entity: PC1 -0.0638, PC2 -0.0371, PC3 +0.0101.
- seed43122 ex-Entity: PC1 +0.0875, PC2 +0.1820, PC3 -0.1039.

Rows omitting GlobalPIQA also use restricted scale1p75 item mechanism and cross route tradeoff axes with large missing loadings, so projection fractions are descriptive only. scale1p75 item mechanism and cross route tradeoff axes do not establish a shared intervention manifold here.

## transition result absorbed

reference and margin state (`experiments/archive/representation_and_objectives/data/conservation_transition_readout`) reinforces the correction. Official macro Entity V-B is real and positive across operation strata (all18 +2.994 pp; zero-op +3.376; nonzero +2.918), but paired binding-state flow does not move examples into both-correct retained-state form: late mean V-B Δ both-correct is -1.389 pp, not-both→both +2.257 pp, both→not-both +3.646 pp, with dominant unaffected-only→affected-only flow +20.312 pp. This means Entity V-B should not by itself trigger the already materialized permuted-companion H100 run.

## Clean training state

The first MAX-geometry clean control completed training to 100M:

- run: `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022`
- word exposure: 100,000,000
- updates: 2,552
- checkpoints: 10, chck_10M through chck_100M, all expected present
- parameter count: 34,467,424
- tokenizer label: `compliant16k_reinvest10M`
- loss_first 9.826911926269531; loss_last 2.446653127670288
- preflight recipe has `extra_init_seed=43022` and `train_rng_seed=43023`; `scientific_metrics.json` stores `seed=43`, which matches the metrics convention for first-basin MAX view/repeat/breadth, not a mismatch.

Matched-geometry V-C recomputation remains conditional on completion of the clean-control scoring.

## How to read incoming results

- Matched-clean scores should be read primarily as a content-placement test: does MAX view or breadth retain a broad ex-Entity V-C/B-C advantage after row/update matching?
- Full breadth scores should be read as V-B, B-C, B-R, V-R by checkpoint and family, with ex-Entity and Entity-operation splits before any broad mechanism claim.
- If broad ex-Entity V-B remains near zero, compact own-source relatedness remains an Entity/state surface effect; the general program should move toward content admission or toward a narrower state/selector boundary, not permuted-companion training.
- If broad ex-Entity V-B becomes robustly positive in the completed breadth table, then second-basin breadth or the already materialized permuted compact companion becomes scientifically justified because it would separate own-source correspondence from compact-text fertility/multiset effects.
