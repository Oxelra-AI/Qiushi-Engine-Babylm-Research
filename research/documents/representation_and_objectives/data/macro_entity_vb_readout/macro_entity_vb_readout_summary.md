# earlier analysis macro Entity V/B/R readout

This file-only readout integrates official Entity split tables with the CPU binding affected/unaffected probe. It separates the official Entity surface from the counterbalanced binding substrate before deciding whether compact source views support correspondence or merely redistribute finite-budget decisions.

## Coarse Entity operation rows (late 80/90/100M)

| contrast | official all18 Δ pp | zero-op Δ pp | nonzero Δ pp | balanced zero/nonzero Δ pp | pred-option0 shift on all18 |
|---|---:|---:|---:|---:|---:|
| V-R | +4.018 | -9.661 | +6.753 | -1.454 | NA |
| V-B | +2.994 | +3.376 | +2.918 | +3.147 | NA |
| B-R | +1.023 | -13.037 | +3.836 | -4.601 | NA |

## Entity by number of operations

| group | V-R Δ pp | V-B Δ pp | B-R Δ pp |
|---|---:|---:|---:|
| numops_0 | -9.661 | +3.376 | -13.037 |
| numops_1 | +1.967 | +0.724 | +1.243 |
| numops_2 | +3.854 | +1.390 | +2.464 |
| numops_3 | +8.032 | +1.994 | +6.038 |
| numops_4 | +9.343 | +5.159 | +4.184 |
| numops_5 | +10.571 | +5.321 | +5.250 |

## Entity split families

| split group | V-R Δ pp | V-B Δ pp | B-R Δ pp |
|---|---:|---:|---:|
| split_ambiref | +3.574 | +2.948 | +0.625 |
| split_move_contents | +4.119 | +2.688 | +1.431 |
| split_regular | +4.360 | +3.346 | +1.014 |
| ambiref_zero_ops | -4.528 | +3.740 | -8.268 |
| ambiref_nonzero_ops | +5.194 | +2.790 | +2.404 |
| move_contents_zero_ops | -10.013 | +2.907 | -12.920 |
| move_contents_nonzero_ops | +6.945 | +2.644 | +4.302 |
| regular_zero_ops | -14.442 | +3.482 | -17.924 |
| regular_nonzero_ops | +8.120 | +3.319 | +4.801 |

## Counterbalanced binding substrate

| contrast | affected Δ pp | unaffected Δ pp | EEBF Δ pp | affected margin Δ | unaffected margin Δ | signed margin Δ |
|---|---:|---:|---:|---:|---:|---:|
| V-R | +21.354 | -19.792 | +0.781 | +0.796 | -0.860 | +1.656 |
| V-B | +6.944 | -4.167 | +1.389 | +0.107 | +0.062 | +0.045 |
| B-R | +14.410 | -15.625 | -0.608 | +0.689 | -0.922 | +1.611 |

## Broad-family visible vector rows

| contrast | set | net | RMS | |net|/RMS | vector |
|---|---|---:|---:|---:|---|
| D1_BminusCold | stable5_exEntity | +0.896 | +1.293 | +0.693 | BLiMP:+2.590, COMPS:+0.830, EWoK:+0.970, Reading:+0.140, Supplement:-0.050 |
| D1_BminusR | stable5_exEntity | +0.389 | +0.932 | +0.418 | BLiMP:+1.140, COMPS:+1.310, EWoK:-0.780, Reading:-0.445, Supplement:+0.720 |
| D1_VminusB | stable6 | +0.465 | +1.586 | +0.293 | BLiMP:-0.870, COMPS:-1.180, EWoK:-0.150, Entity:+3.100, Reading:+0.070, Supplement:+1.820 |
| D1_VminusB | stable5_exEntity | -0.062 | +1.048 | +0.059 | BLiMP:-0.870, COMPS:-1.180, EWoK:-0.150, Reading:+0.070, Supplement:+1.820 |
| D1_VminusCold | stable5_exEntity | +0.834 | +1.177 | +0.708 | BLiMP:+1.720, COMPS:-0.350, EWoK:+0.820, Reading:+0.210, Supplement:+1.770 |
| D1_VminusR | stable5_exEntity | +0.327 | +1.229 | +0.266 | BLiMP:+0.270, COMPS:+0.130, EWoK:-0.930, Reading:-0.375, Supplement:+2.540 |

## Scientific reading

- Official Entity V-R is strongly operation-skewed in the first basin: zero-op -9.661 pp versus nonzero +6.753 pp. B-R has the same broad shape: zero-op -13.037 pp versus nonzero +3.836 pp.
- Official Entity V-B is positive on both coarse strata: zero-op +3.376 pp and nonzero +2.918 pp. This means V-B is not simply the same zero/nonzero arithmetic as V-R/B-R on the official Entity surface.
- The counterbalanced binding substrate changes that reading: late V-B affected gain is +6.944 pp, unaffected change is -4.167 pp, and balanced EEBF movement is only +1.389 pp. Thus the source-view advantage on binding rows is small and offset-shaped, not a clean affected-with-retained-unaffected record-selection result.
- Visible broad V-B at 80M is not a broad ex-Entity gain: stable5 ex-Entity net is -0.062 pp while RMS is about 1.048 pp. B-C_old is positive on ex-Entity, so broad total V-C can arise from admitting different experience or geometry without showing own-source companion correspondence.
- conservation-vector file reports ex-Entity V-R net across doses [[0.042352, -0.18675000000000044], [0.07712, -0.04362500000000047], [0.111872, 0.06362500000000026]]; the non-Entity mean remains small relative to family-vector motion. This supports a finite-budget redistribution view: many interventions move capabilities between families or item decisions more than they raise all families together.
- A permuted compact companion remains the cleanest future asymmetry only if updated broad V-B survives after missing breadth/clean scoring, or if a stronger binding/state panel shows V-B without an unaffected cost. Current binding evidence alone does not warrant an H100 permuted run.

## Files

- entity_group_summary_csv: `experiments/archive/representation_and_objectives/data/macro_entity_vb_readout/entity_group_late_summary.csv`
- entity_identity_residuals_csv: `experiments/archive/representation_and_objectives/data/macro_entity_vb_readout/entity_identity_residuals.csv`
- binding_late_summary_csv: `experiments/archive/representation_and_objectives/data/macro_entity_vb_readout/binding_late_summary.csv`
- visible_broad_vectors_csv: `experiments/archive/representation_and_objectives/data/macro_entity_vb_readout/visible_broad_vectors.csv`
- summary_json: `experiments/archive/representation_and_objectives/data/macro_entity_vb_readout/macro_entity_vb_readout_summary.json`
- summary_md: `research/documents/representation_and_objectives/data/macro_entity_vb_readout/macro_entity_vb_readout_summary.md`
- source_entity_contrasts: `experiments/archive/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_contrast_rows.csv`
- source_binding_contrasts: `experiments/archive/representation_and_objectives/data/breadth_binding_affunaff_probe/binding_contrast_deltas.csv`
- source_broad_vectors: `experiments/archive/frontier_consolidation/data/conservation_vector_readout/visible_reference_net_norm.csv`
