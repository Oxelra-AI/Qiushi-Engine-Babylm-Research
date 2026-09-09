# axis noise floor and clean training state axis/noise-floor readout

File-only analysis of already scored tables. It asks whether broad ex-Entity family-vector movement is a reproducible low-dimensional trade-off, or is better treated as the same-coordinate basin floor before incoming breadth and matched-clean results are read.

## Seed reproducibility of the 1x DeBERTa V-R vector

| window | family set | seed43022 net/RMS | seed43122 net/RMS | cosine | centered cosine | ||diff|| / mean ||seed|| | vector reading |
|---|---|---:|---:|---:|---:|---:|---|
| common10_80 | stable6 | -0.0625/0.5356 | +0.0191/0.6577 | -0.0918 | -0.0890 | 1.4841 | not reproduced |
| common10_80 | stable5_exEntity | -0.1868/0.5308 | -0.1514/0.6060 | -0.4232 | -0.5639 | 1.6886 | not reproduced |
| common10_80 | stable4_exEntity_noReading | -0.3406/0.5534 | -0.1059/0.6567 | -0.3700 | -0.6033 | 1.6581 | not reproduced |
| full10_100 | stable6 | -0.0142/0.5608 | +0.0991/0.7311 | -0.0218 | -0.0186 | 1.4414 | not reproduced |
| full10_100 | stable5_exEntity | -0.1698/0.5105 | -0.0945/0.6433 | -0.5292 | -0.6196 | 1.7524 | not reproduced |
| full10_100 | stable4_exEntity_noReading | -0.3208/0.5279 | -0.0197/0.6918 | -0.4779 | -0.6238 | 1.7247 | not reproduced |
| late80_100 | stable6 | +0.2581/0.6775 | +0.3053/0.9415 | +0.4887 | +0.4175 | 1.0497 | partially aligned |
| late80_100 | stable5_exEntity | +0.0557/0.4778 | +0.1357/0.8932 | +0.1901 | +0.1756 | 1.3558 | not reproduced |
| late80_100 | stable4_exEntity_noReading | -0.0392/0.4879 | +0.3292/0.9462 | +0.3700 | +0.4258 | 1.2409 | partially aligned |
| mature70_100 | stable6 | +0.2944/0.6377 | +0.4085/1.0955 | +0.6017 | +0.5218 | 1.0101 | partially aligned |
| mature70_100 | stable5_exEntity | +0.1107/0.4404 | +0.1692/0.9618 | +0.2719 | +0.2389 | 1.3446 | partially aligned |
| mature70_100 | stable4_exEntity_noReading | +0.0175/0.4289 | +0.3337/1.0472 | +0.4522 | +0.4637 | 1.2670 | partially aligned |

For the dose-matched common10_80 window, the two 1x seed vectors point in different directions. This is the low-cost exclusion that earlier analysis lacked: ex-Entity V-R movement should not be promoted to a shared rotation unless a direction reproduces beyond this floor.

## Entity explains the apparent dose growth in V-R displacement

| dose | rho | stable6 net | RMS6 | ex-Entity net | RMS5 | implied |Entity| | actual Entity | identity error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dose1 | 0.042352 | -0.0625 | 0.5356 | -0.1868 | 0.5308 | 0.5588 | +0.5587 | 2.22e-16 |
| dose1p82 | 0.077120 | +0.1414 | 0.6939 | -0.0436 | 0.5920 | 1.0662 | +1.0662 | 2.22e-16 |
| dose2p64 | 0.111872 | +0.3968 | 0.9912 | +0.0636 | 0.5729 | 2.0625 | +2.0625 | 0.00e+00 |

The displacement growth from dose1 to MAX is therefore the same Entity component already shown to be an operation-sensitive allocation in two DeBERTa basins. Outside Entity, the V-R RMS stays roughly 0.53-0.59 and the net remains close to zero.

## Key projections on scale1p75 item mechanism and cross route tradeoff PCA axes

scale1p75 item mechanism and cross route tradeoff PC1/PC2/PC3 explain approximately 42.9%, 28.0%, and 17.4% of archive delta-vector variance. For rows without GlobalPIQA, projections use the available six-column subspace, so they are descriptive coordinates rather than a complete seven-column reconstruction.

| label | net | RMS | PC1 proj/cos | PC2 proj/cos | PC3 proj/cos | vector |
|---|---:|---:|---:|---:|---:|---|
| seed43022_common10_80_VminusR | -0.0625 | 0.5356 | -0.5910/-0.502 | -0.1972/-0.230 | +0.7147/+0.617 | `{"BLiMP": -0.20375000000000032, "COMPS": 0.28999999999999915, "EWoK": -0.8825000000000012, "Entity": 0.5587499999999999, "Reading": 0.4287500000000003, "Supplement": -0.5662500000000001}` |
| seed43122_common10_80_VminusR | +0.0191 | 0.6577 | -0.4074/-0.282 | -0.1522/-0.145 | +0.8536/+0.600 | `{"BLiMP": -0.8275000000000006, "COMPS": -0.20999999999999996, "EWoK": 0.9425000000000008, "Entity": 0.8712500000000005, "Reading": -0.33312500000000034, "Supplement": -0.32874999999999854}` |
| dose1:V_minus_R:stable6 | -0.0625 | 0.5356 | -0.5910/-0.502 | -0.1972/-0.230 | +0.7147/+0.617 | `{"BLiMP": -0.20375000000000032, "COMPS": 0.28999999999999915, "EWoK": -0.8825000000000012, "Entity": 0.5587499999999999, "Reading": 0.4287500000000003, "Supplement": -0.5662500000000001}` |
| dose1p82:V_minus_R:stable6 | +0.1414 | 0.6939 | +0.3370/+0.221 | -0.8267/-0.745 | +0.7898/+0.526 | `{"BLiMP": 0.5712499999999991, "COMPS": 0.017499999999998295, "EWoK": -1.165, "Entity": 1.0662499999999997, "Reading": 0.13187499999999996, "Supplement": 0.22625000000000028}` |
| dose2p64:V_minus_R:stable6 | +0.3968 | 0.9912 | +0.7102/+0.326 | -1.3104/-0.826 | +1.4068/+0.656 | `{"BLiMP": -0.20500000000000096, "COMPS": 0.09250000000000025, "EWoK": -0.6062499999999993, "Entity": 2.062499999999999, "Reading": -0.06687500000000024, "Supplement": 1.1037500000000016}` |
| D1_VminusB:chck_80M:stable6 | +0.4650 | 1.5864 | +0.9660/+0.277 | -1.8229/-0.718 | +1.9826/+0.578 | `{"BLiMP": -0.8700000000000045, "COMPS": -1.1799999999999997, "EWoK": -0.15000000000000568, "Entity": 3.1000000000000014, "Reading": 0.0699999999999994, "Supplement": 1.8200000000000003}` |
| D1_BminusCold:chck_80M:stable6 | +0.7717 | 1.1820 | +1.6599/+0.639 | -0.7881/-0.417 | -0.2186/-0.085 | `{"BLiMP": 2.5900000000000034, "COMPS": 0.8299999999999983, "EWoK": 0.970000000000006, "Entity": 0.14999999999999858, "Reading": 0.14000000000000057, "Supplement": -0.05000000000000426}` |
| D1_VminusCold:chck_80M:stable6 | +1.2367 | 1.7075 | +2.6259/+0.699 | -2.6110/-0.956 | +1.7640/+0.478 | `{"BLiMP": 1.7199999999999989, "COMPS": -0.3500000000000014, "EWoK": 0.8200000000000003, "Entity": 3.25, "Reading": 0.20999999999999996, "Supplement": 1.769999999999996}` |

The projections do not rescue a single shared broad rotation. Dose V-R changes PC mixture with dose; the visible V-B row is dominated by Entity and Supplement with BLiMP/COMPS losses; B-C points along a different content-placement direction. scale1p75 item mechanism and cross route tradeoff archive vectors occupy multiple axes and include opposing signs.

## Scientific reading

- The earlier analysis all-family V-R RMS growth is not a broad conservation-vector law: the identity |Entity| = sqrt(6*RMS6^2 - 5*RMS5_exEntity^2) recovers the Entity term at each dose, while ex-Entity RMS stays in a narrow band.
- For earlier analysis 1x common10_80 V-R, the two seed mean stable6 vectors have cosine -0.0918 and ||diff||/mean||seed|| 1.4841; ex-Entity cosine is -0.4232 with ratio 1.6886. This does not support a reproduced broad rotation direction.
- For the full 10M-100M ladder, stable6 seed-vector cosine is -0.0218; for mature70_100 it is +0.6017; late80_100 rises to +0.4887 mainly because both seeds share positive Entity/Supplement tendencies while other families still differ.
- scale1p75 item mechanism and cross route tradeoff PCA axes are useful as descriptive coordinates, but the current data do not show that unrelated interventions share one reliable manifold. External dose and V-B vectors project onto these axes in different mixtures; scale1p75 item mechanism and cross route tradeoff archive rows themselves contain several opposing directions rather than a single reusable route.
- The safer current interpretation is that ex-Entity V-R movement is at, or close to, the same-coordinate basin floor until a repeated direction is shown. The high-value incoming result remains the matched clean/content-admission leg and the full breadth readout; any broad mechanism reading must be conditioned on this seed-floor comparison.

## Files
- seed_reproducibility_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/seed_1x_vector_reproducibility.csv`
- checkpoint_alignment_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/checkpoint_seed_vector_alignment.csv`
- axis_projection_rows_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/axis_projection_rows.csv`
- axis_projection_group_summary_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/axis_projection_group_summary.csv`
- key_axis_projection_rows_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/key_axis_projection_rows.csv`
- nearest_step107_archive_directions_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/nearest_archive_directions.csv`
- dose_entity_component_identity_csv: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/dose_entity_component_identity.csv`
- summary_json: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/axis_noise_floor_summary.json`
- summary_md: `research/documents/frontier_consolidation/data/axis_noise_floor_readout/axis_noise_floor_summary.md`
- plots: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/seed_1x_vector_reproducibility.png`
- plots: `experiments/archive/frontier_consolidation/data/axis_noise_floor_readout/axis_projection_pc1_pc2.png`

No model loading, training, official evaluation, GPU work, GlobalPIQA/SuperGLUE/AoA work, upload, or leaderboard action occurred.
