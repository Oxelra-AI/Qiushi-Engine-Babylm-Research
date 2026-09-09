# seed2 entity consistency 20m training validation — seed2 Entity consistency available-coordinate manifest

Evidence JSON: `experiments/archive/initial_model_studies/data/seed2_entity_consistency_available_coordinate_manifest.json`

Direct local checkpoint evaluation: model_path_or_name is hf_model/chck_*M; no local revision selection is used.

- consistency_10M: BLiMP 54.51, Supp 53.40, Entity 17.01, COMPS 50.02, GPIQA 34.18, Reading 7.46
- consistency_20M: BLiMP 59.33, Supp 56.27, Entity 17.68, COMPS 50.87, GPIQA 30.70, Reading 7.42
- shuffled_10M: BLiMP 54.74, Supp 51.47, Entity 17.13, COMPS 50.22, GPIQA 32.70, Reading 7.50
- shuffled_20M: BLiMP 59.84, Supp 56.44, Entity 17.47, COMPS 50.44, GPIQA 28.73, Reading 8.45
