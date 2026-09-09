# entity consistency 20m training validation — Entity consistency available-coordinate manifest

Evidence JSON: `experiments/archive/initial_model_studies/data/entity_consistency_available_coordinate_manifest.json`

Direct local checkpoint evaluation: model_path_or_name is hf_model/chck_*M; no local revision selection is used.

- consistency_10M: BLiMP 55.45, Supp 54.20, Entity 17.49, COMPS 50.48, GPIQA 32.16, Reading 8.54
- consistency_20M: BLiMP 61.04, Supp 57.61, Entity 17.87, COMPS 50.59, GPIQA 35.15, Reading 7.68
- shuffled_10M: BLiMP 55.18, Supp 53.73, Entity 17.43, COMPS 50.25, GPIQA 30.20, Reading 8.68
- shuffled_20M: BLiMP 59.00, Supp 58.59, Entity 17.48, COMPS 50.62, GPIQA 31.23, Reading 7.89
