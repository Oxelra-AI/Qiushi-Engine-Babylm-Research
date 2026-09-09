# dualview pending panel interim dual-view 20M parameter geometry

## Pairwise global geometry

| comparison | group | params | rel-L2 | cosine | diff-norm | mean |
|---|---|---:|---:|---:|---:|---:|
| aligned_vs_shuffled | all_params | 35463008 | 0.485054 | 0.882054907 | 93.1665 | 0.0113139 |
| aligned_vs_shuffled | stock_all | 34467424 | 0.47405 | 0.887712569 | 83.5879 | 0.010711 |
| aligned_vs_shuffled | adapter_all | 995584 | 0.540217 | 0.851250047 | 41.1467 | 0.0321874 |
| aligned_vs_shuffled | adapter_up | 495360 | 1.16814 | 0.264567959 | 28.1045 | 0.0312205 |
| aligned_vs_shuffled | adapter_down | 492544 | 0.98786 | 0.489290797 | 29.8097 | 0.0331241 |
| aligned_vs_shuffled | stock_embeddings | 8111040 | 0.496482 | 0.877544175 | 43.8616 | 0.0119597 |
| aligned_vs_shuffled | stock_attention | 11089920 | 0.452666 | 0.897675523 | 45.4307 | 0.00936207 |
| aligned_vs_shuffled | stock_ffn_output | 14772480 | 0.491536 | 0.878627722 | 54.1078 | 0.0111043 |
| aligned_vs_shuffled | stock_lm_head | 248224 | 0.20482 | 0.979103709 | 5.78114 | 0.00882348 |
| aligned_vs_mlm_only | all_params | 35463008 | 0.594678 | 0.819977272 | 116.101 | 0.0146861 |
| aligned_vs_mlm_only | stock_all | 34467424 | 0.600789 | 0.812845265 | 110.418 | 0.0142885 |
| aligned_vs_mlm_only | adapter_all | 995584 | 0.544752 | 0.876817867 | 35.8766 | 0.0284529 |
| aligned_vs_mlm_only | adapter_up | 495360 | 2.56552 | 0.024889180 | 23.8811 | 0.026925 |
| aligned_vs_mlm_only | adapter_down | 492544 | 1.19985 | 0.491628035 | 26.2058 | 0.0296249 |
| aligned_vs_mlm_only | stock_embeddings | 8111040 | 0.616822 | 0.803531015 | 56.8146 | 0.0155753 |
| aligned_vs_mlm_only | stock_attention | 11089920 | 0.627696 | 0.790734522 | 67.8581 | 0.0148824 |
| aligned_vs_mlm_only | stock_ffn_output | 14772480 | 0.578921 | 0.831561451 | 63.7548 | 0.0130243 |
| aligned_vs_mlm_only | stock_lm_head | 248224 | 0.256101 | 0.966649986 | 7.50044 | 0.0114306 |
| shuffled_vs_mlm_only | all_params | 35463008 | 0.600711 | 0.816739495 | 117.278 | 0.0148203 |
| shuffled_vs_mlm_only | stock_all | 34467424 | 0.603046 | 0.811331647 | 110.833 | 0.0143686 |
| shuffled_vs_mlm_only | adapter_all | 995584 | 0.582204 | 0.864048189 | 38.3431 | 0.0304588 |
| shuffled_vs_mlm_only | adapter_up | 495360 | 2.75239 | 0.020282821 | 25.6205 | 0.0288995 |
| shuffled_vs_mlm_only | adapter_down | 492544 | 1.27707 | 0.462496719 | 27.8923 | 0.0316142 |
| shuffled_vs_mlm_only | stock_embeddings | 8111040 | 0.61421 | 0.804207299 | 56.574 | 0.0155134 |
| shuffled_vs_mlm_only | stock_attention | 11089920 | 0.630014 | 0.788991292 | 68.1087 | 0.0149673 |
| shuffled_vs_mlm_only | stock_ffn_output | 14772480 | 0.584897 | 0.828873070 | 64.413 | 0.0131711 |
| shuffled_vs_mlm_only | stock_lm_head | 248224 | 0.260131 | 0.965575014 | 7.61845 | 0.0115663 |

## Adapter RMS by arm

| arm | adapter_all rms | adapter_up rms | adapter_down rms | stock_all rms |
|---|---:|---:|---:|---:|
| aligned | 0.0747747 | 0.0315779 | 0.0409877 | 0.030054 |
| shuffled | 0.0763356 | 0.0341839 | 0.0429972 | 0.0300341 |
| mlm_only | 0.0660045 | 0.0132257 | 0.0311205 | 0.0313051 |

Output JSON: `experiments/archive/frontier_consolidation/data/dualview_parameter_geometry/dualview_parameter_geometry.json`
