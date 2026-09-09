# hybrid cleanqwen semantic view candidate WWM-to-token recipe effect vs clean-Qwen fixed WWM

Target summary: `experiments/archive/representation_and_objectives/data/wwm_to_token_postswitch_fullzeroshot_reading/qwen_wwm_to_token_postswitch_trajectory_summary.json`

Baseline summary: `experiments/archive/compact_experience/data/trajectory_screen/clean_qwen_seed43022_trajectory_summary.json`

This compares official-compatible zero-shot columns plus Reading at matched checkpoints. It does not include SuperGLUE or AoA.

| checkpoint | WWM->token equal7 | clean-Qwen WWM equal7 | delta | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chck_80M | 43.1521 | 42.2036 | 0.9486 | 0.4000 | 1.4400 | 1.8000 | -0.8600 | -0.3100 | 4.9850 | -0.8150 |
| chck_90M | 43.1793 | 42.7564 | 0.4229 | 0.4100 | -0.3500 | 1.7900 | 0.5700 | -0.4900 | 1.5150 | -0.4850 |
| chck_100M | 43.1443 | 43.1129 | 0.0314 | 0.5100 | -0.2100 | 1.4800 | -0.0300 | -0.4700 | -0.4700 | -0.5900 |

Best WWM->token checkpoint by equal7: `chck_90M` = 43.1793.
Best matched delta checkpoint: `chck_80M` = 0.9486.
Clean-Qwen WWM best in baseline summary: `chck_100M` = 43.112857142857145.

JSON: `experiments/archive/representation_and_objectives/data/wwm_to_token_postswitch_fullzeroshot_reading/wwm_to_token_vs_clean_qwen_delta_summary.json`
