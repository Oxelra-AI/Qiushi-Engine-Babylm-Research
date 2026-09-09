# debertav2 b256 aoa local ckpts result — DeBERTa-v2 b256 AoA with direct local checkpoint loading

Evidence JSON: `experiments/archive/initial_model_studies/data/debertav2_b256_aoa_local_ckpts_result.json`
Surprisal JSON: `experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_aoa_local_ckpts/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json`
Score JSON: `experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_aoa_local_ckpts/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json`

AoA: **-0.1745**
Rows: 124640 across 19 checkpoints

This run overrides only checkpoint resolution: it loads `hf_model/chck_*M` directories directly and preserves official AoA target words, surprisal computation, and curve-fitness scoring.
