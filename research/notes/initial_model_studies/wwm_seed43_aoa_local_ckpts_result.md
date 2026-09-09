# wwm seed43 aoa local ckpts result — AoA with direct local checkpoint loading

Evidence JSON: `experiments/archive/initial_model_studies/data/wwm_seed43_aoa_local_ckpts_result.json`
Surprisal JSON: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/eval_aoa_local_ckpts/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json`
Score JSON: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/eval_aoa_local_ckpts/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json`

AoA: **0.0000**
Rows: 124640 across 19 checkpoints

This run loads local `hf_model/chck_*M` directories directly and preserves official AoA target words, surprisal computation, and curve-fitness scoring.
