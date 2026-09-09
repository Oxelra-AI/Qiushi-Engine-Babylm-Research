# compact view core group route and replication AoA local checkpoints with explicit min_context

Evidence JSON: `experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/parts/AoA/eval/aoa_outputs/scale1p75_chck82_independent__AoA/aoa_local_ckpts_minctx0.json`
Surprisal JSON: `experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/parts/AoA/eval/aoa_outputs/scale1p75_chck82_independent__AoA/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json`
Score JSON: `experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/parts/AoA/eval/aoa_outputs/scale1p75_chck82_independent__AoA/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json`

min_context: **0**
AoA curve_fitness: **0.000000**
Rows: 152095 across 19 checkpoints; row_count_values=[8005]

This run loads local `hf_model/chck_*M` directories directly and uses explicit AoA row-count settings. For current `collate_preds.py`, official-row-count compatibility is 8005 predictions/checkpoint.
