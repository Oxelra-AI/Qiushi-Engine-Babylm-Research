# globalpiqa practical repair blueprint — repaired AoA with direct local checkpoint loading

Evidence JSON: `experiments/archive/representation_and_objectives/data/semantic_view_full_eval/aoa_outputs/semantic_view_treatment__chck_80M/aoa_local_ckpts.json`
Surprisal JSON: `experiments/archive/representation_and_objectives/data/semantic_view_full_eval/aoa_outputs/semantic_view_treatment__chck_80M/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json`
Score JSON: `experiments/archive/representation_and_objectives/data/semantic_view_full_eval/aoa_outputs/semantic_view_treatment__chck_80M/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json`

AoA: **0.0000**
Rows: 124640 across 19 checkpoints

This run loads local `hf_model/chck_*M` directories directly and preserves official AoA target words, surprisal computation, and curve-fitness scoring.
