# clean qwen experiment state — repaired AoA with direct local checkpoint loading

Evidence JSON: `experiments/archive/compact_experience/data/full_eval_candidates/aoa_outputs/context_cap120_control_seed43022_80M/aoa_local_ckpts.json`
Surprisal JSON: `experiments/archive/compact_experience/data/full_eval_candidates/aoa_outputs/context_cap120_control_seed43022_80M/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json`
Score JSON: `experiments/archive/compact_experience/data/full_eval_candidates/aoa_outputs/context_cap120_control_seed43022_80M/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json`

AoA: **-0.2086**
Rows: 124640 across 19 checkpoints

This run loads local `hf_model/chck_*M` directories directly and preserves official AoA target words, surprisal computation, and curve-fitness scoring.
