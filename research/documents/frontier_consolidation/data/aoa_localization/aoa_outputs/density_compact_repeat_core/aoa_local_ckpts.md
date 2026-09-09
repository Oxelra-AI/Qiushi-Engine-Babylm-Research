# earlier analysis — repaired AoA with direct local checkpoint loading

Evidence JSON: `experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_compact_repeat_core/aoa_local_ckpts.json`
Surprisal JSON: `experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_compact_repeat_core/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/surprisal.json`
Score JSON: `experiments/archive/frontier_consolidation/data/aoa_localization/aoa_outputs/density_compact_repeat_core/hf_model_local_ckpts/main/zero_shot/mlm/AoA_word/aoa_score.json`

AoA: **-0.1890**
Rows: 124640 across 19 checkpoints

This run loads local `hf_model/chck_*M` directories directly and preserves official AoA target words, surprisal computation, and curve-fitness scoring.
