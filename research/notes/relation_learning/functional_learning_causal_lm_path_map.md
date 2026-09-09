# accidental target context audit path map for functional_learning causal-LM bridge

Purpose: provide exact reusable artifacts for a possible causal next-token locality bridge of CLEAN / REPEAT / REPEAT_SPLIT under the same text-stream relation manipulation.

## Text streams

Original frontier_consolidation dose2p64x row-holdout pools:

- CLEAN 100M stream: `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_100M.jsonl`
- REPEAT 100M stream: `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_100M.jsonl`
- VIEW 100M stream, if needed: `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_100M.jsonl`
- Metadata and SHA records: `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json`

Split-locality pools from relation_learning:

- REPEAT_SPLIT 100M stream: `experiments/archive/relation_learning/data/split_inwindow_rowholdout_pools/compact_repeat_split_dose2p64x_100M.jsonl`
- VIEW_SPLIT 100M stream, if needed: `experiments/archive/relation_learning/data/split_inwindow_rowholdout_pools/compact_view_split_dose2p64x_100M.jsonl`
- Metadata and SHA records: `experiments/archive/relation_learning/data/split_inwindow_rowholdout_pools/split_inwindow_rowholdout_metadata.json`

Useful SHA prefixes already verified in relation_learning:

- REPEAT_SPLIT 100M SHA: `427302c903d40ea23c4afc4afd12b9bb3fc0f33cd69bca3bbf8fcb3d32fd3345`
- Tokenizer JSON SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Tokenizer and existing recipes

- Shared BabyLM-compliant tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- DeBERTa split recipe launchers for reference: `experiments/archive/relation_learning/scripts/train_split_inwindow_control.py` and `experiments/archive/relation_learning/scripts/train_split_seed_replicate.py`
- RoBERTa trainer/recipe reference: `experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py`; relation_learning launcher for REPEAT_SPLIT RoBERTa: `experiments/archive/relation_learning/scripts/train_roberta_repeat_split.py`
- Causal GPT trainer present in frontier_consolidation: `experiments/archive/frontier_consolidation/scripts/causal_gpt_trainer.py`; prior causal data/tokenizer helper: `experiments/archive/frontier_consolidation/scripts/causal_data_and_tokenizer.py`

## Existing model run directories if needed for comparison

- DeBERTa CLEAN seed43022: `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022`
- DeBERTa REPEAT seed43022: `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022`
- DeBERTa REPEAT_SPLIT seed43022: `experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43022`
- DeBERTa CLEAN seed43122: `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122`
- DeBERTa REPEAT seed43122: `experiments/archive/frontier_consolidation/training/runs/full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122`
- DeBERTa REPEAT_SPLIT seed43122: `experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43122`
- RoBERTa CLEAN seed43022: `experiments/archive/frontier_consolidation/training/runs/roberta_clean_dose2p64x_matched_rowholdout_100M_seed43022`
- RoBERTa REPEAT seed43022: `experiments/archive/frontier_consolidation/training/runs/roberta_repeat_dose2p64x_matched_rowholdout_100M_seed43022`

## T/U/N and related scorer/readout assets

- Base held-out compact rewrite and natural copy scorer: `experiments/archive/relation_learning/scripts/heldout_copy_rewrite_entity_ablation.py`
- Split seed43122 wrapper showing how to point the base scorer at new runs: `experiments/archive/relation_learning/scripts/score_split_seed43122.py`
- Neutral-source anchor scorer: `experiments/archive/relation_learning/scripts/neutral_anchor_rewrite_probe.py`; original three-seed version: `experiments/archive/relation_learning/scripts/original_threeseed_neutral_anchor.py`
- Two-seed DeBERTa split integration: `experiments/archive/relation_learning/scripts/integrate_split_seed_replication.py`; summary note `research/notes/relation_learning/split_seed_replication.md`
- Original three-seed T/U/N note: `research/notes/relation_learning/original_threeseed_neutral_anchor.md`
- Seed43022 split T/U/N note: `research/notes/relation_learning/neutral_anchor_rewrite_probe.md`

## Current relation_learning status relevant to avoiding duplicated work

- Hash-mixed same-window DeBERTa training was running at this stage.
- Official Entity evaluation on seed43122 split DeBERTa arms was running.
- RoBERTa REPEAT_SPLIT seed43022 training was pending.
