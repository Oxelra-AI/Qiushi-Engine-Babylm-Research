# hybrid cleanqwen semantic view candidate hybrid overlap and WWM-to-token training dynamics

## Source overlap

- COMPACT_EXPERIENCE selected clean-Qwen originals: 37,594 pairs, 37,450 unique original texts; SimpleWiki subset 10,810.
- REPRESENTATION_FRONTIER_STUDIES semantic-view selected sources recovered from prompts: 10,840 unique source texts for 16,368 accepted generated views.
- Exact normalized overlap between semantic sources and all clean-Qwen selected originals: 623 texts, word mass 14,425.
- Exact normalized overlap against the clean-Qwen SimpleWiki subset: 623 texts.
This means the hybrid candidate mostly adds new SimpleWiki source texts relative to the selected COMPACT_EXPERIENCE clean-Qwen originals at exact-string level, while still reweighting SimpleWiki rather than broadening outside official sources.

## Hybrid candidate scale

- Clean-Qwen pair words: 1,656,800; semantic packet words: 820,187; combined prefix fraction 24.77%.
- Required construction checks pass: True; rows {'treatment': 70095, 'control': 70095}.

## Training dynamics: fixed WWM vs WWM→token

- WWM→token completed 100,000,000 words in 2515 steps; loss 9.8113 → 2.2665; log modes ['wwm', 'token'].
- Clean-Qwen fixed WWM reference completed 100,000,000 words in 2515 steps; loss 9.8113 → 2.5076.
- Final loss delta WWM→token minus fixed WWM: -0.2411. This lower loss is not by itself a downstream competence result because token masking changes target granularity.
- The managed no-AoA evaluation task will decide whether the recipe change helps BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading.

JSON: `experiments/archive/representation_and_objectives/training/data/hybrid_cleanqwen_semantic_view/hybrid_overlap_and_training_dynamics.json`
