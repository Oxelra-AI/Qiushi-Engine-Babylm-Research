# hybrid cleanqwen semantic view candidate hybrid clean-Qwen + semantic-view candidate

This is a checked candidate corpus, not trained evidence. It should only be trained if the running packet-local semantic-view contrast gives a meaningful positive signal.

## Construction

- Clean-Qwen pair rows inherited from COMPACT_EXPERIENCE: 12,236 rows, 1,656,800 words (16.57%).
- REPRESENTATION_FRONTIER_STUDIES semantic packet rows: 10,840 rows, 820,187 words (8.20%).
- Combined generated/paired prefix: 2,476,987 words (24.77%).
- Official filler from the clean-Qwen filler pool: 7,523,013 words in 47,019 rows; partial last row: True.
- Treatment and control have identical clean-Qwen rows, identical official filler, identical row-length sequence, and matched SimpleWiki packet word totals.

## Use

If the basic semantic-view treatment beats its packet-local control in official-compatible no-AoA evaluation without damaging key columns, this hybrid pair can test whether the mechanism adds on top of the stronger COMPACT_EXPERIENCE clean-Qwen coordinate. If the basic contrast is weak or negative, do not spend H100 time on this hybrid; instead reopen a broader verifiable factual-source route.

Audit JSON: `experiments/archive/representation_and_objectives/training/data/hybrid_cleanqwen_semantic_view/hybrid_materialization_audit.json`

Metadata JSON: `experiments/archive/representation_and_objectives/training/data/hybrid_cleanqwen_semantic_view/hybrid_materialization_metadata.json`
