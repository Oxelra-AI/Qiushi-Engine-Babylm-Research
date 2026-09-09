# fw mechanism arms — FW mechanism source selection

CPU-only source selection and compact-rewrite prompt preparation for mechanism-scale FineWeb source–compact family.

## Source union
- Unique sources: 38,167
- Total source words: 875,891
- Sources with rewrites: 12,152
- Sources needing new rewrites: 26,015

## Budget
- Official BabyLM: 7,919,680 words
- FineWeb pair block: ~1,494,110 words
- Retained Qwen: ~586,201 words
- Neutral topup: 9 words
- Shared tokenizer pool: ~9,381,781 words

## Compact rewrite prompts
- 26,015 prompts prepared
- Source words: 614,161
- Target rewrite words: ~456,550
- Expected ratio: 0.743

## Arm design (shared tokenizer)
All arms share one compliant tokenizer trained on ~9.33M common words (official + retained Qwen + FineWeb source spans + neutral). Arms differ only in what accompanies each FineWeb source sentence:
- **compact_view**: faithful shorter rewrite
- **source_repeat**: repeated source prefix (same words)
- **source_diversity**: different FineWeb source (new propositions)

## Artifacts
- Manifest: `experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_source_selection.json`
- Frozen sources: `experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_frozen_sources.jsonl`
- Compact prompts: `experiments/archive/representation_and_objectives/data/fw_mechanism_source_selection/fw_mechanism_compact_prompts.jsonl`
