# full contrast materialized and verified semantic-view seq256 visibility audit

Trainer-equivalent tokenizer max length: 256 baseline16k tokens, add_special_tokens=False.

Semantic packet rows: 10840; accepted generated views: 16368.

Views fully visible: 16218 / 16368; partially visible: 150; not visible: 0.

Overall generated-view token visibility: 0.9966. Rows with any hidden/partial view: 150; rows with zero visible view tokens: 0.

By type:
- simplification: fully 7065/7065, partial 0, none 0, visible token fraction 1.0000.
- paraphrase: fully 9153/9303, partial 150, none 0, visible token fraction 0.9944.

JSON: `experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1/semantic_view_seq256_visibility_audit.json`
