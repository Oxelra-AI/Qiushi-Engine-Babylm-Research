# full contrast materialized and verified semantic-view seq256 visibility audit

Trainer-equivalent tokenizer max length: 256 baseline16k tokens, add_special_tokens=False.

Semantic packet rows: 10840; accepted generated views: 16394.

Views fully visible: 16233 / 16394; partially visible: 154; not visible: 7.

Overall generated-view token visibility: 0.9961. Rows with any hidden/partial view: 154; rows with zero visible view tokens: 0.

By type:
- simplification: fully 7077/7077, partial 0, none 0, visible token fraction 1.0000.
- paraphrase: fully 9156/9317, partial 154, none 7, visible token fraction 0.9935.

JSON: `experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast/semantic_view_seq256_visibility_audit.json`
