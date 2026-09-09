# fineweb high anchor slice high-anchor FineWeb generation slice

This narrows the repaired high-precision tier for the first Qwen faithfulness run. It removes first/second-person advice, generic starts, title-like rows, low-anchor rows, and low content-score rows that remained after the broader repair.

Input: 9,517 rows / 214,771 words. Kept: 3,985 rows / 97,605 words / 2,268 docs.

Recommended first generation prompts: `experiments/archive/frontier_consolidation/data/fineweb_high_anchor_slice/fineweb_high_anchor_simplification_prompts_pilot1024.jsonl`; larger pilot: `experiments/archive/frontier_consolidation/data/fineweb_high_anchor_slice/fineweb_high_anchor_simplification_prompts_pilot2048.jsonl`.

Use only after the pending SimpleWiki semantic-view evidence supports continuing a generated-view route, or if that evidence is weak and this high-anchor slice is needed to test whether broad factual source-by-rewrite is a better mechanism.

JSON: `experiments/archive/frontier_consolidation/data/fineweb_high_anchor_slice/fineweb_high_anchor_slice_metadata.json`
