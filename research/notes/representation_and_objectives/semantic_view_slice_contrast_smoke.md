# semantic view contrast materialization — matched semantic-view contrast materialization

## Purpose
This constructs the contrast required by the current research state: Qwen simplifications/paraphrases are treated as semantic views of existing SimpleWiki facts, not as a broad new factual source. The control sees the same source rows/articles and exactly the same row-length sequence, but only original source tokens repeated/truncated to the same word mass. Official filler is identical in both arms.

## Corpus evidence
- Accepted generated views: 149 / 256 prompts; by type {'simplification': 68, 'paraphrase': 81}.
- Semantic packet rows: 149; words 9,398 (4.70% of the 200,000-word pool).
- Identical official filler: 190,602 words in 1,192 rows.
- Treatment/control row-length sequence identical: True.
- Unique selected source keys: 149 across 148 SimpleWiki article labels.
- Domain row hits among selected source rows: {"geography_places": 20, "media_culture": 30, "people_history": 40, "quant_numeric": 59, "causal_relational": 18, "science_physical": 11, "institutions_society": 20}

## Interpretation for the next training run
A score difference between these two arms will be interpretable as evidence about linguistic-view transformation of the same source facts under matched word exposure and row lengths, not as broad factual/entity/causal coverage. If this contrast is weak or negative, the data route should reopen a genuinely broader verifiable factual source rather than adding more paraphrases of the same SimpleWiki rows.

Metadata JSON: `experiments/archive/representation_and_objectives/training/data/semantic_view/slice_contrast_smoke/semantic_view_materialization_metadata.json`
