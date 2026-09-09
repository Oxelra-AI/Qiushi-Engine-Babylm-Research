# semantic view contrast materialization — matched semantic-view contrast materialization

## Purpose
This constructs the contrast required by the current research state: Qwen simplifications/paraphrases are treated as semantic views of existing SimpleWiki facts, not as a broad new factual source. It writes three source-only comparison arms. The packet-local balanced-repetition control is the cleanest transformation test: each row uses only its own original source words, starts with the exact source to match the treatment prefix, then uses cyclic offsets for any extra repetitions to reach the exact treatment row length; it preserves source identity, topical boundaries, row length, and repetition while removing generated wording. The stream control chunks a shuffled/cycled stream of the same selected SimpleWiki source texts to the same row lengths; it remains useful for detecting packing effects but can splice unrelated articles within a row. A naive prefix-repeated packet-exact file is kept only for forensic inspection because it over-emphasizes source prefixes. Official filler is identical in all arms.

## Corpus evidence
- Accepted generated views: 16394 / 30000 prompts; by type {'simplification': 7077, 'paraphrase': 9317}.
- Semantic packet rows: 10840; words 820,830 (8.21% of the 10,000,000-word pool).
- Identical official filler: 9,179,170 words in 57,370 rows.
- Treatment/control row-length sequence identical: True.
- Unique selected source keys: 10840 across 7161 SimpleWiki article labels.
- Domain row hits among selected source rows: {"geography_places": 2095, "media_culture": 1618, "causal_relational": 1129, "quant_numeric": 3652, "people_history": 2989, "institutions_society": 1683, "science_physical": 389}

## Interpretation for the next training run
A score difference between these two arms will be interpretable as evidence about linguistic-view transformation of the same source facts under matched word exposure and row lengths, not as broad factual/entity/causal coverage. If this contrast is weak or negative, the data route should reopen a genuinely broader verifiable factual source rather than adding more paraphrases of the same SimpleWiki rows.

Metadata JSON: `experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast/semantic_view_materialization_metadata.json`
