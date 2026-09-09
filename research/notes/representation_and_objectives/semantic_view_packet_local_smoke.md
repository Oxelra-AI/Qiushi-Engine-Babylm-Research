# semantic view contrast materialization — matched semantic-view contrast materialization

## Purpose
This constructs the contrast required by the current research state: Qwen simplifications/paraphrases are treated as semantic views of existing SimpleWiki facts, not as a broad new factual source. It writes three source-only comparison arms. The packet-local cyclic control is the cleanest transformation test: each row uses only its own original source words, cyclically offset and repeated to the exact treatment row length, preserving source identity, topical boundaries, row length, and repetition while removing generated wording. The stream control chunks a shuffled/cycled stream of the same selected SimpleWiki source texts to the same row lengths; it remains useful for detecting packing effects but can splice unrelated articles within a row. A naive prefix-repeated packet-exact file is kept only for forensic inspection because it over-emphasizes source prefixes. Official filler is identical in all arms.

## Corpus evidence
- Accepted generated views: 149 / 256 prompts; by type {'simplification': 68, 'paraphrase': 81}.
- Semantic packet rows: 149; words 9,398 (9.40% of the 100,000-word pool).
- Identical official filler: 90,602 words in 567 rows.
- Treatment/control row-length sequence identical: True.
- Unique selected source keys: 149 across 148 SimpleWiki article labels.
- Domain row hits among selected source rows: {"media_culture": 30, "institutions_society": 20, "people_history": 40, "quant_numeric": 59, "causal_relational": 18, "geography_places": 20, "science_physical": 11}

## Interpretation for the next training run
A score difference between these two arms will be interpretable as evidence about linguistic-view transformation of the same source facts under matched word exposure and row lengths, not as broad factual/entity/causal coverage. If this contrast is weak or negative, the data route should reopen a genuinely broader verifiable factual source rather than adding more paraphrases of the same SimpleWiki rows.

Metadata JSON: `experiments/archive/representation_and_objectives/training/data/semantic_view/slice_packet_local_smoke/semantic_view_materialization_metadata.json`
