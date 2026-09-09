# fineweb factual sentence rewrite source by rewrite refined FineWeb factual sentence source-by-rewrite question

The active H100 work remains the SimpleWiki semantic-view treatment/control pair. This analysis did not start Qwen generation, training, or evaluation.

## Repair over the broad sentence prompt asset
Inspection of the first broad prompts found literary dialogue, context-dependent pronoun starts, trailing abbreviation fragments, and list/table-like rows. Those would make the next experiment less about leader-like factual sentence simplification. This refined asset keeps only factual/expository, self-contained sentences with cleaner punctuation and relation signal.

## Prepared source
- Selected factual sentence sources: 13,198 / 61,511.
- Selected source words: 309,191 / 1,281,741.
- Unique docs: 4,484.
- Mean words/source: 23.43; p95 40; max 48.
- Full prompt file: `experiments/archive/representation_and_objectives/training/data/fineweb_factual_sentence_rewrite/fineweb_factual_complete_sentence_simplification_prompts_all.jsonl`.
- Pilot prompt file: `experiments/archive/representation_and_objectives/training/data/fineweb_factual_sentence_rewrite/fineweb_factual_complete_sentence_simplification_prompts_pilot8192.jsonl`.
- Metadata: `experiments/archive/representation_and_objectives/training/data/fineweb_factual_sentence_rewrite/factual_source_by_rewrite_prompt_metadata.json`. Samples: `experiments/archive/representation_and_objectives/training/data/fineweb_factual_sentence_rewrite/factual_prompt_samples.json`.

## Next scientific use
If the active same-source SimpleWiki contrast is weak, use these prompts for a small Qwen faithfulness slice first. If source/output screening shows high acceptance, materialize a matched FineWeb source-by-rewrite contrast: treatment is original factual sentence plus accepted simplification; control is the same original sentence with row length matched by same-source repetition; remaining official filler is identical. This tests whether faithful simplification coupled to broad factual web experience is useful beyond broader factual source exposure alone.

At the observed SimpleWiki simplification length scale, accepting all refined sources would provide roughly 587,463 paired source+rewrite words, with planning range 541,084--633,842.
