# fineweb factual sentence rewrite source by rewrite FineWeb complete-sentence source-by-rewrite question

The active H100 work remains the matched SimpleWiki semantic-view treatment/control pair. No new GPU generation, training, or evaluation job was started here.

## Why this asset exists
The public leader's README describes sentence-level FineWeb originals followed by Qwen-generated simplifications. The exact train file remains unavailable locally, so the legal route is to reconstruct the mechanism with cached public FineWeb sentences and count every generated token as training data if used. The earlier cached fineweb quality audit fragment prompts are superseded: this file uses complete sentence-like FineWeb spans.

## Prepared sources and prompts
- Source sentences selected: 59,769 from 5,419 docs.
- Source word mass: 1,246,944.
- Mean sentence length: 20.86 words; p95 39 words; max 55 words.
- Full simplification prompts: `experiments/archive/representation_and_objectives/training/data/fineweb_sentence_rewrite/fineweb_complete_sentence_simplification_prompts_all.jsonl` (59,769).
- Pilot prompt subset for a future faithfulness slice: `experiments/archive/representation_and_objectives/training/data/fineweb_sentence_rewrite/fineweb_complete_sentence_simplification_prompts_pilot8192.jsonl` (8,192).
- Metadata: `experiments/archive/representation_and_objectives/training/data/fineweb_sentence_rewrite/source_by_rewrite_prompt_metadata.json`. Samples: `experiments/archive/representation_and_objectives/training/data/fineweb_sentence_rewrite/prompt_samples.json`.

## Scientific comparison to run only after current evidence arrives
If the SimpleWiki same-source semantic-view contrast is weak, the next stronger question is not a raw source swap. It is: does broad factual FineWeb experience become more useful when the same source sentences are coupled to faithful simpler rewrites? The matched contrast should use the identical sentence set in both arms. Treatment rows concatenate original sentence plus accepted simplification. Control rows use the same original sentence and match the treatment row length by same-source repetition. The remaining official filler must be identical.

## Expected scale
Using the observed SimpleWiki simplification length ratio as only a planning prior, accepting all selected sentence rewrites would give roughly 2,369,194 pair words at a 0.90 rewrite/source ratio, with a plausible range 2,182,152--2,556,235. This is substantially broader FineWeb factual coverage than the current SimpleWiki semantic packet mass and directly tests a leader-like data structure.

## Before any H100 training
Generate only after the active pair and frontier_consolidation evidence make this the best next allocation. Then screen source/output pairs for number preservation, entity retention, no added facts by automated proxies, complete sentence form, length range, non-copy transformation, and no prompt artifacts. Materialize matched rows only from accepted rewrites; verify exact 10M words, matched row lengths, identical filler, source reconstruction, and seq256 visibility before training.


## fineweb factual sentence rewrite source by rewrite refinement note
The broad prompt asset in this note is superseded for future H100 use by the stricter factual/expository prompt asset at `experiments/archive/representation_and_objectives/training/data/fineweb_factual_sentence_rewrite`, because sample inspection found dialogue/fragments/list-like sources.
