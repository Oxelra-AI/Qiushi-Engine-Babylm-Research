# bidir ranking contextual training and eval plan contextual cap-120 scientific interpretation

This note records the current scientific meaning of the contextual one-pair cap-120 arm before training.

## Why cap-120 is worth training now

clean qwen control interpretation and next mechanism showed that same-window original--generated second-view correspondence is an active component: clean aligned beat selected-original duplication and separated coexistence under corrected local official-style evaluation. The remaining weakness is the topology: clean qwen compliance and validity packed several pairs into relation-only rows, which plausibly removes much of the natural syntactic, discourse, and register context needed for BLiMP, EWoK, COMPS, and Reading.

The cap-120 arm keeps the generated-word dose fixed at the clean clean qwen compliance and validity level (1,656,800 pair words = 16.568% of the 10M pool) but changes the experience geometry: each selected original--rewrite pair is embedded in its own real official row context, one pair per row. This directly tests whether correspondence can act as a scaffold while official context preserves ordinary language competence.

## Materialization facts

- Metadata: `data/contextual_one_pair/cap120/contextual_one_pair_cap120_metadata.json`.
- Treatment/control both have exact 10M pools, exact 100M exposure files, and identical row length sequence.
- 37,594 contextual pair rows, one pair per row, no row longer than 160 words.
- Contextual pair rows contain 4,511,360 words: 1,656,800 generated/original pair words and 2,854,560 official context words.
- Official filler contributes 5,488,640 words from full official rows; selected source rows did not have to be reused as filler.
- Treatment/control training files are present and non-empty; treatment 100M SHA256 is `6c90cb644121f4747d5a901206b35d118cb3b6959d856a827f81191ff5903640`, matched official 100M SHA256 is `ddd8291f76171359f720aecfbb1d882219a6cab810f7dc629ff81b0e0a25564a`.

## Tokenization measurements before training

File: `data/contextual_one_pair/cap120/contextual_cap120_tokenization_audit.json`.

- Treatment 10M has 71,898 rows, mean tokenized length with special tokens 207.61, median 201, p95 290; 13,727 rows exceed 256 tokens.
- Matched official control has mean tokenized length 209.19, median 206, p95 285; 10,916 rows exceed 256 tokens.
- The contextual pair rows themselves are much safer: only 108 of 37,594 exceed 256 tokens.
- Original side is truncated in 1 pair row; rewrite side in 12 pair rows; original visibility mean 0.999997, rewrite visibility mean 0.999922.

Interpretation: row-level tokenized truncation exists in both treatment and official control because some 160-word official rows tokenize long under the baseline16k tokenizer, but the paired relation itself is almost always visible. The treatment/control comparison remains meaningful as a recipe-level test; pair-side truncation is unlikely to explain a positive effect.

## How to read the first result

The first cap-120 training wave should be interpreted in three coordinates:

1. Treatment minus its length-matched official control: does the full contextual one-pair recipe improve sample-efficient transfer under matched word count, row length, seed, tokenizer, model, and exposure?
2. Treatment minus clean qwen compliance and validity clean aligned seed43022: does adding real official context recover BLiMP, EWoK, COMPS, and Reading while preserving Supplement, Entity, and SuperGLUE?
3. Treatment absolute full score: does it approach or exceed the visible 41.8 leader under corrected nine-column scoring?

A strong scientific pattern would show positive movement in most of BLiMP, EWoK, COMPS, and Reading relative to clean qwen compliance and validity clean aligned, while retaining positive treatment-control gains in Supplement, Entity, and SuperGLUE. If the gain is mostly SuperGLUE/Entity or AoA with flat or worse BLiMP/EWoK/COMPS/Reading, it is another redistribution rather than the recovery mechanism we want.

## Remaining controls if cap-120 is promising

The current official control matches length sequence and total words, but it does not isolate the native-context contribution. If cap-120 improves, the most informative next controls are:

- same pair and same row length, but unrelated same-source official context around the pair;
- same true official context and same position, but original+original instead of original+rewrite;
- cap-80/cap-160 or lower pair-dose deep-context variants to find whether context amount or pair density drives recovery;
- replicate the best contextual arm on seed43122 before treating the effect as robust.

These controls should still use no official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream evaluation outputs for data design.
