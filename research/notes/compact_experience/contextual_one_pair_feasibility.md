# Feasibility of the contextual one-pair same-window arm

One hypothesis is that the current clean-Qwen pair rows may hurt Reading/BLiMP/EWoK/COMPS because they pack unrelated pairs into synthetic rows (`x1 rewrite1 x2 rewrite2 x3 rewrite3`), creating frequent semantic restarts inside one attention window. A higher-value mechanism test is to preserve the same-window second-view event but embed each pair in natural official context:

`left context ; original ; rewrite ; right context`

rather than multi-pair concatenations.

## Existing evidence for the problem

clean qwen compliance and validity metadata:

- selected pairs: 37,594
- selected pair words: 1,656,800 (16.568% of the 10M pool)
- qwen pair rows: 12,236
- mean pair-row words: 135.40
- average pairs per pair row: about 37,594 / 12,236 = 3.07

Thus many training examples contain multiple unrelated pair events, often from mixed sources. This can plausibly explain why same-window correspondence helps Supplement/Entity/SuperGLUE but costs natural Reading and relation-like columns.

## What existing artifacts preserve

Files inspected:

- `data/qwen_clean_aligned/selected_pairs.jsonl`
- `data/qwen_clean_aligned/qwen_pair_packed_rows_meta.jsonl`
- `data/qwen_clean_aligned/clean_materialization_metadata.json`
- `data/mixture/official_pool.jsonl`
- `data/qwen_aligned/source_sentences.jsonl`
- `data/qwen_clean_aligned/extra_source_sentences.jsonl`
- `scripts/extract_and_prompt.py`
- `scripts/extract_extra_clean_prompts.py`

`selected_pairs.jsonl` preserves `pair_id`, `original`, `rewrite`, `original_words`, `rewrite_words`, `source`, `example_id`, and cohort. clean qwen compliance and validity extra source files preserve `sentence_idx`; base initial rewrite source files do **not** preserve `sentence_idx`, but the original sentence text and `example_id` are present and can likely be matched back inside the 160-word official row.

`official_pool.jsonl` is ordered 160-word official chunks with `example_id` and `source`. Consecutive `example_id`s are often contiguous chunks from the same source stream; within a row, sentence splitting can recover local sentence context.

## Feasible contextualization levels

### Level 1: within-row sentence-neighbor context

For every selected pair:

1. Locate the official 160-word row by `example_id`.
2. Split the row into sentences using the same or stricter sentence splitter used by the initial and clean-materialization constructors.
3. Locate the original sentence:
   - for clean qwen compliance and validity extra cohorts, use preserved `sentence_idx` where available and verify exact text match;
   - for initial rewrite base cohort, match normalized original text among split row sentences; if ambiguous or not found, reject or keep in packed baseline subset.
4. Build a row with nearest official sentence context around the pair until max 160 words:
   - left neighbor(s), original, rewrite, right neighbor(s), preserving official order around original and inserting rewrite immediately after original.
5. Keep at most one pair per row.

This tests whether the same-window second-view signal survives inside more natural local discourse geometry.

### Level 2: cross-row neighbor context

If within-row context is too short, use adjacent official rows with consecutive `example_id` and same `source` to draw left/right words. This is more natural for chunk boundaries but requires careful source continuity auditing, especially when source streams concatenate documents.

### Level 3: exact source-total matched contextual arm

After creating one-pair contextual rows, fill the remaining 10M pool with official rows not used as pair sources, and match source word totals as closely as possible to clean qwen compliance and validity clean-Qwen. Because one-pair rows will include extra official context from selected rows, source totals and selected-original exposure change unless audited. A clean design should record what is preserved exactly and what differs:

- pair identity set and pair words;
- one-pair-per-row topology;
- exact 10M and 100M word totals;
- same tokenizer/model/seed/training recipe;
- source word totals or controlled residuals;
- selected source row exposure.

## Scientific value and needed controls

Primary arm:

- `one_pair_contextual_aligned`: one selected original+Qwen rewrite per row, surrounded by official context.

Controls if budget permits:

- `one_pair_contextual_duplicate`: same context, original duplicated instead of rewrite, to separate generated second-view from repetition under natural context.
- `one_pair_contextual_shuffled`: same context shape but rewrite from another selected item, to isolate identity-correct correspondence.

The decisive comparison is not only against official-only but against the existing packed clean-Qwen arm. If contextual aligned keeps Entity/SuperGLUE/Supplement gains while improving Reading/BLiMP/EWoK/COMPS, then the principle becomes sharper: **attention-visible semantic equivalence is useful, but must be embedded in natural discourse geometry rather than synthetic multi-pair rows**.

## Current caution

Do not call this a natural-context arm until the matching audit proves that context is drawn from the actual official row/neighbor around the source sentence. clean qwen compliance and validity artifacts are promising but not sufficient by themselves. The next implementation should first produce an audit over all 37,594 selected pairs:

- matched sentence index success rate;
- ambiguous/no-match rate by cohort/source;
- available left/right context word statistics;
- resulting row word statistics;
- source word totals and selected row reuse.

This should be built after the current trajectory screen and bidirectional materialization/training decision, because bidirectional pair order is the lower-cost immediate clean causal test.
