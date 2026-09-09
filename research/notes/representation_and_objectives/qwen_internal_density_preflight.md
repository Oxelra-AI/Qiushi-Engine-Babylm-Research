# qwen internal density preflight Qwen-internal density preflight

CPU-only redesign work. No GPU, generation, official evaluation, or training was launched; the pending compact reinvest full eval summary reinvest results remain the next decisive evidence.

## Why this redesign is different from the failed compact FineWeb replacement
- Full compact_view_core did not merely have an AoA problem: replacing its AoA by zero would lift Overall only to about 41.40 because its full NLP average is essentially tied to clean-Qwen while it loses Supplement, GlobalPIQA, and SuperGLUE.
- The safer density question is therefore internal to the already-successful COMPACT_EXPERIENCE clean-Qwen allocation: shorten redundant generated rewrites while preserving all original sides and all official filler rows, then use the recovered words for additional official rows rather than a new FineWeb replacement block.

## Clean-Qwen substrate facts
- Selected Qwen pairs: 37594 pairs / 1,656,800 words (16.568% of the pool).
- Existing official filler that this redesign preserves: 8,343,200 words.
- Selected official example IDs: 25,486; full official rows not already in filler: 10,355 rows / 1,656,800 words.

## Preferred if FineWeb-reinvest full vector fails: len_ratio>=0.90, target rewrite/source ratio 0.60
- Pairs to compact: 26,567; expected saved words: 233,903 (2.339% of 10M).
- Addable full official 160-word rows: 1,461; exact-materialization slack to absorb by length adjustment: 143 words.
- Compacted-subset current rewrite words -> target compact rewrite words: 592,318 -> 358,415.
- Top-up row source counts: {'bnc_spoken': 124, 'childes': 264, 'gutenberg': 453, 'open_subtitles': 323, 'simple_wiki': 295, 'switchboard': 2}.
- Affordance top-up score mean/p95: 0.2145/0.2774; neutral source-matched mean/p95: 0.0727/0.1450.
- After-the-fact AoA-target exposure audit using a rewrite-suffix surrogate: added top-up minus removed suffix = 9592 target occurrences; this is not a selection signal and must be repeated after actual compact generation.

## Other CPU-only scenarios
- all_pairs_r060: compact 36,914 pairs, save 283,761 words, add 1,773 official rows, slack 81.
- len90_r060: compact 26,567 pairs, save 233,903 words, add 1,461 official rows, slack 143.
- len90_r065: compact 26,567 pairs, save 204,010 words, add 1,275 official rows, slack 10.
- len90_content80_r060: compact 10,144 pairs, save 96,190 words, add 601 official rows, slack 30.
- pilot4096_len90_r060: compact 4,096 pairs, save 59,167 words, add 369 official rows, slack 127.

## Files written for future low-cost checks
- Preferred compaction manifest: `experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_compaction_manifest.jsonl`
- Preferred relation/action top-up rows: `experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_affordance_topup_rows.jsonl`
- Source-matched neutral rows for a future matched comparison: `experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_neutral_source_matched_rows.jsonl`
- 512-prompt slice for a future compact-generation audit: `experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/preferred_len90_r060_compaction_prompt_slice512.jsonl`
- Machine-readable summary: `experiments/archive/representation_and_objectives/data/qwen_internal_density_preflight/qwen_internal_density_preflight.json`

## Scientific use
This preflight is a design asset, not a training result. It should be used only after the pending reinvest full vector is read. If reinvest lacks a substantial full-NLP gain or inherits negative AoA, the FineWeb-replacement endpoint should be retired and this Qwen-internal density route can be tested first by a small compact-generation slice and a 10M dry-run audit. If reinvest unexpectedly has strong full-NLP with neutral AoA, preserve that endpoint instead of diverting to this redesign.
