# live FineWeb relation-dense source pool

Scanned 930 live FineWeb-Edu docs / 708,520 document words / 42,899 raw sentence candidates.

Strict-anchor pool: 6,812 rows / 162,847 words (yield 22.98% of scanned doc words).

Relation-dense pool: 3,414 rows / 80,459 words (yield 11.36% of scanned doc words).

Dense domain counts: `{'quant': 806, 'history_society': 773, 'science': 569, 'geography': 521}`.

Flag counts seen: `{'weak_relation_structure': 17071, 'symbol_or_index_like': 2415, 'pronoun_instruction_heavy': 1752, 'many_digit_tokens': 1462, 'low_alpha': 941, 'unbalanced_or_fragmentary_quote': 865, 'web_instruction_or_advice': 570, 'urlish': 273, 'event_listing_or_contact_fragment': 250, 'html': 28, 'nonlatin': 19, 'web_boilerplate': 10, 'mojibake': 3, 'byline_or_metadata': 2}`.

Scientific use: this is the minimal non-GPU substrate for a future faithful-rewrite pilot or larger source-breadth corpus if the seqsafe96 and view/compact results warrant it. It does not justify training by itself.

Relation-dense JSONL: `experiments/archive/representation_and_objectives/data/live_fineweb_relation_dense_pool/live_fineweb_relation_dense_pool.jsonl`

Strict-anchor JSONL: `experiments/archive/representation_and_objectives/data/live_fineweb_relation_dense_pool/live_fineweb_strict_anchor_pool.jsonl`

Summary: `experiments/archive/representation_and_objectives/data/live_fineweb_relation_dense_pool/live_fineweb_relation_dense_pool_summary.json`

## Data Availability

The managed task `s15_t50_tool1` exited with code 134 after printing the successful summary payload because Python aborted during shutdown with `Fatal Python error: PyGILState_Release` in the `datasets`/`pyarrow` stack. The artifacts above were verified after the abort: `strict_anchor_pool.jsonl` has 6,812 lines and `relation_dense_pool.jsonl` has 3,414 lines. Treat the source-pool files and summary as usable CPU extraction evidence, but treat the command exit status as an infrastructure/shutdown issue to avoid repeating with the same `datasets` finalization pattern at larger scale.
