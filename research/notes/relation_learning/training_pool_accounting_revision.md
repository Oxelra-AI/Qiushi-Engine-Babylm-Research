# Training Pool and Exposure Accounting Revision

Scientific status: corrected corpus accounting, not a new corpus or training run.

The full base pool contains 64,740 rows, replacing the earlier count of 64,381. Its recorded source counts are:

| Source | Rows |
|---|---:|
| childes | 16,093 |
| qwen_pair_packed | 12,236 |
| gutenberg | 11,621 |
| open_subtitles | 11,434 |
| simple_wiki | 6,619 |
| bnc_spoken | 3,599 |
| cleanqwen_fineweb_compact_view_reinvest | 3,005 |
| switchboard | 132 |
| neutral_cleanqwen_topup_compact_reinvest::open_subtitles | 1 |

The Stage III continuation prefix contains 20,475 rows, including 940 FineWeb-derived rows. Prefix composition is not interchangeable with the complete pool composition. Similarly, unique corpus size is not cumulative training exposure: multiple presentations and generated derivative rewrites must remain visible in the accounting.

The original prediction-package revisions also distinguish full filtered evaluations from fast screens. For example, the 59,875-item BLiMP result is not the roughly 13,400-item fast screen. A summary file is not a complete AoA trajectory. The intermediate 18-point shared-ancestry package and the later standalone 19-checkpoint package must retain distinct provenance; the final clarification is recorded separately.

This note records construction and measurement identity. It does not grant redistribution rights for underlying text or assert challenge approval. Scientific data builders and manifests can be retained without redistributing raw source corpora.

Related rationale: [rewrite construction](rewrite_generation_design_rationale.md), [AoA coordinate distinction](aoa_standalone_and_shared_trajectory_identity.md), [corrected state-use construction](state_use_realized_construction.md).
