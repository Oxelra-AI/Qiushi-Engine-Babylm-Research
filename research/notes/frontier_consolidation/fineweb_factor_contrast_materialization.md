# fineweb factor contrast bestview probe FineWeb factor contrast materialization

This is a research-facing corpus factorization, not a final submission artifact. It separates broad factual source content from generated view structure.

Accepted input: `experiments/archive/frontier_consolidation/data/high_anchor_fineweb_generation_analysis/fineweb_high_anchor_accepted_rewrites.jsonl`. Selected 324 pairs / 16,640 source+view words, 0.166% of the 10M pool.

Arms:

- `official_lengthmatched`: official words chunked to the same changed-block row lengths plus identical official filler.
- `fineweb_packet_local_repeat`: FineWeb sources with packet-local source repetition filling the second-view word budget plus identical filler.
- `fineweb_source_rewrite`: FineWeb sources with accepted simplified rewrites plus identical filler.

Changed block rows: 127; filler rows: 62,396. Row-length sequence identical: True. Filler identical after prefix: {'repeat_vs_rewrite': True, 'official_vs_rewrite': True}.

Metadata: `experiments/archive/frontier_consolidation/data/fineweb_factor_contrast/fineweb_factor_contrast_metadata.json`
