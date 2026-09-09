# compact core full loss anatomy Qwen-pair density redesign budget

CPU-only route arithmetic over the inherited COMPACT_EXPERIENCE clean-Qwen selected pairs. This estimates a safer density redesign: release word budget inside the already-beneficial same-window Qwen-pair block instead of replacing an additional broad official slice with FineWeb packets. It does not generate compacted text or prove fidelity.

## Current inherited pair block
- 37594 selected Qwen pairs, 25486 unique official example IDs, 1,656,800 pair words = 16.568% of the strict-small pool.
- Original side 849,013 words; generated side 807,787 words; weighted rewrite/source ratio 0.951.
- Pair words by source: gutenberg 704,709, simple_wiki 463,379, open_subtitles 249,118, bnc_spoken 162,137, childes 75,334, switchboard 2,123

## If Qwen rewrites were compacted while originals and official filler stayed protected
- Target rewrite/source 0.50: projected rewrite words 433,248, pair block 1,282,261, saved 374,539 words (3.75% of corpus), enough for about 16584 additional mean-length originals; saved mass is 88.4% of the density cleanqwen overlay medium riskhard FineWeb changed block.
- Target rewrite/source 0.55: projected rewrite words 484,079, pair block 1,333,092, saved 323,708 words (3.24% of corpus), enough for about 14334 additional mean-length originals; saved mass is 76.4% of the density cleanqwen overlay medium riskhard FineWeb changed block.
- Target rewrite/source 0.60: projected rewrite words 524,579, pair block 1,373,592, saved 283,208 words (2.83% of corpus), enough for about 12540 additional mean-length originals; saved mass is 66.9% of the density cleanqwen overlay medium riskhard FineWeb changed block.
- Target rewrite/source 0.65: projected rewrite words 568,575, pair block 1,417,588, saved 239,212 words (2.39% of corpus), enough for about 10592 additional mean-length originals; saved mass is 56.5% of the density cleanqwen overlay medium riskhard FineWeb changed block.
- Target rewrite/source 0.70: projected rewrite words 612,163, pair block 1,461,176, saved 195,624 words (1.96% of corpus), enough for about 8662 additional mean-length originals; saved mass is 46.2% of the density cleanqwen overlay medium riskhard FineWeb changed block.
- Target rewrite/source 0.75: projected rewrite words 650,509, pair block 1,499,522, saved 157,278 words (1.57% of corpus), enough for about 6964 additional mean-length originals; saved mass is 37.1% of the density cleanqwen overlay medium riskhard FineWeb changed block.

## Conservative subset compaction
- len_ratio_ge_0p90 at ratio 0.60: 26567 pairs, saved 233,903 words (~10357 mean-length originals).
- len_ratio_ge_0p90 at ratio 0.65: 26567 pairs, saved 204,010 words (~9033 mean-length originals).
- len_ratio_0p75_0p90 at ratio 0.60: 8677 pairs, saved 46,059 words (~2039 mean-length originals).
- len_ratio_0p75_0p90 at ratio 0.65: 8677 pairs, saved 35,095 words (~1554 mean-length originals).
- content_overlap_ge_0p80 at ratio 0.60: 12371 pairs, saved 107,626 words (~4766 mean-length originals).
- content_overlap_ge_0p80 at ratio 0.65: 12371 pairs, saved 92,226 words (~4084 mean-length originals).
- content_overlap_ge_0p80_and_len_ge_0p90 at ratio 0.60: 10144 pairs, saved 96,190 words (~4259 mean-length originals).
- content_overlap_ge_0p80_and_len_ge_0p90 at ratio 0.65: 10144 pairs, saved 83,621 words (~3703 mean-length originals).

## Why this differs from the failed FineWeb overlay
- density cleanqwen overlay medium riskhard compact_core displaced 423,520 official clean-Qwen words: childes 137,760, gutenberg 99,520, open_subtitles 97,760, simple_wiki 56,640, bnc_spoken 30,720, switchboard 1,120.
- Its compact_core inserted 353,945 FineWeb source+view words plus 69,575 neutral clean-Qwen top-up words. The compact core full loss anatomy loss anatomy suggests that replacing the official slice may be entangled with the compact-view mechanism.
- Qwen-internal density would protect the full official source distribution and ask a cleaner question: can shorter faithful second views inside an already successful paired block buy additional reusable official/practical/event evidence without losing QA-congruence and RTE/MRPC calibration?

## Research implication
- If pending reinvest/full-AoA evidence shows compact views remain scientifically useful but FineWeb replacement hurts complete surface, the next experiment should be a small pilot for compacting inherited Qwen pairs, not another FineWeb overlay. The minimum-cost path is to generate/accept only a few thousand high-redundancy Qwen-pair compact views, materialize a 10M probe that preserves official filler and row geometry, and screen it before any 100M run.
- If pending evidence instead shows compression itself broadly causes AoA/SuperGLUE harm independent of source displacement, this Qwen-internal route should not be trained; the saved arithmetic is a design option, not authorization.

Machine-readable JSON: `experiments/archive/frontier_consolidation/data/qwen_pair_density_redesign_budget/qwen_pair_density_redesign_budget.json`
