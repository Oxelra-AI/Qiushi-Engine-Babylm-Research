# endpoint consumption and u256 tail composition — U256 tail balance profile

This CPU-only profile compares the row256-visible prefix mass with the U256-recovered suffix mass in the exact legal 10M pool. Features are broad corpus-text descriptors only.

- Rows: 64,740; recovered-tail rows: 15,117 (23.3503%).
- Row256 active BPE tokens: 14,294,893; U256 recovered BPE tokens: 369,626; active-token ratio: 1.025857.
- Recovered full words: 197,806; boundary hidden BPE pieces: 10,461.
- Global suffix-minus-prefix per-word shifts: action +0.0001, physical +0.0009, spatial -0.0057, material -0.0000, quantitative +0.0166, mental/social -0.0030.
- Global suffix-minus-prefix row fractions: cleanish -0.1515, encoding-noise -0.0012, allcaps/glued +0.0639, CHILDES-like +0.4041, subtitle-like -0.1151, list/TOC-like -0.1615.

Largest source shares of recovered tokens:
- childes: 223,846 tokens, share 60.56%, source raw fraction 5.25%.
- simple_wiki: 74,970 tokens, share 20.28%, source raw fraction 4.43%.
- open_subtitles: 38,835 tokens, share 10.51%, source raw fraction 1.47%.
- gutenberg: 29,993 tokens, share 8.11%, source raw fraction 1.21%.
- qwen_pair_packed: 1,279 tokens, share 0.35%, source raw fraction 0.06%.
- cleanqwen_fineweb_compact_view_reinvest: 636 tokens, share 0.17%, source raw fraction 0.10%.

Interpretation: use this when the U256 endpoint score arrives. A positive endpoint would mean this small, source-tail-concentrated visibility repair added useful credit; a negative or mixed endpoint should be read against the suffix enrichment/noise/source mix rather than as evidence against compact-view reinvestment itself.

JSON: `experiments/archive/frontier_consolidation/data/u256_tail_balance_profile/u256_tail_balance_profile.json`
