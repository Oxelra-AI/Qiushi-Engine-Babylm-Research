# fw comparison mechanical audit — compact vs source-breadth mechanical audit

- Both 10M arms have 64,183 rows and 10,000,000 words; row word sequence match: True.
- Source labels match except the FineWeb arm label: True.
- Shared tokenizer `experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer` has len=16384, tokenizer.json SHA `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`.
- Compact tokens/word visible: 1.4449; breadth tokens/word visible: 1.4419.
- Compact rows over 256 tokens: 16,294; breadth rows over 256 tokens: 15,951.
- WWM groups visible per pass: compact 9,790,415, breadth 9,794,105.
- Breadth companion source hashes overlapping selected compact-pair source hashes: 0.
- Exact score-text overlap common sources: n7: 21 comps/28 unique, n8: 6 comps/7 unique, n10: 0 comps/0 unique.
- Exact score-text overlap compact rewrites: n7: 7 comps/8 unique, n8: 1 comps/1 unique, n10: 0 comps/0 unique.
- Exact score-text overlap source-breadth companions: n7: 19 comps/28 unique, n8: 7 comps/11 unique, n10: 1 comps/1 unique.

JSON: `experiments/archive/representation_and_objectives/data/fw_comparison_mechanical_audit/fw_compact_breadth_mechanical_audit.json`
