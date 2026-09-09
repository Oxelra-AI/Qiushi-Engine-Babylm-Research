# fw breadth boundary audit — compact vs repaired whole-sentence source-breadth check

- Both 10M arms have 64,183 rows and 10,000,000 words; row word sequence match: True.
- Source labels match except the FineWeb arm label: True.
- FineWeb rows: compact 5,785, repaired breadth 5,785; repaired breadth rows marked whole-sentence: 5,785.
- Shared tokenizer `experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer` has len=16384, tokenizer.json SHA `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`.
- Compact visible tokens/word: 1.4449; repaired breadth visible tokens/word: 1.4412.
- Compact rows over 256 tokens: 16,294; repaired breadth rows over 256 tokens: 15,907.
- WWM groups visible per pass: compact 9,790,415, repaired breadth 9,794,415.
- Repaired breadth companion source hashes overlapping selected compact-pair source hashes: 0.
- 100M repaired breadth stream: 100,000,000 words / 641,830 rows; pass orders match existing compact stream: True.
- Exact score-text overlap common sources: n7 21 comps/28 unique, n8 6 comps/7 unique, n10 0 comps/0 unique.
- Exact score-text overlap compact rewrites: n7 7 comps/8 unique, n8 1 comps/1 unique, n10 0 comps/0 unique.
- Exact score-text overlap repaired breadth companions: n7 20 comps/27 unique, n8 6 comps/8 unique, n10 0 comps/0 unique.

Use the repaired whole-sentence breadth stream for the first compact-vs-breadth H100 comparison; do not use the fw comparison mechanical audit sliced breadth stream.

JSON: `experiments/archive/representation_and_objectives/data/fw_compact_vs_wholesentence_breadth_check/fw_compact_vs_wholesentence_breadth_check.json`
