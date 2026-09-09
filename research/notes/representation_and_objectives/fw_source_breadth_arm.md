# fw comparison mechanical audit — FineWeb source-breadth arm

The compact arm is compared against an independent FineWeb source-breadth arm rather than literal repetition as the first expensive data test. The breadth arm keeps the same base filler rows, the same selected first-source spans, the same 813,005-word FineWeb block, the same row word sequence, and the same shared 16k tokenizer; only the 318,851 compact rewrite words are replaced by independently selected FineWeb source words.

- Common FineWeb source words retained: 494,154
- Compact rewrite words replaced by breadth: 318,851
- FineWeb block words: 813,005
- Reused filler words: 9,186,995
- Total arm words: 10,000,000
- Selected independent breadth sources: 13,836 rows / 318,851 words, exact strategy `tail_subset_full_sentences`
- Row word sequence matches compact arm: yes (64,183 rows)
- Shared tokenizer: `experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer`

Manifest: `experiments/archive/representation_and_objectives/data/fw_source_breadth_arm/fw_source_breadth_arm_manifest.json`
10M breadth arm: `experiments/archive/representation_and_objectives/data/fw_source_breadth_arm/fw_preserved_source_breadth_10M.jsonl`
