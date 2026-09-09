# clean natural nearduplicate adjacency CLEAN row-internal near-duplicate adjacency audit

This Execute note measures the natural dose of the relation-typed composition phenomenon in the CLEAN stream before any new natural-composition training is designed. The 100M training stream is a 10x repeat of the 10M stream, so unique row-internal relation opportunities are counted on the 10M file and exposure counts multiply by ten.

## Definition

Rows are split into sentence/utterance-like spans. A span is eligible when it has at least `8` content tokens after stopword removal. A near-duplicate hit at threshold `τ` is a pair of eligible spans in the same row with content-overlap coefficient `|A∩B|/min(|A|,|B|) >= τ`. I also wrote a fixed-window sensitivity scan with `12` content-token windows and stride `12`. This is a lexical dose measurement, not a trained-model result.

## Main measurement

Input stream: `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl`; SHA256 `ed9d6a2c550c44c0...`. It contains `65313` rows and `10,000,000` words. The designed MAX intervention dose used for the causal mechanism arms is `33291` selected source/companion relations occupying `1,118,587` words per 10M stream before 10x repetition.

Sentence-span hits:

| subset | τ=0.50 unique hits / 10M | τ=0.50 hits / 1M words | τ=0.70 unique hits / 10M | rows with τ=0.50 hit | 100M exposure count at τ=0.50 |
|---|---:|---:|---:|---:|---:|
| natural BabyLM subcorpora only | 4581 | 549.070 | 1569 | 2709 | 45810 |
| designed qwen_pair_packed block | 22528 | 13597.296 | 15808 | 10983 | 225280 |
| all CLEAN rows | 27109 | 2710.900 | 17377 | 13692 | 271090 |

Fixed-window sensitivity for natural-only τ=0.50 finds `51734` unique hits, `6200.738` per 1M words. The two views differ because sentence splitting catches local restatements/utterance repetition, while fixed windows catch repeated lexical blocks independent of punctuation.

## By subcorpus

| source | τ=0.50 sentence hits / 10M | hits / 1M source words | rows with hit | mean hit overlap |
|---|---:|---:|---:|---:|
| childes | 1245 | 458.731 | 736 | 0.672 |
| open_subtitles | 221 | 114.807 | 144 | 0.694 |
| bnc_spoken | 709 | 1170.371 | 352 | 0.611 |
| gutenberg | 351 | 179.107 | 301 | 0.583 |
| simple_wiki | 2050 | 1835.875 | 1172 | 0.666 |
| switchboard | 5 | 226.408 | 4 | 0.544 |
| qwen_pair_packed | 22528 | 13597.296 | 10983 | 0.774 |

## Scientific reading

At τ=0.50, the natural BabyLM subcorpora contribute `4581` row-internal lexical near-duplicate span pairs per 10M stream, whereas the protected `qwen_pair_packed` block contributes `22528` under the same sentence-span definition. Compare this with the designed causal arms' `33291` selected source/companion relations per 10M stream. Thus the practical claim should distinguish (i) the deliberately installed compact relation dose, (ii) the existing designed Qwen-pair block already in CLEAN, and (iii) the smaller or larger natural BabyLM row-internal dose measured here. The paper should not imply that the natural stream necessarily has the same relation dose as the intervention without this table.

If the natural-only τ=0.50 hit count is large enough for a training intervention, the next construction is a CLEAN_NATURAL_SPLIT arm that preserves row texts and word budget as much as possible while moving one member of each naturally co-occurring high-overlap span pair to a separate row. Its readouts should be pre-stated against CLEAN: lower exact copy gain and improved true-source use on source-absent changed/substituted targets would indicate that baseline adjacency itself pays the exact-recurrence cost. If the measured natural dose is too low relative to the 1.82x/2.64x designed doses, the stronger extent test is an ICLM-style composed-window manipulation with near-duplicate filtering on versus off.

## Files

- Candidate distribution table: `experiments/archive/relation_learning/data/clean_natural_adjacency_audit/candidate_span_pair_distribution.csv`
- Threshold hit table: `experiments/archive/relation_learning/data/clean_natural_adjacency_audit/threshold_hit_summary.csv`
- Examples: `experiments/archive/relation_learning/data/clean_natural_adjacency_audit/near_duplicate_examples.jsonl`
- Global JSON: `experiments/archive/relation_learning/data/clean_natural_adjacency_audit/clean_nearduplicate_adjacency_result.json`
- Figure: `experiments/archive/relation_learning/figures/clean_natural_adjacency_audit/clean_neardup_dose_by_source_sentence.png`
- Figure: `experiments/archive/relation_learning/figures/clean_natural_adjacency_audit/clean_neardup_dose_by_source_sentence.pdf`
- Figure: `experiments/archive/relation_learning/figures/clean_natural_adjacency_audit/clean_neardup_dose_by_source_fixed_window.png`
- Figure: `experiments/archive/relation_learning/figures/clean_natural_adjacency_audit/clean_neardup_dose_by_source_fixed_window.pdf`
