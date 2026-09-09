# clean natural nearduplicate adjacency refined CLEAN natural near-duplicate adjacency audit

The first clean natural nearduplicate adjacency audit measured model-visible lexical overlap and intentionally left corpus scaffolding visible. Inspecting examples showed that CHILDES speaker labels and bracketed action annotations inflate some high-overlap counts. This refined pass estimates language-content adjacency by stripping `*MOT:`/`*CHI:`-style labels, `%` tier labels, and bracketed action/annotation spans before tokenization. The model-visible table remains useful as an upper-bound exposure measurement because those tokens are in the stream; the refined table is the better guide for a natural-language composition intervention.

## Refined sentence-span dose

| subset | τ | unique hits / 10M | hits / 1M words | rows with hit | exact-sequence hits | nonidentical high-overlap hits | nonidentical fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| visible:NATURAL_ONLY | 0.50 | 4036 | 483.747 | 2380 | 200 | 3824 | 0.947 |
| visible:NATURAL_ONLY | 0.70 | 1452 | 174.034 | 710 | 200 | 1240 | 0.854 |
| visible:NATURAL_ONLY | 0.80 | 984 | 117.940 | 446 | 200 | 772 | 0.785 |
| visible:DESIGNED_QWEN_PAIR_PACKED | 0.50 | 22311 | 13466.321 | 10927 | 124 | 22043 | 0.988 |
| visible:DESIGNED_QWEN_PAIR_PACKED | 0.70 | 15677 | 9462.216 | 8773 | 124 | 15409 | 0.983 |
| visible:DESIGNED_QWEN_PAIR_PACKED | 0.80 | 11544 | 6967.648 | 7254 | 124 | 11276 | 0.977 |
| visible:ALL | 0.50 | 26347 | 2634.700 | 13307 | 324 | 25867 | 0.982 |
| visible:ALL | 0.70 | 17129 | 1712.900 | 9483 | 324 | 16649 | 0.972 |
| visible:ALL | 0.80 | 12528 | 1252.800 | 7700 | 324 | 12048 | 0.962 |
| refined:NATURAL_ONLY | 0.50 | 3508 | 420.462 | 2107 | 79 | 3420 | 0.975 |
| refined:NATURAL_ONLY | 0.70 | 1232 | 147.665 | 615 | 79 | 1144 | 0.929 |
| refined:NATURAL_ONLY | 0.80 | 813 | 97.445 | 390 | 79 | 725 | 0.892 |
| refined:DESIGNED_QWEN_PAIR_PACKED | 0.50 | 22117 | 13349.227 | 10873 | 120 | 21853 | 0.988 |
| refined:DESIGNED_QWEN_PAIR_PACKED | 0.70 | 15574 | 9400.048 | 8728 | 120 | 15310 | 0.983 |
| refined:DESIGNED_QWEN_PAIR_PACKED | 0.80 | 11486 | 6932.641 | 7227 | 120 | 11222 | 0.977 |
| refined:ALL | 0.50 | 25625 | 2562.500 | 12980 | 199 | 25273 | 0.986 |
| refined:ALL | 0.70 | 16806 | 1680.600 | 9343 | 199 | 16454 | 0.979 |
| refined:ALL | 0.80 | 12299 | 1229.900 | 7617 | 199 | 11947 | 0.971 |

## By natural subcorpus after stripping

| source | τ=0.50 hits / 10M | hits / 1M words | rows with hit | exact-sequence | nonidentical | τ=0.70 hits |
|---|---:|---:|---:|---:|---:|---:|
| childes | 234 | 86.219 | 176 | 21 | 211 | 115 |
| open_subtitles | 191 | 99.223 | 132 | 17 | 170 | 80 |
| bnc_spoken | 701 | 1157.165 | 346 | 0 | 701 | 150 |
| gutenberg | 326 | 166.351 | 280 | 6 | 320 | 48 |
| simple_wiki | 2051 | 1836.770 | 1169 | 35 | 2013 | 839 |
| switchboard | 5 | 226.408 | 4 | 0 | 5 | 0 |

## Scientific reading

After stripping annotation scaffolding, natural BabyLM subcorpora still contain `3508` high-overlap sentence-span pairs at τ=0.50 and `1232` at τ=0.70 per unique 10M stream. These become `35080` and `12320` pair exposures in the repeated 100M training stream. The designed causal intervention used `33291` selected same-window relations per 10M stream, so the natural language-content dose is material but several-fold smaller than the deliberate compact relation dose. The `qwen_pair_packed` block in CLEAN remains much denser (`22117` refined τ=0.50 hits), so the baseline already contains a large designed restatement/repetition block in addition to natural BabyLM adjacency.

The next natural-composition construction should not claim it is pure BabyLM if it includes `qwen_pair_packed`; it should either (a) split only refined NATURAL_ONLY high-overlap sentence pairs to test whether ordinary row composition contributes a smaller but measurable copy/restatement pressure, or (b) explicitly split the qwen-pair block as a baseline-designed-adjacency control. Because the refined natural dose is far below the 2.64x designed dose, any training readout should be framed as a sensitivity/extent measurement, not as a prerequisite for the already established DeBERTa compact locality result.

## Files

- Refined threshold table: `experiments/archive/relation_learning/data/clean_natural_adjacency_refined/threshold_hit_summary_refined.csv`
- Candidate distribution: `experiments/archive/relation_learning/data/clean_natural_adjacency_refined/candidate_distribution_refined.csv`
- Examples: `experiments/archive/relation_learning/data/clean_natural_adjacency_refined/near_duplicate_examples_refined.jsonl`
- Result JSON: `experiments/archive/relation_learning/data/clean_natural_adjacency_refined/refined_clean_nearduplicate_adjacency_result.json`
- First model-visible audit: `experiments/archive/relation_learning/data/clean_natural_adjacency_audit/threshold_hit_summary.csv`
- Figure: `experiments/archive/relation_learning/figures/clean_natural_adjacency_refined/clean_sentence_neardup_visible.png`
- Figure: `experiments/archive/relation_learning/figures/clean_natural_adjacency_refined/clean_sentence_neardup_visible.pdf`
- Figure: `experiments/archive/relation_learning/figures/clean_natural_adjacency_refined/clean_sentence_neardup_refined.png`
- Figure: `experiments/archive/relation_learning/figures/clean_natural_adjacency_refined/clean_sentence_neardup_refined.pdf`
