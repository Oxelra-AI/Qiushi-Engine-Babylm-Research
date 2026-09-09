# cached fineweb quality audit cached FineWeb complete sentence sources

Input: `experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl`

Accepted 61,511 sentence sources, 1,281,741 words, across 5,539 docs from 11,032 single-doc cached rows (1,765,120 words).

| metric | value |
|---|---:|
| split candidates | 92,691 |
| accepted sentences | 61,511 |
| accepted words | 1,281,741 |
| unique docs | 5,539 |
| mean sentence words | 20.84 |

Top rejection reasons: [["too_short", 14605], ["bad_end", 10436], ["bad_start", 9407], ["catalog_like", 2579], ["symbol_or_index_like", 2085], ["too_long", 1815], ["many_digit_tokens", 1427], ["low_alpha", 1221], ["colon_heavy", 529], ["url_or_email", 264]]

Interpretation: this is the preferred source for any future Qwen FineWeb simplification/paraphrase generation because it supplies complete sentence-like spans, not arbitrary row fragments. It is not a trained result.

JSON: `experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/sentence_source_summary.json`

JSONL: `experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/fineweb_sentence_sources.jsonl`
