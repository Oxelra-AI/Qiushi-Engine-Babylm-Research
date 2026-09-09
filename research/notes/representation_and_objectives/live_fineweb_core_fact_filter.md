# live fineweb core fact filter live FineWeb core factual filter

This CPU-only pass tightens the live FineWeb sentence pool after sample inspection found web instructions, licenses, and advice/contact fragments in the relation-dense set. It is preparation for a possible source+faithful-view corpus, not model evidence.

Strict input: 6,812 rows / 162,847 words.

Core-fact retained: 2,994 rows / 77,353 words (47.50% of strict words), across 617 docs.

Doc-cap8 pool: 2,178 rows / 56,206 words; doc-cap4 pool: 1,608 rows / 41,608 words.

Core domain counts: `{'quant': 619, 'history_society': 565, 'geography': 488, 'science': 436}`.

Main rejection reasons: `{'low_core_fact_score': 2082, 'no_explicit_relation_verb': 2050, 'first_second_person_heavy': 360, 'trailing_colon_title': 280, 'quote_report_fragment': 26, 'too_digit_dense': 18, 'no_entity_or_number_anchor': 6, 'nonlatin': 4, 'low_alpha': 4, 'copyright_license_boilerplate': 3, 'imperative_advice_start': 1, 'explicit_web_action': 1}`.

Use: if pending source-breadth trajectory justifies scaling, run a larger extraction with this stricter notion (or stronger) before Qwen generation. Do not use the relation-dense pool directly as if it were clean factual data.

Summary JSON: `experiments/archive/representation_and_objectives/data/live_fineweb_core_fact_filter/live_fineweb_core_fact_filter_summary.json`

Core pool: `experiments/archive/representation_and_objectives/data/live_fineweb_core_fact_filter/live_fineweb_core_fact_pool.jsonl`

Samples: `experiments/archive/representation_and_objectives/data/live_fineweb_core_fact_filter/live_fineweb_core_fact_sample.json` and `experiments/archive/representation_and_objectives/data/live_fineweb_core_fact_filter/live_fineweb_core_reject_sample.json`
