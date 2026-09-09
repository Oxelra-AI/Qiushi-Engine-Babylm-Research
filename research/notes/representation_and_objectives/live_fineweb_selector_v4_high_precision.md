# live fineweb core scale probe live FineWeb selector v4 high precision

This CPU-only pass refines selector-v3 after sample reading showed remaining rows with unresolved `them/it/this/their`, discourse-open starts, attribution/news timing, catalogues, and fandom/procedural residue. It does not train or score a model.

Input v3 stable pool: 23,364 rows / 538,124 words.

V4 kept: 13,509 rows / 292,118 words (54.28% of v3 stable words).

Doc caps: cap12 11,596 rows / 252,118 words; cap8 10,564 rows / 230,568 words; cap4 8,293 rows / 182,470 words; cap2 5,668 rows / 125,923 words.

Yield per scanned document word: kept=7.776%, doccap12=6.711%, doccap8=6.138%, doccap4=4.857%, doccap2=3.352%.

Projected scanned document words needed:

| target source words | v4 cap8 | v4 cap4 | v4 uncapped |
|---:|---:|---:|---:|
| 1,000,000 | 16,292,426 | 20,587,012 | 12,859,570 |
| 1,750,000 | 28,511,745 | 36,027,270 | 22,504,248 |
| 2,500,000 | 40,731,064 | 51,467,529 | 32,148,926 |
| 3,500,000 | 57,023,490 | 72,054,540 | 45,008,497 |

Type counts: {"quantitative": 5020, "stable_relational_other": 3886, "causal_mechanistic": 2649, "spatial_geographic": 2215, "biographical_historical": 1694, "definition_taxonomy": 569}

Main v4 rejection reasons: {"third_person_pronoun_or_possessive": 6834, "many_anchors_without_strong_relation": 2067, "discourse_dependent_start": 1177, "unresolved_nominal_reference": 570, "catalogue_or_list_structure": 257, "attributed_modal_relative_time": 161, "extraction_or_web_residue": 124, "media_fandom_or_fiction": 113, "ambiguous_passive_relation": 58, "no_reusable_relation": 41, "too_few_specific_anchors": 33}

Scientific use: use this as the high-precision single-sentence candidate source pool if the repaired cached-source training supports continuing FineWeb. Compare it with v3/v2 samples before bulk generation; v4 is safer for faithful rewriting but lower-yield and still needs tokenizer exposure and benchmark-overlap checks in any materialized arms.

Summary JSON: `experiments/archive/representation_and_objectives/data/live_fineweb_selector_v4_high_precision/live_fineweb_selector_v4_high_precision_summary.json`

Kept sample: `experiments/archive/representation_and_objectives/data/live_fineweb_selector_v4_high_precision/live_fineweb_selector_v4_kept_sample.json`

Rejected sample: `experiments/archive/representation_and_objectives/data/live_fineweb_selector_v4_high_precision/live_fineweb_selector_v4_rejected_sample.json`
