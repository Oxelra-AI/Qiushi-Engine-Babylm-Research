# live fineweb core scale probe live FineWeb selector v5 strict-stable

This CPU-only selector repairs the visible v4 weaknesses by rejecting first/second-person source wording, third-person pronoun dependence, attribution/news/modality markers, discourse-open starts, catalogues, fandom/procedural rows, extraction/reference residue, and ambiguous passive fragments. It does not train or score a model.

Input v3 stable pool: 23,364 rows / 538,124 words.

V5 kept: 11,415 rows / 242,601 words (45.08% of v3 stable words).

Doc caps: cap12 9,916 rows / 211,877 words; cap8 9,098 rows / 194,670 words; cap4 7,272 rows / 156,246 words; cap2 5,117 rows / 111,235 words.

Yield per scanned document word: kept=6.458%, doccap12=5.640%, doccap8=5.182%, doccap4=4.159%, doccap2=2.961%.

| target source words | v5 cap8 | v5 cap4 | v5 uncapped |
|---:|---:|---:|---:|
| 1,000,000 | 19,296,820 | 24,042,292 | 15,484,322 |
| 1,750,000 | 33,769,435 | 42,074,011 | 27,097,563 |
| 2,500,000 | 48,242,051 | 60,105,731 | 38,710,805 |
| 3,500,000 | 67,538,871 | 84,148,023 | 54,195,127 |

Type counts: {"quantitative": 4270, "stable_relational_other": 3368, "causal_mechanistic": 2151, "spatial_geographic": 1795, "biographical_historical": 1477, "definition_taxonomy": 473}

Main rejection reasons: {"third_person_pronoun_or_possessive": 6834, "first_second_person": 2113, "many_anchors_without_strong_relation": 2057, "unresolved_nominal_reference": 1321, "discourse_dependent_start": 1196, "attributed_modal_relative_time": 910, "extraction_or_reference_residue": 318, "catalogue_or_list_structure": 257, "media_fandom_or_fiction": 113, "ambiguous_passive_relation": 70, "no_reusable_relation": 42, "too_few_specific_anchors": 34, "title_like_without_finite_relation": 34, "nonlatin_heavy": 29}

Scientific use: v5 is a safer single-sentence source substrate than v2/v3/v4 for faithful-view generation, but it is lower-yield and still derived from an anchor-like stream. If the repaired cached-source run supports FineWeb continuation, the next extractor should apply v5-like rules during streaming and optionally add a separate generic-definition stream rather than relying only on proper-name/number anchors.

Summary JSON: `experiments/archive/representation_and_objectives/data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_summary.json`

Head sample: `experiments/archive/representation_and_objectives/data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_head_sample.json`

Random kept sample: `experiments/archive/representation_and_objectives/data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_kept_sample.json`

Rejected sample: `experiments/archive/representation_and_objectives/data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_rejected_sample.json`
