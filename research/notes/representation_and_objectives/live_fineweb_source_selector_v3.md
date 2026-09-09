# live fineweb core scale probe live FineWeb source selector v3

This CPU-only repair responds to source-sample reading: it cleans extraction artifacts, separates attributed or time-relative text from stable factual rows, rejects unresolved antecedents, labels proposition types, and adds within-document redundancy control. It does not train or score a model.

Input strict-anchor-like pool: 106,118 rows / 2,201,820 original words from 3,756,512 scanned document words.

Accepted before redundancy control: 23,468 rows / 540,166 repaired words. Stable after within-doc redundancy: 23,364 rows / 538,124 words.

Doc caps: cap12 17,828 rows / 411,078 words; cap8 15,583 rows / 359,452 words; cap4 11,236 rows / 259,177 words; cap2 7,032 rows / 162,676 words.

Stable yield per scanned document word: stable=14.325%, stable_doccap12=10.943%, stable_doccap8=9.569%, stable_doccap4=6.899%, stable_doccap2=4.331%.

Projected scanned document words needed for future source budgets:

| target source words | stable cap8 | stable cap4 | stable uncapped |
|---:|---:|---:|---:|
| 1,000,000 | 10,450,664 | 14,494,002 | 6,980,755 |
| 1,750,000 | 18,288,662 | 25,364,504 | 12,216,322 |
| 2,500,000 | 26,126,659 | 36,235,005 | 17,451,888 |
| 3,500,000 | 36,577,323 | 50,729,008 | 24,432,644 |

Final proposition types: {"quantitative": 7670, "stable_relational_other": 7088, "causal_mechanistic": 4775, "spatial_geographic": 3592, "biographical_historical": 2968, "catalogue_list": 888, "definition_taxonomy": 830}

Main rejection reasons: {"no_reusable_relation": 67915, "attributed_modal_or_relative_time": 14820, "no_terminal_sentence_punctuation": 10148, "unresolved_sentence_initial_reference": 7895, "too_few_anchors_for_non_definition": 5609, "quote_fragment": 4389, "unbalanced_parentheses_or_quotes": 3743, "malformed_spacing": 2868, "too_digit_dense": 2461, "catalogue_or_entity_list": 2069, "contact_or_listing": 834, "html_or_markup": 821, "length_outside_10_42": 768, "web_or_navigation": 686, "media_fandom": 469, "url_or_email": 458}

Scientific reading: this selector is more aligned with self-contained, stable, masked-LM-useful factual relations than the earlier anchor-heavy pool, but its yield is lower. It should be sample-read before any bulk rewriting, and the repaired cached-source training result should still decide whether FineWeb source breadth deserves the next H100 allocation.

Summary JSON: `experiments/archive/representation_and_objectives/data/live_fineweb_source_selector_v3/live_fineweb_source_selector_v3_summary.json`

Stable sample: `experiments/archive/representation_and_objectives/data/live_fineweb_source_selector_v3/live_fineweb_selector_v3_stable_sample.json`

Rejected sample: `experiments/archive/representation_and_objectives/data/live_fineweb_source_selector_v3/live_fineweb_selector_v3_rejected_sample.json`
