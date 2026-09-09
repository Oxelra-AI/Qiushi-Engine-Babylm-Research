# live fineweb core scale probe live FineWeb quality tiers

This CPU-only pass refines the live fineweb core scale probe live core pool after sample reading showed residual citation fragments, fiction/media rows, malformed punctuation, and advice-like statements. It does not train or score a model.

Input live fineweb core scale probe near-dedup core pool: 16,817 rows / 435,210 words.

Quality-v2 kept: 11,564 rows / 297,648 words (68.39% of input core words).

Doc caps: cap12 9,421 rows / 242,860 words; cap8 8,580 rows / 221,250 words; cap4 6,735 rows / 173,903 words; cap2 4,628 rows / 119,779 words.

Main rejection reasons: `{'no_strong_relation_or_rich_anchor_set': 1950, 'no_terminal_sentence_punctuation': 765, 'malformed_spacing_or_tokenization': 749, 'first_second_person': 601, 'too_few_specific_anchors': 502, 'too_digit_dense_for_v2': 457, 'uncertain_or_normative': 393, 'quote_fragment': 384, 'unbalanced_parentheses_or_quotes': 276, 'citation_parenthetical_or_et_al': 203, 'starts_with_attribution': 177, 'media_or_fandom': 151, 'list_or_heading_residue': 43, 'low_alpha_for_v2': 7}`.

Kept domain counts: `{'quant': 2603, 'history_society': 2517, 'geography': 2455, 'science': 1447}`.

Scientific use: a future large live FineWeb arm should choose between the broader live fineweb core scale probe cap8 pool and this stricter quality-v2 pool after reading their samples and after the repaired cached-source result arrives. The stricter pool raises self-contained factual precision but reduces usable word yield, so it is a source-quality/coverage tradeoff rather than automatic progress.

Summary JSON: `experiments/archive/representation_and_objectives/data/live_fineweb_quality_tiers/live_fineweb_quality_tiers_summary.json`

Kept sample: `experiments/archive/representation_and_objectives/data/live_fineweb_quality_tiers/live_fineweb_quality_v2_kept_sample.json`

Rejected sample: `experiments/archive/representation_and_objectives/data/live_fineweb_quality_tiers/live_fineweb_quality_v2_rejected_sample.json`
