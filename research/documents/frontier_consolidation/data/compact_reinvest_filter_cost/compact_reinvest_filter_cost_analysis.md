# earlier analysis — compact_view_reinvest selected-packet filter-cost map

CPU-only analysis of the frozen selected compact-reinvest changed block. This does not alter the corpus. It estimates how costly stricter post-filters would be if the temporal DiD points to semantic-view fragility.

Selected reinvest packets: 12155 pairs, 423511 pair words, 261803 source words, 161708 rewrite words.

## Mechanical hazard flags on selected reinvest packets
- content_recall_lt_0p70: 7500 rows (61.70%), 269768 pair words (63.70%)
- content_recall_lt_0p65: 5980 rows (49.20%), 217576 pair words (51.37%)
- content_recall_lt_0p60: 3921 rows (32.26%), 143454 pair words (33.87%)
- content_recall_lt_0p55: 2487 rows (20.46%), 91909 pair words (21.70%)
- lost_modal_or_hedge: 1566 rows (12.88%), 60379 pair words (14.26%)
- length_ratio_lt_0p50: 988 rows (8.13%), 35058 pair words (8.28%)
- new_causal_marker_without_source: 641 rows (5.27%), 23305 pair words (5.50%)
- rewrite_starts_unresolved_pronoun: 455 rows (3.74%), 14980 pair words (3.54%)
- lost_negation: 412 rows (3.39%), 16250 pair words (3.84%)
- length_ratio_lt_0p45: 389 rows (3.20%), 12984 pair words (3.07%)
- question_force_flip: 354 rows (2.91%), 12029 pair words (2.84%)
- lost_attribution: 353 rows (2.90%), 13532 pair words (3.20%)
- entity_recall_lt_1: 218 rows (1.79%), 10077 pair words (2.38%)
- gained_negation: 128 rows (1.05%), 4666 pair words (1.10%)

## Candidate repair-rule word costs
- semantic_force_minimal: remove 3020 rows (24.85%), 113451 pair words (26.79%); keep 310060 pair words
- semantic_force_plus_pronoun: remove 3359 rows (27.63%), 124403 pair words (29.37%); keep 299108 pair words
- content_ge_0p60_plus_force: remove 5722 rows (47.08%), 209499 pair words (49.47%); keep 214012 pair words
- content_ge_0p65_plus_force: remove 7264 rows (59.76%), 263915 pair words (62.32%); keep 159596 pair words
- strict_surface_and_force: remove 7502 rows (61.72%), 272248 pair words (64.28%); keep 151263 pair words
- aggressive_content_ge_0p70_surface_force: remove 8582 rows (70.60%), 307888 pair words (72.70%); keep 115623 pair words

## Domain composition by pair words
- no_domain: 227057 pair words
- science_physical: 61669 pair words
- causal_relational: 57737 pair words
- quant_numeric: 38894 pair words
- institutions_society: 34235 pair words
- geography_places: 30507 pair words
- media_culture: 14240 pair words
- people_history: 14116 pair words

## Interpretation
- what_this_is: mechanical filter-cost map over selected compact packets; not a semantic-error prevalence estimate
- main_repair_signal: If DiD confirms treatment-specific Supplement/EWoK fragility, a plausible repair is to hard-filter force/attribution/modal/causal/pronoun hazards before reinvesting saved words, while preserving compactness and source multiplier.
- risk_of_overfiltering: Content thresholds above 0.65 may remove many pair words and reduce source-diversity gain; use measured costs before any new training allocation.

Machine-readable output: `experiments/archive/frontier_consolidation/data/compact_reinvest_filter_cost/compact_reinvest_filter_cost_analysis.json`
