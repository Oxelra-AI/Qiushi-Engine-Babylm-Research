# earlier analysis — compact-view filter refill feasibility

Accepted medium compact rows: 18682; selected reinvest rows: 12155 (423511 pair words); unused accepted rows: 6527 (221722 pair words).

## Rule feasibility
- semantic_force_minimal: remove 3018 selected rows / 113387 pair words; unused passing pool 4769 rows / 156174 pair words; available:needed=1.38; refill=True; core_removed_words=94233, added_removed_words=19154
- semantic_force_plus_pronoun: remove 3357 selected rows / 124339 pair words; unused passing pool 4519 rows / 148447 pair words; available:needed=1.19; refill=True; core_removed_words=103277, added_removed_words=21062
- content_ge_0p60_plus_force: remove 5721 selected rows / 209463 pair words; unused passing pool 2408 rows / 78950 pair words; available:needed=0.38; refill=False; core_removed_words=176337, added_removed_words=33126
- content_ge_0p65_plus_force: remove 7263 selected rows / 263879 pair words; unused passing pool 1952 rows / 62850 pair words; available:needed=0.24; refill=False; core_removed_words=221684, added_removed_words=42195
- strict_surface_and_force: remove 7501 selected rows / 272212 pair words; unused passing pool 1853 rows / 59120 pair words; available:needed=0.22; refill=False; core_removed_words=228432, added_removed_words=43780

## Interpretation
- semantic_force_rules_are_refillable: The targeted semantic-force filters can be refilled from unused accepted compact rows without new generation if available_to_needed_ratio > 1.
- content_thresholds_are_expensive: Rules with content_recall floors may remove much of the selected compact block and could weaken source-diversity gains even if refillable.
- not_training_authorization: Use this only after exact DiD/full-vector evidence shows semantic repair is worth a new training allocation.

Machine-readable output: `experiments/archive/frontier_consolidation/data/filter_refill_feasibility/filter_refill_feasibility.json`
