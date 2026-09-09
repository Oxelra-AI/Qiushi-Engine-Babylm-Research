# semantic repair frontier and refill plan — EWoK concept exposure under candidate semantic refills

Weak lexical exposure check: counts official EWoK ConceptA/ConceptB token hits in source/rewrite text for the original selected compact block and candidate repaired blocks.

## force_only
Final rows 12517; final pair words 423518; replacements 3380 rows / 113394 words.
Most negative official EWoK-DiD domains and exposure deltas:
- material-dynamics: official DiD -17.53; coverage_delta +0.067; source_hit_rows_delta -1; rewrite_hit_rows_delta -2; either_hit_rows_delta -1
- physical-dynamics: official DiD -10.83; coverage_delta +0.091; source_hit_rows_delta -18; rewrite_hit_rows_delta -13; either_hit_rows_delta -19
- spatial-relations: official DiD -8.37; coverage_delta +0.000; source_hit_rows_delta +44; rewrite_hit_rows_delta +59; either_hit_rows_delta +44
- physical-interactions: official DiD -5.94; coverage_delta -0.043; source_hit_rows_delta -66; rewrite_hit_rows_delta +8; either_hit_rows_delta -52
- social-relations: official DiD -2.13; coverage_delta +0.000; source_hit_rows_delta -18; rewrite_hit_rows_delta -14; either_hit_rows_delta -15

## force_plus_pronoun
Final rows 12520; final pair words 423512; replacements 3722 rows / 124340 words.
Most negative official EWoK-DiD domains and exposure deltas:
- material-dynamics: official DiD -17.53; coverage_delta +0.067; source_hit_rows_delta +2; rewrite_hit_rows_delta -1; either_hit_rows_delta +3
- physical-dynamics: official DiD -10.83; coverage_delta +0.091; source_hit_rows_delta -22; rewrite_hit_rows_delta -18; either_hit_rows_delta -24
- spatial-relations: official DiD -8.37; coverage_delta +0.000; source_hit_rows_delta +52; rewrite_hit_rows_delta +63; either_hit_rows_delta +54
- physical-interactions: official DiD -5.94; coverage_delta -0.043; source_hit_rows_delta -64; rewrite_hit_rows_delta -9; either_hit_rows_delta -48
- social-relations: official DiD -2.13; coverage_delta +0.000; source_hit_rows_delta -12; rewrite_hit_rows_delta -11; either_hit_rows_delta -9

## force_surface
Final rows 12619; final pair words 423516; replacements 3660 rows / 121572 words.
Most negative official EWoK-DiD domains and exposure deltas:
- material-dynamics: official DiD -17.53; coverage_delta +0.067; source_hit_rows_delta +2; rewrite_hit_rows_delta -2; either_hit_rows_delta +2
- physical-dynamics: official DiD -10.83; coverage_delta +0.091; source_hit_rows_delta -14; rewrite_hit_rows_delta -14; either_hit_rows_delta -16
- spatial-relations: official DiD -8.37; coverage_delta +0.000; source_hit_rows_delta +24; rewrite_hit_rows_delta +40; either_hit_rows_delta +26
- physical-interactions: official DiD -5.94; coverage_delta -0.043; source_hit_rows_delta -62; rewrite_hit_rows_delta -17; either_hit_rows_delta -56
- social-relations: official DiD -2.13; coverage_delta +0.000; source_hit_rows_delta -17; rewrite_hit_rows_delta -15; either_hit_rows_delta -14

## Scientific read
- This protects against a bad repair that exactly refills word count while removing terms from material/physical/spatial EWoK domains.
- The measurement is lexical and weak; it should be combined with exact DiD and semantic-hazard evidence before any new training.

Machine-readable output: `experiments/archive/frontier_consolidation/data/repair_ewok_concept_exposure/repair_ewok_concept_exposure.json`
