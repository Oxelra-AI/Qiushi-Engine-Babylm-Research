# semantic repair frontier and refill plan — domain-aware compact-view refill plan

CPU-only replacement planning over existing accepted compact rows. No training corpus was changed.

## force_only
Removed 113387 pair words; chose 113394 replacement words; final-minus-original 7 words; same-domain shortfall before global fill 1055.
Final candidate block: 12517 rows / 423518 pair words; mean content 0.676, mean length ratio 0.640. Replacement rows: 3380 / 113394 words.
Largest primary-domain shifts after refill:
- science_physical: -814 words (-0.19 percentage points)
- no_domain: 678 words (+0.16 percentage points)
- geography_places: 312 words (+0.07 percentage points)
- quant_numeric: -241 words (-0.06 percentage points)
- media_culture: 27 words (+0.01 percentage points)
Replacement rows: `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/force_only_replacement_rows.csv`

## force_plus_pronoun
Removed 124339 pair words; chose 124340 replacement words; final-minus-original 1 words; same-domain shortfall before global fill 3602.
Final candidate block: 12520 rows / 423512 pair words; mean content 0.668, mean length ratio 0.636. Replacement rows: 3722 / 124340 words.
Largest primary-domain shifts after refill:
- no_domain: 3223 words (+0.76 percentage points)
- science_physical: -2590 words (-0.61 percentage points)
- quant_numeric: -847 words (-0.20 percentage points)
- geography_places: 342 words (+0.08 percentage points)
- media_culture: -165 words (-0.04 percentage points)
Replacement rows: `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/force_plus_pronoun_replacement_rows.csv`

## force_surface
Removed 121567 pair words; chose 121572 replacement words; final-minus-original 5 words; same-domain shortfall before global fill 2919.
Final candidate block: 12619 rows / 423516 pair words; mean content 0.670, mean length ratio 0.637. Replacement rows: 3660 / 121572 words.
Largest primary-domain shifts after refill:
- no_domain: 2845 words (+0.67 percentage points)
- science_physical: -1778 words (-0.42 percentage points)
- quant_numeric: -778 words (-0.18 percentage points)
- media_culture: -363 words (-0.09 percentage points)
- people_history: 30 words (+0.01 percentage points)
Replacement rows: `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/force_surface_replacement_rows.csv`

## force_content_ge_0p45
Removed 113387 pair words; chose 106242 replacement words; final-minus-original -7145 words; same-domain shortfall before global fill 13744.
Final candidate block: 12338 rows / 416366 pair words; mean content 0.681, mean length ratio 0.641. Replacement rows: 3201 / 106242 words.
Largest primary-domain shifts after refill:
- no_domain: 5980 words (+2.36 percentage points)
- science_physical: -5021 words (-1.05 percentage points)
- causal_relational: -4501 words (-0.85 percentage points)
- quant_numeric: -2142 words (-0.41 percentage points)
- institutions_society: -1350 words (-0.22 percentage points)
Replacement rows: `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/force_content_ge_0p45_replacement_rows.csv`

## Scientific read
- A semantically safer compact-view repair is mechanically possible without new teacher generation for force-only, force-plus-pronoun, and force-plus-entity/number-surface rules.
- Adding even a low 0.45 content floor makes the unused passing pool insufficient, so content-threshold repair would shrink or regenerate the changed block rather than preserve the current rate–distortion idea.
- Because EWoK fragility is concentrated in physical/spatial/material relations, any later corpus repair should protect science_physical and causal_relational exposure while filtering assertion-force errors.

Machine-readable output: `experiments/archive/frontier_consolidation/data/domain_aware_refill_plan/domain_aware_refill_plan.json`
