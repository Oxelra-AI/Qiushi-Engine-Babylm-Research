# earlier analysis item flip edge stratification

If losses are enriched among edge-touching pairs beyond their base-correct fraction, boundary-token redistribution is implicated; if not, losses are not localized by this edge test.

Anchor: `chck82`.

## Primary summaries for first/last 3-token touch

| comparison | column | payload Δ | reconstructed Δ | gains | losses | loss edge frac | base-correct edge frac | edge loss rate | non-edge loss rate | worst groups |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| coherent86_s43022_minus_chck82 | Supplement | +0.702 | -0.192 | 27 | 37 | 0.838 | 0.878 | 0.009 | 0.013 | subject_aux_inversion -14/3867, turn_taking +0/280, hypernym +1/842, qa_congruence_tricky +1/165 |
| coherent86_s43022_minus_chck82 | EWoK | -0.035 | +0.053 | 117 | 113 | 0.389 | 0.395 | 0.029 | 0.030 | social-properties -4/328, physical-dynamics -1/120, spatial-relations -2/490, quantitative-properties -1/314 |
| dense64_minus_chck82 | Supplement | +0.102 | -0.287 | 60 | 75 | 0.893 | 0.878 | 0.020 | 0.017 | hypernym -7/842, subject_aux_inversion -9/3867, qa_congruence_tricky +0/165, turn_taking +0/280 |
| dense64_minus_chck82 | EWoK | -0.135 | +0.223 | 281 | 264 | 0.386 | 0.395 | 0.068 | 0.070 | social-properties -9/328, physical-dynamics -3/120, physical-interactions -9/556, quantitative-properties -2/314 |
| dense65_minus_chck82 | Supplement | +0.152 | -0.326 | 58 | 75 | 0.907 | 0.878 | 0.020 | 0.015 | hypernym -7/842, subject_aux_inversion -12/3867, qa_congruence_tricky +0/165, turn_taking +1/280 |
| dense65_minus_chck82 | EWoK | -0.285 | +0.105 | 280 | 272 | 0.404 | 0.395 | 0.073 | 0.070 | social-properties -10/328, physical-dynamics -3/120, physical-interactions -8/556, quantitative-properties -2/314 |
| clean64_minus_chck82 | Supplement | +0.342 | -0.172 | 47 | 56 | 0.893 | 0.878 | 0.015 | 0.013 | hypernym -2/842, subject_aux_inversion -9/3867, turn_taking +0/280, qa_congruence_tricky +1/165 |
| clean64_minus_chck82 | EWoK | -0.235 | +0.039 | 218 | 215 | 0.353 | 0.395 | 0.051 | 0.060 | social-properties -10/328, physical-dynamics -2/120, physical-relations -5/818, physical-interactions -3/556 |
| clean65_minus_chck82 | Supplement | +0.352 | -0.192 | 46 | 56 | 0.893 | 0.878 | 0.015 | 0.013 | subject_aux_inversion -11/3867, hypernym -1/842, turn_taking +0/280, qa_congruence_tricky +1/165 |
| clean65_minus_chck82 | EWoK | -0.335 | +0.000 | 217 | 217 | 0.378 | 0.395 | 0.055 | 0.059 | social-properties -10/328, physical-dynamics -2/120, social-interactions -2/294, physical-relations -5/818 |

## Notes

- `edge_touch` means at least one word-level difference between the alternatives lies in the first or last k content tokens of either sentence/string.
- EWoK follows the validated earlier analysis reconstruction: good is `Context1 Target1`, bad is `Context2 Target1`.
- Full examples and k=5 summaries are in the JSON and CSV files.

JSON: `experiments/archive/relation_learning/data/item_edge_vs_parent/item_edge_flip_stratification.json`
