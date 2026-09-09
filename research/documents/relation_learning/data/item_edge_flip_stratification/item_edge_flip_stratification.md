# earlier analysis item flip edge stratification

If losses are enriched among edge-touching pairs beyond their base-correct fraction, boundary-token redistribution is implicated; if not, losses are not localized by this edge test.

Anchor: `coherent86_s43022`.

## Primary summaries for first/last 3-token touch

| comparison | column | payload Δ | reconstructed Δ | gains | losses | loss edge frac | base-correct edge frac | edge loss rate | non-edge loss rate | worst groups |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| coherent_special_98097_minus_coherent86_s43022 | Supplement | -2.650 | -1.303 | 65 | 133 | 0.962 | 0.878 | 0.038 | 0.011 | qa_congruence_tricky -12/165, qa_congruence_easy -2/64, hypernym -19/842, subject_aux_inversion -36/3867 |
| coherent_special_98097_minus_coherent86_s43022 | EWoK | -0.330 | -0.276 | 419 | 440 | 0.448 | 0.393 | 0.132 | 0.105 | physical-dynamics -3/120, social-properties -5/328, social-relations -18/1548, social-interactions -2/294 |
| coherent_special_98098_minus_coherent86_s43022 | Supplement | -2.900 | -1.686 | 61 | 149 | 0.899 | 0.878 | 0.040 | 0.032 | qa_congruence_tricky -11/165, qa_congruence_easy -3/64, hypernym -20/842, subject_aux_inversion -56/3867 |
| coherent_special_98098_minus_coherent86_s43022 | EWoK | -0.800 | -0.722 | 439 | 494 | 0.425 | 0.393 | 0.140 | 0.123 | physical-dynamics -4/120, material-dynamics -22/770, social-interactions -7/294, social-relations -15/1548 |
| half_98097_minus_coherent86_s43022 | Supplement | -2.910 | -1.533 | 156 | 236 | 0.763 | 0.878 | 0.053 | 0.119 | qa_congruence_easy -4/64, qa_congruence_tricky -7/165, hypernym -17/842, subject_aux_inversion -50/3867 |
| half_98097_minus_coherent86_s43022 | EWoK | +0.640 | -0.131 | 709 | 719 | 0.331 | 0.393 | 0.159 | 0.208 | physical-relations -25/818, material-properties -4/170, material-dynamics -15/770, agent-properties -19/2210 |
| half_98098_minus_coherent86_s43022 | Supplement | -3.480 | -1.725 | 149 | 239 | 0.791 | 0.878 | 0.056 | 0.106 | qa_congruence_tricky -11/165, qa_congruence_easy -4/64, hypernym -20/842, subject_aux_inversion -53/3867 |
| half_98098_minus_coherent86_s43022 | EWoK | +0.350 | -0.210 | 697 | 713 | 0.339 | 0.393 | 0.162 | 0.204 | material-dynamics -25/770, physical-relations -25/818, material-properties -4/170, social-interactions -5/294 |

## Notes

- `edge_touch` means at least one word-level difference between the alternatives lies in the first or last k content tokens of either sentence/string.
- EWoK follows the validated earlier analysis reconstruction: good is `Context1 Target1`, bad is `Context2 Target1`.
- Full examples and k=5 summaries are in the JSON and CSV files.

JSON: `experiments/archive/relation_learning/data/item_edge_flip_stratification/item_edge_flip_stratification.json`
