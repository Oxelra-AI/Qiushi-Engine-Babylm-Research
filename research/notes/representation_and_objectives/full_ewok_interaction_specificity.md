# full ewok interaction specificity — full EWoK interaction-specificity measurement

Existing checkpoints only; no training and no official benchmark run. The measurement asks whether EWoK errors are conditional target-reversal failures: changing context fails to reorder the two targets after target priors cancel.

Rows scored per model: `7618` of `7618`. Device `cpu`, threads `16`.

| model | saved accuracy | saved wrong | stable conditional-reversal failures | frac among saved wrong | interaction median all | interaction median saved-wrong | both target margins positive saved-wrong | sign match official |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legal40_depth_12x384_43022 | 0.5128642688369651 | 3711 | 2494 | 0.6720560495823228 | 0.1037634015083313 | -0.7359655052423477 | 0.03125842091080571 | 1.0 |
| legal40_8x480_43022 | 0.5112890522446837 | 3723 | 2550 | 0.684931506849315 | 0.02262457855977118 | -0.7870388847077265 | 0.0319634703196347 | 1.0 |

Two-model overlap: stable conditional-reversal failure in both models on `1471` / `7618` rows (0.19309530060383304).

Interpretation: a surviving future route would need corpus-derived two-context/two-target structures in which target priors and lexical plausibility cancel. Broad sentence-level replacement is not supported by this measurement alone.

JSON: `experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_fullcpu.json`

Records CSV: `experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_records_fullcpu.csv`

By-domain CSV: `experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_by_domain_fullcpu.csv`

By-context-type CSV: `experiments/archive/representation_and_objectives/data/full_ewok_interaction_specificity/ewok_full_interaction_specificity_by_context_type_fullcpu.csv`
