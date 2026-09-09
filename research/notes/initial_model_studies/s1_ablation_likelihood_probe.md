# counterfactual propagation closed — primary-likelihood s1 ablation probe

Model: `experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M`
Data: `experiments/archive/initial_model_studies/data/counterfactual_revision_109/linked_definition_counterfactual_v3_seed1093_target100000_actual.jsonl`
Evidence JSON: `experiments/archive/initial_model_studies/data/s1_ablation_likelihood_probe.json`

Deltas are target-token mean logprob with true s1 minus no-s1 or wrong-s1 contexts. Positive means the true previous sentence helps predict the masked s2 target.

| target type | n | true-no mean | true-no positive frac | true-wrong mean | true-wrong positive frac | mean target tokens |
|---|---:|---:|---:|---:|---:|---:|
| initial_dependent | 64 | +2.3706 | 0.969 | +0.0004 | 0.453 | 1.00 |
| first_s2_content | 64 | +0.3841 | 0.703 | +0.2993 | 0.578 | 1.14 |
| last_s2_content | 64 | +0.2526 | 0.578 | +0.3591 | 0.641 | 1.53 |
| first_two_s2_content_window | 61 | +0.3833 | 0.754 | +0.2745 | 0.656 | 4.33 |

Interpretation note: this probes whether the existing protected WWM model already uses true s1 for candidate s2 spans. Strongly positive dependency-specific deltas would motivate XSpan/Prefix-LM construction; near-zero or negative deltas mean the current span design does not create load-bearing cross-sentence likelihood.
