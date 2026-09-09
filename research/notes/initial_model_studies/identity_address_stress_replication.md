# identity address stress replication identity-address stress replication

Decision: **IDENTITY_ADDRESS_STATE_RETRIEVAL_REPLICATED_ON_STRESS_TESTS**

Fresh tests vary entity count, overwrite depth, distractors, names, objects, and templates.
No BabyLM evaluation data is used.

| split | exact seed42/43 | displaced correct-address seed42/43 |
|---|---:|---:|
| variable_entities | 1.000 / 0.987 | 0.903 / 0.947 |
| multiple_overwrites | 1.000 / 0.980 | 0.997 / 1.000 |
| long_distractors | 1.000 / 0.993 | 0.937 / 0.985 |

Gate components: {'exact_pass': True, 'displaced_pass': True, 'causal_pass': True}

Next action: Promote the mechanism family to a natural-text gate, not to an SOTA claim: implement word-span addressing, measure activation coverage and MLM retention, and compare true-address, shuffled-address, and standard WWM from matched initialization.

Evidence JSON: `experiments/archive/initial_model_studies/data/identity_address_stress_replication.json`
