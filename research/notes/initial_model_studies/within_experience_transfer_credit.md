# within experience transfer credit within-experience transfer-credit gate

Decision: **REJECT_CURRENT_WITHIN_EXPERIENCE_TRANSFER_CREDIT_SIGNAL**

One bounded update used mask set A; evaluation used lexically disjoint mask set B.
No BabyLM evaluation text or labels were used.

| checkpoint | aligned gain | aligned-same-bag mean (CI95) | positive | aligned-cross mean (CI95) | pass |
|---|---:|---:|---:|---:|---:|
| wwm_seed42_60M | +0.00212813 | -0.00002456 ([-0.00029702, +0.00024370]) | 0.425 | +0.00048262 ([-0.00077579, +0.00172515]) | False |
| wwm_seed43_60M | +0.00218652 | -0.00015718 ([-0.00042700, +0.00010638]) | 0.475 | +0.00124177 ([+0.00003581, +0.00251376]) | False |

This gate tests whether a new credit-assignment target exists; it is not a model score.

Next action: Do not build the meta-objective from this measurement. Reopen the bottleneck at representation formation rather than manufacturing another contrastive loss.

Evidence JSON: `experiments/archive/initial_model_studies/data/within_experience_transfer_credit.json`
