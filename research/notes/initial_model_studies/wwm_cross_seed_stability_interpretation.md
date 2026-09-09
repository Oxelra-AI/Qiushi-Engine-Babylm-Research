# wwm cross seed stability interpretation — Cross-seed stability of whole-word masking at 1M

Seed-42 evidence: `experiments/archive/initial_model_studies/data/masked_1m_grid_pos512_profile.json`
Seed-43 evidence: `experiments/archive/initial_model_studies/data/masked_1m_seed43_profile.json`

Both comparisons isolate WWM against token-level masking under the same model family, tokenizer, exposure, position capacity, official `mlm` backend, and paired data/order within seed. Seed 43 changes only the random condition.

| column | seed42 WWM-token | seed43 WWM-token | mean | positive both seeds |
|---|---:|---:|---:|---:|
| blimp_fast | 1.83 | 2.47 | 2.15 | True |
| supplement_fast | 0.40 | -1.20 | -0.40 | False |
| ewok_fast | 0.09 | 4.91 | 2.50 | True |
| entity_tracking_fast | 0.85 | 0.17 | 0.51 | True |
| comps | -0.13 | -0.12 | -0.12 | False |
| reading_eye_tracking | -0.09 | -0.03 | -0.06 | False |
| reading_self_paced | -0.23 | 0.03 | -0.10 | False |

## Interpretation

Positive in both seeds: blimp_fast, ewok_fast, entity_tracking_fast.
The WWM effect is cross-seed positive for BLiMP, EWoK, and Entity Tracking, but not for Supplement, COMPS, or Reading. The Entity gain is modest but repeated; WWM alone does not close the large gap to the current top system, so subsequent work should treat WWM as a fixed base and decompose the next factors from the SOTA route one at a time.
