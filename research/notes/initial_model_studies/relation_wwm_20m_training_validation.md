# relation wwm 20m training validation — relation-WWM 20M training validation

Evidence JSON: `experiments/archive/initial_model_studies/data/relation_wwm_20m_training_validation.json`

Both 20M runs are complete and loadable. These are mechanism-screening artifacts, not BabyLM score evidence until evaluated.

| quantity | relation_wwm | shuffled control |
|---|---:|---:|
| word exposure | 20000000 | 20000000 |
| optimizer steps | 489 | 489 |
| LR schedule total | 2442 | 2442 |
| checkpoints | chck_10M, chck_20M | chck_10M, chck_20M |
| loss first→last | 9.7824→3.6498 | 9.7788→3.5584 |
| selected group fraction | 0.150 | 0.150 |
| candidate relation fraction | 0.080 | 0.080 |
| selected relation fraction | 0.151 | 0.080 |
| candidate entity fraction | 0.252 | 0.252 |
| selected entity fraction | 0.442 | 0.252 |
| selected state fraction | 0.0094 | 0.0044 |

Next evidence required: official-compatible scores at chck_10M and chck_20M for Entity, GlobalPIQA, Reading, BLiMP, Supplement, and COMPS, compared to each other and to the ordinary WWM trajectory.
