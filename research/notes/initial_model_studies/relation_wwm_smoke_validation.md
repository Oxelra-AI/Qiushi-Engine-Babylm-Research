# relation wwm 20m launch — relation-WWM smoke validation

Evidence JSON: `experiments/archive/initial_model_studies/data/relation_wwm_smoke_validation.json`

Both smokes completed with ai_lab-accepted artifacts and loadable HF checkpoints.

| metric | relation_wwm | shuffled control |
|---|---:|---:|
| word exposure | 20000 | 20000 |
| optimizer steps | 8 | 8 |
| selected groups / candidate groups | 3000/20000 | 3000/20000 |
| selected group fraction | 0.150 | 0.150 |
| candidate relation fraction | 0.088 | 0.088 |
| selected relation fraction | 0.166 | 0.088 |
| candidate entity fraction | 0.277 | 0.277 |
| selected entity fraction | 0.464 | 0.295 |
| candidate state fraction | 0.0031 | 0.0031 |
| selected state fraction | 0.0070 | 0.0047 |
| loss first→last | 9.782→7.870 | 9.788→7.869 |

Interpretation: exact-K mask density is correct in both arms. The relation arm selects relation/entity/state groups more often than the shuffled-weight control while preserving the same candidate text and selected-group count. This is sufficient to launch a 20M falsification experiment; it is not yet evidence of BabyLM score improvement.
