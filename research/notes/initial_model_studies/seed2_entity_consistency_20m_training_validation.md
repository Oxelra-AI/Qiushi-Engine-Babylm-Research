# entity consistency 20m training validation — Entity Mention Consistency 20M training validation

Evidence JSON: `experiments/archive/initial_model_studies/data/seed2_entity_consistency_20m_training_validation.json`

Both arms are valid/loadable 20M artifacts; this is training validation, not task-score evidence.

| quantity | consistency | shuffled_pair | delta c-s |
|---|---:|---:|---:|
| word exposure | 20000000 | 20000000 | 0 |
| steps | 489 | 489 | 0 |
| aux pairs | 250137 | 250137 | 0 |
| aux candidates | 1001472 | 1001472 | 0 |
| mean pair gap | 38.072 | 38.072 | 0.000 |
| mean aux loss/active step | 1.683 | 7.572 | -5.888 |
| final MLM loss | 3.7536 | 3.7448 | +0.0089 |
| final total loss | 3.8213 | 4.1165 | -0.2951 |

Projection heads are outside `hf_model`; checkpoint hashes differ between 10M and 20M within both arms.
