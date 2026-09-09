# entity consistency 20m training validation — Entity Mention Consistency 20M training validation

Evidence JSON: `experiments/archive/initial_model_studies/data/entity_consistency_20m_training_validation.json`

Both arms are valid/loadable 20M artifacts; this is training validation, not task-score evidence.

| quantity | consistency | shuffled_pair | delta c-s |
|---|---:|---:|---:|
| word exposure | 20000000 | 20000000 | 0 |
| steps | 489 | 489 | 0 |
| aux pairs | 250139 | 250139 | 0 |
| aux candidates | 1001472 | 1001472 | 0 |
| mean pair gap | 38.140 | 38.140 | 0.000 |
| mean aux loss/active step | 1.697 | 7.571 | -5.874 |
| final MLM loss | 3.6328 | 3.6340 | -0.0012 |
| final total loss | 3.6938 | 4.0074 | -0.3136 |

Projection heads are outside `hf_model`; checkpoint hashes differ between 10M and 20M within both arms.
