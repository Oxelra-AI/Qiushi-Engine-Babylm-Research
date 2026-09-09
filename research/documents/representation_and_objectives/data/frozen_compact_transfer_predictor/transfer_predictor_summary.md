# frozen transfer test and route decision frozen compact-transfer predictor

No training or model evaluation was run. This joins existing item predictions to ask whether DeBERTa compact-responsive items/subtasks predict RoBERTa and causal-GPT movement.

## Key item-level net movements

| item pool | n | DeBERTa view-repeat | RoBERTa compact-repeat | RoBERTa factorial interaction |
|---|---:|---:|---:|---:|
| stable_nonBLiMP (Supplement+EWoK+COMPS) | 92378 | 0.3053 | 0.0108 | 1.6108 |
| stable_with_BLiMP | 105778 | 0.2590 | 0.0596 | 1.7168 |

## Column table

| column | n | D view-repeat | D view-adj | D adj-repeat | R compact-repeat | R factorial |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 13400 | -0.0597 | -1.6194 | 1.5597 | 0.3955 | 2.4478 |
| COMPS | 91028 | 0.2713 | 0.1791 | 0.0923 | -0.0143 | 1.5633 |
| EWoK | 1100 | 2.1818 | 4.6364 | -2.4545 | 2.9091 | 5.7273 |
| Supplement | 250 | 4.4000 | 3.6000 | 0.8000 | -3.6000 | 0.8000 |

## Subtask-vector correlations

| comparison | n subtasks | pearson | weighted pearson |
|---|---:|---:|---:|
| deberta_view_repeat_vs_roberta_compact_repeat | 87 | 0.06033600955078231 | 0.3083853044520361 |
| deberta_view_adjbreak_vs_roberta_factorial_interaction | 87 | -0.2949934836189176 | -0.05988531114946998 |
| deberta_adjbreak_repeat_vs_roberta_compact_repeat | 87 | -0.18612035646027142 | -0.07873753234678897 |
| deberta_view_repeat_vs_roberta_factorial_interaction | 87 | -0.19320161847674713 | 0.08978679224558174 |

## Frozen-evidence reading

A compact-transfer predictor would require DeBERTa gains or DeBERTa source-own-adjacency gains to select stable-family items/subtasks that move coherently in RoBERTa or in the causal-GPT reciprocity surface. The full JSON contains conditional item means by DeBERTa selector value and the family-level GPT comparison.

JSON: `experiments/archive/representation_and_objectives/data/frozen_compact_transfer_predictor/transfer_predictor_summary.json`
Subtask table: `experiments/archive/representation_and_objectives/data/frozen_compact_transfer_predictor/subtask_alignment_table.csv`
