# earlier analysis MAX-register readout

File-only readout; no model loading, training, official evaluation, GPU, GlobalPIQA, SuperGLUE, AoA, upload, or leaderboard action.

Data state: `partial`; register stable cells 0/96.

## Trajectory contrasts over complete common-window checkpoints

| contrast | complete ck count | cheap6 | exEntity5 | Entity | BLiMP | Supplement | EWoK | COMPS | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adult_minus_clean_mean | 0 |  |  |  |  |  |  |  |  |
| adult_minus_view | 0 |  |  |  |  |  |  |  |  |
| child_minus_adult | 0 |  |  |  |  |  |  |  |  |
| child_minus_clean_mean | 0 |  |  |  |  |  |  |  |  |
| child_minus_view | 0 |  |  |  |  |  |  |  |  |
| view_minus_clean_mean | 6 | 0.7867 | 0.5680 | 1.8800 | 0.7167 | 1.1567 | 0.5650 | 0.0550 | 0.3467 |

## Interpretation

Primary contrast is `child_minus_adult`: positive means removing developmental/speech hurts less or adult-prose removal hurts more; negative means developmental/speech removal has higher opportunity cost. Arm-minus-view locates each pure-register removal arm around the trained proportional MAX view midpoint. Arm/view-minus-clean uses whatever clean anchors are available and must be read with the reported clean spread.

JSON: `experiments/archive/frontier_consolidation/data/register_max_readout/register_max_readout_summary.json`
Contrasts: `experiments/archive/frontier_consolidation/data/register_max_readout/contrast_rows.csv`
