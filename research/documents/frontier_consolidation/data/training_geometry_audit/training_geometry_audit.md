# earlier analysis training-geometry audit

Static CPU/file-only audit; no task status was queried and no training/evaluation was started.

## Main correction
The repaired MAX 2.64x streams have 653,130 rows at 100M words. With batch256 this implies 2,552 optimizer updates, while the old 1x 100M streams have 647,400 rows and 2,529 updates.

Therefore the MAX training verification should expect 2,552 actual updates if all rows are consumed, while checking that `lr_total_steps` was the intended 2,529 recipe value.

## Interpretation
MAX view and MAX repeat have identical rows, words, batch count, and LR horizon; view-minus-repeat remains the cleanest estimate of semantic re-expression at MAX dose.
MAX repeat-minus-clean0 and MAX view-minus-clean0 compare different row/update geometries as well as different text. Interpret this as the value of admitting the matched MAX source-packet stream under the fixed training recipe, not a pure source-content scalar.

JSON: `experiments/archive/frontier_consolidation/data/training_geometry_audit/training_geometry_audit.json`
