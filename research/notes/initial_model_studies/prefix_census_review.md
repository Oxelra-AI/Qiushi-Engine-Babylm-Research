# prefix census critical review critical review of prefix target census v2 prefix census

JSON: `experiments/archive/initial_model_studies/data/prefix_census_review.json`
Rows reviewed: `experiments/archive/initial_model_studies/data/prefix_target_census_v2_fixedpos_rows.csv`

## Control-overlap summary (threshold 0.05, sign-stable across seed42/seed43)

| exposure | same+ | same+ frac | same+ but deleted not | same+ but cross not | same+ but block not | robust all controls + | corr same/deleted | corr same/cross | corr same/block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40M | 155 | 0.258 | 80 | 72 | 126 | 13 (0.022) | 0.717 | 0.536 | 0.253 |
| 80M | 164 | 0.273 | 84 | 79 | 136 | 18 (0.030) | 0.765 | 0.642 | 0.207 |
| 100M | 172 | 0.287 | 82 | 83 | 146 | 13 (0.022) | 0.754 | 0.636 | 0.202 |

## Review interpretation

The fixed-position repair removes the largest v1 artifact, but the v2 positive population must not yet be treated as semantic/entity-state evidence. The same-source effect is strongly correlated with deleted, cross-source, and block-shuffled variants; a large robust-all-controls subset means many targets are helped by the true local prefix relative to every corrupted prefix, but this can still reflect local lexical/topic/history coherence rather than entity-state binding. The positives not shared by deleted/cross/block are the more relevant targets for high-precision route design and require row-level inspection.

Representative decoded cases are stored in the JSON for qualitative inspection.
