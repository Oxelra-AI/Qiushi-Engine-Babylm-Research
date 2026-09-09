# corrected two component frame Wikipedia holdout near-duplicate check
Created: 2026-09-06T13:31:26Z

Checked 1200 WikiLarge probe target sentences against:
- simple_wiki.train.txt: 0 sentences
- CLEAN 10M pool: 752614 sentences

## Threshold analysis

| Threshold | N above (simple_wiki) | Frac | N above (CLEAN pool) | Frac |
|-----------|----------------------|------|---------------------|------|
| 0.5 | 0 | 0.0 | 12 | 0.01 |
| 0.6 | 0 | 0.0 | 5 | 0.0042 |
| 0.7 | 0 | 0.0 | 0 | 0.0 |
| 0.8 | 0 | 0.0 | 0 | 0.0 |
| 0.9 | 0 | 0.0 | 0 | 0.0 |
| 1.0 | 0 | 0.0 | 0 | 0.0 |

## Interpretation

At Jaccard ≥ 0.8, 0 of 1200 probe targets have a near-duplicate in the CLEAN pool.
Near-duplicate contamination appears limited at this threshold.
