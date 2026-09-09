# earlier analysis argument-slot type sharing versus identity randomization

Device `cpu`, elapsed 811.65 s, 4 seeds per cell.

## Held-family argument-slot vocabulary coverage

| mode | held slot types | shared with train | type coverage | occurrence coverage | mean train occ/shared type |
|---|---:|---:|---:|---:|---:|
| natural_disjoint | 400 | 87 | 0.217 | 0.271 | 47.6 |
| natural_shared_pool | 57 | 57 | 1.000 | 1.000 | 150.2 |
| natural_shared_large | 155 | 93 | 0.600 | 0.578 | 34.8 |
| family_alias | 60 | 60 | 1.000 | 1.000 | 150.0 |
| per_item_alias | 64 | 64 | 1.000 | 1.000 | 150.0 |

## Transfer to held-family probe predicates

| mode | arm | fit | train | probe-held | anchor-held | probe-trainfam | held-template |
|---|---|---:|---:|---:|---:|---:|---:|
| natural_disjoint | zero | 3/4 | 0.958 | 0.539 | 0.602 | 0.516 | 0.546 |
| natural_disjoint | true | 4/4 | 1.000 | 0.595 | 0.641 | 0.600 | 0.583 |
| natural_disjoint | shuffled | 4/4 | 0.998 | 0.477 | 0.648 | 0.462 | 0.542 |
| natural_shared_pool | zero | 4/4 | 1.000 | 0.741 | 0.917 | 0.736 | 0.693 |
| natural_shared_pool | true | 4/4 | 1.000 | 0.829 | 0.889 | 0.821 | 0.736 |
| natural_shared_pool | shuffled | 4/4 | 0.999 | 0.316 | 0.914 | 0.316 | 0.604 |
| natural_shared_large | zero | 4/4 | 1.000 | 0.598 | 0.704 | 0.631 | 0.607 |
| natural_shared_large | true | 4/4 | 1.000 | 0.639 | 0.715 | 0.652 | 0.605 |
| natural_shared_large | shuffled | 4/4 | 0.999 | 0.424 | 0.832 | 0.421 | 0.639 |
| family_alias | zero | 4/4 | 1.000 | 0.653 | 0.883 | 0.672 | 0.668 |
| family_alias | true | 4/4 | 1.000 | 0.854 | 0.924 | 0.854 | 0.707 |
| family_alias | shuffled | 4/4 | 1.000 | 0.323 | 0.917 | 0.311 | 0.757 |
| per_item_alias | zero | 4/4 | 0.997 | 0.774 | 0.963 | 0.776 | 0.833 |
| per_item_alias | true | 4/4 | 0.998 | 0.911 | 0.966 | 0.911 | 0.782 |
| per_item_alias | shuffled | 4/4 | 0.999 | 0.291 | 0.968 | 0.287 | 0.800 |

## Aligned minus anti-aligned, against slot-sharing

| mode | zero | true | shuffled | true-shuffled | held slot type coverage |
|---|---:|---:|---:|---:|---:|
| natural_disjoint | 0.539 | 0.595 | 0.477 | +0.118 | 0.217 |
| natural_shared_pool | 0.741 | 0.829 | 0.316 | +0.513 | 1.000 |
| natural_shared_large | 0.598 | 0.639 | 0.424 | +0.215 | 0.600 |
| family_alias | 0.653 | 0.854 | 0.323 | +0.530 | 1.000 |
| per_item_alias | 0.774 | 0.911 | 0.291 | +0.620 | 1.000 |

Summary JSON: `experiments/archive/representation_and_objectives/data/slot_type_sharing/slot_type_sharing_summary.json`
