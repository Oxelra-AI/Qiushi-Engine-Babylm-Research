# earlier analysis admission/saturation readout

CPU/file-only readout of existing pools and score tables. No model loading, training, evaluation, GPU work, GlobalPIQA/SuperGLUE/AoA, upload, leaderboard action, or final-facing writing occurred.

## Score-side reason for the test

Before matched-clean scoring lands, existing old-clean tables already indicate that companion form is not the broad ex-Entity carrier: MAX common10_80 ex-Entity V-C_old is +0.5981, R-C_old is +0.5345, and V-R is only +0.0636. The single visible breadth point at 80M gives B-C_old +0.896 and V-B -0.062. Thus V/B/R are clustered relative to clean while differing from one another on the same scale as the seed/basin spread measured in axis noise floor and clean training state.

## Admitted FineWeb coverage across dose

| dose | rho | pair words | words mult | unique docs | docs mult | content types | type mult | old V-C exEnt | old R-C exEnt | old V-R exEnt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dose1 | 0.042352 | 423,511 | 1.000 | 4,529 | 1.000 | 26,909 | 1.000 | +0.4889 | +0.6756 | -0.1868 |
| dose1p82 | 0.077120 | 771,199 | 1.821 | 5,089 | 1.124 | 36,016 | 1.338 | +0.3100 | +0.3536 | -0.0436 |
| dose2p64 | 0.111872 | 1,118,587 | 2.641 | 5,261 | 1.162 | 43,540 | 1.618 | +0.5981 | +0.5345 | +0.0636 |

## Incremental coverage

| dose label | added pair words vs previous | new content types vs previous | new types per 100k added words | unique doc gain |
|---|---:|---:|---:|---:|
| dose1_view_repeat_pairset | 423,511 | 26,909 | 6353.8 | 4,529 |
| dose1p82_view_repeat_pairset | 347,688 | 9,107 | 2619.3 | 560 |
| dose2p64_view_repeat_pairset | 347,388 | 7,524 | 2165.9 | 172 |

## MAX form overlap

- max_source_vs_rewrite_content_jaccard: 0.7965778594395958
- max_pair_all_vs_breadth_content_jaccard: 0.39840008707009145
- max_source_vs_breadth_content_jaccard: 0.41017855437792416
- max_rewrite_vs_breadth_content_jaccard: 0.39936571327890996
- breadth_content_types_not_in_max_pair: 11588
- max_pair_content_types_not_in_breadth: 21577

## Scientific reading

The existing score side and file side point to a sharper mechanism test: the load-bearing quantity is likely the admitted FineWeb slice replacing clean-Qwen words, not whether the admitted slice is shown as own-source compact re-expression, breadth, or exact source duplication. The dose file statistics now make the coming matched-clean readout interpretable as a cluster test: compute V-C, B-C, and R-C against the geometry-matched clean in one common window, then compare their within-cluster spread to the axis noise floor and clean training state seed/noise context. If all three are similarly positive and the gain changes weakly from 4.2% to 11.2% while admitted types/docs grow sublinearly, the principle becomes a finite-budget content-admission/coverage-threshold effect. If only one arm survives matched-clean, the carrier returns to companion form and must be decomposed further.

## Files

- `experiments/archive/frontier_consolidation/data/admission_saturation_readout/admission_saturation_summary.json`
- `experiments/archive/frontier_consolidation/data/admission_saturation_readout/admitted_pair_increment_metrics.csv`
- `experiments/archive/frontier_consolidation/data/admission_saturation_readout/score_coverage_saturation_rows.csv`
- `experiments/archive/frontier_consolidation/data/admission_saturation_readout/domain_composition_rows.csv`
