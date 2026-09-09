# extractive surface and readout repair extractive surface/discontinuity atlas

Created UTC: `2026-09-02T06:33:11Z`

CPU-only corpus geometry; no training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.

`prefix_repeat_local` is a local structural baseline only; the historical selected repeat arm is hash-rotated cyclic repetition after correction.

## Pooled copied-adjacency summary

| variant | absent content/content | pooled gap=1 frac | pooled skip frac | mean skip frac | mean source span frac |
|---|---:|---:|---:|---:|---:|
| compact | 0.1729 | 0.7775 | 0.2225 | 0.2506 | 0.8072 |
| prefix_repeat_local | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.6246 |
| extractive_balanced | 0.0000 | 0.5278 | 0.4722 | 0.4850 | 0.9715 |
| extractive_wide | 0.0000 | 0.6465 | 0.3535 | 0.3641 | 0.9281 |

## Compact-relative deltas

- `extractive_balanced_minus_compact`: {"mean_gap1_frac": -0.23439884670810474, "mean_skip_frac": 0.2343988467081049, "mean_source_span_frac": 0.1642676829033869, "pooled_absent_content_frac": -0.1729200494858115, "pooled_skip_frac": 0.24978750694690027}
- `extractive_wide_minus_compact`: {"mean_gap1_frac": -0.11353553970090713, "mean_skip_frac": 0.11353553970090718, "mean_source_span_frac": 0.12086709697202369, "pooled_absent_content_frac": -0.1729200494858115, "pooled_skip_frac": 0.13106689158342455}
- `prefix_repeat_local_minus_compact`: {"mean_gap1_frac": 0.25057285872778456, "mean_skip_frac": -0.25057285872778445, "mean_source_span_frac": -0.18267676586904125, "pooled_absent_content_frac": -0.1729200494858115, "pooled_skip_frac": -0.22246135785560556}

## Interpretation

If extractive arms score far below compact, this atlas helps attribute the deficit to the bundled absence of generated fluent re-expression/source-absent content/context recomposition, not simply to missing source-tail coverage. If a source-only arm scores near compact despite high skip-bigram fragmentation, source-word selection/coverage is more sufficient than expected.
