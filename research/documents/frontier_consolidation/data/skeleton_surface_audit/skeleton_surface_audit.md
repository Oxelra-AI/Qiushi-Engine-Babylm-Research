# source wide skeleton recurrence integrated skeleton surface audit

Surface/tokenizer audit of exact-length skeleton variants; no model scoring.

| variant | n | function-word fraction | punctuation/word | mean source gap | max-gap mean | LCS with compact | tokens/word | Δ function vs compact | Δ gap vs compact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 12155 | 29.35% | 0.168484 | 2.451194 | 6.365772 | 99.99% | 1.543941 | 0.00% | 0.000000 |
| prefix_repeat | 12155 | 45.57% | 0.081662 | 1.002065 | 1.022213 | 44.74% | 1.333681 | 16.22% | -1.449129 |
| spread_even | 12155 | 42.66% | 0.159390 | 1.710561 | 2.161744 | 46.97% | 1.424425 | 13.31% | -0.740633 |
| content_spread | 12155 | 21.77% | 0.175301 | 1.669764 | 3.933278 | 51.62% | 1.545202 | -7.58% | -0.781430 |
| scored_source_skeleton | 12155 | 13.73% | 0.190676 | 1.582533 | 3.304648 | 54.88% | 1.587973 | -15.62% | -0.868661 |
| oracle_compact_projection | 12155 | 37.39% | 0.141902 | 1.554527 | 4.291814 | 71.38% | 1.473935 | 8.04% | -0.896668 |

## Scientific reading
The source-only variants intentionally improve source-tail content coverage, but the scored/content skeletons are more telegraphic than generated compact views: fewer function words and larger source-position gaps indicate a stronger distribution shift. This does not rule them out as a discriminating short screen, but it argues against treating coverage alone as sufficient. A training test should compare at least compact, prefix-repeat, and one source-only skeleton, and should stop early if source-only skeleton harms broad cheap7 despite high coverage.

Examples: `experiments/archive/frontier_consolidation/data/skeleton_surface_audit/surface_gap_examples.jsonl`
JSON: `experiments/archive/frontier_consolidation/data/skeleton_surface_audit/skeleton_surface_audit.json`
