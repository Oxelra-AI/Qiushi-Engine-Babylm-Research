# Compact Rewrite Quality Pilot (n=26567)

## Word Savings
| Metric | Value |
|---|---|
| Total current words | 592318 |
| Total compact words | 328548 |
| Total saved | 263770 |
| Effective compression | 0.5547 |
| Target hit rate | 0.7277 |
| Mean savings ratio | 0.5607 |

## Entity/Number Preservation
| Metric | Compact | Current |
|---|---|---|
| Entity recall (n=15727) | 0.8518 | 0.9868 |
| Number recall (n=4544) | 0.963 | N/A |

## Structural Preservation
| Marker | N pairs | Mean preservation |
|---|---|---|
| Negation | 5005 | 0.4584 |
| Modality | 5248 | 0.3245 |
| Causal/temporal | 12182 | 0.4149 |

## Expression Diversity (Jaccard overlap)
| Comparison | Mean overlap |
|---|---|
| Compact vs Original | 0.5096 |
| Current vs Original | 0.5342 |
| Compact vs Current | 0.3650 |

## By Source
| Source | N | Compression | Target hit | Entity recall | Negation pres |
|---|---|---|---|---|---|
| bnc_spoken | 1922 | 0.537 | 0.802 | 0.907 | 0.454 |
| childes | 1472 | 0.557 | 0.749 | 0.239 | 0.530 |
| gutenberg | 9246 | 0.534 | 0.810 | 0.932 | 0.417 |
| open_subtitles | 5203 | 0.548 | 0.752 | 0.919 | 0.514 |
| simple_wiki | 8706 | 0.602 | 0.605 | 0.898 | 0.475 |
| switchboard | 18 | 0.518 | 0.944 | 0.700 | 0.400 |
