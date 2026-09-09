# Compact Rewrite Quality Pilot (n=512)

## Word Savings
| Metric | Value |
|---|---|
| Total current words | 19761 |
| Total compact words | 9571 |
| Total saved | 10190 |
| Effective compression | 0.4843 |
| Target hit rate | 0.7520 |
| Mean savings ratio | 0.4845 |

## Entity/Number Preservation
| Metric | Compact | Current |
|---|---|---|
| Entity recall (n=385) | 0.7503 | 0.9744 |
| Number recall (n=95) | 0.9347 | N/A |

## Structural Preservation
| Marker | N pairs | Mean preservation |
|---|---|---|
| Negation | 150 | 0.3827 |
| Modality | 141 | 0.3511 |
| Causal/temporal | 298 | 0.3628 |

## Expression Diversity (Jaccard overlap)
| Comparison | Mean overlap |
|---|---|
| Compact vs Original | 0.4760 |
| Current vs Original | 0.5937 |
| Compact vs Current | 0.3646 |

## By Source
| Source | N | Compression | Target hit | Entity recall | Negation pres |
|---|---|---|---|---|---|
| bnc_spoken | 91 | 0.462 | 0.857 | 0.844 | 0.319 |
| childes | 85 | 0.487 | 0.765 | 0.386 | 0.559 |
| gutenberg | 123 | 0.465 | 0.805 | 0.874 | 0.323 |
| open_subtitles | 86 | 0.466 | 0.674 | 0.801 | 0.452 |
| simple_wiki | 109 | 0.533 | 0.624 | 0.856 | 0.402 |
| switchboard | 18 | 0.516 | 0.944 | 0.900 | 0.400 |
