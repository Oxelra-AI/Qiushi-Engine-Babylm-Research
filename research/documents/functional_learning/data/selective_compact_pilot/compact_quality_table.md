# Compact Rewrite Quality Pilot (n=512)

## Word Savings
| Metric | Value |
|---|---|
| Total current words | 19761 |
| Total compact words | 17445 |
| Total saved | 2316 |
| Effective compression | 0.8828 |
| Target hit rate | 0.0332 |
| Mean savings ratio | 0.8774 |

## Entity/Number Preservation
| Metric | Compact | Current |
|---|---|---|
| Entity recall (n=385) | 0.9599 | 0.9744 |
| Number recall (n=95) | 0.9939 | N/A |

## Structural Preservation
| Marker | N pairs | Mean preservation |
|---|---|---|
| Negation | 150 | 0.7567 |
| Modality | 141 | 0.7494 |
| Causal/temporal | 298 | 0.6808 |

## Expression Diversity (Jaccard overlap)
| Comparison | Mean overlap |
|---|---|
| Compact vs Original | 0.6836 |
| Current vs Original | 0.5937 |
| Compact vs Current | 0.7775 |

## By Source
| Source | N | Compression | Target hit | Entity recall | Negation pres |
|---|---|---|---|---|---|
| bnc_spoken | 91 | 0.872 | 0.044 | 0.970 | 0.707 |
| childes | 85 | 0.888 | 0.024 | 0.900 | 0.794 |
| gutenberg | 123 | 0.917 | 0.033 | 0.989 | 0.839 |
| open_subtitles | 86 | 0.833 | 0.046 | 0.964 | 0.583 |
| simple_wiki | 109 | 0.859 | 0.018 | 0.975 | 0.843 |
| switchboard | 18 | 0.908 | 0.056 | 1.000 | 0.600 |
