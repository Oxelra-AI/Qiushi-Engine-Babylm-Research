# consolidated margin and preservation evidence — Relational Content Preservation in Compact Views

Analyzed 18682 accepted compact rewrites.

General word retention rate: 0.6104

Rows with any dynamic verb in source: 1557 (8.3%)
Rows with >3 dynamic verbs in source: 2 (0.0%)


## Per-category marker preservation

Category              SourceTotal RewriteTotal  Retention     Excess   MeanPres MedianPres   RowsWSrc
dynamic_verbs                1749         1594     0.9114    +0.3010     0.8158     1.0000       1557
spatial_preps                3058         1677     0.5484    -0.0620     0.4581     0.0000       2753
causal_markers               1941         1735     0.8939    +0.2835     0.5070     0.5000       1797
temporal_markers             4089         2829     0.6919    +0.0815     0.5426     0.5000       3537
static_property             24749         9831     0.3972    -0.2132     0.3928     0.0000      16925


## Differential Preservation Interpretation

Dynamic verb retention: 0.9114 (excess vs general: +0.3010)
Static property retention: 0.3972 (excess vs general: -0.2132)
Spatial prep retention: 0.5484 (excess: -0.0620)
Causal marker retention: 0.8939 (excess: +0.2835)
Temporal marker retention: 0.6919 (excess: +0.0815)

Dynamic-vs-static retention gap: +0.5142
→ Compact views preferentially lose static property markers.


Machine-readable: `experiments/archive/frontier_consolidation/data/relational_preservation/relational_preservation.json`
