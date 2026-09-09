# earlier analysis AoA across-pass schedule predictor

Predicted official-AoA movement for byte-identical 100M stream reorderings that redistribute CHILDES-enriched rows across ten 10M-word phases; rows are not added or removed.

Current actual reproduction: clipped 0.0000, r -0.0361, p 0.5913, n 223.
Current whole-stream log frequency vs child AoA r -0.0714; vs measured model AoA r -0.6214.
CHILDES log frequency vs child AoA r -0.2047; CHILDES-minus-whole vs child AoA r -0.3225.
Official-only 10M log frequency vs child AoA r -0.0721, p 0.1472, n 406.

## Best legal predictions

| mode | beta | clipped score | r | p | n | mean |dlog| 20M | mean |dlog| 50M |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_childes_taper_rowsort_ratio_f40_d40 | -0.75 | 0.0000 | 0.0996 | 0.1436 | 217 | 0.7709 | 0.4073 |
| source_childes_taper_rowsort_ratio_f40_d25 | -0.75 | 0.0000 | 0.0977 | 0.1479 | 221 | 0.7133 | 0.3844 |
| global_sort__childes_ratio | -0.75 | 0.0000 | 0.0897 | 0.1780 | 227 | 0.8393 | 0.4545 |
| source_childes_taper_rowsort_ratio_f55_d40 | -0.75 | 0.0000 | 0.0808 | 0.2359 | 217 | 0.7364 | 0.3860 |
| source_childes_taper_rowsort_ratio_f55_d25 | -0.75 | 0.0000 | 0.0612 | 0.3687 | 218 | 0.6837 | 0.3677 |
| source_childes_taper_rowsort_ratio_f55_d40 | -0.5 | 0.0000 | 0.0564 | 0.3968 | 228 | 0.7364 | 0.3860 |
| source_childes_taper_rowsort_ratio_f40_d25 | -0.5 | 0.0000 | 0.0472 | 0.4814 | 225 | 0.7133 | 0.3844 |
| source_childes_taper_rowsort_ratio_f40_d40 | -0.5 | 0.0000 | 0.0416 | 0.5353 | 224 | 0.7709 | 0.4073 |
| global_sort__childes_ratio | -0.5 | 0.0000 | 0.0346 | 0.6044 | 226 | 0.8393 | 0.4545 |
| source_childes_taper_f70_d40 | -0.75 | 0.0000 | 0.0315 | 0.6405 | 222 | 0.1628 | 0.0736 |
| source_childes_taper_f70_d60 | -0.75 | 0.0000 | 0.0309 | 0.6473 | 222 | 0.2173 | 0.0860 |
| source_childes_taper_f55_d40 | -0.75 | 0.0000 | 0.0292 | 0.6655 | 221 | 0.2324 | 0.1074 |
| global_sort__cdi_density | -0.75 | 0.0000 | 0.0280 | 0.6991 | 193 | 0.7219 | 0.4109 |
| source_childes_taper_f55_d25 | -0.75 | 0.0000 | 0.0238 | 0.7254 | 221 | 0.1558 | 0.0797 |
| source_childes_taper_f25_d15 | -0.75 | 0.0000 | 0.0222 | 0.7418 | 222 | 0.1519 | 0.0852 |
| source_childes_taper_f25_d40 | -0.75 | 0.0000 | 0.0217 | 0.7481 | 222 | 0.3557 | 0.1702 |
| source_childes_taper_rowsort_comp_f55_d40 | -0.5 | 0.0000 | 0.0209 | 0.7533 | 229 | 0.9514 | 0.4945 |
| source_childes_taper_f40_d60 | -0.75 | 0.0000 | 0.0184 | 0.7842 | 223 | 0.3868 | 0.1627 |
| source_childes_taper_rowsort_ratio_f55_d25 | -0.5 | 0.0000 | 0.0175 | 0.7945 | 225 | 0.6837 | 0.3677 |
| source_childes_taper_f55_d60 | -0.75 | 0.0000 | 0.0162 | 0.8120 | 219 | 0.3057 | 0.1258 |

## Oracle-only predictions

| mode | beta | clipped score | r | p | n |
|---|---:|---:|---:|---:|---:|
| global_sort__child_early | -0.75 | 0.3300 | 0.3300 | 0.0000 | 228 |
| global_sort__child_early | -0.5 | 0.1901 | 0.1901 | 0.0037 | 231 |
| global_sort__child_early | -0.36 | 0.1519 | 0.1519 | 0.0214 | 229 |
| global_sort__child_early | -0.2 | 0.0000 | 0.0544 | 0.4157 | 226 |
| global_sort__child_early | -0.1 | 0.0000 | -0.0089 | 0.8941 | 227 |

A schedule with a positive stored score would still require a trunk timing screen and endpoint-column checks.  If the best legal rows remain nonsignificant, the measured source-enrichment ceiling is not by itself enough to authorize H100 trunk spending.

Full JSON: `experiments/archive/relation_learning/data/aoa_across_pass_schedule_predictor/summary.json`
