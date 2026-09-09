# evidence decomposition and missing controls format endpoint context-gain coordinate

context_gain = isolation_loss - row_context_loss. Negative endpoint-minus-anchor delta_context_gain means the endpoint relies less on adjacent row context than the anchor on the same Strict-complement spans.

Scored 80 Strict-complement sentence spans.

## Endpoint summaries

| endpoint | n | row loss | isolated loss | context gain | se | removed |
|---|---:|---:|---:|---:|---:|---:|
| chck82_slow_scale1p75 | 77 | 3.1741 | 4.1045 | 0.9304 | 0.1856 | 3 |
| coherent86_repaired_alpha075 | 77 | 3.1586 | 4.0979 | 0.9392 | 0.1824 | 3 |
| dense_focus_seed62064_u0080 | 77 | 3.2335 | 4.1976 | 0.9640 | 0.1816 | 3 |
| dense_focus_seed62065_u0080 | 77 | 3.2372 | 4.2001 | 0.9629 | 0.1819 | 3 |
| sparse_focus_seed62064_u0080 | 77 | 3.1712 | 4.1087 | 0.9374 | 0.1802 | 3 |

## Paired contrasts

| endpoint | anchor | n | d_row | d_iso | d_context_gain | se | frac higher |
|---|---|---:|---:|---:|---:|---:|---:|
| chck82_slow_scale1p75 | coherent86_repaired_alpha075 | 77 | +0.0154 | +0.0066 | -0.0088 | 0.0109 | 0.442 |
| dense_focus_seed62064_u0080 | coherent86_repaired_alpha075 | 77 | +0.0749 | +0.0997 | +0.0248 | 0.0279 | 0.506 |
| dense_focus_seed62065_u0080 | coherent86_repaired_alpha075 | 77 | +0.0786 | +0.1022 | +0.0236 | 0.0282 | 0.506 |
| sparse_focus_seed62064_u0080 | coherent86_repaired_alpha075 | 77 | +0.0126 | +0.0108 | -0.0018 | 0.0057 | 0.481 |

JSON: `experiments/archive/functional_learning/data/context_gain_dense_endpoints/sample80/format_context_gain_summary.json`
