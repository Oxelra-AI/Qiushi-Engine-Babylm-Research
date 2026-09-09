# earlier analysis context-vs-isolation coordinate

context_gain = isolation_loss - row_context_loss. A positive A-minus-B delta_context_gain means A benefits more than B from adjacent row context on the same Strict-complement sentence spans; compare row_context and isolation loss deltas to separate in-context fit from context-free fit.

Scored 24 sentence spans, arms: OFF43022, ALN43022, SHUF43022, SEP43022, DUP43022, checkpoints: chck_100M.

## Mean over scored checkpoints
- compact_experience seed43022 ALN-minus-DUP: row_context -0.2252, isolation -0.1347, context_gain +0.0905
- compact_experience seed43022 ALN-minus-OFF: row_context -0.6082, isolation -0.0384, context_gain +0.5698
- compact_experience seed43022 ALN-minus-SEP: row_context -0.7709, isolation +0.0075, context_gain +0.7784
- compact_experience seed43022 ALN-minus-SHUF: row_context -0.6890, isolation -0.0766, context_gain +0.6124
- compact_experience seed43022 DUP-minus-OFF: row_context -0.3830, isolation +0.0963, context_gain +0.4793
- compact_experience seed43022 SEP-minus-OFF: row_context +0.1627, isolation -0.0458, context_gain -0.2086
- compact_experience seed43022 SHUF-minus-OFF: row_context +0.0808, isolation +0.0383, context_gain -0.0426
- compact_experience seed43022 SHUF-minus-SEP: row_context -0.0819, isolation +0.0841, context_gain +0.1660

## Per-checkpoint selected contrasts
- compact_experience seed43022 chck_100M ALN-minus-OFF: d_row -0.6082, d_iso -0.0384, d_gain +0.5698, n=24
- compact_experience seed43022 chck_100M SHUF-minus-OFF: d_row +0.0808, d_iso +0.0383, d_gain -0.0426, n=24
- compact_experience seed43022 chck_100M ALN-minus-SHUF: d_row -0.6890, d_iso -0.0766, d_gain +0.6124, n=24
- compact_experience seed43022 chck_100M DUP-minus-OFF: d_row -0.3830, d_iso +0.0963, d_gain +0.4793, n=24
- compact_experience seed43022 chck_100M ALN-minus-DUP: d_row -0.2252, d_iso -0.1347, d_gain +0.0905, n=24
- compact_experience seed43022 chck_100M SEP-minus-OFF: d_row +0.1627, d_iso -0.0458, d_gain -0.2086, n=24
- compact_experience seed43022 chck_100M ALN-minus-SEP: d_row -0.7709, d_iso +0.0075, d_gain +0.7784, n=24
- compact_experience seed43022 chck_100M SHUF-minus-SEP: d_row -0.0819, d_iso +0.0841, d_gain +0.1660, n=24
