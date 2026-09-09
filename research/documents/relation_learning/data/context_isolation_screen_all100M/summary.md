# earlier analysis context-vs-isolation coordinate

context_gain = isolation_loss - row_context_loss. A positive A-minus-B delta_context_gain means A benefits more than B from adjacent row context on the same Strict-complement sentence spans; compare row_context and isolation loss deltas to separate in-context fit from context-free fit.

Scored 651 sentence spans, arms: OFF43022, ALN43022, SHUF43022, SEP43022, DUP43022, OFF43122, ALN43122, SHUF43122, C43022, R43022, V43022, RS43022, VS43022, C43122, R43122, V43122, RS43122, VS43122, base43022, dose21_43022, dose25_43022, base43122, dose21_43122, dose25_43122, checkpoints: chck_100M.

## Mean over scored checkpoints
- crv seed43022 REPEAT-minus-CLEAN: row_context +0.0379, isolation +0.1293, context_gain +0.0915
- crv seed43022 REPEAT-minus-REPEAT_SPLIT: row_context +0.0062, isolation +0.1288, context_gain +0.1227
- crv seed43022 REPEAT_SPLIT-minus-CLEAN: row_context +0.0317, isolation +0.0005, context_gain -0.0312
- crv seed43022 VIEW-minus-CLEAN: row_context +0.0525, isolation +0.0656, context_gain +0.0130
- crv seed43022 VIEW-minus-REPEAT: row_context +0.0147, isolation -0.0638, context_gain -0.0784
- crv seed43022 VIEW-minus-VIEW_SPLIT: row_context +0.0418, isolation +0.0363, context_gain -0.0055
- crv seed43022 VIEW_SPLIT-minus-CLEAN: row_context +0.0107, isolation +0.0293, context_gain +0.0186
- crv seed43122 REPEAT-minus-CLEAN: row_context -0.0066, isolation +0.0469, context_gain +0.0535
- crv seed43122 REPEAT-minus-REPEAT_SPLIT: row_context -0.0285, isolation +0.0520, context_gain +0.0805
- crv seed43122 REPEAT_SPLIT-minus-CLEAN: row_context +0.0219, isolation -0.0051, context_gain -0.0270
- crv seed43122 VIEW-minus-CLEAN: row_context -0.0134, isolation +0.0550, context_gain +0.0684
- crv seed43122 VIEW-minus-REPEAT: row_context -0.0068, isolation +0.0081, context_gain +0.0149
- crv seed43122 VIEW-minus-VIEW_SPLIT: row_context +0.0245, isolation +0.0179, context_gain -0.0065
- crv seed43122 VIEW_SPLIT-minus-CLEAN: row_context -0.0378, isolation +0.0371, context_gain +0.0750
- dose seed43022 dose21-minus-base0: row_context +0.0288, isolation +0.0902, context_gain +0.0614
- dose seed43022 dose25-minus-base0: row_context +0.0439, isolation +0.1193, context_gain +0.0754
- dose seed43022 dose25-minus-dose21: row_context +0.0151, isolation +0.0290, context_gain +0.0140
- dose seed43122 dose21-minus-base0: row_context +0.0031, isolation -0.0514, context_gain -0.0545
- dose seed43122 dose25-minus-base0: row_context +0.0318, isolation -0.0061, context_gain -0.0379
- dose seed43122 dose25-minus-dose21: row_context +0.0287, isolation +0.0453, context_gain +0.0166
- compact_experience seed43022 ALN-minus-DUP: row_context -0.1820, isolation -0.1230, context_gain +0.0589
- compact_experience seed43022 ALN-minus-OFF: row_context -0.2114, isolation +0.1352, context_gain +0.3465
- compact_experience seed43022 ALN-minus-SEP: row_context -0.4918, isolation +0.1986, context_gain +0.6904
- compact_experience seed43022 ALN-minus-SHUF: row_context -0.1747, isolation +0.1963, context_gain +0.3709
- compact_experience seed43022 DUP-minus-OFF: row_context -0.0294, isolation +0.2582, context_gain +0.2876
- compact_experience seed43022 SEP-minus-OFF: row_context +0.2804, isolation -0.0634, context_gain -0.3439
- compact_experience seed43022 SHUF-minus-OFF: row_context -0.0367, isolation -0.0611, context_gain -0.0244
- compact_experience seed43022 SHUF-minus-SEP: row_context -0.3171, isolation +0.0023, context_gain +0.3195
- compact_experience seed43122 ALN-minus-OFF: row_context -0.0737, isolation +0.1391, context_gain +0.2128
- compact_experience seed43122 ALN-minus-SHUF: row_context -0.1546, isolation +0.1562, context_gain +0.3109
- compact_experience seed43122 SHUF-minus-OFF: row_context +0.0809, isolation -0.0171, context_gain -0.0981

## Per-checkpoint selected contrasts
- crv seed43022 chck_100M VIEW-minus-CLEAN: d_row +0.0525, d_iso +0.0656, d_gain +0.0130, n=639
- crv seed43022 chck_100M REPEAT-minus-CLEAN: d_row +0.0379, d_iso +0.1293, d_gain +0.0915, n=639
- crv seed43022 chck_100M VIEW_SPLIT-minus-CLEAN: d_row +0.0107, d_iso +0.0293, d_gain +0.0186, n=639
- crv seed43022 chck_100M REPEAT_SPLIT-minus-CLEAN: d_row +0.0317, d_iso +0.0005, d_gain -0.0312, n=639
- crv seed43022 chck_100M VIEW-minus-VIEW_SPLIT: d_row +0.0418, d_iso +0.0363, d_gain -0.0055, n=639
- crv seed43022 chck_100M REPEAT-minus-REPEAT_SPLIT: d_row +0.0062, d_iso +0.1288, d_gain +0.1227, n=639
- crv seed43022 chck_100M VIEW-minus-REPEAT: d_row +0.0147, d_iso -0.0638, d_gain -0.0784, n=639
- crv seed43122 chck_100M VIEW-minus-CLEAN: d_row -0.0134, d_iso +0.0550, d_gain +0.0684, n=639
- crv seed43122 chck_100M REPEAT-minus-CLEAN: d_row -0.0066, d_iso +0.0469, d_gain +0.0535, n=639
- crv seed43122 chck_100M VIEW_SPLIT-minus-CLEAN: d_row -0.0378, d_iso +0.0371, d_gain +0.0750, n=639
- crv seed43122 chck_100M REPEAT_SPLIT-minus-CLEAN: d_row +0.0219, d_iso -0.0051, d_gain -0.0270, n=639
- crv seed43122 chck_100M VIEW-minus-VIEW_SPLIT: d_row +0.0245, d_iso +0.0179, d_gain -0.0065, n=639
- crv seed43122 chck_100M REPEAT-minus-REPEAT_SPLIT: d_row -0.0285, d_iso +0.0520, d_gain +0.0805, n=639
- crv seed43122 chck_100M VIEW-minus-REPEAT: d_row -0.0068, d_iso +0.0081, d_gain +0.0149, n=639
- dose seed43022 chck_100M dose21-minus-base0: d_row +0.0288, d_iso +0.0902, d_gain +0.0614, n=639
- dose seed43022 chck_100M dose25-minus-base0: d_row +0.0439, d_iso +0.1193, d_gain +0.0754, n=639
- dose seed43022 chck_100M dose25-minus-dose21: d_row +0.0151, d_iso +0.0290, d_gain +0.0140, n=639
- dose seed43122 chck_100M dose21-minus-base0: d_row +0.0031, d_iso -0.0514, d_gain -0.0545, n=639
- dose seed43122 chck_100M dose25-minus-base0: d_row +0.0318, d_iso -0.0061, d_gain -0.0379, n=639
- dose seed43122 chck_100M dose25-minus-dose21: d_row +0.0287, d_iso +0.0453, d_gain +0.0166, n=639
- compact_experience seed43022 chck_100M ALN-minus-OFF: d_row -0.2114, d_iso +0.1352, d_gain +0.3465, n=634
- compact_experience seed43022 chck_100M SHUF-minus-OFF: d_row -0.0367, d_iso -0.0611, d_gain -0.0244, n=634
- compact_experience seed43022 chck_100M ALN-minus-SHUF: d_row -0.1747, d_iso +0.1963, d_gain +0.3709, n=634
- compact_experience seed43022 chck_100M DUP-minus-OFF: d_row -0.0294, d_iso +0.2582, d_gain +0.2876, n=634
- compact_experience seed43022 chck_100M ALN-minus-DUP: d_row -0.1820, d_iso -0.1230, d_gain +0.0589, n=634
- compact_experience seed43022 chck_100M SEP-minus-OFF: d_row +0.2804, d_iso -0.0634, d_gain -0.3439, n=634
- compact_experience seed43022 chck_100M ALN-minus-SEP: d_row -0.4918, d_iso +0.1986, d_gain +0.6904, n=634
- compact_experience seed43022 chck_100M SHUF-minus-SEP: d_row -0.3171, d_iso +0.0023, d_gain +0.3195, n=634
- compact_experience seed43122 chck_100M ALN-minus-OFF: d_row -0.0737, d_iso +0.1391, d_gain +0.2128, n=634
- compact_experience seed43122 chck_100M SHUF-minus-OFF: d_row +0.0809, d_iso -0.0171, d_gain -0.0981, n=634
- compact_experience seed43122 chck_100M ALN-minus-SHUF: d_row -0.1546, d_iso +0.1562, d_gain +0.3109, n=634
