# copy and relevant update result Entity relevant-update and copy-window readout

## Training packet design facts

- Changed rows scanned: 7,924; changed source/companion pairs: 33,291.
- VIEW and REPEAT rows rebuilt exactly from `source_text + companion`: 7,923/7,924 and 7,923/7,924.
- Pair-local construction count: 33,291/33,291 (100.00%).
- Each individual source+companion pair fits the 256-token tokenizer window: VIEW 100.00%, REPEAT 100.00%. Packed rows are longer: full packed-row fit is VIEW 97.49%, REPEAT 98.90%; mean/max packed-row token lengths are VIEW 205.3/340 and REPEAT 197.8/328.
- Interpretation: the changed-block construction really is source plus its own companion, and the REPEAT companion is a rotated exact source-token segment. Because multiple pairs can be packed into a row, not every entire packed row is visible under seq256; the pair-level copy signal is nevertheless canonical for most individual source+companion packets.

## Entity variable audit

- Official-filtered Entity items: 6,780. Parsed relevant-update count mismatches with the dataset `numops`: 247.
  First mismatches: [{'uid': 'move_contents_2_ops', 'entity_type': 'move_contents', 'reported_numops': 2, 'item_index': 33, 'sample_id': 1196, 'example_id': 1257, 'query_box': 2, 'total_ops': 4, 'relevant_updates': 1, 'irrelevant_ops': 3, 'ops_after_last_relevant': 0, 'ops_before_first_relevant': 3, 'last_relevant_is_final_op': 1, 'prefix_words': 100, 'prefix_chars': 472, 'prefix_tokens_untruncated': 130, 'prefix_tokens_trunc256': 130, 'prefix_truncated_at_256': 0, 'stale_initial': 'the note', 'stale_available': 1, 'stale_is_gold': 0, 'gold': 'the boat.', 'prefix_words_bin': 'q2', 'prefix_tokens_bin': 'q2'}, {'uid': 'move_contents_2_ops', 'entity_type': 'move_contents', 'reported_numops': 2, 'item_index': 37, 'sample_id': 1410, 'example_id': 1265, 'query_box': 4, 'total_ops': 4, 'relevant_updates': 1, 'irrelevant_ops': 3, 'ops_after_last_relevant': 1, 'ops_before_first_relevant': 2, 'last_relevant_is_final_op': 0, 'prefix_words': 93, 'prefix_chars': 438, 'prefix_tokens_untruncated': 122, 'prefix_tokens_trunc256': 122, 'prefix_truncated_at_256': 0, 'stale_initial': 'the dress and the medicine and the wire', 'stale_available': 0, 'stale_is_gold': 0, 'gold': 'the ball.', 'prefix_words_bin': 'q1', 'prefix_tokens_bin': 'q1'}, {'uid': 'move_contents_2_ops', 'entity_type': 'move_contents', 'reported_numops': 2, 'item_index': 42, 'sample_id': 1199, 'example_id': 1272, 'query_box': 2, 'total_ops': 5, 'relevant_updates': 1, 'irrelevant_ops': 4, 'ops_after_last_relevant': 0, 'ops_before_first_relevant': 4, 'last_relevant_is_final_op': 1, 'prefix_words': 88, 'prefix_chars': 433, 'prefix_tokens_untruncated': 121, 'prefix_tokens_trunc256': 121, 'prefix_truncated_at_256': 0, 'stale_initial': 'the medicine', 'stale_available': 1, 'stale_is_gold': 0, 'gold': 'the painting.', 'prefix_words_bin': 'q1', 'prefix_tokens_bin': 'q1'}]
- Thus the earlier depth variable is best read as number of state updates touching the queried box, not merely total operation sentences or context length.

## Late official accuracy by relevant updates

| seed | group | contrast | delta acc pct | acc a | acc b | n | total ops mean | words mean |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 43022 | rel_updates_0 | RminusC | +8.78 | 48.24 | 39.45 | 1541 | 3.39 | 88.7 |
| 43022 | rel_updates_0 | RminusV | +9.69 | 48.24 | 38.55 | 1541 | 3.39 | 88.7 |
| 43022 | rel_updates_0 | VminusC | -0.91 | 38.55 | 39.45 | 1541 | 3.39 | 88.7 |
| 43022 | rel_updates_1 | RminusC | -1.39 | 14.08 | 15.47 | 1323 | 6.32 | 112.0 |
| 43022 | rel_updates_1 | RminusV | -2.19 | 14.08 | 16.28 | 1323 | 6.32 | 112.0 |
| 43022 | rel_updates_1 | VminusC | +0.81 | 16.28 | 15.47 | 1323 | 6.32 | 112.0 |
| 43022 | rel_updates_2 | RminusC | -0.18 | 18.80 | 18.98 | 1266 | 8.17 | 125.8 |
| 43022 | rel_updates_2 | RminusV | -4.13 | 18.80 | 22.93 | 1266 | 8.17 | 125.8 |
| 43022 | rel_updates_2 | VminusC | +3.95 | 22.93 | 18.98 | 1266 | 8.17 | 125.8 |
| 43022 | rel_updates_3 | RminusC | -5.83 | 17.80 | 23.63 | 1230 | 9.44 | 137.8 |
| 43022 | rel_updates_3 | RminusV | -8.46 | 17.80 | 26.26 | 1230 | 9.44 | 137.8 |
| 43022 | rel_updates_3 | VminusC | +2.63 | 26.26 | 23.63 | 1230 | 9.44 | 137.8 |
| 43022 | rel_updates_4 | RminusC | -1.59 | 23.71 | 25.30 | 1112 | 10.02 | 142.9 |
| 43022 | rel_updates_4 | RminusV | -9.14 | 23.71 | 32.85 | 1112 | 10.02 | 142.9 |
| 43022 | rel_updates_4 | VminusC | +7.55 | 32.85 | 25.30 | 1112 | 10.02 | 142.9 |
| 43022 | rel_updates_5 | RminusC | -2.06 | 21.65 | 23.70 | 308 | 10.44 | 146.2 |
| 43022 | rel_updates_5 | RminusV | -9.74 | 21.65 | 31.39 | 308 | 10.44 | 146.2 |
| 43022 | rel_updates_5 | VminusC | +7.68 | 31.39 | 23.70 | 308 | 10.44 | 146.2 |
| 43122 | rel_updates_0 | RminusC | +9.09 | 48.39 | 39.30 | 1541 | 3.39 | 88.7 |
| 43122 | rel_updates_0 | RminusV | +9.15 | 48.39 | 39.24 | 1541 | 3.39 | 88.7 |
| 43122 | rel_updates_0 | VminusC | -0.06 | 39.24 | 39.30 | 1541 | 3.39 | 88.7 |
| 43122 | rel_updates_1 | RminusC | -3.22 | 14.11 | 17.33 | 1323 | 6.32 | 112.0 |
| 43122 | rel_updates_1 | RminusV | -2.67 | 14.11 | 16.78 | 1323 | 6.32 | 112.0 |
| 43122 | rel_updates_1 | VminusC | -0.55 | 16.78 | 17.33 | 1323 | 6.32 | 112.0 |
| 43122 | rel_updates_2 | RminusC | -1.79 | 18.14 | 19.93 | 1266 | 8.17 | 125.8 |
| 43122 | rel_updates_2 | RminusV | -3.19 | 18.14 | 21.33 | 1266 | 8.17 | 125.8 |
| 43122 | rel_updates_2 | VminusC | +1.40 | 21.33 | 19.93 | 1266 | 8.17 | 125.8 |
| 43122 | rel_updates_3 | RminusC | -3.09 | 18.43 | 21.52 | 1230 | 9.44 | 137.8 |
| 43122 | rel_updates_3 | RminusV | -6.40 | 18.43 | 24.82 | 1230 | 9.44 | 137.8 |
| 43122 | rel_updates_3 | VminusC | +3.31 | 24.82 | 21.52 | 1230 | 9.44 | 137.8 |
| 43122 | rel_updates_4 | RminusC | -0.24 | 23.86 | 24.10 | 1112 | 10.02 | 142.9 |
| 43122 | rel_updates_4 | RminusV | -8.24 | 23.86 | 32.10 | 1112 | 10.02 | 142.9 |
| 43122 | rel_updates_4 | VminusC | +8.00 | 32.10 | 24.10 | 1112 | 10.02 | 142.9 |
| 43122 | rel_updates_5 | RminusC | +2.06 | 27.71 | 25.65 | 308 | 10.44 | 146.2 |
| 43122 | rel_updates_5 | RminusV | -3.68 | 27.71 | 31.39 | 308 | 10.44 | 146.2 |
| 43122 | rel_updates_5 | VminusC | +5.74 | 31.39 | 25.65 | 308 | 10.44 | 146.2 |

## Zero-relevant-update split by total operation count

| seed | group | contrast | delta acc pct | acc a | acc b | n | words mean |
|---:|---|---|---:|---:|---:|---:|---:|
| 43022 | rel0_total_ops_0 | RminusC | +5.70 | 44.19 | 38.49 | 304 | 61.5 |
| 43022 | rel0_total_ops_0 | RminusV | +2.30 | 44.19 | 41.89 | 304 | 61.5 |
| 43022 | rel0_total_ops_1 | RminusC | +7.00 | 43.45 | 36.45 | 257 | 69.7 |
| 43022 | rel0_total_ops_1 | RminusV | +4.80 | 43.45 | 38.65 | 257 | 69.7 |
| 43022 | rel0_total_ops_10 | RminusC | +6.48 | 40.74 | 34.26 | 36 | 143.7 |
| 43022 | rel0_total_ops_10 | RminusV | +5.56 | 40.74 | 35.19 | 36 | 143.7 |
| 43022 | rel0_total_ops_11 | RminusC | -3.12 | 30.21 | 33.33 | 32 | 149.0 |
| 43022 | rel0_total_ops_11 | RminusV | -1.04 | 30.21 | 31.25 | 32 | 149.0 |
| 43022 | rel0_total_ops_12 | RminusC | +7.78 | 44.44 | 36.67 | 30 | 157.4 |
| 43022 | rel0_total_ops_12 | RminusV | +11.11 | 44.44 | 33.33 | 30 | 157.4 |
| 43022 | rel0_total_ops_2 | RminusC | +6.60 | 46.70 | 40.09 | 212 | 77.9 |
| 43022 | rel0_total_ops_2 | RminusV | +6.76 | 46.70 | 39.94 | 212 | 77.9 |
| 43022 | rel0_total_ops_3 | RminusC | +8.33 | 49.42 | 41.09 | 172 | 85.9 |
| 43022 | rel0_total_ops_3 | RminusV | +8.91 | 49.42 | 40.50 | 172 | 85.9 |
| 43022 | rel0_total_ops_4 | RminusC | +8.14 | 49.62 | 41.48 | 131 | 93.4 |
| 43022 | rel0_total_ops_4 | RminusV | +12.72 | 49.62 | 36.90 | 131 | 93.4 |
| 43022 | rel0_total_ops_5 | RminusC | +15.06 | 56.73 | 41.67 | 104 | 100.6 |
| 43022 | rel0_total_ops_5 | RminusV | +18.91 | 56.73 | 37.82 | 104 | 100.6 |
| 43022 | rel0_total_ops_6 | RminusC | +13.26 | 59.85 | 46.59 | 88 | 107.6 |
| 43022 | rel0_total_ops_6 | RminusV | +19.32 | 59.85 | 40.53 | 88 | 107.6 |
| 43022 | rel0_total_ops_7 | RminusC | +25.25 | 59.60 | 34.34 | 66 | 117.8 |
| 43022 | rel0_total_ops_7 | RminusV | +27.27 | 59.60 | 32.32 | 66 | 117.8 |
| 43022 | rel0_total_ops_8 | RminusC | +8.47 | 50.26 | 41.80 | 63 | 127.5 |
| 43022 | rel0_total_ops_8 | RminusV | +21.69 | 50.26 | 28.57 | 63 | 127.5 |
| 43022 | rel0_total_ops_9 | RminusC | +17.39 | 60.87 | 43.48 | 46 | 134.9 |
| 43022 | rel0_total_ops_9 | RminusV | +22.46 | 60.87 | 38.41 | 46 | 134.9 |
| 43122 | rel0_total_ops_0 | RminusC | +10.42 | 46.82 | 36.40 | 304 | 61.5 |
| 43122 | rel0_total_ops_0 | RminusV | +5.81 | 46.82 | 41.01 | 304 | 61.5 |
| 43122 | rel0_total_ops_1 | RminusC | +5.84 | 45.40 | 39.56 | 257 | 69.7 |
| 43122 | rel0_total_ops_1 | RminusV | +6.36 | 45.40 | 39.04 | 257 | 69.7 |
| 43122 | rel0_total_ops_10 | RminusC | -0.00 | 42.59 | 42.59 | 36 | 143.7 |
| 43122 | rel0_total_ops_10 | RminusV | +9.26 | 42.59 | 33.33 | 36 | 143.7 |
| 43122 | rel0_total_ops_11 | RminusC | +8.33 | 41.67 | 33.33 | 32 | 149.0 |
| 43122 | rel0_total_ops_11 | RminusV | +10.42 | 41.67 | 31.25 | 32 | 149.0 |
| 43122 | rel0_total_ops_12 | RminusC | +4.44 | 45.56 | 41.11 | 30 | 157.4 |
| 43122 | rel0_total_ops_12 | RminusV | +7.78 | 45.56 | 37.78 | 30 | 157.4 |
| 43122 | rel0_total_ops_2 | RminusC | +6.76 | 48.11 | 41.35 | 212 | 77.9 |
| 43122 | rel0_total_ops_2 | RminusV | +9.43 | 48.11 | 38.68 | 212 | 77.9 |
| 43122 | rel0_total_ops_3 | RminusC | +15.89 | 54.46 | 38.57 | 172 | 85.9 |
| 43122 | rel0_total_ops_3 | RminusV | +12.02 | 54.46 | 42.44 | 172 | 85.9 |
| 43122 | rel0_total_ops_4 | RminusC | +5.60 | 42.49 | 36.90 | 131 | 93.4 |
| 43122 | rel0_total_ops_4 | RminusV | +5.85 | 42.49 | 36.64 | 131 | 93.4 |
| 43122 | rel0_total_ops_5 | RminusC | +13.46 | 52.56 | 39.10 | 104 | 100.6 |
| 43122 | rel0_total_ops_5 | RminusV | +14.10 | 52.56 | 38.46 | 104 | 100.6 |
| 43122 | rel0_total_ops_6 | RminusC | +3.79 | 51.89 | 48.11 | 88 | 107.6 |
| 43122 | rel0_total_ops_6 | RminusV | +11.36 | 51.89 | 40.53 | 88 | 107.6 |
| 43122 | rel0_total_ops_7 | RminusC | +19.19 | 59.09 | 39.90 | 66 | 117.8 |
| 43122 | rel0_total_ops_7 | RminusV | +20.71 | 59.09 | 38.38 | 66 | 117.8 |
| 43122 | rel0_total_ops_8 | RminusC | +12.17 | 47.09 | 34.92 | 63 | 127.5 |
| 43122 | rel0_total_ops_8 | RminusV | +15.87 | 47.09 | 31.22 | 63 | 127.5 |
| 43122 | rel0_total_ops_9 | RminusC | +5.80 | 52.17 | 46.38 | 46 | 134.9 |
| 43122 | rel0_total_ops_9 | RminusV | +2.90 | 52.17 | 49.28 | 46 | 134.9 |

## Scientific reading

The official Entity depth result is already organized by relevant updates to the queried box. The key alternative to test is therefore whether REPEAT's no-update advantage persists when irrelevant operations and longer contexts are present, and whether the arm crossover starts when the queried state is actually changed. The tables above preserve that split for the two existing seeds and late checkpoints. Stale-initial percentages remain a secondary error clue because the eligible stale option is not present for every item and the wrong-answer denominator changes with accuracy.
