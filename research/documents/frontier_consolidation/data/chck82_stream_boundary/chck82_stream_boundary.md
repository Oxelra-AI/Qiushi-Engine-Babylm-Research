# chck82 frozen private tail design chck_82M stream boundary

Status: **PASS**

- Boundary update: `2074`; loader step: `2074`.
- Previous cumulative words: `81972552`; boundary cumulative words: `82012495`; target overshoot: `12495`.
- Remaining legal charged words after chck_82M: `17987505`.
- Tail should start at next loader step `2075`; if using batch256 sequential rows, skip `530944` rows.

## Checks
- `boundary_matches_metrics_chck82_words`: `True`
- `boundary_crosses_target_from_below`: `True`
- `remaining_positive`: `True`
- `remaining_within_cap`: `True`
- `training_log_has_expected_updates`: `True`
- `all_checks_passed`: `True`

For a frozen-82M tail, the slow function already consumed the full batch ending at the boundary update. Any remaining-exposure private training should start from the next stream batch/row and cap total new charged words at 17,987,505 so the full model remains at or below 100M counted-word exposure.

JSON: `experiments/archive/frontier_consolidation/data/chck82_stream_boundary/chck82_stream_boundary.json`
