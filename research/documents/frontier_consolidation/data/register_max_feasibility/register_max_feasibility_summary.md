# earlier analysis MAX register-pair feasibility

File-only. No generation, streaming, model loading, training, evaluation, upload, or leaderboard action.

## Target MAX FineWeb block
Rows: 7923; words: 1118587; rho: 0.111859; word-length types: 76.

## Existing MAX displaced prefix mixture
Developmental/speech words (CHILDES + OpenSubtitles + BNC Spoken + Switchboard): 706267 (0.631392).
Adult-prose words (Gutenberg + SimpleWiki): 412320 (0.368608).
Other/protected qwen words in prefix: 0; qwen: 0.

## Feasibility
Exact non-qwen word-histogram feasible for both skewed arms: True.
Naive sorted-within-length position pairing:
```json
{
  "mean_abs_row_distance": 54.782531869241446,
  "median_abs_row_distance": 0,
  "p90_abs_row_distance": 0,
  "max_abs_row_distance": 5755,
  "same_position_overlap": 7498
}
```

## Skewed selection summaries
```json
{
  "childspeech": {
    "rows": 7923,
    "words": 1118587,
    "dev_speech_words": 722850,
    "adult_prose_words": 395737,
    "qwen_pair_packed_words": 0,
    "other_words": 0,
    "dev_speech_fraction": 0.6462170577702048,
    "adult_prose_fraction": 0.3537829422297953,
    "qwen_pair_packed_fraction": 0.0,
    "row_index_min": 0,
    "row_index_max": 8229,
    "row_index_mean": 4036.1027388615425,
    "row_index_sd": 2340.0122476696665
  },
  "adultprose": {
    "rows": 7923,
    "words": 1118587,
    "dev_speech_words": 677437,
    "adult_prose_words": 441150,
    "qwen_pair_packed_words": 0,
    "other_words": 0,
    "dev_speech_fraction": 0.6056185169325229,
    "adult_prose_fraction": 0.3943814830674771,
    "qwen_pair_packed_fraction": 0.0,
    "row_index_min": 0,
    "row_index_max": 8771,
    "row_index_mean": 4088.9238924649753,
    "row_index_sd": 2378.371644752733
  }
}
```

Length table: `experiments/archive/frontier_consolidation/data/register_max_feasibility/length_feasibility_rows.csv`
JSON: `experiments/archive/frontier_consolidation/data/register_max_feasibility/register_max_feasibility_summary.json`
