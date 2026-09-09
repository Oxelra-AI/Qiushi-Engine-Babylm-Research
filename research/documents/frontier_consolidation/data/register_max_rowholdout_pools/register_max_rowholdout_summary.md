# earlier analysis MAX register-skewed row-holdout pools

File-only materialization from existing base and MAX compact-view files. No generation, streaming, model loading, training, evaluation, upload, or leaderboard action.

## Scientific contrast
Both arms admit the identical matched max dose execution note MAX FineWeb compact-view block (7,923 rows, 1,118,587 words; rho=0.111859). The childspeech arm removes only CHILDES/OpenSubtitles/BNC/Switchboard 160-word rows from the original clean base; the adultprose arm removes only Gutenberg/SimpleWiki 160-word rows. The original trained MAX view arm is the proportional row-holdout midpoint.

## Geometry
```json
{
  "total_rows_10m": 65313,
  "total_words_10m": 10000000,
  "pair_rows": 7923,
  "selected_pair_words": 1118587,
  "topup_words": 133,
  "rho_pair_words": 0.1118587,
  "heldout_rows_per_arm": 6992,
  "heldout_budget_words_per_arm": 1118720,
  "row_word_count_sequence_identical_to_step256_max_view": true,
  "fineweb_pair_rows_identical_to_step256_max_view": true,
  "qwen_pair_packed_rows_never_selected": true
}
```

## Position matching
```json
{
  "target_basis": "nearest to original matched max dose execution note proportional MAX heldout base-row indices",
  "child_to_original_distance": {
    "mean": 0.5755148741418764,
    "median": 0.0,
    "p90": 2,
    "max": 9
  },
  "adult_to_original_distance": {
    "mean": 1.908895881006865,
    "median": 1.0,
    "p90": 5,
    "max": 28
  },
  "child_adult_sorted_pair_distance": {
    "mean": 2.339387871853547,
    "median": 2.0,
    "p90": 5,
    "max": 28
  }
}
```

## Arm summaries
```json
{
  "childspeech": {
    "rows": 6992,
    "words": 1118720,
    "source_words": {
      "childes": 578080,
      "open_subtitles": 411040,
      "bnc_spoken": 124160,
      "switchboard": 5440
    },
    "dev_speech_words": 1118720,
    "adult_prose_words": 0,
    "other_words": 0,
    "dev_speech_fraction": 1.0,
    "adult_prose_fraction": 0.0,
    "row_index_min": 1,
    "row_index_max": 64373,
    "row_index_mean": 32385.605835240276,
    "row_index_sd": 18587.81652052028
  },
  "adultprose": {
    "rows": 6992,
    "words": 1118720,
    "source_words": {
      "gutenberg": 710560,
      "simple_wiki": 408160
    },
    "dev_speech_words": 0,
    "adult_prose_words": 1118720,
    "other_words": 0,
    "dev_speech_fraction": 0.0,
    "adult_prose_fraction": 1.0,
    "row_index_min": 4,
    "row_index_max": 64379,
    "row_index_mean": 32386.208667048057,
    "row_index_sd": 18587.83682899887
  }
}
```

Topup rows are only 133 clean words per arm and are reported separately in metadata; the admitted FineWeb rows are identical.

Metadata: `experiments/archive/frontier_consolidation/data/register_max_rowholdout_pools/register_max_rowholdout_metadata.json`
Selection records: `experiments/archive/frontier_consolidation/data/register_max_rowholdout_pools/regmax_selection_records.jsonl`
