# earlier analysis register-displacement feasibility

CPU/file-only; no model loading/training/eval/generation/streaming.

## Existing quarter admitted FineWeb block
Rows: 758, words: 105962

## Original clean rows displaced by existing quarter
```json
{
  "n_rows": 758,
  "words": 105962,
  "mean_words": 139.7915567282322,
  "min_words": 91,
  "max_words": 160,
  "component_words": {
    "gutenberg": 24480,
    "simple_wiki": 13760,
    "bnc_spoken": 8800,
    "open_subtitles": 23040,
    "childes": 35882
  },
  "group_rows_threshold_0p5": {
    "adult_gut_simple": 268,
    "mixed_adult_lean": 23,
    "child_sub": 426,
    "other_or_balanced": 13,
    "mixed_child_sub_lean": 28
  },
  "child_sub_words": 58922,
  "adult_gut_simple_words": 38240,
  "other_words": 8800
}
```

## Candidate exact length-match feasibility
```json
{
  "child_sub_lean_selected_rows": 757,
  "adult_lean_selected_rows": 756,
  "target_rows": 758,
  "child_deficits": [
    {
      "words": 95,
      "need": 1,
      "available": 0,
      "deficit": 1
    }
  ],
  "adult_deficits": [
    {
      "words": 91,
      "need": 1,
      "available": 0,
      "deficit": 1
    },
    {
      "words": 92,
      "need": 1,
      "available": 0,
      "deficit": 1
    }
  ],
  "child_deficit_total": 1,
  "adult_deficit_total": 2,
  "child_selected_summary": {
    "n_rows": 757,
    "words": 105867,
    "mean_words": 139.85072655217965,
    "min_words": 91,
    "max_words": 160,
    "component_words": {
      "childes": 60480,
      "open_subtitles": 45387
    },
    "group_rows_threshold_0p5": {
      "child_sub": 757
    },
    "child_sub_words": 105867,
    "adult_gut_simple_words": 0,
    "other_words": 0
  },
  "adult_selected_summary": {
    "n_rows": 756,
    "words": 105779,
    "mean_words": 139.91931216931218,
    "min_words": 95,
    "max_words": 160,
    "component_words": {
      "gutenberg": 67893,
      "simple_wiki": 37779,
      "childes": 29,
      "open_subtitles": 1,
      "bnc_spoken": 77
    },
    "group_rows_threshold_0p5": {
      "adult_gut_simple": 756
    },
    "child_sub_words": 30,
    "adult_gut_simple_words": 105672,
    "other_words": 77
  },
  "intersection_selected_rows": 0
}
```
