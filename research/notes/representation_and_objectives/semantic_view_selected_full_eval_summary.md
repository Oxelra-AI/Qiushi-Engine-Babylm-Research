# cached fineweb broad source candidate semantic-view selected full-evaluation summary

Summary JSON: `experiments/archive/representation_and_objectives/data/semantic_view_full_eval/semantic_view_selected_full_eval_summary.json`

| target | endpoint | Overall | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | SuperGLUE | Reading | AoA leaderboard | AoA raw | submit-ready |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| original_packet_local__chck_80M | chck_80M | 38.554 | 65.97 | 63.65 | 48.24 | 16.15 | 51.6 | 38.08 | 68.1796 | 7.37 | -12.2539 | -0.122539 | True |
| semantic_view_treatment__chck_80M | chck_80M | 40.1399 | 66.26 | 61.74 | 49.51 | 18.25 | 52.2 | 37.13 | 68.3595 | 7.8 | 0.0 | 0.0 | True |

## Same-endpoint treatment minus packet-local deltas

```json
{
  "semantic_view_treatment__chck_80M_minus_original_packet_local__chck_80M": {
    "BLiMP": 0.29,
    "Supplement": -1.91,
    "EWoK": 1.27,
    "Entity": 2.1,
    "COMPS": 0.6,
    "GlobalPIQA": -0.945,
    "SuperGLUE": 0.179913,
    "Reading": 0.435,
    "AoA": 12.253908,
    "Overall": 1.58598,
    "NLP_average": 0.226416,
    "Human_like_average": 6.344454
  }
}
```

Best full-eval target: `semantic_view_treatment__chck_80M` Overall 40.13994328378824. Public reference: 41.8.
