# frozen82 short tail plan frozen-82M private-tail panel analysis

Status: **COMPLETE**
Route read: `does_not_support_more_tail_exposure_in_this_form`

Available: `['aligned', 'shuffled', 'neutral']`; pending: `[]`

Protected chck_82M cheap7: `43.95944987645173`; Overall: `41.942481167385985`

## Scores
| arm | cheap7 | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | tail main | tail aux | total words |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned | 43.91285714285714 | 69.19 | 59.27 | 51.81 | 28.95 | 52.36 | 37.09 | 8.72 | 3954265 | 38708 | 86005468 |
| shuffled | 44.01285714285714 | 69.23 | 59.81 | 51.44 | 26.77 | 52.65 | 39.565 | 8.625 | 3954265 | 38653 | 86005413 |
| neutral | 43.95944987645173 | 68.49128403651986 | 62.9378112562002 | 50.05545332553276 | 28.314041930298774 | 52.19117509443596 | 37.57766990291262 | 8.148713589261902 | 0 | 0 | 82012495 |

## Decisions
- `aligned_preserves_chck82_cheap7_within_0p3`: `True`
- `aligned_no_large_fragile_damage_vs_chck82`: `False`
- `true_correspondence_beats_shuffled_on_tail_cheap7`: `False`
- `aligned_beats_neutral_carrier_on_tail_cheap7`: `False`
- `aligned_lower_true_source_free_nll_than_shuffled`: `False`

## Comparisons
```json
{
  "aligned_minus_shuffled": {
    "cheap7_delta": -0.10000000000000142,
    "column_deltas": {
      "BLiMP": -0.04000000000000625,
      "Supplement": -0.5399999999999991,
      "EWoK": 0.37000000000000455,
      "Entity": 2.1799999999999997,
      "COMPS": -0.28999999999999915,
      "GlobalPIQA": -2.4749999999999943,
      "Reading": 0.09500000000000064
    }
  },
  "aligned_minus_neutral": {
    "cheap7_delta": -0.04659273359458638,
    "column_deltas": {
      "BLiMP": 0.6987159634801401,
      "Supplement": -3.6678112562001957,
      "EWoK": 1.7545466744672424,
      "Entity": 0.6359580697012248,
      "COMPS": 0.16882490556403695,
      "GlobalPIQA": -0.48766990291261436,
      "Reading": 0.5712864107380984
    }
  },
  "shuffled_minus_neutral": {
    "cheap7_delta": 0.053407266405415044,
    "column_deltas": {
      "BLiMP": 0.7387159634801463,
      "Supplement": -3.1278112562001965,
      "EWoK": 1.384546674467238,
      "Entity": -1.544041930298775,
      "COMPS": 0.4588249055640361,
      "GlobalPIQA": 1.98733009708738,
      "Reading": 0.47628641073809774
    }
  }
}
```

## Source-free probe key deltas
```json
{
  "aligned_model_minus_shuffled_model_on_true_free_view": 0.018026061741252875,
  "aligned_model_minus_shuffled_model_on_true_conditioned_view": -0.14902118248988838,
  "aligned_model_minus_base_on_true_free_view": -0.7687184490164416,
  "aligned_model_minus_base_on_true_conditioned_view": -0.6209096274664696,
  "shuffled_model_minus_base_on_shuffled_free_view": -0.7867445107576945,
  "aligned_model_minus_shuffled_model_on_shuffled_free_view": 0.018026061741252875
}
```

The frozen-tail panel protects the verified chck_82M slow function and asks whether a fresh private source-correspondence pathway can add source-free competence without broad-score erosion. Full remaining exposure should follow only from mature-state preservation plus aligned-over-shuffled/source-free evidence.

JSON: `experiments/archive/frontier_consolidation/data/frozen82_tail_panel_analysis/frozen82_tail_panel_analysis.json`
