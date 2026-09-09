# static decoy contingency validation frozen82 tail score decision

Status: **COMPLETE**
Route read: `generic_shuffled_private_tail_endpoint_candidate`

Protected chck82 Overall: `41.942481167385985`; cheap7 `43.95944987645173`; SuperGLUE `69.7661813713118`; AoA `0.0`

## Arms

| arm | cheap7 | required SG to tie chck82 | SuperGLUE | projected Overall AoA0 | delta vs chck82 | score read |
|---|---:|---:|---:|---:|---:|---|
| aligned | 43.91285714285714 | 70.0923305064739 | 69.77796826428681 | 41.9075520293652 | -0.0349291380207859 | above_41p8_but_below_chck82_before_aoa |
| shuffled | 44.01285714285714 | 69.39233050647391 | 69.7661813713118 | 41.984020152367975 | 0.04153898498199027 | candidate_exceeds_chck82_before_aoa |

## Source-free probe

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

## Decisions

- `aligned_preserves_chck82_cheap7_within_0p3`: `True`
- `aligned_no_large_fragile_damage_vs_chck82`: `False`
- `true_correspondence_beats_shuffled_on_tail_cheap7`: `False`
- `aligned_beats_neutral_carrier_on_tail_cheap7`: `False`
- `aligned_lower_true_source_free_nll_than_shuffled`: `False`
- `aligned_true_free_nll_delta_vs_shuffled`: `0.018026061741252875`
- `aligned_true_cond_nll_delta_vs_shuffled`: `-0.14902118248988838`
- `intended_correspondence_source_free_supported`: `False`

Aligned correspondence is supported only if it beats shuffled both on source-free NLL and endpoint score. A shuffled endpoint candidate can improve the score but would be generic private-tail adaptation, not evidence for true correspondence transfer.

JSON: `experiments/archive/frontier_consolidation/data/frozen82_tail_score_decision/frozen82_tail_score_decision.json`
