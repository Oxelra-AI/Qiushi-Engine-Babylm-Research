# dualview pending panel interim dual-view complete panel analysis — dualview_panel_complete

Created: `2026-08-31T22:58:09Z`

## Route reading

`true_correspondence_used_but_current_budget_tradeoff_under_mlm_only`

## Cheap7 and columns

| Arm | cheap7 | vs spatial repair route status 20M | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned | 38.762857 | -0.900714 | 56.7300 | 51.7600 | 51.1100 | 17.9100 | 50.2000 | 36.1050 | 7.5250 |
| shuffled | 38.541429 | -1.122143 | 56.5400 | 52.8200 | 51.3100 | 17.3500 | 49.8000 | 34.6200 | 7.3500 |
| mlm_only | 39.786429 | +0.122857 | 59.5400 | 58.4300 | 50.1000 | 18.3900 | 50.7000 | 32.6650 | 8.6800 |

### Pairwise cheap7 deltas

| contrast | cheap7 delta |
|---|---:|
| aligned_minus_shuffled | +0.221429 |
| aligned_minus_mlm_only | -1.023571 |
| shuffled_minus_mlm_only | -1.245000 |

## Mechanism/accounting context

- mean_main_loss_aligned_minus_shuffled: `-0.03165926903548211`
- final_main_loss_aligned_minus_shuffled: `-0.060129642486572266`
- mean_aux_loss_aligned_minus_shuffled: `-0.06168294309748925`
- dual_aux_words: `966720`
- displaced_main_words: `978416`
- displaced_main_rows_count: `6344`
- aligned_vs_shuffled_stock_rel_l2: `0.47405023870515933`
- aligned_vs_shuffled_adapter_rel_l2: `0.5402174137040273`
- aligned_vs_mlm_stock_rel_l2: `0.6007885878779412`
- aligned_vs_mlm_adapter_rel_l2: `0.5447522688455644`

## Most negative fragile sentinel nets

### aligned_vs_step35
- EWoK / material-dynamics: -32 over n=770
- Entity / regular_3_ops: -21 over n=425
- Entity / regular_5_ops: -16 over n=94
- EWoK / physical-dynamics: -14 over n=120
- Supplement / subject_aux_inversion: -12 over n=3867
- Supplement / qa_congruence_easy: -6 over n=64
- Supplement / qa_congruence_tricky: -5 over n=165
- EWoK / quantitative-properties: 5 over n=314

### shuffled_vs_step35
- Entity / regular_5_ops: -18 over n=94
- Entity / regular_3_ops: -18 over n=425
- EWoK / material-dynamics: -15 over n=770
- Supplement / qa_congruence_easy: -6 over n=64
- Entity / regular_4_ops: 0 over n=388
- EWoK / physical-dynamics: 1 over n=120
- Supplement / qa_congruence_tricky: 2 over n=165
- EWoK / quantitative-properties: 4 over n=314

### mlm_only_vs_step35
- EWoK / material-dynamics: -71 over n=770
- Supplement / subject_aux_inversion: -31 over n=3867
- EWoK / quantitative-properties: -3 over n=314
- EWoK / physical-dynamics: -3 over n=120
- Entity / regular_5_ops: -3 over n=94
- Entity / regular_4_ops: -3 over n=388
- Entity / regular_3_ops: 0 over n=425
- Supplement / qa_congruence_easy: 2 over n=64

### shuffled_vs_aligned
- EWoK / spatial-relations: -15 over n=490
- EWoK / social-relations: -8 over n=1548
- EWoK / material-properties: -6 over n=170
- Entity / regular_4_ops: -5 over n=388
- EWoK / physical-interactions: -4 over n=556
- Entity / regular_5_ops: -2 over n=94
- EWoK / quantitative-properties: -1 over n=314
- Supplement / qa_congruence_easy: 0 over n=64

### mlm_only_vs_aligned
- EWoK / material-dynamics: -39 over n=770
- Supplement / subject_aux_inversion: -19 over n=3867
- EWoK / spatial-relations: -17 over n=490
- EWoK / material-properties: -12 over n=170
- EWoK / physical-interactions: -9 over n=556
- EWoK / quantitative-properties: -8 over n=314
- EWoK / social-relations: -8 over n=1548
- Entity / regular_4_ops: -8 over n=388

## Artifact paths

- base_payload: `experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json`
- aligned_payload: `experiments/archive/frontier_consolidation/data/dualview_aligned_20m_eval/per_target/dualview_aligned_20M.json`
- shuffled_payload: `experiments/archive/frontier_consolidation/data/dualview_shuffled_20m_eval/per_target/dualview_shuffled_20M.json`
- mlm_only_payload: `experiments/archive/frontier_consolidation/data/dualview_mlm_only_20m_eval/per_target/dualview_mlm_only_20M.json`
- aligned_summary: `experiments/archive/frontier_consolidation/data/dualview_aligned_20m_summary/dualview_aligned_20M_summary.json`
- shuffled_summary: `experiments/archive/frontier_consolidation/data/dualview_shuffled_20m_summary/dualview_shuffled_20M_summary.json`
- mlm_only_summary: `experiments/archive/frontier_consolidation/data/dualview_mlm_only_20m_summary/dualview_mlm_only_20M_summary.json`
- sentinel_json: `experiments/archive/frontier_consolidation/data/dualview_sentinel_compare/dualview_panel_complete.json`
- sentinel_md: `research/documents/frontier_consolidation/data/dualview_sentinel_compare/dualview_panel_complete.md`
- training_audit: `experiments/archive/frontier_consolidation/data/dualview_training_audit/aligned_shuffled_mlm_only_completed.json`
- budget_audit: `experiments/archive/frontier_consolidation/data/dualview_budget_substitution/budget_substitution_audit.json`
- parameter_geometry: `experiments/archive/frontier_consolidation/data/dualview_parameter_geometry/dualview_parameter_geometry.json`
- training_trajectory: `experiments/archive/frontier_consolidation/data/dualview_training_trajectory/aligned_vs_shuffled_training_trajectory.json`
- sentinel_stdout: `experiments/archive/frontier_consolidation/data/dualview_complete_panel_analysis/sentinel_stdout.log`
- sentinel_stderr: `experiments/archive/frontier_consolidation/data/dualview_complete_panel_analysis/sentinel_stderr.log`
- out_json: `experiments/archive/frontier_consolidation/data/dualview_complete_panel_analysis/dualview_panel_complete.json`
- out_md: `research/documents/frontier_consolidation/data/dualview_complete_panel_analysis/dualview_panel_complete.md`
