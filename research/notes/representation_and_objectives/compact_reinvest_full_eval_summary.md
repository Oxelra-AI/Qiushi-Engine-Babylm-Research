# compact reinvest full eval summary compact_view_reinvest full evaluation summary

Summary JSON: `experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/compact_reinvest_full_eval_summary.json`
Per-target dir: `experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/per_target`

## Completed rows

| target | Overall | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | SuperGLUE | Reading | AoA | submit-ready | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| compact_view_reinvest | 42.0868 | 66.8700 | 63.2800 | 53.6700 | 27.7500 | 51.9700 | 35.6200 | 71.3811 | 8.2400 | 0.0000 | True |  |

## Key contrasts

- **compact_view_reinvest_minus_visible_leader**: BLiMP=-0.33, Supplement=7.27, EWoK=-2.4, Entity=-0.7, COMPS=-1.6, SuperGLUE=1.591071, GlobalPIQA=-4.05, Reading=2.82, AoA=0.0, Overall=0.286786, NLP_average=None, Human_like_average=None
- **compact_view_reinvest_minus_compact_experience_clean_qwen**: BLiMP=0.03, Supplement=0.44, EWoK=3.48, Entity=1.99, COMPS=0.19, SuperGLUE=1.072455, GlobalPIQA=-1.0, Reading=0.48, AoA=0.0, Overall=0.742495, NLP_average=None, Human_like_average=None
- **compact_view_reinvest_full_minus_core_fast_overlap**: BLiMP=-0.06, Supplement=-2.32, EWoK=2.12, Entity=0.45, COMPS=-0.21, SuperGLUE=None, GlobalPIQA=0.485, Reading=-0.01, AoA=None, Overall=None, NLP_average=None, Human_like_average=None

## Interpretation

{
  "full_overall": 42.086785719138156,
  "submit_ready_overall": true,
  "aoa_status": "official_aoa_done",
  "seven_column_sum": 307.4,
  "superglue_plus_aoa": 71.38107147224343,
  "passes_visible_41p8": true,
  "passes_round_42p0": true,
  "scientific_reading": "Full compact_view_reinvest score should be read as the SOTA-facing endpoint; core view/repeat remains the mechanism comparison."
}
