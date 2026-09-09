# compact core full eval projection thresholds density compact full-evaluation summary

Summary JSON: `experiments/archive/frontier_consolidation/data/density_full_eval/density_full_eval_summary.json`
Per-target dir: `experiments/archive/frontier_consolidation/data/density_full_eval/per_target`

## Completed target rows

| target | Overall | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | SuperGLUE | Reading | AoA | submit-ready | missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| compact_view_core | 39.9877 | 67.1500 | 61.8500 | 51.2600 | 27.8500 | 52.1800 | 35.1350 | 68.9012 | 8.2500 | -12.6873 | True |  |

## Key contrasts

- **compact_view_core_minus_compact_experience_clean_qwen**: BLiMP=0.31, Supplement=-0.99, EWoK=1.07, Entity=2.09, COMPS=0.4, SuperGLUE=-1.407463, GlobalPIQA=-1.485, Reading=0.49, AoA=-12.687286, Overall=-1.356639, NLP_average=-0.00178, Human_like_average=-6.098643
- **compact_view_core_minus_visible_leader**: BLiMP=-0.05, Supplement=5.84, EWoK=-4.81, Entity=-0.6, COMPS=-1.39, SuperGLUE=-0.888847, GlobalPIQA=-4.535, Reading=2.83, AoA=-12.687286, Overall=-1.812348, NLP_average=None, Human_like_average=None

## Reading rule

The compact-core mechanism is judged first by `compact_view_core_minus_compact_repeat_core` and then by full comparison with the inherited COMPACT_EXPERIENCE clean-Qwen result and visible leader. Reinvestment, if present, is an extension rather than a verdict on the compact-core mechanism.
