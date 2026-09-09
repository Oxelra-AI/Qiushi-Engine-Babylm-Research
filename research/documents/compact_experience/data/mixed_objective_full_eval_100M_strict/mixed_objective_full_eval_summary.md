# mixed objective measurement and mechanism plan mixed-objective true-100M full evaluation

Output root: `experiments/archive/compact_experience/data/mixed_objective_full_eval_100M_strict`

Clean-Qwen reference recomputed from `experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned.json`: Overall 41.344290664796, equal7 43.112857142857.
Visible leader reference from `experiments/archive/compact_experience/data/babylm2026_surface/strict_small_top20.json`: wwm_curriculum_simplification_40k displayed Overall 41.8 (rounded display).

AoA is used only as a terminal aggregate readout from the frozen complete checkpoint ladder. Both causal fractions were fixed before evaluation; no endpoint or fraction was selected from AoA.

SuperGLUE in this local scorer is the unweighted mean of validation accuracies over BoolQ, MultiRC, RTE, WSC, MRPC, QQP, and MNLI, matching `calculate_results_from_pred.py::_calculate_glue_results`; `print_results_table.py` has a separate markdown-table metric map that uses F1 for MRPC/QQP and is recorded as a remaining server-equivalence caveat.

| target | Overall | Δ clean | Δ visible leader | equal7 | Δ clean equal7 | SuperGLUE | AoA raw | AoA lb | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mixed_causal15_100M | 41.20913349834475 | -0.13515716645098053 | -0.590866501655249 | 42.978571428571435 | -0.1342857142857099 | 70.0322014851027 | 0.0 | 0.0 | 66.66 | 59.9 | 48.57 | 26.01 | 51.88 | 39.105000000000004 | 8.725 | yes |
| mixed_causal50_100M | 40.099923146601064 | -1.2443675181946645 | -1.700076853398933 | 41.924285714285716 | -1.1885714285714286 | 67.42930831940954 | 0.0 | 0.0 | 66.25 | 59.3 | 50.77 | 22.44 | 51.9 | 34.65 | 8.16 | yes |

## Interpretation scope

Scores apply first to the full mixed-objective package: causal visibility, uncorrupted inputs, dense next-token targets, fewer MLM updates, and changed gradient statistics. Target density and objective complementarity require a separate matched control before a deeper mechanism assignment.
