# scale1p75 item mechanism and cross route tradeoff score-vector tradeoff analysis

CPU-only analysis of existing official-compatible cheap-column summaries. Deltas are against matched spatial repair route status legal references at the same exposure when available.

## Records

| label | family | exposure | Δcheap7 | ΔBLiMP | ΔSupp | ΔEWoK | ΔEntity | ΔCOMPS | ΔGP | ΔRead | strong gains | strong losses |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| RTD/GDES lambda1 | objective | 20M | +0.1107 | +0.47 | +1.49 | -0.66 | +0.70 | -0.02 | -0.97 | -0.23 | Supplement +1.49 | - |
| Muon wdmatched lr0.008 | optimizer | 20M | +1.3600 | +1.46 | +2.65 | -0.03 | +2.25 | +0.19 | +4.41 | -1.41 | BLiMP +1.46, Supplement +2.65, Entity +2.25, GlobalPIQA +4.41 | Reading -1.41 |
| LAMB lr0.007 | optimizer | 20M | -2.4979 | -5.95 | -5.50 | -1.79 | -1.51 | -0.26 | -0.95 | -1.53 | - | BLiMP -5.95, Supplement -5.50, EWoK -1.79, Entity -1.51, Reading -1.53 |
| Adapter scale1.75 train | architecture | 20M | +0.6421 | -0.08 | +0.82 | +1.49 | -0.43 | -0.29 | +2.95 | +0.03 | EWoK +1.49, GlobalPIQA +2.95 | - |
| Adapter scale2.00 train | architecture | 20M | +0.0593 | +0.23 | +0.76 | -0.19 | -1.20 | +0.20 | +0.92 | -0.31 | - | Entity -1.20 |
| Minfreq50 support-floor | tokenizer | 80M | -0.1679 | -0.42 | -2.29 | -1.44 | +0.25 | -0.03 | +2.54 | +0.22 | GlobalPIQA +2.54 | Supplement -2.29, EWoK -1.44 |
| Word-mean MLM | objective | 80M | -0.2457 | -1.56 | -1.71 | -1.81 | -1.15 | +0.51 | +4.53 | -0.52 | GlobalPIQA +4.53 | BLiMP -1.56, Supplement -1.71, EWoK -1.81, Entity -1.15 |
| Strict content-innovation WWM | objective | 80M | -1.0529 | -0.46 | -2.72 | -1.85 | +0.94 | -0.75 | -2.45 | -0.08 | - | Supplement -2.72, EWoK -1.85, GlobalPIQA -2.45 |
| Muon wdmatched lr0.008 | optimizer | 80M | -0.1607 | +0.24 | -0.23 | -0.29 | -2.89 | -0.41 | +3.48 | -1.03 | GlobalPIQA +3.48 | Entity -2.89, Reading -1.03 |
| Muon20->AdamW | optimizer_switch | 80M | -0.1436 | +0.27 | +1.32 | -0.93 | -3.47 | +0.30 | +2.03 | -0.52 | Supplement +1.32, GlobalPIQA +2.03 | Entity -3.47 |
| Muon40->AdamW | optimizer_switch | 80M | -0.5964 | -0.13 | +1.02 | -0.79 | -1.84 | -0.27 | -1.45 | -0.72 | Supplement +1.02 | Entity -1.84, GlobalPIQA -1.45 |
| Expanded FineWeb compact-view | data | 100M | +0.1757 | +1.00 | -2.31 | -0.14 | +0.96 | -0.17 | +2.57 | -0.68 | GlobalPIQA +2.57 | Supplement -2.31 |
| Expanded FineWeb source-breadth | data | 100M | -0.3686 | +2.02 | -1.48 | +0.11 | -3.52 | -0.68 | +1.00 | -0.03 | BLiMP +2.02, GlobalPIQA +1.00 | Supplement -1.48, Entity -3.52 |
| Adapter scale1.75 train | architecture | 50M | +0.4114 | +1.84 | -0.04 | -0.73 | +2.79 | +0.73 | -1.47 | -0.23 | BLiMP +1.84, Entity +2.79 | GlobalPIQA -1.47 |

## Mature summary

- n: 8
- mean_delta: {'BLiMP': 0.11982500000000051, 'Supplement': -1.0489249999999988, 'EWoK': -0.8932999999999991, 'Entity': -1.3401999999999994, 'COMPS': -0.187075000000001, 'GlobalPIQA': 1.532350000000001, 'Reading': -0.42267499999999947}
- mean_cheap7_delta: -0.3199999999999994
- count_with_globalpiqa_gain_ge_1: 6
- count_with_entity_loss_le_m1: 5
- count_with_supp_or_ewok_loss_le_m1: 5
- count_all_columns_positive: 0
- best_mature_record: Expanded FineWeb compact-view 100M
- best_mature_delta: 0.17571428571428527

## Principal components

- PC1 variance_fraction=0.429, loadings: BLiMP:+0.553, Supplement:+0.665, EWoK:+0.210, Entity:+0.104, COMPS:+0.047, GlobalPIQA:+0.441, Reading:+0.027
- PC2 variance_fraction=0.280, loadings: BLiMP:-0.276, Supplement:-0.189, EWoK:+0.020, Entity:-0.557, COMPS:-0.013, GlobalPIQA:+0.757, Reading:-0.063
- PC3 variance_fraction=0.174, loadings: BLiMP:-0.134, Supplement:-0.308, EWoK:-0.067, Entity:+0.809, COMPS:+0.075, GlobalPIQA:+0.469, Reading:-0.046

## Column correlation

| col | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | +1.00 | +0.66 | +0.53 | +0.21 | +0.11 | +0.16 | +0.42 |
| Supplement | +0.66 | +1.00 | +0.55 | +0.09 | +0.34 | +0.22 | +0.07 |
| EWoK | +0.53 | +0.55 | +1.00 | -0.02 | -0.12 | +0.33 | +0.14 |
| Entity | +0.21 | +0.09 | -0.02 | +1.00 | +0.34 | -0.14 | +0.05 |
| COMPS | +0.11 | +0.34 | -0.12 | +0.34 | +1.00 | +0.24 | -0.07 |
| GlobalPIQA | +0.16 | +0.22 | +0.33 | -0.14 | +0.24 | +1.00 | -0.17 |
| Reading | +0.42 | +0.07 | +0.14 | +0.05 | -0.07 | -0.17 | +1.00 |

## Interpretation

- Across 8 mature legal-coordinate records, the mean cheap7 delta is -0.3200; none has all seven columns positive (0).
- GlobalPIQA gains >= +1 occur in 6 mature records, but Entity losses <= -1 occur in 5 and Supplement/EWoK losses <= -1 occur in 5; the pattern is not broad expansion.
- Best mature cheap7 movement in this set is Expanded FineWeb compact-view 100M at +0.1757, still far below the +0.697 cheap7 equivalent needed if SuperGLUE/AoA stay flat.
- PC1 explains 0.429 of delta-vector variance; oriented positive with GlobalPIQA, strongest positive loadings [('Supplement', 0.6653590781728816), ('BLiMP', 0.5526785985040139), ('GlobalPIQA', 0.4407244030535238)], strongest negative loadings [('Reading', 0.027376507773209872), ('COMPS', 0.04716672166052529), ('Entity', 0.10356509622843599)].
- Current scale1.75 at 50M has cheap7 delta +0.4114, with gains {'BLiMP': 1.8400000000000034, 'Entity': 2.789999999999999} and losses {'GlobalPIQA': -1.4749999999999943}; it is a mixed positive point, not an already broad mature repair.

## Sources
- RTD/GDES lambda1 20M: `experiments/archive/frontier_consolidation/data/mlm_rtd_20M_summary/mlm_rtd_20M_summary.json`
- Muon wdmatched lr0.008 20M: `experiments/archive/frontier_consolidation/data/muon_wdmatched_20M_eval/muon_wdmatched_20M_comparison.json`
- LAMB lr0.007 20M: `experiments/archive/frontier_consolidation/data/lamb_20M_eval_merged/lamb_20M_eval_merged.json`
- Adapter scale1.75 train 20M: `experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_merged/scaled_train_20M_merged.json`
- Adapter scale2.00 train 20M: `experiments/archive/frontier_consolidation/data/scaled_train_20M_eval_merged/scaled_train_20M_merged.json`
- Minfreq50 support-floor 80M: `experiments/archive/frontier_consolidation/data/minfreq50_decision/minfreq50_decision.json`
- Word-mean MLM 80M: `experiments/archive/frontier_consolidation/data/wordmean_70_80M_eval/per_target/wordmean_mlm_seed43022_80M.json`
- Strict content-innovation WWM 80M: `experiments/archive/frontier_consolidation/data/strict_innovation_70_80M_eval/per_target/strict_content_innovation_seed43022_80M.json`
- Muon wdmatched lr0.008 80M: `experiments/archive/frontier_consolidation/data/muon_wdmatched_mature_eval/muon_wdmatched_mature_comparison.json`
- Muon20->AdamW 80M: `experiments/archive/frontier_consolidation/data/muon_switch_actual_merged/actual_switch_eval_merged.json`
- Muon40->AdamW 80M: `experiments/archive/frontier_consolidation/data/muon_switch_actual_merged/actual_switch_eval_merged.json`
- Expanded FineWeb compact-view 100M: `experiments/archive/frontier_consolidation/data/fw_absolute_decision/fw_absolute_decision.json`
- Expanded FineWeb source-breadth 100M: `experiments/archive/frontier_consolidation/data/fw_absolute_decision/fw_absolute_decision.json`
- Adapter scale1.75 train 50M: `experiments/archive/frontier_consolidation/data/scale1p75_trajectory/scale1p75_trajectory.json`
