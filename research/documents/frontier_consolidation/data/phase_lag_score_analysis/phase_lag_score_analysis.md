# earlier analysis phase-lag score analysis

Scored intervals: 11. Source atlas: `experiments/archive/frontier_consolidation/data/epoch_phase_exposure_atlas/epoch_phase_exposure_atlas.json`.

## Compact-rich versus compact-poor intervals

Threshold `family_frac__compact_pair_block >= 0.1`:
- high: n=3, intervals=['chck_70M→chck_72M', 'chck_80M→chck_82M', 'chck_90M→chck_92M'], mean Δcheap7=0.16999999999999935, mean Δcheap6_noGP=0.06416666666666988, mean Δrelation_state=-0.00666666666666534
- low: n=8, intervals=['chck_72M→chck_74M', 'chck_74M→chck_76M', 'chck_76M→chck_78M', 'chck_78M→chck_80M', 'chck_82M→chck_84M', 'chck_84M→chck_86M', 'chck_86M→chck_88M', 'chck_88M→chck_90M'], mean Δcheap7=0.023392857142857437, mean Δcheap6_noGP=0.07854166666666629, mean Δrelation_state=0.05687499999999979

Threshold `family_frac__compact_pair_block >= 0.01`:
- high: n=5, intervals=['chck_70M→chck_72M', 'chck_78M→chck_80M', 'chck_80M→chck_82M', 'chck_88M→chck_90M', 'chck_90M→chck_92M'], mean Δcheap7=0.036285714285715184, mean Δcheap6_noGP=0.011333333333335815, mean Δrelation_state=-0.06699999999999733
- low: n=6, intervals=['chck_72M→chck_74M', 'chck_74M→chck_76M', 'chck_76M→chck_78M', 'chck_82M→chck_84M', 'chck_84M→chck_86M', 'chck_86M→chck_88M'], mean Δcheap7=0.08595238095238027, mean Δcheap6_noGP=0.12736111111111015, mean Δrelation_state=0.1283333333333315

## Local 78M--92M sequence

| interval | compact frac | qwen frac | official frac | Δcheap7 | Δcheap6_noGP | Δrelation_state |
|---|---:|---:|---:|---:|---:|---:|
| chck_78M→chck_80M | 1.70% | 16.94% | 81.36% | 0.10999999999999943 | 0.045000000000001705 | 0.32500000000000284 |
| chck_80M→chck_82M | 19.67% | 13.95% | 66.38% | 0.14642857142857224 | 0.25833333333333286 | 0.4650000000000034 |
| chck_82M→chck_84M | 0.00% | 17.49% | 82.51% | 0.16499999999999915 | 0.10249999999999915 | 0.14000000000000057 |
| chck_84M→chck_86M | 0.00% | 17.53% | 82.47% | -0.35285714285714675 | -0.08083333333333798 | 0.03499999999999659 |
| chck_86M→chck_88M | 0.00% | 16.94% | 83.06% | -0.060000000000002274 | 0.013333333333335418 | -0.09000000000000341 |
| chck_88M→chck_90M | 1.93% | 16.87% | 81.20% | -0.4385714285714215 | -0.1808333333333323 | -0.6399999999999935 |
| chck_90M→chck_92M | 19.45% | 13.98% | 66.57% | 0.09499999999999886 | -0.05333333333332746 | -0.23499999999999943 |

## Reading

The compact block is deterministically at the start of each 10M cycle, and compact-rich intervals sometimes precede gains, but this small late-window sequence does not support a simple recent-source-slice explanation of the 84M peak: the 82M→84M gain itself occurs in a no-compact interval, other no-compact intervals can be positive or negative, and compact-rich intervals are not uniformly broad-positive. Treat source phase as geometry context, not causal evidence.

Figure: `experiments/archive/frontier_consolidation/data/phase_lag_score_analysis/reference_late_phase_cheap7.png`

JSON: `experiments/archive/frontier_consolidation/data/phase_lag_score_analysis/phase_lag_score_analysis.json`
