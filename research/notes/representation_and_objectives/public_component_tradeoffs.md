# babysteps public method reading public strict-small component tradeoffs
Analyzed `experiments/archive/representation_and_objectives/data/babylm2026_live_surface/strict_small_top30.json` against inherited COMPACT_EXPERIENCE `qwen_clean_aligned` (Overall 41.3443).
Strict-small leader: wwm_curriculum_simplification_40k at 41.80.

## Leader minus ours
| column | ours | leader | delta | overall contribution |
|---|---:|---:|---:|---:|
| BLiMP | 66.840 | 67.200 | +0.360 | +0.040 |
| Supplement | 62.840 | 56.010 | -6.830 | -0.759 |
| EWoK | 50.190 | 56.070 | +5.880 | +0.653 |
| Entity | 25.760 | 28.450 | +2.690 | +0.299 |
| COMPS | 51.780 | 53.570 | +1.790 | +0.199 |
| GlobalPIQA | 36.620 | 39.670 | +3.050 | +0.339 |
| SuperGLUE | 70.309 | 69.790 | -0.519 | -0.058 |
| Reading | 7.760 | 5.420 | -2.340 | -0.260 |
| AoA | 0.000 | 0.000 | +0.000 | +0.000 |

## Top30 empirical tradeoff clues
Knowledge-cluster mean (EWoK/Entity/COMPS/GlobalPIQA) correlation with Supplement among top30: 0.078.
Knowledge-cluster mean correlation with Reading among top30: 0.070.
Knowledge-cluster mean correlation with Overall among top30: 0.664.

Models in top30 with Supplement>=60 and Reading>=7:
- BabySteps_MurphysLaw-10M-mixed: Overall 40.86, cluster 42.29, cluster delta vs ours +1.20, Supp 63.93, Reading 7.67.
- instanton-hybrid-dialogue: Overall 40.43, cluster 39.31, cluster delta vs ours -1.78, Supp 61.06, Reading 7.14.
- FACTORIZED Natural Dense: Overall 39.80, cluster 39.08, cluster delta vs ours -2.01, Supp 60.63, Reading 7.87.

High-EWoK public models (EWoK>=55):
- wwm_curriculum_simplification_40k: Overall 41.80, EWoK 56.07, Supp 56.01, Entity 28.45, GPIQA 39.67, Reading 5.42.

## Research consequence
The public tradeoff picture reinforces the component gap analysis: the SOTA path is not to copy the leader wholesale, because we already have rare Supplement/Reading strength. The next H100 allocation should seek a narrow factual-breadth intervention on top of the protected clean-Qwen/developmental mixture, then evaluate whether EWoK/Entity/COMPS/GlobalPIQA move without erasing Supplement/Reading. Same-source semantic-view evidence remains scientifically informative but low-ceiling for the actual 41.8 gap.

Full JSON: `experiments/archive/representation_and_objectives/data/public_component_tradeoffs/public_component_tradeoffs.json`
