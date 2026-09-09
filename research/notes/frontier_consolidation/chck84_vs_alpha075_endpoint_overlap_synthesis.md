# chck84 vs alpha075 endpoint overlap synthesis `chck_84M` versus coherent86 alpha0.75 endpoint overlap

This CPU/file-only step compared two already-existing endpoint functions on the seven cheap columns and raw classification items:

- `chck_84M`: ordinary checkpoint on the legal scale1.75 seed43022 residual-adapter training trajectory, 84,028,405 charged words. partial deberta grid and endpoint branch identity audit: model SHA256 `2217917c687faf4de26ef6f381be3048d0bd66b2025c06382683c24d78e8d8c9`, trusted `AdapterDebertaV2ForMaskedLM`, 35,463,008 parameters.
- coherent86 alpha0.75: frozen chck82 slow anchor plus coherent private replay with inference-time private scale 0.75. Truthful carrier SHA256 `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`; model weights SHA256 `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`.

The analysis reuses `chck84_item_movement_analyzer.py` scoring definitions and parses the alpha0.75 truthful carrier directly. Outputs are in `data/chck84_vs_alpha075_item_overlap/`.

## Aggregate cheap-task comparison

| endpoint | cheap7 | cheap6 no GlobalPIQA | cheap5 no GlobalPIQA/Reading | EWoK+Entity | BLiMP+COMPS |
|---|---:|---:|---:|---:|---:|
| chck82 | 43.959634 | 45.023294 | 52.397953 | 39.184748 | 60.341230 |
| chck84 | 44.123626 | 45.124004 | 52.517804 | 39.324289 | 60.228378 |
| alpha0.75 | 44.181208 | 45.117558 | 52.508070 | 39.170918 | 60.281573 |

Alpha0.75 beats chck84 by only +0.0576 cheap7. That advantage disappears when GlobalPIQA is removed: alpha0.75 is -0.0064 on cheap6-no-GP and -0.0097 on cheap5-no-GP/Reading. Relation/state favors the ordinary trained checkpoint: alpha0.75 is -0.1534 on EWoK+Entity relative to chck84.

Per-column alpha0.75 minus chck84:

- BLiMP +0.2648
- Supplement +0.1517
- EWoK -0.0539
- Entity -0.2529
- COMPS -0.1584
- GlobalPIQA +0.4417
- Reading +0.0100

## Raw item overlap

Across 170,722 classification items, chck84 and alpha0.75 disagree on 8,588 items. Relative to chck82:

- chck84 gains 4,443 raw items and loses 4,501.
- alpha0.75 gains 2,332 raw items and loses 2,418.
- shared gains are only 1,276; gain Jaccard 0.232.
- shared losses are 1,277; loss Jaccard about 0.23.
- alpha0.75-correct/chck84-wrong items = 4,280; chck84-correct/alpha0.75-wrong = 4,308; net alpha0.75 minus chck84 raw items = -28.

The two functions therefore reach similar aggregate cheap scores through substantially different item allocations, not through one endpoint strictly containing the other.

## Scientific reading

`chck_84M` is the cleaner endpoint branch for the ordinary training trajectory: it improves chck82 broadly but shallowly and preserves more relation/state signal than alpha0.75. Alpha0.75 remains the numerically stronger local cheap7/Overall(AoA0) carrier before chck84 SuperGLUE is known, but its advantage over chck84 is small and GlobalPIQA/BLiMP/Supplement-tilted; it does not dominate the ordinary checkpoint once volatile columns are removed.

This should shape the pending endpoint decision without reopening private-scale tuning. If `chck_84M` SuperGLUE clears the earlier analysis thresholds (>=68.6173 to beat chck82, >=69.135 for Overall 42.0, >=70.2242 to match alpha0.75 projected Overall), it becomes a strong ordinary-training endpoint branch. If it falls below those thresholds, alpha0.75 remains the stronger local endpoint carrier, but still as endpoint engineering rather than the desired transferable learning principle.

The transferable-science question remains separate. `chck_84M` is still one seed/mask trajectory; the already-trained scale1.75 seed43122 common-grid scoring is the important robustness test for whether the late competence-allocation phase and its family profile survive a different init + mask stream. No compact ordered/scrambled training should start before the delivered grid evidence and directional-fork scores are read.
