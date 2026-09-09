# support error conditioned probe — minfreq50 support-floor: CLOSED by real 70M/80M scores

Training and selected-evaluation results were available. Cheap official-compatible columns only (no SuperGLUE, no AoA); this is a screen, not a complete-endpoint judgment, but it is sufficient to decide continue/stop.

## Scores (minfreq50 init-matched, seed43022)

| exposure | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | mean7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70M | 65.50 | 58.65 | 49.52 | 26.54 | 52.02 | 37.12 | 8.545 | 42.5564 |
| 80M | 65.69 | 58.37 | 49.57 | 27.31 | 51.90 | 38.12 | 8.505 | 42.7807 |

## minfreq50 minus spatial repair route status token-mean reinvest reference

| exposure | Δmean7 | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70M | **-0.0521** | +0.11 | -0.66 | -0.95 | -0.44 | +0.20 | +1.57 | -0.195 |
| 80M | **-0.1679** | -0.42 | **-2.29** | **-1.44** | +0.25 | -0.03 | +2.54 | +0.215 |

## Reading (decisive interpretation)

- minfreq50 is **worse than the token-mean reinvest reference at both exposures**,
  and the gap **widens** with training (-0.052 -> -0.168). It does not mature
  toward parity.
- The only positive column is **GlobalPIQA (+2.54 at 80M)**, driven entirely by
  **GlobalPIQA_nonparallel +8.0** while **GlobalPIQA_parallel is -2.92**. Core
  language/evidence columns **Supplement -2.29 and EWoK -1.44** are damaged — the
  exact columns the legal endpoint must strengthen.
- This is the **same redistributive signature as global word-mean MLM** (related experiments): a representation/optimization change that trades away BLiMP/Supplement/
  EWoK for GlobalPIQA-nonparallel. Two independent interventions now share this
  failure mode.
- Against the wordmean failure anatomy calibration (needed cheap7 mean gain ~ +0.6972 to reach 41.8),
  minfreq50 is **negative**. It cannot close the SOTA gap even if it fully matured.

## Decision

**Close minfreq50 support-floor.** No 100M continuation, no seed change, no
relabel, no LR retune (timely-termination rule). The spatial repair route status token-mean reinvest
compact-view trajectory remains the strongest legal mechanism (80M mean7 42.9486,
+1.35 over clean), but its own 80M->100M cheap7 maturation was only +0.057, so
neither minfreq50 nor plain continuation of the same recipe reaches 41.8.

## Pattern that should now guide route selection

Both legal-coordinate interventions that changed **credit scale/direction**
(word-mean) or **segmentation/support** (minfreq50) redistribute toward
GlobalPIQA-nonparallel and away from BLiMP/Supplement/EWoK. legal40k
representation package showed a complementary tradeoff (better language columns,
worse GlobalPIQA/Entity/SuperGLUE). The legal deficit is NOT closed by broad
representation/segmentation reshuffles; it needs an intervention that adds genuine
learnable structure on the load-bearing syntax/QA-congruence/EWoK-dynamics columns
WITHOUT the GlobalPIQA-nonparallel-for-language trade.

The consistency decoy diagnostic and route correction-corrected consistency evidence removed the "absent pair-specificity"
rationale: true source<->rewrite identity is already learned almost perfectly by
80M, so a naive positive-only consistency loss has little headroom. do not preassign consistency; reopen the representation/optimization
choice around a genuinely UNSATURATED quantity with a credible path to broad
transfer, incorporating real sequence result before the next expensive run.
