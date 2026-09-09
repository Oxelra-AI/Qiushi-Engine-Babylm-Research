# frequency concentration probe HALF_VIEW curve probe
Created: 2026-09-06T12:51:58Z

HALF_VIEW was trained after the hash-mixed arm to separate a half-dose restatement curve from active exact/restatement composition. It uses the same deterministic half assignment as hash-mix: the rewrite-assigned half receives local source+rewrite pairs; the other selected-source slots receive neutral no-exact companions. This note scores C/R/V/HM/HV with the same records so the compact designed-register curve and the off-axis hash-mixed placement are directly comparable.

## Records and arms

- Per model records: copy 4000, compact T/U 11784, compact N 5892, Wikipedia 10800.
- Checkpoints: chck_80M, chck_90M, chck_100M.
- C is zero designed compact-relation dose; HV is 16.7K local compact restatements with no exact companion; V is 33.3K local compact restatements; HM is off-axis because it has the same rewrite half as HV plus 16.6K local exact recurrence companions.

## Late-checkpoint absolute source-use readouts

Compact values use `A_T=N−T`, `A_U=N−U`, and `G=U−T`; positive `A_T` means the true source helps more relative to the neutral ordinary source.

| role | compact nonoverlap A_T | compact nonoverlap A_U | compact nonoverlap G | compact overlap A_T | natural-copy gain |
|---|---:|---:|---:|---:|---:|
| C | +1.5267 | +0.3054 | +1.2213 | +6.4843 | +3.8990 |
| HV | +1.5444 | +0.1548 | +1.3896 | +6.5568 | +3.8776 |
| V | +2.2371 | +0.2943 | +1.9428 | +7.0712 | +4.3037 |
| HM | +1.8723 | +0.3852 | +1.4871 | +6.8838 | +4.4247 |
| R | +0.8439 | +0.3406 | +0.5033 | +6.3097 | +4.4211 |

## Dose and off-axis contrasts

| contrast | compact nonoverlap ΔA_T | compact nonoverlap ΔA_U | compact nonoverlap ΔG | compact overlap ΔA_T | Wikipedia overlap Δ(N−T) | Wikipedia nonoverlap Δ(N−T) | natural-copy Δgain |
|---|---:|---:|---:|---:|---:|---:|---:|
| HVminusC | +0.0177 | -0.1505 | +0.1683 | +0.0725 | +0.0324 ± 0.0484 | -0.0407 ± 0.0339 | -0.0215 |
| VminusC | +0.7104 | -0.0110 | +0.7214 | +0.5870 | +0.2978 ± 0.0576 | +0.0485 ± 0.0356 | +0.4047 |
| HMminusC | +0.3456 | +0.0798 | +0.2658 | +0.3996 | +0.3243 ± 0.0536 | -0.2136 ± 0.0372 | +0.5257 |
| HVminusV | -0.6926 | -0.1395 | -0.5532 | -0.5145 | -0.2654 ± 0.0505 | -0.0893 ± 0.0327 | -0.4261 |
| HMminusHV | +0.3279 | +0.2304 | +0.0975 | +0.3271 | +0.2918 ± 0.0467 | -0.1728 ± 0.0332 | +0.5471 |
| RminusC | -0.6828 | +0.0353 | -0.7180 | -0.1746 | +0.1955 ± 0.0523 | -0.2073 ± 0.0367 | +0.5221 |

## Scientific reading

On the designed FineWeb-register compact nonoverlap readout, HALF_VIEW is near CLEAN rather than half-way to full VIEW: HV−C ΔA_T is +0.0177, while V−C is +0.7104. The half-dose/no-exact result therefore does not support a concave saturation curve. It is more consistent with a threshold-like or density-dependent conversion in which this particular 16.7K local-restatement subset is insufficient to install the compact true-source routine measured by the T/U/N probe.

Hash-mix is above the half-dose no-exact point on compact nonoverlap A_T (HM−HV ΔA_T = +0.3279), but still below full VIEW (HM−V ΔA_T = −0.3647). Thus the earlier HM shortfall relative to VIEW cannot be explained by half-dose alone, and it also cannot be described as exact recurrence suppressing the half-dose signal relative to HV. The sound reading is that adding local exact-recurrence companions changes the off-axis composition: it adds copy behavior, raises compact A_T above HALF_VIEW, and introduces the source-absent Wikipedia cost. It does not reproduce full restatement competence.

The Wikipedia readout must be kept separate from the compact curve because it is near the inherited Qwen/SimpleWiki register rather than the designed compact register. The paper-level principle should therefore distinguish reach: corresponding restatement competence is strongest in the format/register practiced, while exact recurrence liability has traveled across compact, Wikipedia-substitution, and causal next-token readouts.

## Output files

- `experiments/archive/relation_learning/data/half_view_curve_probe/compact_TUN_late_roles.csv`
- `experiments/archive/relation_learning/data/half_view_curve_probe/compact_TUN_late_contrasts.csv`
- `experiments/archive/relation_learning/data/half_view_curve_probe/wikipedia_late_contrasts.csv`
- `experiments/archive/relation_learning/data/half_view_curve_probe/copy_late_contrasts.csv`
