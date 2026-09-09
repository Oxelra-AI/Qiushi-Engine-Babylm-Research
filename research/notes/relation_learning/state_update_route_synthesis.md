# state update route synthesis synthesis: state-update replacement route and next screens
This note integrates the earlier analysis/052 held-out T/U/N margin readout with the earlier analysis official Entity deployment strata. It is a research-facing route note; it does not run new training or official evaluation.
## Main route judgment
The plain-use state-update arm should not be advanced as a direct practical SOTA route. It replaced inherited `qwen_pair_packed` ALN companions and trained for only `99,909,920` words, and its strongest two-seed signal is not entity-gated state revision. The powered template margins move target-update use flat/negative while raising source-state retention in co-established distractor packets. Entity strata then show seed-basin-dependent policy overwrite rather than additive competence.
## Entity side-by-side: arm pulls different bases toward a shallower policy
At `chck_86M`, the intervention compresses large between-seed differences in relevant-update strata but does so by damaging the seed that had more relevant-update competence.
| group | n | base 43022 -> int | base 43122 -> int | delta 43022 | delta 43122 | seed-range change |
|---|---:|---:|---:|---:|---:|---:|
| `ALL` | 6780 | 28.20 -> 25.84 | 26.05 -> 26.31 | -2.360 | +0.265 | -1.681 |
| `rel_eq0` | 1541 | 37.51 -> 39.07 | 40.69 -> 38.55 | +1.557 | -2.141 | -2.661 |
| `rel_eq0_irrelevant_ops_gt0` | 1237 | 38.00 -> 38.97 | 40.58 -> 38.32 | +0.970 | -2.264 | -1.940 |
| `rel_ge1` | 5239 | 25.46 -> 21.95 | 21.74 -> 22.71 | -3.512 | +0.973 | -2.959 |
| `rel_ge3` | 2650 | 30.57 -> 25.70 | 25.85 -> 25.85 | -4.868 | +0.000 | -4.566 |
| `stale_available_not_gold` | 1221 | 27.76 -> 23.01 | 23.42 -> 24.24 | -4.750 | +0.819 | -3.112 |

Negative seed-range change means the two seeds became more similar after intervention; for `rel_ge1` and `rel_ge3` this convergence is not competence gain, because seed43022 drops sharply and seed43122 is only flat/small-positive. The same qualitative pattern persists at final, especially for `rel_ge3` (seed43022 28.49 -> 25.74; seed43122 26.45 -> 26.34).
## Margin evidence
| seed | ck | packet | n | T update | U retention | T retention | Tupdate+Uret | Tret+Uret | T-N update | T-U update | T-N retention |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 43022 | chck_86M | UNCHANGED_DISTRACTOR_USE | 200 | NA | +0.096 | +0.065 | NA | +0.081 | NA | NA | +0.136 |
| 43022 | chck_86M | UPDATED_USE | 1125 | -0.031 | +0.050 | NA | +0.009 | NA | -0.110 | -0.122 | NA |
| 43022 | final | UNCHANGED_DISTRACTOR_USE | 200 | NA | +0.066 | +0.029 | NA | +0.047 | NA | NA | +0.147 |
| 43022 | final | UPDATED_USE | 1125 | -0.017 | +0.034 | NA | +0.009 | NA | -0.152 | -0.177 | NA |
| 43122 | chck_86M | UNCHANGED_DISTRACTOR_USE | 200 | NA | +0.025 | +0.117 | NA | +0.071 | NA | NA | +0.262 |
| 43122 | chck_86M | UPDATED_USE | 1125 | -0.115 | +0.021 | NA | -0.047 | NA | -0.365 | -0.317 | NA |
| 43122 | final | UNCHANGED_DISTRACTOR_USE | 200 | NA | +0.030 | +0.131 | NA | +0.080 | NA | NA | +0.260 |
| 43122 | final | UPDATED_USE | 1125 | -0.086 | +0.002 | NA | -0.042 | NA | -0.295 | -0.248 | NA |

The neutral-anchored terms reinforce the caution: relation-specific update use (`T-N update`) is negative in both seeds for UPDATED packets, while relation-specific retention (`T-N retention`) is positive on the co-established distractor packets. A scalar average can hide the tradeoff.
## Consequences for future practical arms
1. Do not spend coherent replay, full Overall, or another two-seed 100M run on a larger version of this replacement packet family. A distractor-heavy version would make the same coarse source-retention relation denser and risks further damage to legitimate target updates.
2. Entity aggregate should not be the primary screen for one-percent relation interventions. Its between-seed dispersion is larger than the expected companion-dose effect. It remains a deployment readout after a low-noise source-use or broad-fit signal exists.
3. The best evidence-backed practical route is to preserve existing ALN and add more correctly aligned local restatement from filler. Existing ALN improves ordinary held-out MLM at two seeds (about -0.0125/-0.0128 nats) and has large in-family source-recurring T/U/N margins; SHUF shows correctness of correspondence matters; split controls show locality matters.
4. The best mechanism-science route is a same-source contrastive entity-binding packet family, added from filler rather than replacing ALN, where the only sufficient predictor is which entity receives the update. Its first result should be the vector of neutral-adjusted update and retention margins, not Entity aggregate.

## Output tables
- Entity side-by-side CSV: `experiments/archive/relation_learning/data/route_synthesis/entity_side_by_side_key_groups.csv`
- Margin selected metrics CSV: `experiments/archive/relation_learning/data/route_synthesis/state_margin_selected_metrics.csv`
- Raw base recency selected CSV: `experiments/archive/relation_learning/data/route_synthesis/raw_base_recency_selected.csv`
