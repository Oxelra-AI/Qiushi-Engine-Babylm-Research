# held fitted direction test — parent-anchored continuation result so far

## Scientific question

Starting from exact coherent86 alpha0.75, clean d component ablation ordinary continuation kept the successful private tensors but regularized the current private-on model toward the private-off 82M carrier on neutral examples. That preservation reference may oppose the function that produced the inherited SOTA. held fitted direction test therefore changed only the neutral KL target to a frozen copy of coherent86 private-on and added a no-neutral-KL bracket.

The test asks whether additional legal suffix experience can be accumulated onto an already useful private correction when preservation is centered on the parent function, rather than pulling back toward the carrier or leaving the adapter unconstrained.

## Implemented arms

All arms start from `models/frontier`, use compact-view-reinvest rows after `skip_rows=556791`, train existing private adapters only, use the same WWM masking seed and cumulative cosine schedule offset, and stop at the legal 100M cap.

| arm | neutral term | run |
|---|---|---|
| carrier-anchor standard (clean d component ablation) | `KL(private-off carrier || current private-on)` | `training/runs/coherent86_continue_standard_seed43023` |
| parent-anchor standard (held fitted direction test) | `KL(frozen coherent86 private-on || current private-on)` | `training/runs/coherent86_continue_parent_anchor_seed43023` |
| no-KL bracket (held fitted direction test) | none | `training/runs/coherent86_continue_no_kl_seed43023` |

independent_review verification (`data/external/independent_review01_verifier1_integration.md`) found no evidence of KL reversal, teacher leakage, private-state leakage, masking-RNG perturbation, or legal-accounting mismatch relative to clean d component ablation. It correctly warned that held fitted direction test tests parent-on reference versus carrier-off reference, and that no-KL is needed to distinguish beneficial anchoring from simply removing carrier shrinkage.

## Integrity and mechanistic checks

`data/integrity_checks/integrity_checks.md` records:

- Parent SHA256 `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`.
- 48 private-adapter tensor keys, 0 missing private keys, 0 unexpected keys.
- Executed private scales after load: eight layers all `0.75`.
- First-update pairing between clean d component ablation standard, held fitted direction test parent-anchor, and held fitted direction test no-KL is exact for source rows, macro words, tail words, total words, target count, schedule index, LR, and main loss.
- First neutral loss differs as intended: clean d component ablation carrier anchor `0.001525`, parent anchor approximately `0`, no-KL `0`.

`data/anchor_gradient_probe/anchor_gradient_probe.md` shows that at the exact parent and same fixed suffix microbatch, private-gradient L2 is:

| loss | value | private grad L2 |
|---|---:|---:|
| WWM main | 2.785809 | 0.561155 |
| carrier-off KL | 0.0017507 | 0.010025 |
| parent-on KL | ~0 | 0.000000466 |

Thus the carrier-off KL already exerts a small but real shrinkage force at the coherent86 parent, while parent-on anchoring is centered on the inherited function.

CPU geometry confirms parent-on anchoring reduces final adapter displacement relative to no-KL, though modestly:

| arm | 87M relative Δ | final relative Δ | final cos(child,parent) |
|---|---:|---:|---:|
| carrier-anchor standard | 0.0709 | 0.1334 | 0.9913 |
| parent-anchor | 0.0741 | 0.1319 | 0.9915 |
| no-KL | 0.0755 | 0.1368 | 0.9908 |

## Common-screen results

Parent common-screen reference from causal interface trajectory: equal7 `44.5643` with BLiMP 69.17, Supplement 66.40, EWoK 49.82, Entity 27.78, COMPS 52.05, GlobalPIQA mean 38.565, Reading 8.165.

Selected continuation ladder:

| arm | best checkpoint | equal7 | Δ vs parent | main changes vs parent |
|---|---|---:|---:|---|
| carrier-anchor standard | std_87M | 44.5543 | -0.0100 | Supplement +0.80, EWoK +0.36, Reading +0.08; Entity -0.24, GlobalPIQA -1.00 |
| std87 alpha0.5 scale check | std_87M_alpha050 | 44.5671 | +0.0029 | essentially measurement-level mean change; Entity -0.21, GlobalPIQA -1.00 |
| no-KL | no_kl_87M | 44.5086 | -0.0557 | Supplement +0.80, EWoK +0.18, Reading +0.09; Entity -0.45, GlobalPIQA -1.00 |
| parent-anchor | pa_87M | 44.6121 | +0.0479 | Supplement +0.80, EWoK +0.18, Reading +0.075; Entity -0.21, GlobalPIQA -0.50 |

The complete collation is in `data/parent_anchor_collated/parent_anchor_collated.md` and JSON beside it. Parent-anchor at 87M is the first continuation point in this comparison that beats the coherent86 cheap7 reference by a nonzero but still small amount on this screen. Its advantage over the no-KL bracket is much larger (`+0.1036` equal7 at 87M), and it also beats the carrier-anchor standard at 87M by `+0.0579`. This supports the narrow mechanism that preserving the parent function is a better consolidation reference than either carrier-off shrinkage or no regularization for the earliest legal suffix continuation.

However, this is not yet a trustworthy BabyLM advance. The pa87 gain is selected from a ladder and still trades down Entity (`27.57` vs `27.78`) and GlobalPIQA (`38.065` vs `38.565`) relative to parent while improving Supplement, EWoK, COMPS, and Reading. Later parent-anchor checkpoints fall below parent despite better GlobalPIQA at 94M/98M/final, mainly because EWoK and Entity decay. The result therefore suggests a narrow early continuation window and a preservation-reference effect, not a solved general learning principle.

## Pending checks

Two comparisons were in progress at the time:

- pa87 alpha0.5 common screen, testing whether the pa87 gain is scale-sensitive.
- fixed-probe drift for pa87/final and no-KL 87M/final, testing whether parent-anchor learns suffix MLM while reducing function drift relative to no-KL/carrier-anchor.

Interpretation after these checks:

- If pa87 alpha0.5 materially improves the frontier and fixed probes show better CE with limited parent KL, then run a stronger official-style evaluation and item-level comparison.
- If the gain remains small and column-trade based, treat parent anchoring as useful consolidation evidence but not as the central empirical breakthrough.
- Neither outcome supported returning to raw carrier-error weighting. The more promising next scientific work is either (i) combine parent-function anchoring with the changed-form state-update data if that stream survives screening, or (ii) develop a relation/update-resolvable target selection method rather than confidence-only weighting.
