# compact mixture model and predictions VIEW_SPLIT integration

VIEW_SPLIT seed43022 preserves the selected source and rewrite texts and the 100M budget, but prevents each source/rewrite pair from co-occurring in one training row. It decides whether VIEW's positive content-conditioning benefit is local to in-window restatement practice or can be learned from cross-row paraphrase exposure.

## 1. Token-level term decomposition

Late 80M/90M/100M compact-rewrite gains, seed43022:

| role | token class | gain U−T | true-source NLL | unrelated-source NLL |
|---|---|---:|---:|---:|
| C | nonoverlap | +1.2119 | +7.0073 | +8.2192 |
| R | nonoverlap | +0.4602 | +7.4553 | +7.9155 |
| RS | nonoverlap | +1.1806 | +6.4604 | +7.6410 |
| V | nonoverlap | +1.8956 | +5.8285 | +7.7241 |
| VS | nonoverlap | +1.2417 | +6.1555 | +7.3972 |
| C | overlap | +6.3015 | +1.8430 | +8.1445 |
| R | overlap | +6.0336 | +1.7511 | +7.7847 |
| RS | overlap | +5.5246 | +1.8880 | +7.4125 |
| V | overlap | +6.8527 | +0.9068 | +7.7595 |
| VS | overlap | +5.6531 | +1.7396 | +7.3926 |

Key contrasts:

| contrast | token class | gain Δ | true-source NLL Δ | unrelated-source NLL Δ |
|---|---|---:|---:|---:|
| RSminusC | nonoverlap | -0.0313 | -0.5469 | -0.5782 |
| RminusC | nonoverlap | -0.7517 | +0.4480 | -0.3037 |
| VSminusC | nonoverlap | +0.0298 | -0.8518 | -0.8220 |
| VSminusR | nonoverlap | +0.7815 | -1.2998 | -0.5183 |
| VSminusRS | nonoverlap | +0.0611 | -0.3048 | -0.2438 |
| VSminusV | nonoverlap | -0.6539 | +0.3270 | -0.3269 |
| VminusC | nonoverlap | +0.6837 | -1.1788 | -0.4951 |
| RSminusC | overlap | -0.7769 | +0.0449 | -0.7320 |
| RminusC | overlap | -0.2679 | -0.0919 | -0.3598 |
| VSminusC | overlap | -0.6484 | -0.1035 | -0.7519 |
| VSminusR | overlap | -0.3805 | -0.0115 | -0.3921 |
| VSminusRS | overlap | +0.1285 | -0.1484 | -0.0199 |
| VSminusV | overlap | -1.1996 | +0.8327 | -0.3669 |
| VminusC | overlap | +0.5512 | -0.9362 | -0.3850 |

On token-nonoverlap targets, original V−C was +0.6837 because VIEW improved the true-source term more than the unrelated-source term (T −1.1788, U −0.4951). VIEW_SPLIT−CLEAN is only +0.0298: it improves both T and U by nearly the same amount (T about −0.852, U about −0.822). Thus the VIEW positive content-conditioning residual disappears when source and rewrite are split across rows, even though the split arm has broad better fit to these compact rewrite targets. VS−V is −0.6539 on nonoverlap gain: relative to original VIEW, splitting makes the true-source term worse and the unrelated-source term better by roughly equal opposite amounts. This is the positive-side analogue of the REPEAT_SPLIT result.

## 2. Pair and strict-word summaries

| contrast | pair gain mean | pair gain median | pair excess mean | strict-word gain | strict-word excess |
|---|---:|---:|---:|---:|---:|
| VminusC | +0.6711 | +0.6679 | -0.6711 | +0.5400 | -0.5400 |
| VSminusC | +0.0040 | +0.0416 | -0.0040 | -0.0809 | +0.0809 |
| VSminusV | -0.6671 | -0.5759 | +0.6671 | -0.6209 | +0.6209 |
| RminusC | -0.7852 | -0.6067 | +0.7852 | -0.5883 | +0.5883 |
| RSminusC | -0.0470 | -0.0165 | +0.0470 | -0.0964 | +0.0964 |

The near-zero VS−C nonoverlap result is preserved after pair aggregation and strict word filtering. It is not a single-token artifact. Original V−C remains strongly positive in the same analyses, so the loss of the residual is specific to splitting source/rewrite co-occurrence.

## 3. Pair-bootstrap uncertainty

| contrast | mean gain Δ | median pair | 95% bootstrap interval | P(|mean|≤0.25) | P(mean in original VIEW range) |
|---|---:|---:|---:|---:|---:|
| VSminusC | +0.0040 | +0.0416 | [-0.0779, +0.0865] | 1.000 | 0.000 |
| VSminusV | -0.6671 | -0.5759 | [-0.7587, -0.5752] | 0.000 | 0.000 |
| VminusC | +0.6711 | +0.6679 | [+0.5746, +0.7676] | 0.000 | 0.402 |

This quantifies held-out pair uncertainty only, not training-seed variability. The second split seed is still needed. But at seed43022, VS−C is decisively inside the pre-stated near-CLEAN band and outside the original VIEW band.

## 4. Natural-copy component check

| role | raw gain | normalized gain | source-present NLL | no-source/control NLL |
|---|---:|---:|---:|---:|
| C | +3.6869 | +0.9040 | +0.3913 | +4.0782 |
| R | +4.1864 | +0.9537 | +0.2032 | +4.3896 |
| RS | +3.4863 | +0.8734 | +0.5054 | +3.9918 |
| V | +4.0140 | +0.9131 | +0.3820 | +4.3960 |
| VS | +3.7865 | +0.8944 | +0.4471 | +4.2336 |

| contrast | raw gain Δ | normalized gain Δ | source-present NLL Δ | control NLL Δ |
|---|---:|---:|---:|---:|
| RSminusC | -0.2005 | -0.0307 | +0.1141 | -0.0864 |
| RminusC | +0.4996 | +0.0497 | -0.1881 | +0.3115 |
| VSminusC | +0.0997 | -0.0097 | +0.0558 | +0.1554 |
| VSminusV | -0.2274 | -0.0187 | +0.0651 | -0.1624 |
| VminusC | +0.3271 | +0.0091 | -0.0093 | +0.3178 |

VIEW_SPLIT has only a small raw and normalized copy-gain advantage over CLEAN; like original VIEW, its source-present repeated NLL is not meaningfully better than CLEAN. The copy-probe conclusion remains dominated by original REPEAT's source-present improvement; VIEW/VS copy statements should stay component-qualified.

## 5. Entity cue-ablation split readout

The split Entity-cue ablation is not an official Entity accuracy evaluation, but it checks whether split arms create DeBERTa-like update cue use in the same probe family.

| split contrast | group | margin full Δ | no-initial Δ | no-last Δ | no-all Δ |
|---|---|---:|---:|---:|---:|
| CminusR | ALL | +0.4136 | -0.4296 | -0.0520 | -0.7186 |
| CminusR | rel_ge2 | +0.4097 | -0.4243 | -0.0590 | -0.7345 |
| CminusR | rel_ge3 | +0.4220 | -0.5361 | -0.0580 | -0.9384 |
| RminusC | ALL | -0.4136 | +0.4296 | +0.0520 | +0.7186 |
| RminusC | rel_ge2 | -0.4097 | +0.4243 | +0.0590 | +0.7345 |
| RminusC | rel_ge3 | -0.4220 | +0.5361 | +0.0580 | +0.9384 |
| VminusC | ALL | +0.5841 | -0.4065 | -0.3190 | +0.2090 |
| VminusC | rel_ge2 | +0.5770 | -0.4081 | -0.3115 | +0.2235 |
| VminusC | rel_ge3 | +0.6282 | -0.3245 | -0.3584 | +0.3456 |

## Scientific reading

At seed43022 both sides of the relation-practice account are now local in the same operational sense. Exact local recurrence creates source-specific identity misfire; local restatement creates the source-specific content-conditioning residual. When either companion is split into separate rows, the source-specific residual collapses toward zero, while broad T/U target fit can improve because the same text exposure no longer supplies the local shortcut. This supports the training-window relation-practice principle more strongly than a generic paraphrase/repetition exposure account.

The result should still be stated with the following boundaries: it is one split seed so far; row splitting changes positions/format/spacing as well as co-occurrence; the mixture equation remains phenomenological; and second split seeds plus a natural variation-set probe are needed before claiming a general training law beyond this BabyLM construction.

Data outputs: `experiments/archive/relation_learning/data/view_split_integration`.
