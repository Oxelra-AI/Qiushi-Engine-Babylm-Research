# seed43222 threearm probe decomposition seed43222 CLEAN baseline probe decomposition

This analysis merges seed43222 VIEW/REPEAT probe rows from research synthesis with the completed roberta probe result parallel CLEAN run scored in seed43222 threearm probe decomposition. It is the first seed43222 test of whether REPEAT falls below CLEAN on nonidentical true-source use, the pair level relation robustness active-cost center.

## Rewrite content-conditioning

| group | contrast | gain Δ | true-source Δ | unrelated-source Δ | excess true-source cost | n |
|---|---|---:|---:|---:|---:|---:|
| ALL | RminusC | -0.7761 | +0.3081 | -0.4680 | +0.7761 | 5892 |
| ALL | VminusC | +0.6058 | -1.0916 | -0.4858 | -0.6058 | 5892 |
| ALL | VminusR | +1.3819 | -1.3997 | -0.0177 | -1.3819 | 5892 |
| token_nonoverlap | RminusC | -0.8503 | +0.4156 | -0.4347 | +0.8503 | 2732 |
| token_nonoverlap | VminusC | +0.8392 | -1.3680 | -0.5288 | -0.8392 | 2732 |
| token_nonoverlap | VminusR | +1.6896 | -1.7836 | -0.0940 | -1.6896 | 2732 |
| token_overlap | RminusC | -0.7119 | +0.2151 | -0.4968 | +0.7119 | 3160 |
| token_overlap | VminusC | +0.4040 | -0.8526 | -0.4486 | -0.4040 | 3160 |
| token_overlap | VminusR | +1.1160 | -1.0678 | +0.0482 | -1.1160 | 3160 |

For token-nonoverlap targets, seed43222 reproduces the active cost of exact recurrence: R-C gain is -0.8503, because REPEAT is worse than CLEAN on the true-source term by +0.4156 while the unrelated-source term differs by only -0.4347. This gives a third DeBERTa seed for the below-baseline recurrence cost.

VIEW also remains above CLEAN on the same targets: V-C gain is +0.8392, with true-source improvement larger than unrelated-source improvement. Thus the positive VIEW-side content-conditioning result is now three-seed DeBERTa evidence, while RoBERTa remains weak on V-C after register fit is subtracted.

## Held-out natural copy gain

| group | contrast | gain Δ | repeated-term Δ | unrepeated-term Δ | n |
|---|---|---:|---:|---:|---:|
| ALL | RminusC | +0.5594 | -0.2278 | +0.3315 | 2000 |
| ALL | VminusC | +0.3147 | -0.0349 | +0.2798 | 2000 |
| ALL | RminusV | +0.2447 | -0.1930 | +0.0517 | 2000 |
| span_1 | RminusC | +0.6467 | -0.2889 | +0.3578 | 1000 |
| span_1 | VminusC | +0.3247 | -0.0184 | +0.3063 | 1000 |
| span_1 | RminusV | +0.3220 | -0.2705 | +0.0515 | 1000 |
| span_4 | RminusC | +0.4721 | -0.1668 | +0.3053 | 1000 |
| span_4 | VminusC | +0.3047 | -0.0513 | +0.2534 | 1000 |
| span_4 | RminusV | +0.1674 | -0.1155 | +0.0519 | 1000 |

The R-C copy advantage at seed43222 is smaller than prior seeds but positive on the aggregate: +0.5594. This supports the copy side directionally but reinforces the research synthesis caution that installed probe quantities are more stable than Entity magnitude, and copy conversion varies more than rewrite conditioning.

## Entity cue-ablation with CLEAN baseline

| group | contrast | full-margin Δ | no-initial effect Δ | no-last effect Δ | no-all-updates effect Δ | n |
|---|---|---:|---:|---:|---:|---:|
| ALL | VminusC | +0.2273 | -0.2947 | -0.0321 | +0.5575 | 1221 |
| ALL | RminusC | -0.4740 | +0.5820 | +0.2256 | -0.2232 | 1221 |
| ALL | VminusR | +0.7013 | -0.8767 | -0.2577 | +0.7807 | 1221 |
| rel_ge2 | VminusC | +0.2176 | -0.2822 | -0.0240 | +0.5734 | 1205 |
| rel_ge2 | RminusC | -0.4724 | +0.5861 | +0.2273 | -0.2276 | 1205 |
| rel_ge2 | VminusR | +0.6901 | -0.8683 | -0.2512 | +0.8010 | 1205 |
| rel_ge3 | VminusC | +0.1419 | -0.1394 | -0.0585 | +0.6847 | 716 |
| rel_ge3 | RminusC | -0.4705 | +0.6371 | +0.2634 | -0.2678 | 716 |
| rel_ge3 | VminusR | +0.6124 | -0.7764 | -0.3219 | +0.9525 | 716 |

The full gold-over-stale margins at seed43222 do not preserve a simple V>C>R ladder against CLEAN: V-R remains positive, but C is not a passive midpoint in this cue-ablation subset. This is another reason to keep Entity as a downstream correlate with direction-robust V-R structure, not the primary installed quantity.

## Files

- Output directory: `experiments/archive/relation_learning/data/seed43222_threearm_probe_decomposition`
- VIEW/REPEAT source rows: `experiments/archive/relation_learning/data/seed43222_probes`
- CLEAN source rows: `experiments/archive/relation_learning/data/seed43222_parallel_clean_probes`
