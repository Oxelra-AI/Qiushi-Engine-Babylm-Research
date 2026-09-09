# argument map corrections direct target-probability readout for compact rewrites

Created: 2026-09-06T10:28:08Z

This readout converts existing true-source/unrelated-source NLL rows into target probabilities. For tokenizer-overlap targets the target token ID is present in the true source; for tokenizer-nonoverlap targets it is absent. It tests whether REPEAT's source-content mass elevation corresponds to higher probability on the correct source-present token inside nonidentical rewrite contexts.

## Across-seed contrasts

| token class | contrast | Δ true target prob | Δ true-source NLL | Δ source-conditioned gain U−T |
|---|---|---:|---:|---:|
| overlap | RminusC | +0.05532 ± 0.02927 (+/+/+) | +0.01979 ± 0.16976 (-/-/+) | -0.42106 ± 0.25203 (-/-/-) |
| overlap | VminusC | +0.16111 ± 0.00701 (+/+/+) | -0.92198 ± 0.06344 (-/-/-) | +0.53783 ± 0.12764 (+/+/+) |
| overlap | VminusR | +0.10578 ± 0.02403 (+/+/+) | -0.94177 ± 0.11444 (-/-/-) | +0.95889 ± 0.14918 (+/+/+) |
| nonoverlap | RminusC | -0.00608 ± 0.00203 (-/-/-) | +0.51890 ± 0.15174 (+/+/+) | -0.88178 ± 0.14833 (-/-/-) |
| nonoverlap | VminusC | +0.03753 ± 0.00220 (+/+/+) | -1.30038 ± 0.10555 (-/-/-) | +0.80558 ± 0.10902 (+/+/+) |
| nonoverlap | VminusR | +0.04361 ± 0.00418 (+/+/+) | -1.81928 ± 0.21263 (-/-/-) | +1.68735 ± 0.25086 (+/+/+) |

Reading: in the nonidentical rewrite format, REPEAT does not become a reliable precise answer copier. On tokenizer-overlap targets where the answer token is present in the source, R−C raises direct true-source target probability slightly in all three seeds (+0.055 ± 0.029), but still has negative source-conditioned gain in all three seeds (−0.421 ± 0.252). Thus the source-present answer receives some probability, but not enough to make true-source use beneficial relative to CLEAN in the nonidentical rewrite format. On nonoverlap targets, R lowers true-source target probability and increases true-source NLL in all three seeds. VIEW raises true-source target probability and source-conditioned gain on both token classes, with the larger gain on nonoverlap tokens. The correct mechanism phrase is therefore a source-recognition trigger whose diffuse source-content pull transfers to related spans, while precise answer copying remains format-bound to exact natural-repeat contexts.

Data: `experiments/archive/relation_learning/data/integrated_scope_and_mechanism/rewrite_target_probability_across_seed.csv`
