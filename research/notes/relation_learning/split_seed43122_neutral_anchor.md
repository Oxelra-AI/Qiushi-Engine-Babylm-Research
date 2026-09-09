# anchor corrected price interpretation neutral-source anchor for rewrite conditioning

This run adds a third context to the existing compact-rewrite probe. T is the true source, U is an unrelated compact source, and N is length-matched ordinary held-out text in the same source slot. N separates broad target fit from dependence on having another compact source-like neighbor.

Records: 5892 neutral rows over 1626 pairs; ordinary held-out source windows: 6992. The T/U merge recovered all matching rows.

## Late token-nonoverlap terms

| role | n | T | U | N | U-T | T-N | U-N |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 2732 | +6.9534 | +8.2254 | +8.5077 | +1.2720 | -1.5543 | -0.2823 |
| R | 2732 | +7.6465 | +7.8752 | +8.1910 | +0.2286 | -0.5445 | -0.3158 |
| RS | 2732 | +6.5728 | +7.6550 | +7.9929 | +1.0822 | -1.4200 | -0.3378 |
| V | 2732 | +5.5990 | +7.7648 | +7.9770 | +2.1657 | -2.3779 | -0.2122 |
| VS | 2732 | +6.1370 | +7.3735 | +7.7197 | +1.2365 | -1.5827 | -0.3462 |

Here T-N < 0 means the related source helps compared with ordinary text; U-N > 0 means the unrelated compact source hurts compared with ordinary text.

## Local-versus-split reading

| contrast | ΔN | ΔU | ΔT | Δ(U-T) | Δ(T-N) | Δ(U-N) | pair ΔN | pair Δ(U-N) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RminusRS | +0.1982 | +0.2202 | +1.0737 | -0.8535 | +0.8755 | +0.0220 | +0.2035 | +0.0257 |
| VminusVS | +0.2573 | +0.3913 | -0.5379 | +0.9293 | -0.7952 | +0.1340 | +0.2522 | +0.1358 |

If the earlier U gap came from weaker learning of the companion content, local-minus-split should remain visible on N. If it came mainly from unrelated-neighbor interference, ΔN should be small while Δ(U-N) grows for the local arm. The result above decides which reading better fits each relation.

## Arm-versus-CLEAN anchor terms

| contrast | ΔN | ΔT | ΔU | Δ(U-T) | Δ(T-N) | Δ(U-N) |
|---|---:|---:|---:|---:|---:|---:|
| RSminusC | -0.5148 | -0.3806 | -0.5704 | -0.1898 | +0.1343 | -0.0555 |
| RminusC | -0.3167 | +0.6931 | -0.3502 | -1.0433 | +1.0098 | -0.0335 |
| VSminusC | -0.7880 | -0.8164 | -0.8519 | -0.0355 | -0.0284 | -0.0639 |
| VminusC | -0.5307 | -1.3544 | -0.4606 | +0.8938 | -0.8236 | +0.0701 |

Data outputs: `experiments/archive/relation_learning/data/neutral_anchor_rewrite_probe`.
