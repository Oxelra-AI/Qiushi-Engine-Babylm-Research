# anchor corrected price interpretation neutral-source anchor for rewrite conditioning

This run adds a third context to the existing compact-rewrite probe. T is the true source, U is an unrelated compact source, and N is length-matched ordinary held-out text in the same source slot. N separates broad target fit from dependence on having another compact source-like neighbor.

Records: 5892 neutral rows over 1626 pairs; ordinary held-out source windows: 6992. The T/U merge recovered all matching rows.

## Late token-nonoverlap terms

| role | n | T | U | N | U-T | T-N | U-N |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 2732 | +7.0073 | +8.2192 | +8.5265 | +1.2119 | -1.5192 | -0.3073 |
| R | 2732 | +7.4553 | +7.9155 | +8.2916 | +0.4602 | -0.8364 | -0.3762 |
| RS | 2732 | +6.4604 | +7.6410 | +7.9889 | +1.1806 | -1.5286 | -0.3479 |
| V | 2732 | +5.8285 | +7.7241 | +8.0592 | +1.8956 | -2.2307 | -0.3351 |
| VS | 2732 | +6.1555 | +7.3972 | +7.7904 | +1.2417 | -1.6349 | -0.3932 |

Here T-N < 0 means the related source helps compared with ordinary text; U-N > 0 means the unrelated compact source hurts compared with ordinary text.

## Local-versus-split reading

| contrast | ΔN | ΔU | ΔT | Δ(U-T) | Δ(T-N) | Δ(U-N) | pair ΔN | pair Δ(U-N) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RminusRS | +0.3027 | +0.2745 | +0.9949 | -0.7204 | +0.6922 | -0.0282 | +0.3033 | -0.0291 |
| VminusVS | +0.2688 | +0.3269 | -0.3270 | +0.6539 | -0.5958 | +0.0581 | +0.2837 | +0.0533 |

If the earlier U gap came from weaker learning of the companion content, local-minus-split should remain visible on N. If it came mainly from unrelated-neighbor interference, ΔN should be small while Δ(U-N) grows for the local arm. The result above decides which reading better fits each relation.

## Arm-versus-CLEAN anchor terms

| contrast | ΔN | ΔT | ΔU | Δ(U-T) | Δ(T-N) | Δ(U-N) |
|---|---:|---:|---:|---:|---:|---:|
| RSminusC | -0.5375 | -0.5469 | -0.5782 | -0.0313 | -0.0094 | -0.0407 |
| RminusC | -0.2348 | +0.4480 | -0.3037 | -0.7517 | +0.6828 | -0.0689 |
| VSminusC | -0.7360 | -0.8518 | -0.8220 | +0.0298 | -0.1157 | -0.0859 |
| VminusC | -0.4672 | -1.1788 | -0.4951 | +0.6837 | -0.7115 | -0.0278 |

Data outputs: `experiments/archive/relation_learning/data/neutral_anchor_rewrite_probe`.
