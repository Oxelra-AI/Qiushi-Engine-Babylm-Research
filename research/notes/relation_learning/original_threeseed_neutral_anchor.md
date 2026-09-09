# anchor corrected price interpretation original-arm three-seed neutral anchor

N is the target NLL with length-matched ordinary held-out text in the source slot, scored on the same compact-rewrite masked targets as T and U. This checks whether the original three-seed source-specific effects survive a neutral anchor.

Neutral records: 5892 over 1626 pairs; ordinary source windows: 6992.

## Token-nonoverlap across-seed contrasts

| contrast | seeds | ΔT | ΔU | ΔN | Δ(U-T) | Δ(T-N) | Δ(U-N) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RminusC | 3 | +0.5189±0.1517 | -0.3629±0.0664 | -0.2949±0.0527 | -0.8818±0.1483 | +0.8138±0.1729 | -0.0679±0.0340 |
| VminusC | 3 | -1.3004±0.1056 | -0.4948±0.0341 | -0.5033±0.0326 | +0.8056±0.1090 | -0.7970±0.0758 | +0.0085±0.0536 |
| VminusR | 3 | -1.8193±0.2126 | -0.1319±0.0521 | -0.2084±0.0273 | +1.6874±0.2509 | -1.6109±0.2196 | +0.0765±0.0321 |

## Per-seed token-nonoverlap contrasts

| seed | contrast | ΔT | ΔU | ΔN | Δ(U-T) | Δ(T-N) | Δ(U-N) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 43022 | RminusC | +0.4480 | -0.3037 | -0.2348 | -0.7517 | +0.6828 | -0.0689 |
| 43022 | VminusC | -1.1788 | -0.4951 | -0.4672 | +0.6837 | -0.7115 | -0.0278 |
| 43022 | VminusR | -1.6268 | -0.1914 | -0.2324 | +1.4354 | -1.3944 | +0.0410 |
| 43122 | RminusC | +0.6931 | -0.3502 | -0.3167 | -1.0433 | +1.0098 | -0.0335 |
| 43122 | VminusC | -1.3544 | -0.4606 | -0.5307 | +0.8938 | -0.8236 | +0.0701 |
| 43122 | VminusR | -2.0475 | -0.1104 | -0.2141 | +1.9371 | -1.8334 | +0.1037 |
| 43222 | RminusC | +0.4156 | -0.4347 | -0.3333 | -0.8503 | +0.7489 | -0.1014 |
| 43222 | VminusC | -1.3680 | -0.5288 | -0.5121 | +0.8392 | -0.8560 | -0.0167 |
| 43222 | VminusR | -1.7836 | -0.0940 | -0.1788 | +1.6896 | -1.6048 | +0.0847 |

## Reading

Across all three DeBERTa seeds, original REPEAT's negative compact-rewrite gain relative to CLEAN is a true-source effect relative to N: R−C has positive Δ(T-N) in every seed, while Δ(U-N) is small and not the source of the effect. VIEW's positive compact-rewrite gain is likewise true-source help relative to N in every seed, while Δ(U-N) remains small. The T/U sign reversal therefore survives the neutral anchor as a relation-specific true-source effect rather than an unrelated-neighbor artifact.

Data outputs: `experiments/archive/relation_learning/data/original_threeseed_neutral_anchor`.
