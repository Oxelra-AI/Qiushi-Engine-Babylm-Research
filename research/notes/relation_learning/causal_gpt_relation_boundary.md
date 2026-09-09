# causal gpt relation boundary causal-GPT objective-boundary readout

Created: 2026-09-06T12:30:11Z

This scores the completed GPT2LMHead C/R/RS runs with a left-context next-token readout. It is not directly comparable in level to MLM T/U/N because the causal model cannot use right context; only signs and relative C/R/RS structure are used as objective-boundary evidence.

Records per model: copy 4000, compact T/U 11784, compact N 5892, Wikipedia 10800.

## Compact FineWeb-register T/U/N, token-nonoverlap

| contrast | ΔA_T | ΔA_U | ΔG |
|---|---:|---:|---:|
| RminusC | -1.1532 | +0.0806 | -1.2338 |
| RSminusC | +0.1068 | +0.1784 | -0.0716 |
| RminusRS | -1.2599 | -0.0978 | -1.1621 |

## Wikipedia/Simple-English source use by target class

| contrast | overlap Δ(T vs N) | nonoverlap Δ(T vs N) |
|---|---:|---:|
| RminusC | +0.0416 ± 0.0500 | -0.5355 ± 0.0351 |
| RSminusC | -0.2289 ± 0.0475 | -0.1794 ± 0.0274 |
| RminusRS | +0.2705 ± 0.0484 | -0.3561 ± 0.0329 |

## Held-out natural-copy gain

| contrast | Δ copy gain |
|---|---:|
| RminusC | +1.7855 |
| RSminusC | +0.2517 |
| RminusRS | +1.5338 |

## Reading

If the causal RminusC compact changed-form cost and RminusRS locality contrast match the MLM direction, exact-recurrence locality is not limited to bidirectional masking. If not, the current principle must keep objective/left-context deployment as an explicit boundary.

Output CSVs are in `experiments/archive/relation_learning/data/causal_gpt_relation_boundary`.
