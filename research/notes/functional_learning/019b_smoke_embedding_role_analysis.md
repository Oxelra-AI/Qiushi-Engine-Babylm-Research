# Step019b analysis: separating held-row drift from distributed specialization

Read result JSON: `experiments/archive/functional_learning/data/revision_019b_smoke/results.json`.

## Branch means

| branch | n | train strong | held high | final train4 | final held4 | final heldB | final heldSel | blocked train4 | held-input L2 | be500 held4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied_carry_full | 1 | 0 | 0 | 0.208 | 0.271 | 0.003 | -0.001 | 0.198 | 0.006 | 0.271 |
| tied_carry_freeze_held_shared | 1 | 0 | 0 | 0.208 | 0.271 | 0.003 | -0.001 | 0.198 | 0.000 | 0.271 |
| untied_reset_full | 1 | 0 | 0 | 0.250 | 0.229 | 0.003 | 0.002 | 0.240 | 0.000 | 0.229 |

## Invariants checked

- Branch epoch-0 records reproduce preparation metrics with max core absolute difference `0.000`.
- Untied `final_FF` hybrids reproduce final tied behavior with max core absolute difference `0.000`.
- Frozen held-input/shared rows have maximum final held-input L2 `0.000` across frozen branches.
- budget matched design note comparison for `tied_carry_full`:
  - seed 42: matched epochs 0, max core absolute difference ``.

## Per-seed endpoint deltas

| seed | branch | prep h4 | final train4 | final h4 | Δh4 | final hB | ΔhB | final hSel | ΔhSel | final hL2 | blocked train4 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | tied_carry_full | 0.250 | 0.208 | 0.271 | 0.021 | 0.003 | -0.004 | -0.001 | -0.001 | 0.006 | 0.198 |
| 42 | tied_carry_freeze_held_shared | 0.250 | 0.208 | 0.271 | 0.021 | 0.003 | -0.004 | -0.001 | -0.001 | 0.000 | 0.198 |
| 42 | untied_reset_full | 0.250 | 0.250 | 0.229 | -0.021 | 0.003 | -0.005 | 0.002 | 0.003 | 0.000 | 0.240 |

## Tied endpoint hybrids

PF restores only held input rows to preparation values in an untied final copy; FP restores only held output rows; PP restores both. The reverse row transplant inserts final held input rows into the preparation network.

| seed | branch | prep h4 | FF h4 | PF h4 | FP h4 | PP h4 | prep+final-input h4 | PF gap h4 | PF gap hB | PF gap hSel | reverse Δh4 | reverse ΔhB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | tied_carry_full | 0.250 | 0.271 | 0.271 | 0.271 | 0.271 | 0.250 | -0.000 | 0.000 | 0.003 | 0.000 | -0.000 |
| 42 | tied_carry_freeze_held_shared | 0.250 | 0.271 | 0.271 | 0.271 | 0.271 | 0.250 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Scientific reading rule

A selective PF rescue would show that the final contextual network and output readout still know how to use the preparation-era held-symbol input code. A reverse transplant that damages the preparation network would show that the final held input rows are sufficient to break a network that still implements held binding. If neither occurs while trained binding remains strong, then held-transfer loss is not explained by held input rows alone and the next work should measure distributed contextual/readout specialization, attribute/RWT geometry, or training-state effects.
