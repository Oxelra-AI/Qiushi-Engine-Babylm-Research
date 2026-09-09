# common r1 interface comparison result — Common R1 interface comparison: result and confound analysis

## Result

Script: `scripts/common_r1_interface_comparison.py`
Result: `data/common_r1_interface_comparison.json`

| Arm | Intermediate acc | Test both_correct | Train both_correct | Test I |
|---|---:|---:|---:|---:|
| A: Bidir dense intermediate supervision | 1.000 (from ep1) | 0.000 | 0.000 | ~0 |
| B: Dense causal prefix (from scratch) | N/A | 0.020 | 0.020 | 0.025 |

## Critical confound: Arm A intermediate supervision was pre-solved

Arm A's intermediate supervision queries achieve 1.0 accuracy from epoch 1 with loss 0.0455→0.0.
This happened because:

1. The queries ask "After this change, <obj> is in [MASK]" where the answer location is literally in the operation text of the same sentence.
2. The DeBERTa checkpoint (chck_5M) was PRE-TRAINED on 5M words of exactly this R1 ordered-dynamic text distribution.
3. Therefore the pre-trained model already solves intermediate state queries perfectly — the supervision provides ZERO new learning signal.

This means Arm A did not actually test whether intermediate supervision can teach compositional state tracking. It tested whether a model that already predicts local state can be made to predict global compositional state by doing more of the same — and unsurprisingly, it cannot.

## What a CORRECT Arm A would need

For intermediate supervision to provide meaningful compositional training pressure, the queries must:

1. NOT be solvable from local operation text alone
2. REQUIRE resolving indirect references across the full prefix history
3. Test state at points where the model does NOT already achieve high accuracy

Concrete corrected designs:
- Ask about entity X's location after operation t, where X was moved by an EARLIER indirect operation and has since been referenced only by location
- Require the model to track which entity is at each location after a SEQUENCE of operations
- Include commuting/non-commuting operation pairs where the same query has different answers depending on operation order
- Start from a FRESH (randomly initialized) encoder so the supervision is genuinely informative

## What happened with Arm B

The fresh causal model (8-layer, 480-dim, ~30M params) trained from scratch on only ~3000 texts was insufficient:
- Loss decreased from 0.94→0.52 (moderate text modeling, not converged)
- both_correct fluctuated 0.01-0.04 (noise level)
- A from-scratch model on 300 scenarios × 30 epochs ≈ 18000 gradient steps is far too little to learn compositional reasoning

This does NOT show that causal scoring cannot learn state composition. It shows that a randomly initialized model on very limited data doesn't learn it in 30 epochs.

## What this step DOES establish

1. The R1 chck_5M DeBERTa already perfectly predicts intermediate states from local context (inter_acc=1.0).
2. Despite perfect intermediate state prediction, it CANNOT rank counterfactual pairs (both_correct=0.0).
3. This confirms the corrected order gradient probe result/224 finding from a different angle: the model can read local state but cannot compose state across operations for order-dependent ranking.
4. Dense intermediate supervision at the level of "predict locally visible state" adds nothing beyond what pre-training already provided.

## Route implication

The experiment is inconclusive about the mechanism comparison because of confounded arms. The next attempt must either:

1. **Fresh-init bidirectional + compositional intermediate supervision**: Initialize DeBERTa from scratch (not the R1-pretrained checkpoint) and supervise with truly compositional state queries that cannot be solved from local text.
2. **Properly scaled causal arm**: Either use a pre-trained causal model (like the existing RecGPT checkpoint) or train the causal model with significantly more data/epochs.
3. **Or pivot entirely**: Since the R1 micro-world has now consumed related experiments without producing transferable evidence, and since RecGPT already DEMONSTRATES that a causal/recursive model reaches Overall 41.53 on the official evaluation, consider whether the R1 mechanism comparison is worth continued investment vs. directly building/testing a causal model on official data and evaluating on official columns.

The accumulated R1 evidence (related experiments) establishes that:
- Plain WWM cannot learn state composition from R1 data (r1 learning response result)
- Final-answer ranking objectives cannot break the mirror (related experiments)
- Layer-mixing readouts cannot extract order information (layer mixing learning response result)
- Dense intermediate supervision at the local-state level adds nothing (common r1 interface comparison result)
- A from-scratch causal model on limited R1 data also doesn't learn it (common r1 interface comparison result)

But RecGPT on official data reaches BLiMP 73, COMPS 55, GlobalPIQA 41, Overall 41.53 — suggesting the BabyLM SOTA gap may be better attacked through a well-designed causal model on official data than through continued R1 micro-world mechanism probes.
