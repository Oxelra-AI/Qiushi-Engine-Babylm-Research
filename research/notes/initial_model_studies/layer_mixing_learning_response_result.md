# layer mixing learning response result — Layer-mixing learning-response result and route implication

## Experiment

Script: `scripts/layer_mixing_learning_response.py`
Result: `data/layer_mixing_learning_response.json`

Three arms on the R1 ordered-dynamic chck_5M checkpoint, using 200 train pairs
(train-vocab entities/locations) and 100 held-out test pairs (different entities/locations).
30 training epochs, LR 1e-3, position-aligned scoring, paired counterfactual loss.

## Results

| Arm | Train both_correct | Test both_correct | Train I | Test I |
|---|---:|---:|---:|---:|
| 1. Final MLM head (frozen control) | 0.000 | 0.000 | ~0 | ~0 |
| 2. Layer-mixing readout (frozen encoder) | 0.000 | 0.000 | ~0 | ~0 |
| 3. Layer-mixing readout (unfrozen encoder) | 0.760 | 0.120 | 13.91 | 0.022 |

## Scientific interpretation

### What Arm 2 proves

The corrected order gradient probe result shallow order signal (L2=0.19 at layer 0) is NOT usable by a trained readout
with full layer-mixing, learned LayerNorms, and an MLP output head. After 30 epochs on
200 pairs with a direct paired objective, held-out both_correct stays exactly 0.0.

**This directly answers the mechanistic question "is there a hidden resource waiting to be tapped?"
Answer: No.** The existing frozen bidirectional encoder does not contain accessible order
information at the query position, even when the readout is freely trained to find it.

### What Arm 3 shows

The unfrozen encoder CAN be shaped to discriminate training examples:
- Train both_correct reaches 0.76-0.82 (strong memorization)
- But test both_correct is only 0.12 (weak generalization)
- Train-test gap: 0.64 (massive overfitting)

This means:
- DeBERTa's architecture is not structurally incapable of representing order
- But what it learns is specific to training entities/locations, not a compositional order mechanism
- 200 pairs and 30 epochs produce entity/template-specific discrimination, not transferable state tracking
- The 0.12 held-out success is above zero but could be chance (12/100 pairs, needs binomial test)

### What this does NOT settle

- Whether more training pairs (1000+) or intermediate state supervision would improve generalization
- Whether the poor generalization is a fundamental bidirectional-architecture limit or a training-signal/data limit
- Whether a causal model would do better on the SAME held-out test with comparable compute
- Whether R1 success transfers to official Entity Tracking at all

## Route implication

The experiment supports the following priority ordering:

1. **A causal/recursive interface screen** is now the highest-priority mechanism to test, because:
   - Frozen bidirectional readout is completely dead (Arm 2 = 0)
   - Unfrozen bidirectional mostly memorizes (Arm 3 train/test gap = 0.64)
   - A causal interface structurally avoids the query-position order-collapse by conditioning only on prefixes
   - RecGPT demonstrates the causal phenotype reaches near-leader Overall

2. **Bidirectional intermediate supervision** remains a secondary possibility but must FIRST demonstrate held-out generalization (test both_correct > 0.30) before being considered for official-scale work. The proposed Arm 3 with explicit intermediate state targets was not built in this step.

3. **Do not scale any R1-only route to 100M** before establishing that R1 success transfers to official columns. RecGPT reaches near-leader Overall with Entity only 16.59 — causal scoring helps grammar/COMPS/GlobalPIQA, not necessarily Entity tracking.

## Relation to SOTA goal

The BabyLM SOTA gap is primarily in Entity (-5.83), GlobalPIQA (-4.03), EWoK (-3.88).
RecGPT shows a causal route can address GlobalPIQA (+5.05) and potentially BLiMP (+6.35),
COMPS (+3.24) through a different phenotype. It does NOT address Entity.

The route that best serves Overall SOTA is therefore:
- Build a causal/recursive model to capture RecGPT-like BLiMP/COMPS/GlobalPIQA/Reading gains
- Test whether any intermediate-supervision trick can also improve Entity from the same architecture
- Compare against the protected DeBERTa on full official evaluation before committing to 100M

## Supersession

This note provides the decisive learning-response evidence that intermediate layer order probe result needed before a route decision. Combined with:
- counterfactual order diagnosis: final-layer order collapse
- corrected order gradient probe result: corrected all-layer decay + weak gradient leverage
- layer mixing learning response result: frozen readout failure + unfrozen memorization without generalization

The bidirectional final-answer-based entity-state route is now closed with converging evidence from representation (counterfactual order diagnosis and intermediate-layer probes), gradient (corrected order gradient probe result), and learning (layer mixing learning response result) levels. The next main route should change the scoring interface.
