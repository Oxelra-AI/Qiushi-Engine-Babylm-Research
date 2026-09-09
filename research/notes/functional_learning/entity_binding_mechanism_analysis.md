# credit allocation binding pilot entity-binding mechanism analysis

## What the counterfactual diagnostic established

The inherited coherent86 model was scored on 8 paired contrastive packets with five conditions per pair:
1. UPDATE: target entity updated → ask target (answer = new state)
2. RETAIN: distractor entity updated → ask target (answer = source state)
3. NEUTRAL: no update → ask target (answer = source state)
4. CROSS: target updated but use sentence says source state
5. RETAIN_SCORE_NEW: score new_state in distractor context

### Asymmetric binding (not full entity binding)

The model shows strong UPDATE binding (7/8 correct, mean margin +11.8) but **RETAIN margins are systematically below NEUTRAL** in 6/8 pairs. This means:

- The model detects when the target entity name appears in an update and shifts predictions toward the mentioned new state. This is entity-name matching in the update → use sentence pipeline.
- The model does NOT actively filter distractor updates by entity identity. ANY update sentence weakens the source-state prediction, regardless of whether the updated entity matches the queried entity.

Key numbers:

| pair | UPDATE margin | RETAIN margin | NEUTRAL margin | RETAIN < NEUTRAL? |
|---|---:|---:|---:|:---:|
| tp_001 | +2.8 | +4.4 | +4.1 | ✗ (close) |
| tp_002 | +16.4 | +16.9 | +31.7 | ✓ (large gap) |
| tp_003 | +16.6 | +3.4 | +6.4 | ✓ |
| tp_004 | -4.7 | +29.0 | +29.4 | ✓ (close) |
| tp_005 | +17.3 | +13.0 | +17.6 | ✓ |
| tp_006 | +6.0 | +10.1 | +11.5 | ✓ |
| tp_007 | +19.0 | +6.5 | +22.0 | ✓ |
| tp_008 | +21.2 | +14.7 | +14.6 | ✗ (close) |

Mean RETAIN margin (12.2) is substantially below mean NEUTRAL margin (17.1). The model loses ~5 nats of source-state confidence merely from the presence of a distractor update.

### Connection to the synthetic causal-access mechanism

This asymmetry matches the earlier permutation-orbit and interface-reach findings:
- In the synthetic setting, the model could use query-first access to mark relevant context, but full-objective training narrowed which symbols could instantiate the selection interface.
- In natural language, the model uses entity-name co-occurrence as a proxy for "query-first" access: when the queried entity appears in the update sentence, the model shifts toward the update's state words.
- But it lacks a NEGATIVE selection interface: it cannot recognize "this update is about a different entity, so ignore it for predicting the target entity's state."

### What this means for training interventions

1. **The binding gap is specific**: the model needs to learn distractor-update filtering, not just target-update detection.
2. **Paired contrastive training** should explicitly teach: seeing an update about entity D when asked about entity E should NOT shift the prediction away from E's source state.
3. **The metric for success**: after training, RETAIN margin should approach NEUTRAL margin (distractor updates become ignorable), while UPDATE margin stays positive.
4. **MLM mask statistics**: P(useful supervision) ≈ 9-15% per packet per epoch. Over 10 epochs with ~1000 pairs, this gives ~1000-1500 useful supervision events per type. The paired structure forces entity-conditioned routing even though individual answers are copyable from source/update.

### Files
- Counterfactual diagnostic: `data/counterfactual_binding/`
- Baseline binding probe: `data/binding_probe_coherent86/`
- MLM mask analysis: `data/paired_mlm_analysis/`
- Paired packets: `data/paired_packets/`
- Scripts: `scripts/step032_*.py`
