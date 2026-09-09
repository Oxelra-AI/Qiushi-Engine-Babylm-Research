# comparison channel design — Comparison-channel causal intervention design

## Scientific question

Does the comparison graph causally carry the gauge coordinate from anchored
(h0/h2) events to unanchored (h1/h3) events? Or can h1/h3 state orientation
emerge from template regularities, shared-GRU generalization, or state-loss
patterns alone?

## Data structure justification

The substrate training data has a critical asymmetry:
- **State queries (training)**: s_give, s_receive (1024 common background) + 
  h0_dax, h2_norp (128 direct anchors with bridge-sign flip). 
  **No h1_mep or h3_ziv state queries in training.**
- **Comparison edges (training)**: 192 edges linking h0↔h1 (48), h0↔h2 (48), 
  h1↔h2 (48), h2↔h3 (48). These are the ONLY training-time pathway from 
  h0/h2 anchors to h1/h3 events.
- **State queries (eval)**: h0_dax, h1_mep, h2_norp, h3_ziv (512 rows with
  held names). Graph-transfer metric measures h1/h3 eval accuracy.

Without comparison edges, the model has zero supervision for h1/h3 states.
The comparison graph is the sole training-time information channel connecting
h0/h2 state labels to h1/h3 events.

## Interventions

1. **no_cmp**: Remove all 192 training comparisons. Training uses only state loss.
   Prediction: graph_same ≈ constant for both bridge signs (no transport).
   
2. **shuffled_cmp**: Permute comparison labels. Same comparison events/names/structure,
   but labels are randomly reassigned. ~50% of labels change.
   Prediction: graph_same ≈ constant (noisy comparisons don't orient).

3. **full_graph**: All 192 comparisons retained with correct labels.
   Prediction: graph_same=1.0 for +1, 0.0 for -1 (reproduces posalign repair and equality emergence).

## Matched controls

All cells use the same seed (30000), same dual-position learned-equality
architecture (posalign repair and equality emergence), same character-pair pretraining (full alphabet, 60 epochs),
and same relational training (180 epochs). The only difference is the comparison
treatment. Both bridge signs use matched cloned initialization from the same seed.

## Decision criteria

- **Decisive positive**: no_cmp gives graph_same ≈ same for both signs AND full_graph
  gives graph_same reversal → comparison edges causally necessary.
- **Unexpected**: no_cmp gives graph_same reversal → template/state generalization
  sufficient, comparison edges not causally necessary → rethink mechanism.
- **Ambiguous**: partial transport in no_cmp → may need more epochs or a different
  comparison isolation.

## Files

- Script: `training/scripts/comparison_channel_probe.py`
- Analysis: `scripts/pair_analysis.py`
- no_cmp outputs: `data/no_cmp_shared/`
- full_graph outputs: `data/full_graph_shared/`
- shuffled outputs (if needed): `data/shuffled_shared/`
