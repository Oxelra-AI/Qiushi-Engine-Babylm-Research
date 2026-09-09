# triangle dynamics and alpha relaunch compact-view triangle training dynamics

This CPU-only readout compares training logs for the completed matched triangle. It is not a benchmark result.

## Run endpoints
- `compact_view_reinvest_historical`: steps=2529, chck_100M_exists=True, final_loss=2.565617084503174, final_words=100000000
- `compact_repeat_reinvest_gc`: steps=2529, chck_100M_exists=True, final_loss=2.577287197113037, final_words=100000000
- `adjbreak_reinvest_gc`: steps=2529, chck_100M_exists=True, final_loss=2.6621243953704834, final_words=100000000

## Pairwise loss deltas (b minus a)
- `compact_view_reinvest_historical__vs__compact_repeat_reinvest_gc`: mean=-0.008632, median=-0.008364, final=0.011670, last100=-0.008763, fraction_b_higher=0.313, LR_max_delta=0.0
- `compact_view_reinvest_historical__vs__adjbreak_reinvest_gc`: mean=0.072744, median=0.081514, final=0.096507, last100=0.099683, fraction_b_higher=0.899, LR_max_delta=0.0
- `compact_repeat_reinvest_gc__vs__adjbreak_reinvest_gc`: mean=0.081376, median=0.088388, final=0.084837, last100=0.108446, fraction_b_higher=0.945, LR_max_delta=0.0

## Interpretation notes
- Training curves are read only as mechanism context: they share the same 2529-step LR schedule and word-exposure grid, but loss values are on different text streams/objective targets and cannot replace the official-compatible no-AoA endpoint readout.
- Adjbreak minus compact_view mean loss delta=0.072744, final=0.096507, last100=0.099683; positive values mean the adjacency-broken corpus is locally harder for the MLM objective.
- Repeat minus compact_view mean loss delta=-0.008632, final=0.011670, last100=-0.008763.
- Adjbreak minus repeat mean loss delta=0.081376, final=0.084837, last100=0.108446; this isolates source-own adjacency at nearly fixed rewrite marginal more directly than either comparison to view.

JSON: `experiments/archive/representation_and_objectives/data/triangle_training_dynamics/triangle_training_dynamics_mechanism.json`
