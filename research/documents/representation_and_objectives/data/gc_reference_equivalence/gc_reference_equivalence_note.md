# compact triangle implementation guard compact-view GC reference equivalence

Summary JSON: `experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json`

Interpretation: **trajectory_diverged_before_or_at_checked_horizon__retrain_view_under_gc_before_triangle_interpretation**

GC run: `experiments/archive/representation_and_objectives/training/runs/gc_compact_view_reinvest_1M_seed43022`
Historical reference: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022`
Compared steps: 26

Max absolute deltas:
- `loss`: 2.237703800201416
- `lr`: 0.0009933774834437086
- `batch_words`: 0.0
- `cumulative_word_exposure`: 0.0
- `masked_tokens`: 0.0
- `effective_mask_rate`: 0.0

Selected rows:

- step gc/ref 1/1: loss_delta=0.0, cum_words_delta=0.0, masked_tokens_delta=0.0, exact=False
- step gc/ref 2/2: loss_delta=0.0, cum_words_delta=0.0, masked_tokens_delta=0.0, exact=False
- step gc/ref 3/3: loss_delta=-0.5794248580932617, cum_words_delta=0.0, masked_tokens_delta=0.0, exact=False
- step gc/ref 4/4: loss_delta=-0.9817752838134766, cum_words_delta=0.0, masked_tokens_delta=0.0, exact=False
- step gc/ref 5/5: loss_delta=-1.297471046447754, cum_words_delta=0.0, masked_tokens_delta=0.0, exact=False
- step gc/ref 26/26: loss_delta=-1.6002230644226074, cum_words_delta=0.0, masked_tokens_delta=0.0, exact=False

Rule: if this file reports divergence, put the compact-view arm on the same GC implementation before using endpoint gaps as mechanism evidence.
