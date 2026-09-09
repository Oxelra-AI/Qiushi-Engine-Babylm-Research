# earlier analysis repaired compact-view GC reference equivalence

Status: `GC_REFERENCE_EQUIVALENCE_REPAIRED_COMPARISON`.
Interpretation: `trajectory_equivalent_to_checked_horizon`.

The compact triangle implementation guard guard summary was superseded because the 1M GC reference run used a 26-step cosine schedule. The corrected earlier analysis run uses `--lr_total_steps 2529`, matching historical live fineweb core fact filter.
Compared steps: 26; mismatch count at 1e-12: 0; max deltas: `{"loss": 0.0, "lr": 0.0, "batch_words": 0.0, "cumulative_word_exposure": 0.0, "masked_tokens": 0.0, "effective_mask_rate": 0.0}`.
Corrected run: `experiments/archive/representation_and_objectives/training/runs/gc_compact_view_reinvest_1M_lr2529_seed43022`.
Repaired JSON: `experiments/archive/representation_and_objectives/data/gc_reference_equivalence_repair/gc_reference_equivalence_repair.json`. Guard summary overwritten for downstream scripts: `experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json`.
Superseded compact triangle implementation guard file: `experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary_flawed_lr26_superseded.json`.
