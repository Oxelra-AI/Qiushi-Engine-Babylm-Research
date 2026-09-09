# pvdm 80m ewok fourcell reader — PVDM real-batch invariant check

Rows checked: 8,192; batch size 256; target_prob 0.6.

## Aggregate

- selected rows: 6,283/8,192 (0.767)
- selected dependent target groups: 11,450 (1.398/row)
- effective mask rate: treatment 0.1490, control 0.1490
- target mismatches: 0 label/replacement, 0 original-id
- per-row masked token mass diff abs-sum/max: 0 / 0
- background mismatches: 0; anchor action mismatches: 0
- selected categories: {'physical_change': 2922, 'causal_connector': 2485, 'spatial': 2132, 'comparative': 188, 'negation': 1360, 'temporal': 2363}

## Pass conditions

- dependent_targets_identical: True
- masked_mass_equal_per_row: True
- background_replacement_identical: True
- anchor_swap_replacement_actions_matched: True
- treatment_has_material_pivot_visibility_contrast: True
- categories_present: True
- legacy_wwm_callable: True

Ready for two-arm 70M→80M launch: **True**.

JSON: `experiments/archive/representation_and_objectives/data/pvdm_batch_invariants/pvdm_batch_invariant_summary.json`
Batch JSONL: `experiments/archive/representation_and_objectives/data/pvdm_batch_invariants/pvdm_batch_invariant_batches.jsonl`
