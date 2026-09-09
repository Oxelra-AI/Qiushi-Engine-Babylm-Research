# sparse relation aux preflight — sparse relation auxiliary preflight

Rows checked: 8192 from the exact pvdm 80m ewok fourcell reader/121 70M→80M segment.

- Ordinary WWM replay identical across two generators with the same seed: True
- Selected events before cross-target filter: 2592
- Used events after cross-target filter: 507 (fraction 0.19560185185185186)
- Negative targets from a different row: 1.0
- Category counts: {'spatial': 177, 'temporal': 114, 'causal_connector': 164, 'physical_change': 10, 'negation': 42}
- Cross-target match levels: {'3': 162, '2': 30, '0': 52, '4': 257, '1': 6}
- Mean match geometry: {'mean_pair_cost': 2.786982248520714, 'mean_target_freq_abs_delta': 1.834319526627219, 'mean_distance_bin_abs_delta': 0.6785009861932939, 'mean_target_len_abs_delta': 0.6548323471400395, 'mean_pivot_freq_abs_delta': 0.10256410256410256, 'mean_pivot_target_distance_abs_delta': 0.9072978303747534}
- Bad counts: {}

The auxiliary contrast no longer uses the same-row surrogate anchor as the negative target.  In both arms the target is compared against a structure-matched cross-event dependent target; the semantic arm uses the true pivot, and the anchor_permuted arm replaces only that pivot with the same-row matched anchor.

Files: `experiments/archive/representation_and_objectives/data/sparse_relation_aux_preflight/sparse_relation_aux_preflight.json`, `experiments/archive/representation_and_objectives/data/sparse_relation_aux_preflight/prepared_event_samples.jsonl`
