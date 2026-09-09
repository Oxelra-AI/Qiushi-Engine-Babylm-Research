# connectivity substrate construction and audit coordinate-connectivity substrate

Tests whether learner-usable filler connectivity enables role-coordinate induction.
Both conditions have identical formal information (same satisfying assignments).

## Conditions

### CONNECTED
- bridge pairs: [['Mira', 'Omar'], ['Noel', 'Iris'], ['Lena', 'Pavel'], ['Rina', 'Tomas']]
- aligned_state_bridge: 320 supervised + 1024 common = 1344 total
  formal assignments: 1, true=True, inv=False
  full chains (seen→comparison→bridge): 4, partial: 0
  anti-copy: opposite=1.000, same=0.0
- inverted_state_bridge: 320 supervised + 1024 common = 1344 total
  formal assignments: 1, true=False, inv=True
  full chains (seen→comparison→bridge): 4, partial: 0
  anti-copy: opposite=1.000, same=0.0
- heldheld_only: 192 supervised + 1024 common = 1216 total
  formal assignments: 2, true=True, inv=True
  full chains (seen→comparison→bridge): 0, partial: 0
  anti-copy: opposite=1.000, same=n/a

### DISCONNECTED
- bridge pairs: [['Nia', 'Felix'], ['Ava', 'Jonas'], ['Keira', 'Milo'], ['Sara', 'Theo']]
- aligned_state_bridge: 320 supervised + 1024 common = 1344 total
  formal assignments: 1, true=True, inv=False
  full chains (seen→comparison→bridge): 0, partial: 0
  anti-copy: opposite=1.000, same=0.0
- inverted_state_bridge: 320 supervised + 1024 common = 1344 total
  formal assignments: 1, true=False, inv=True
  full chains (seen→comparison→bridge): 0, partial: 0
  anti-copy: opposite=1.000, same=0.0
- heldheld_only: 192 supervised + 1024 common = 1216 total
  formal assignments: 2, true=True, inv=True
  full chains (seen→comparison→bridge): 0, partial: 0
  anti-copy: opposite=1.000, same=n/a

## Global checks
- train_eval_names_disjoint: True
- conn_disc_pairs_disjoint: True
- all_train_names_covered: True
- formal_equiv_aligned_state_bridge: True
- formal_equiv_inverted_state_bridge: True
- formal_equiv_heldheld_only: True
- connected_has_chains_aligned_state_bridge: True
- disconnected_no_chains_aligned_state_bridge: True
- connected_has_chains_inverted_state_bridge: True
- disconnected_no_chains_inverted_state_bridge: True

## Eval suites (shared)
- heldheld_unseen_edge_closure: 128 rows
- mixed_held_seen_orientation: 512 rows
- paired_state_conservation: 1024 rows
- cross_template_state_readout: 512 rows

## Files
- manifest: `experiments/archive/representation_and_objectives/data/coordinate_connectivity_substrate/manifest.json`
- connected: `experiments/archive/representation_and_objectives/data/coordinate_connectivity_substrate/connected`
- disconnected: `experiments/archive/representation_and_objectives/data/coordinate_connectivity_substrate/disconnected`
- eval: `experiments/archive/representation_and_objectives/data/coordinate_connectivity_substrate/eval`
