# accidental target context audit accidental target-token context audit

This data-only readout checks whether compact rewrite masked targets are accidentally visible inside the unrelated compact source slot U or the ordinary-text neutral slot N. The same target positions as the compact T/U/N probe are reconstructed without loading any model.

Rows reconstructed: 5892 from 1626 pairs. Coverage versus T/U file: {'status': 'compared', 'tu_unique_probe_ids': 5892, 'own_unique_probe_ids': 5892, 'missing_from_own': 0, 'extra_in_own': 0, 'tu_rows_path': 'experiments/archive/relation_learning/data/view_split_integration/rewrite_input_rows.csv'}.

| token class | records | pairs | target in true source | target in unrelated source | target in neutral source |
|---|---:|---:|---:|---:|---:|
| nonoverlap | 2732 | 1465 | 0.00% | 0.51% | 0.00% |
| overlap | 3160 | 1619 | 100.00% | 0.54% | 0.00% |
| all | 5892 | 1626 | 53.63% | 0.53% | 0.00% |

For token-nonoverlap rows, true-source target visibility should be zero by construction. If U/N target visibility is also near zero, the T/U/N contrasts are not mainly driven by hidden answer tokens in the alternate source slots. This readout does not distinguish diffuse source-content pull from precise copying; that still requires probability concentration over individual source-content tokens.

Data outputs: `experiments/archive/relation_learning/data/accidental_target_context_audit`.
