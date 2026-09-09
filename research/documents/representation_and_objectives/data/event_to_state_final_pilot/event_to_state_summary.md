# paired world stress teacher and route event-to-state final-match pilot

## Result

- tennis final events: 46972; reversed final pairs: 1472
- BWF final events: 2889; reversed final pairs: 178
- built families: 300 ({'tennis': 220, 'badminton': 80})
- label errors: 0
- sequence probe: {'status': 'completed', 'n_rows': 2400, 'raw_word12': {'mean': 0.49833333333333335, 'sd': 0.002041241452319308, 'folds': [0.4979166666666667, 0.4979166666666667, 0.49583333333333335, 0.4979166666666667, 0.5020833333333333]}, 'raw_char35': {'mean': 0.4929166666666666, 'sd': 0.005034602488997747, 'folds': [0.4895833333333333, 0.4979166666666667, 0.4875, 0.4895833333333333, 0.5]}, 'canonical_word12': {'mean': 0.5, 'sd': 0.0, 'folds': [0.5, 0.5, 0.5, 0.5, 0.5]}, 'canonical_char35': {'mean': 0.5316666666666666, 'sd': 0.013906932723565545, 'folds': [0.5416666666666666, 0.5125, 0.5520833333333334, 0.5229166666666667, 0.5291666666666667]}}

## Scientific meaning

This constructs a distinct source-derived operation: final outcome -> entity state (champion vs runner-up). It does not yet show general data-efficient learning; it gives a next substrate to teacher-check and to use for cross-predicate/event-state transfer if language realization is sound.

## Files

- train: `experiments/archive/representation_and_objectives/data/event_to_state_final_pilot/state_families_train.jsonl`
- held: `experiments/archive/representation_and_objectives/data/event_to_state_final_pilot/state_families_held.jsonl`
- summary: `experiments/archive/representation_and_objectives/data/event_to_state_final_pilot/event_to_state_summary.json`
