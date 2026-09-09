# official remention probe official-text remention-state probe

Evidence JSON: `experiments/archive/initial_model_studies/data/official_remention_probe_smoke.json`

Cases: 120; train 82; test 30; lexical split {'train_words': 66, 'test_words': 28}

| model | best full layer | full bal acc | local bal acc | history gain | best gain layer | best gain |
|---|---:|---:|---:|---:|---:|---:|
| wwm43_40M | 2 | 60.00 | 43.33 | +16.67 | 2 | +16.67 |

Interpretation: a transferable cross-sentence remention-state signal should appear as full-history balanced accuracy above local-only under lexical split. If history gain is near zero or negative, ordinary frozen checkpoints do not contain an easily recoverable official-text state variable for this target.
