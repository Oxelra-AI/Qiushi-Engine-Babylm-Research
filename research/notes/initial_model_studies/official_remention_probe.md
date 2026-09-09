# official remention probe official-text remention-state probe

Evidence JSON: `experiments/archive/initial_model_studies/data/official_remention_probe.json`

Cases: 1400; train 986; test 406; lexical split {'train_words': 589, 'test_words': 252}

| model | best full layer | full bal acc | local bal acc | history gain | best gain layer | best gain |
|---|---:|---:|---:|---:|---:|---:|
| wwm43_40M | 2 | 58.13 | 54.93 | +3.20 | 8 | +6.16 |
| wwm43_80M | 6 | 57.64 | 54.43 | +3.20 | 6 | +3.20 |
| wwm43_100M | 6 | 57.88 | 51.72 | +6.16 | 6 | +6.16 |
| amlm43_40M | 2 | 60.10 | 52.22 | +7.88 | 8 | +8.62 |
| amlm43_100M | 8 | 63.30 | 51.72 | +11.58 | 8 | +11.58 |

Interpretation: a transferable cross-sentence remention-state signal should appear as full-history balanced accuracy above local-only under lexical split. If history gain is near zero or negative, ordinary frozen checkpoints do not contain an easily recoverable official-text state variable for this target.
