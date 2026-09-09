# shared factorization result synthesis factorized probe input audit

## aligned_state_bridge

- train vocab size: 62
- full train+eval vocab size: 94
- eval-only vocab items: 32
- raw eval/train name tokens remaining after replacement: 0 / 0

| suite/kind | unk | total | frac |
|---|---:|---:|---:|
| cross_template_state_readout|state_changed | 256 | 3072 | 0.083 |
| cross_template_state_readout|state_unchanged | 768 | 6656 | 0.115 |
| heldheld_unseen_edge_closure|comparison | 512 | 6144 | 0.083 |
| mixed_held_seen_orientation|comparison | 3072 | 27904 | 0.110 |
| paired_state_conservation|state_changed | 512 | 6144 | 0.083 |
| paired_state_conservation|state_unchanged | 1536 | 13312 | 0.115 |

Top eval unknown tokens: `[('flute', 320), ('badge', 320), ('ticket', 320), ('scarf', 320), ('hammer', 320), ('shell', 320), ('drum', 320), ('booklet', 320), ('goblet', 320), ('key', 320), ('telescope', 320), ('bracelet', 320), ('vase', 320), ('satchel', 320), ('radio', 320), ('painting', 320), ('anchor', 96), ('basketball', 96), ('lamp', 96), ('mirror', 96)]`

## inverted_state_bridge

- train vocab size: 62
- full train+eval vocab size: 94
- eval-only vocab items: 32
- raw eval/train name tokens remaining after replacement: 0 / 0

| suite/kind | unk | total | frac |
|---|---:|---:|---:|
| cross_template_state_readout|state_changed | 256 | 3072 | 0.083 |
| cross_template_state_readout|state_unchanged | 768 | 6656 | 0.115 |
| heldheld_unseen_edge_closure|comparison | 512 | 6144 | 0.083 |
| mixed_held_seen_orientation|comparison | 3072 | 27904 | 0.110 |
| paired_state_conservation|state_changed | 512 | 6144 | 0.083 |
| paired_state_conservation|state_unchanged | 1536 | 13312 | 0.115 |

Top eval unknown tokens: `[('flute', 320), ('badge', 320), ('ticket', 320), ('scarf', 320), ('hammer', 320), ('shell', 320), ('drum', 320), ('booklet', 320), ('goblet', 320), ('key', 320), ('telescope', 320), ('bracelet', 320), ('vase', 320), ('satchel', 320), ('radio', 320), ('painting', 320), ('anchor', 96), ('basketball', 96), ('lamp', 96), ('mirror', 96)]`

## Scientific reading

Under `--vocab-scope train`, eval names are not accessible as individual learned embeddings because the normalizer replaces the candidate with `<cand>` and the other participant with `<other>`. Eval object words become `<unk>` when absent from training vocabulary. Therefore a surviving tied-vs-untied separation in the train-only replication cannot be attributed to eval-token vocabulary access.
