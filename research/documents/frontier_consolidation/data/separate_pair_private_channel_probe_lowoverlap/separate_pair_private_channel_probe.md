# earlier analysis — separate-encoding paired-view private channel

Each source sentence and compact rewrite was encoded as a separate input by the frozen spatial repair route status legal checkpoint. This removes same-row bidirectional attention to the partner text before measuring or learning source/rewrite alignment.

## Legal corpus substrate
- selected pairs: `3779` from `900` packed changed rows; train/test split by row = `630`/`270` rows.
- pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`; tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.
- content_overlap mean: `0.5539`; content_recall mean: `0.6317`.

## Held-out same-row retrieval after removing partner attention
| Channel | same-row top1 | true - max same-row decoy | global top1 | global top5 |
|---|---:|---:|---:|---:|
| identity frozen reps | 1.0000 | 0.5992 | 0.9725 | 0.9956 |
| detached private projection | 0.9991 | 0.6100 | 0.9778 | 0.9938 |

## Private-channel readout
- same-row margin improvement over identity: `0.010736869848236186`.
- same-row top1 improvement over identity: `-0.0008873114463177068`.
- private_channel_signal_present: `True`.
- private_channel_adds_over_identity: `True`.
- Backbone preservation in this test is exact: the spatial repair route status model is frozen, embeddings are detached, and the private channel is not connected to LM logits.

## Scientific use
This is not BabyLM score evidence. It is a cheap corpus-derived test for whether a private fast pathway can learn paired-view structure without displacing the protected spatial repair route status language function. It should guide route construction, not trigger an automatic full run.

JSON: `experiments/archive/frontier_consolidation/data/separate_pair_private_channel_probe_lowoverlap/separate_pair_private_channel_probe.json`
