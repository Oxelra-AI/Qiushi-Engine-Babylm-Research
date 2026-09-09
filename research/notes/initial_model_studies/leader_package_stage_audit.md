# leader package stage audit-125 — leader-package staging audit

Evidence JSON: `experiments/archive/initial_model_studies/data/leader_package_revision_124/model_side/model_side_stage_audit.json`

## Data access status

- Model repo `go76dof/wwm_curriculum_simplification_40k` @ `da3b5c14579cda8f30ab95b6d6843c3fa2f50af5`: accessible (config + tokenizer).
- Dataset repo `go76dof/Fineweb_simplification_pairs`: **GATED** — train file and dataset tokenizer return 403 in this environment. Exact leader training data is not locally reproducible without authorization.

## Model config core

| field | value |
|---|---|
| model_type | `deberta-v2` |
| hidden_size | `384` |
| num_hidden_layers | `12` |
| num_attention_heads | `12` |
| intermediate_size | `1280` |
| vocab_size | `40000` |
| relative_attention | `True` |
| pos_att_type | `['p2c', 'c2p']` |
| max_position_embeddings | `1024` |
| position_buckets | `256` |
| max_relative_positions | `-1` |
| norm_rel_ebd | `layer_norm` |
| layer_norm_eps | `1e-07` |
| hidden_act | `gelu` |

## Tokenizer audit

- SentencePiece load ok, piece size `40000`.
- `19` words -> `20` pieces: ['▁Enlightenment', '▁thinkers', '▁believed', '▁that', '▁using', '▁reason', '▁and', '▁studying', '▁the', '▁world', '▁would', '▁lead', '▁to', '▁scientific', '▁progress', '▁and', '▁better', '▁living', '▁conditions', '.']
- `10` words -> `12` pieces: ['▁The', '▁capital', '▁of', '▁France', '▁is', '▁Paris', '.', '▁It', '▁is', '▁in', '▁Europe', '.']
- `9` words -> `11` pieces: ['▁The', '▁dog', '▁chased', '▁the', '▁ball', '.', '▁It', '▁was', '▁very', '▁happy', '.']

## File hashes

| file | status | size | sha256 |
|---|---|---:|---|
| README.md | ok | 7,277 | `9ac1c1dae652714f...` |
| config.json | ok | 932 | `517f2d4186f1386e...` |
| tokenizer_config.json | ok | 1,287 | `1ef1a7e642808589...` |
| special_tokens_map.json | ok | 286 | `9463f61e1b109a8e...` |
| spm.model | ok | 918,622 | `1157bea377197596...` |

## Decision implication

- The leader is a coupled DeBERTa-v2 MLM + FineWeb simplification-pair data + 40k SentencePiece + LAMB + length/mask curriculum.
- We can reproduce architecture and tokenizer *shape*, but not the exact gated training data.
- Next executable options: (a) seek authorized dataset access; (b) build a legal simplification-pair corpus within strict word accounting from accessible sources; (c) run the leak-tested hybrid pilot in parallel while data access is unresolved.
