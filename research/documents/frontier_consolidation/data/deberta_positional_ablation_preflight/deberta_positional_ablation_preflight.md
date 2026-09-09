# earlier analysis DeBERTa positional ablation preflight

No GPU training/evaluation/upload/leaderboard action occurred.

Legal tokenizer SHA matches expected: `True`
Old mature repeat tokenizer label: `baseline16k` (not legal-matched)

| variant | params | delta vs full | rel_attention | pos_att_type | abs input | role |
|---|---:|---:|---|---|---|---|
| c2p_only_abs | 32620384 | -1847040 | True | c2p | True | retain content->position disentangled channel plus absolute input positions; removes p2c |
| p2c_only_abs | 32620384 | -1847040 | True | p2c | True | retain position->content disentangled channel plus absolute input positions; removes c2p |
| no_disentangle_abs | 30773344 | -3694080 | True | none | True | retain rel_embeddings/absolute input positions but remove both attention score terms; isolates disentangled score contribution |

Required exact estimand: compact-minus-repeat within each ablated architecture under the legal spatial repair route status tokenizer.
The old baseline16k repeat run cannot be reused as the legal reference for a new ablated architecture.
