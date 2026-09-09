# earlier analysis causal official-eval length audit

Tokenizer: `experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer`

| column | n | max | p99 | >256 | >384 | >512 |
|---|---:|---:|---:|---:|---:|---:|
| BLiMP | 119750 | 33 | 23.0 | 0 | 0 | 0 |
| Supplement | 10436 | 35 | 29.0 | 0 | 0 | 0 |
| EWoK | 15236 | 41 | 32.0 | 0 | 0 | 0 |
| Entity | 33900 | 244 | 217.0 | 0 | 0 | 0 |
| COMPS | 182056 | 34 | 29.0 | 0 | 0 | 0 |
| GlobalPIQA_parallel | 412 | 59 | 57.0 | 0 | 0 | 0 |
| GlobalPIQA_nonparallel | 200 | 203 | 101.0 | 0 | 0 | 0 |
| Reading_context_plus_word | 3245 | 17 | 15.0 | 0 | 0 | 0 |

Overall max=244, >256=0, >384=0, >512=0.

JSON: `experiments/archive/frontier_consolidation/data/causal_eval_length_audit/causal_eval_length_audit.json`
