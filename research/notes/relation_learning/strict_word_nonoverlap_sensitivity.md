# pair level relation robustness strict word-level nonoverlap sensitivity

This CPU-only check replays the compact-pair tokenization and filters the already-scored rewrite probe rows to targets whose whole normalized surface word does not occur among normalized source words. It is stricter than the original tokenizer-ID nonoverlap label, though still heuristic for morphology and named entities.

Replay map rows: 11845; strict-word rows: 8605; strict-word pairs: 1624.

| arch | seed | contrast | n tokens min | n pairs min | gain delta | true-source delta | unrelated delta | excess true-source cost | frac tokens excess>0 | checkpoint gains |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| D | 43022 | RminusC | 2184 | 1393 | -0.5883 | +0.3031 | -0.2852 | +0.5883 | 0.601 | chck_100M:-0.519932;chck_80M:-0.723074;chck_90M:-0.521921 |
| D | 43022 | VminusC | 2184 | 1393 | +0.5400 | -1.0679 | -0.5279 | -0.5400 | 0.404 | chck_100M:0.543354;chck_80M:0.517937;chck_90M:0.55869 |
| D | 43022 | VminusR | 2184 | 1393 | +1.1283 | -1.3710 | -0.2427 | -1.1283 | 0.333 | chck_100M:1.06329;chck_80M:1.24101;chck_90M:1.08061 |
| D | 43122 | RminusC | 2184 | 1393 | -0.8296 | +0.5309 | -0.2987 | +0.8296 | 0.620 | chck_100M:-0.851446;chck_80M:-0.790372;chck_90M:-0.847045 |
| D | 43122 | VminusC | 2184 | 1393 | +0.7349 | -1.2256 | -0.4907 | -0.7349 | 0.381 | chck_100M:0.751313;chck_80M:0.715218;chck_90M:0.738238 |
| D | 43122 | VminusR | 2184 | 1393 | +1.5645 | -1.7565 | -0.1920 | -1.5645 | 0.282 | chck_100M:1.60276;chck_80M:1.50559;chck_90M:1.58528 |
| RBT | 43022 | RminusC | 2184 | 1393 | -0.2515 | +0.4327 | +0.1811 | +0.2515 | 0.564 | chck_100M:-0.347378;chck_60M:-0.0617587;chck_70M:-0.154181;chck_80M:-0.346672;chck_90M:-0.347612 |
| RBT | 43022 | VminusC | 2184 | 1393 | +0.0436 | -0.4463 | -0.4027 | -0.0436 | 0.482 | chck_100M:0.0521346;chck_60M:-0.03775;chck_70M:0.105095;chck_80M:0.0477139;chck_90M:0.0507431 |
| RBT | 43022 | VminusR | 2184 | 1393 | +0.2951 | -0.8790 | -0.5839 | -0.2951 | 0.414 | chck_100M:0.399512;chck_60M:0.0240087;chck_70M:0.259276;chck_80M:0.394386;chck_90M:0.398355 |

The main relation-cost result survives the stricter word filter. REPEAT remains worse than CLEAN in source-conditioned use of held-out nonidentical words for both DeBERTa seeds and for RoBERTa, with positive excess true-source cost after subtracting the unrelated-source control. DeBERTa also retains a strong VIEW-over-CLEAN positive conditioning residual under this stricter filter, while RoBERTa's V-C residual remains small. The sample is smaller than the tokenizer-level analysis, so it should be used as a sensitivity check, not as the primary estimate.
