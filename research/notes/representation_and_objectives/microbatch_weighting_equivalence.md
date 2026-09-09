# chunk stream preflight — Microbatch weighting equivalence

CPU-only synthetic tensor test using first-update shapes from the chunk-stream preflight.

Weighted microbatch gradients matched full-batch gradients below 1e-6 for all cases: True.
Naive averaging of microbatch means changed gradients in at least one case: True.

| tokenizer | L | chunks | active tokens | microbatches | targets | max grad diff weighted | max grad diff naive |
|---|---:|---:|---:|---:|---:|---:|---:|
| legal40k | 64 | 998 | 54461 | 16 | 8118 | 1.164e-09 | 3.115e-04 |
| legal40k | 128 | 555 | 49987 | 9 | 7488 | 9.313e-10 | 4.311e-04 |
| legal40k | 256 | 302 | 54397 | 5 | 8154 | 9.313e-10 | 5.394e-04 |
| minfreq25 | 64 | 1011 | 55457 | 16 | 8289 | 1.164e-09 | 2.416e-04 |
| minfreq25 | 128 | 559 | 51798 | 9 | 7828 | 1.048e-09 | 2.749e-04 |
| minfreq25 | 256 | 305 | 56380 | 5 | 8465 | 9.313e-10 | 3.861e-04 |

Scope: this validates masked-target weighting for deterministic forward paths. Real DeBERTa microbatch training still differs from a hypothetical single full-batch pass through dropout and numerical order, so a future run must record realized targets and accumulation depth per update.

JSON: `experiments/archive/representation_and_objectives/data/microbatch_weighting_equivalence/microbatch_weighting_equivalence.json`
CSV: `experiments/archive/representation_and_objectives/data/microbatch_weighting_equivalence/microbatch_weighting_equivalence.csv`
