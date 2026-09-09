# rtd closure and muon route decision Muon spectral update probe

This zero-training measurement estimates the singular-direction concentration of local MLM hidden-matrix updates at existing spatial repair route status legal checkpoints. It uses replayed batches because the original optimizer buffers were not saved.

Probe data: 5184 examples, 799882 words; used 12 batches.

## Core hidden matrices (attention QKV/output + FFN in/out; excluding embeddings, MLM head, DeBERTa position projections)

| checkpoint | momentum stable-rank mean | AdamW-step stable-rank mean | momentum top-8 Frobenius share | cos(momentum, polar) | cos(AdamW-step, polar) |
|---|---:|---:|---:|---:|---:|
| mlm20_step35 | 2.66 | 3.55 | 0.738 | 0.365 | 0.378 |
| mlm80_step35 | 5.06 | 7.54 | 0.482 | 0.545 | 0.568 |

## Family summary

| checkpoint | family | n | momentum stable-rank | AdamW-step stable-rank | momentum top-8 share | cos(momentum, polar) |
|---|---|---:|---:|---:|---:|---:|
| mlm20_step35 | attention_qkv | 24 | 2.92 | 4.10 | 0.784 | 0.309 |
| mlm20_step35 | attention_output | 8 | 2.30 | 2.78 | 0.741 | 0.351 |
| mlm20_step35 | ffn_in | 8 | 3.08 | 4.37 | 0.591 | 0.511 |
| mlm20_step35 | ffn_out | 8 | 1.85 | 1.83 | 0.746 | 0.401 |
| mlm20_step35 | deberta_position_projection | 16 | 1.90 | 2.00 | 0.968 | 0.172 |
| mlm20_step35 | relative_position_embedding | 1 | 1.31 | 3.17 | 0.992 | 0.118 |
| mlm20_step35 | word_embedding | 1 | 3.54 | 4.47 | 0.687 | 0.442 |
| mlm20_step35 | mlm_transform | 1 | 2.38 | 3.25 | 0.732 | 0.357 |
| mlm80_step35 | attention_qkv | 24 | 4.21 | 7.37 | 0.582 | 0.446 |
| mlm80_step35 | attention_output | 8 | 5.53 | 7.56 | 0.455 | 0.536 |
| mlm80_step35 | ffn_in | 8 | 6.53 | 10.60 | 0.332 | 0.704 |
| mlm80_step35 | ffn_out | 8 | 5.69 | 4.95 | 0.360 | 0.696 |
| mlm80_step35 | deberta_position_projection | 16 | 2.64 | 2.76 | 0.903 | 0.241 |
| mlm80_step35 | relative_position_embedding | 1 | 1.51 | 3.48 | 0.982 | 0.164 |
| mlm80_step35 | word_embedding | 1 | 6.59 | 8.60 | 0.400 | 0.640 |
| mlm80_step35 | mlm_transform | 1 | 2.95 | 4.51 | 0.563 | 0.494 |

## Reading for route choice

- Core hidden-matrix local momentum stable rank is 2.66 at 20M and 5.06 at 80M, far below the 480-rank capacity of the square attention matrices and rectangular FFN bottleneck rank.
- The top 8 singular directions carry 0.738 (20M) and 0.482 (80M) of local momentum Frobenius energy across the core hidden matrices.
- This supports a real, testable headroom for Muon-style hidden-matrix update orthogonalization while leaving embeddings, MLM head, norms, and relative-position embedding tables on AdamW.

Elapsed: 108.0s
