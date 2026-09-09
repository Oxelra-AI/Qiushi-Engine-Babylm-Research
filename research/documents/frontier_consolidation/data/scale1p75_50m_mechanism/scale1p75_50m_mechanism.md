# scale1p75 pre80m state scale1.75 50M mechanism readout

## Stock displacement

| comparison | cosine | rel L2 to reference | L2 | top diff group | top group fraction |
|---|---:|---:|---:|---|---:|
| scale1p75_20M_stock_vs_step35_20M | 0.879068 | 0.491312 | 96.562 | embeddings | 0.1440 |
| scale1p75_50M_stock_vs_step35_50M | 0.755852 | 0.698160 | 157.584 | embeddings | 0.1713 |
| scale1p75_stock_50M_vs_20M | 0.888444 | 0.527622 | 103.492 | embeddings | 0.2925 |
| stock_50M_vs_20M | 0.888660 | 0.527036 | 103.583 | embeddings | 0.2927 |

## Stock update alignment from 20M to 50M
- cosine between scale1.75 stock update and spatial repair route status stock update: 0.304957
- scale1.75 stock update norm / spatial repair route status stock update norm: 0.999117
- scale1.75 update norm: 103.492; spatial repair route status update norm: 103.583

## Adapter parameter growth

| checkpoint | adapter RMS all | up RMS | down RMS | layer-norm RMS | up norm sum | down norm sum |
|---|---:|---:|---:|---:|---:|---:|
| scale1p75_20M | 0.066006 | 0.012710 | 0.031639 | 0.700126 | 26.201 | 65.561 |
| scale1p75_50M | 0.066315 | 0.018637 | 0.035987 | 0.681636 | 38.565 | 74.736 |

## Core stock-matrix spectra

| checkpoint | n matrices | stable-rank mean | top8-energy mean |
|---|---:|---:|---:|
| reference_20M | 64 | 33.671 | 0.2873 |
| reference_50M | 64 | 32.754 | 0.2946 |
| scale1p75_20M | 64 | 34.383 | 0.2867 |
| scale1p75_50M | 64 | 32.963 | 0.2935 |

## Top 20M-to-50M update-difference tensors

| tensor | group | diff fraction | update cosine |
|---|---|---:|---:|
| `deberta.embeddings.word_embeddings.weight` | embeddings | 0.1895 | 0.54708 |
| `deberta.encoder.layer.6.intermediate.dense.weight` | layer6_ffn_output | 0.0342 | 0.10583 |
| `deberta.encoder.layer.5.intermediate.dense.weight` | layer5_ffn_output | 0.0339 | 0.10682 |
| `deberta.encoder.layer.4.intermediate.dense.weight` | layer4_ffn_output | 0.0336 | 0.11670 |
| `deberta.encoder.layer.7.intermediate.dense.weight` | layer7_ffn_output | 0.0329 | 0.13540 |
| `deberta.encoder.layer.3.intermediate.dense.weight` | layer3_ffn_output | 0.0329 | 0.12900 |
| `deberta.encoder.layer.2.intermediate.dense.weight` | layer2_ffn_output | 0.0323 | 0.14151 |
| `deberta.encoder.layer.1.intermediate.dense.weight` | layer1_ffn_output | 0.0291 | 0.15906 |
| `deberta.encoder.layer.0.intermediate.dense.weight` | layer0_ffn_output | 0.0248 | 0.20276 |
| `deberta.encoder.layer.6.output.dense.weight` | layer6_ffn_output | 0.0244 | 0.10203 |
| `deberta.encoder.layer.7.output.dense.weight` | layer7_ffn_output | 0.0243 | 0.15236 |
| `deberta.encoder.layer.5.output.dense.weight` | layer5_ffn_output | 0.0232 | 0.09352 |
| `deberta.encoder.layer.4.output.dense.weight` | layer4_ffn_output | 0.0225 | 0.09531 |
| `deberta.encoder.layer.3.output.dense.weight` | layer3_ffn_output | 0.0217 | 0.10625 |
| `deberta.encoder.layer.2.output.dense.weight` | layer2_ffn_output | 0.0208 | 0.11477 |
