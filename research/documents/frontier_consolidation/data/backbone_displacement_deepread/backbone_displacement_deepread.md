# adapter matched horizon plan backbone displacement deep read

This replaces the earlier unweighted mean-over-tensors displacement with whole-vector and group-weighted measurements.

| comparison | whole-vector cosine | rel L2 to reference | L2 | ref norm | top diff group | top group diff fraction |
|---|---:|---:|---:|---:|---|---:|
| live128_20M vs disabled128_20M | 0.89024396 | 0.46809410 | 91.9989 | 196.5393 | embeddings | 0.1478 |
| disabled128_20M vs reference_20M | 1.00000000 | 0.00000000 | 0.0000 | 196.5393 | embeddings | 0.0000 |
| reference_20M vs exact_init_stock | 0.76160741 | 0.85123028 | 127.3651 | 149.6247 | embeddings | 0.2993 |
| reference_21M vs reference_20M | 0.99562767 | 0.09435116 | 18.5437 | 196.5393 | embeddings | 0.2980 |
| reference_50M vs reference_20M | 0.88865996 | 0.52703647 | 103.5834 | 196.5393 | embeddings | 0.2927 |

## Top tensors for live vs disabled by squared-difference fraction

| tensor | group | frac | rel L2 | cosine |
|---|---|---:|---:|---:|
| `deberta.embeddings.word_embeddings.weight` | embeddings | 0.1466 | 0.3908 | 0.923717 |
| `deberta.encoder.layer.3.attention.self.pos_key_proj.weight` | layer3_attention | 0.0374 | 0.7170 | 0.740662 |
| `deberta.encoder.layer.6.attention.self.pos_key_proj.weight` | layer6_attention | 0.0358 | 0.6569 | 0.776139 |
| `deberta.encoder.layer.2.attention.self.pos_key_proj.weight` | layer2_attention | 0.0355 | 0.6831 | 0.774111 |
| `deberta.encoder.layer.4.attention.self.pos_key_proj.weight` | layer4_attention | 0.0334 | 0.6739 | 0.773285 |
| `deberta.encoder.layer.7.attention.self.pos_key_proj.weight` | layer7_attention | 0.0291 | 0.6156 | 0.807704 |
| `deberta.encoder.layer.6.intermediate.dense.weight` | layer6_ffn_output | 0.0257 | 0.5782 | 0.832693 |
| `deberta.encoder.layer.7.intermediate.dense.weight` | layer7_ffn_output | 0.0254 | 0.5745 | 0.835306 |
| `deberta.encoder.layer.7.attention.self.pos_query_proj.weight` | layer7_attention | 0.0253 | 0.7052 | 0.757144 |
| `deberta.encoder.layer.6.attention.self.pos_query_proj.weight` | layer6_attention | 0.0252 | 0.7717 | 0.692669 |
| `deberta.encoder.layer.5.attention.self.pos_key_proj.weight` | layer5_attention | 0.0246 | 0.5140 | 0.866548 |
| `deberta.encoder.layer.5.intermediate.dense.weight` | layer5_ffn_output | 0.0243 | 0.5712 | 0.836509 |
| `deberta.encoder.layer.1.attention.self.pos_key_proj.weight` | layer1_attention | 0.0235 | 0.5537 | 0.845997 |
| `deberta.encoder.layer.4.intermediate.dense.weight` | layer4_ffn_output | 0.0232 | 0.5660 | 0.839447 |
| `deberta.encoder.layer.3.intermediate.dense.weight` | layer3_ffn_output | 0.0219 | 0.5560 | 0.845370 |
| `deberta.encoder.layer.2.attention.self.pos_query_proj.weight` | layer2_attention | 0.0206 | 0.7527 | 0.717859 |
| `deberta.encoder.layer.2.intermediate.dense.weight` | layer2_ffn_output | 0.0200 | 0.5362 | 0.856569 |
| `deberta.encoder.layer.3.attention.self.pos_query_proj.weight` | layer3_attention | 0.0183 | 0.7395 | 0.724864 |
| `deberta.encoder.layer.5.attention.self.pos_query_proj.weight` | layer5_attention | 0.0178 | 0.6382 | 0.795949 |
| `deberta.encoder.layer.0.attention.self.pos_key_proj.weight` | layer0_attention | 0.0175 | 0.5067 | 0.870002 |

## Top tensors for live vs disabled by relative L2

| tensor | group | rel L2 | diff fraction | cosine |
|---|---|---:|---:|---:|
| `deberta.encoder.layer.2.attention.self.pos_key_proj.bias` | layer2_attention | 1.7559 | 0.000000 | 0.122933 |
| `deberta.encoder.layer.6.attention.self.pos_key_proj.bias` | layer6_attention | 1.3220 | 0.000000 | 0.466270 |
| `deberta.encoder.layer.0.attention.self.pos_key_proj.bias` | layer0_attention | 1.2471 | 0.000000 | 0.283404 |
| `deberta.encoder.layer.3.attention.self.pos_key_proj.bias` | layer3_attention | 1.2347 | 0.000000 | 0.153903 |
| `deberta.encoder.layer.4.attention.self.pos_key_proj.bias` | layer4_attention | 1.2306 | 0.000000 | 0.395105 |
| `deberta.encoder.layer.1.attention.self.pos_key_proj.bias` | layer1_attention | 1.0679 | 0.000000 | 0.315305 |
| `deberta.encoder.layer.7.attention.self.pos_key_proj.bias` | layer7_attention | 1.0229 | 0.000000 | 0.532239 |
| `deberta.encoder.layer.5.attention.self.pos_key_proj.bias` | layer5_attention | 0.9556 | 0.000000 | 0.412375 |
| `deberta.encoder.layer.6.attention.self.pos_query_proj.weight` | layer6_attention | 0.7717 | 0.025170 | 0.692669 |
| `deberta.encoder.layer.2.attention.self.query_proj.bias` | layer2_attention | 0.7690 | 0.000009 | 0.773360 |
| `deberta.encoder.layer.2.attention.self.pos_query_proj.weight` | layer2_attention | 0.7527 | 0.020559 | 0.717859 |
| `deberta.encoder.layer.3.attention.self.pos_query_proj.weight` | layer3_attention | 0.7395 | 0.018276 | 0.724864 |
| `deberta.encoder.layer.3.attention.self.pos_key_proj.weight` | layer3_attention | 0.7170 | 0.037441 | 0.740662 |
| `deberta.encoder.layer.2.attention.self.query_proj.weight` | layer2_attention | 0.7129 | 0.009413 | 0.748629 |
| `deberta.encoder.layer.7.attention.self.pos_query_proj.weight` | layer7_attention | 0.7052 | 0.025303 | 0.757144 |
| `deberta.encoder.layer.6.attention.self.value_proj.bias` | layer6_attention | 0.7039 | 0.000000 | 0.721685 |
| `deberta.encoder.layer.3.attention.self.query_proj.weight` | layer3_attention | 0.7014 | 0.008844 | 0.753116 |
| `deberta.encoder.layer.7.attention.self.query_proj.weight` | layer7_attention | 0.6912 | 0.009710 | 0.758290 |
| `deberta.encoder.layer.6.attention.self.query_proj.weight` | layer6_attention | 0.6882 | 0.008535 | 0.763611 |
| `deberta.encoder.layer.2.attention.self.pos_key_proj.weight` | layer2_attention | 0.6831 | 0.035519 | 0.774111 |
