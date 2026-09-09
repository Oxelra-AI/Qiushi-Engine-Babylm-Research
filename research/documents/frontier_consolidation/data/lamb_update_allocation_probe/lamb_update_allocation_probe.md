# lamb update allocation closure LAMB update-allocation probe

Fixed first spatial repair route status WWM batch; local first-step gradient readout. Optimizer moment buffers are not available in HF checkpoints, so this measures the intended update-allocation mechanism rather than exact historical moments.

## All-tensor summary

| model | loss | grad_norm | trust p50 | trust p10-p90 | clipped | full-vector cos vs uniform AdamW | LAMB/AdamW full step norm | multiplier p50 | multiplier p10-p90 | WD/Adam p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact_init | 9.8220 | 2.4778 | 1.0000 | 0.0214-1.0111 | 0 | 0.4433 | 0.28 | 5.00 | 0.11-5.06 | 0.0002 |
| reference_20M | 3.6980 | 1.4093 | 0.0251 | 0.0059-0.9973 | 0 | 0.8192 | 0.19 | 0.13 | 0.03-4.99 | 0.0003 |
| lamb005_14M | 4.6839 | 2.1932 | 0.0211 | 0.0001-0.8723 | 0 | 0.7448 | 0.17 | 0.11 | 0.00-4.36 | 0.0002 |
| lamb007_20M | 6.8260 | 0.4668 | 10.0000 | 0.1220-10.0000 | 130 | 0.7943 | 0.26 | 70.00 | 0.85-70.00 | 3.0024 |

## Broad family trust ratios and step multipliers

### exact_init

| family | tensors | trust median | trust p10-p90 | multiplier median | multiplier p10-p90 | full-vector cos | full stepnorm ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| encoder_attention_out | 16 | 0.5109 | 0.0216-1.0000 | 2.55 | 0.11-5.00 | 0.4479 | 0.27 |
| encoder_attention_qkv | 48 | 0.5186 | 0.0218-1.0000 | 2.59 | 0.11-5.00 | 0.5595 | 0.27 |
| encoder_ffn_in | 16 | 0.5104 | 0.0208-1.0000 | 2.55 | 0.10-5.00 | 0.4512 | 0.25 |
| encoder_ffn_out | 16 | 0.5104 | 0.0208-1.0000 | 2.55 | 0.10-5.00 | 0.6787 | 0.16 |
| encoder_norm_bias | 48 | 1.0000 | 1.0000-1.0132 | 5.00 | 5.00-5.07 | 1.0000 | 5.03 |
| encoder_other | 16 | 0.9944 | 0.9507-1.0231 | 4.97 | 4.75-5.12 | 0.9995 | 4.95 |
| mlm_head | 5 | 1.0000 | 0.4121-1.0066 | 5.00 | 2.06-5.03 | 0.3316 | 1.31 |
| other | 1 | 0.0286 | 0.0286-0.0286 | 0.14 | 0.14-0.14 | 1.0000 | 0.14 |
| other_norm_bias | 2 | 1.0063 | 1.0013-1.0113 | 5.03 | 5.01-5.06 | 1.0000 | 5.03 |
| relative_position | 1 | 0.2838 | 0.2838-0.2838 | 1.42 | 1.42-1.42 | 1.0000 | 1.42 |
| word_embeddings | 1 | 0.0234 | 0.0234-0.0234 | 0.12 | 0.12-0.12 | 1.0000 | 0.12 |

### reference_20M

| family | tensors | trust median | trust p10-p90 | multiplier median | multiplier p10-p90 | full-vector cos | full stepnorm ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| encoder_attention_out | 16 | 0.0168 | 0.0077-0.0253 | 0.08 | 0.04-0.13 | 0.9992 | 0.12 |
| encoder_attention_qkv | 48 | 0.0245 | 0.0043-0.1012 | 0.12 | 0.02-0.51 | 0.9955 | 0.13 |
| encoder_ffn_in | 16 | 0.0208 | 0.0145-0.0273 | 0.10 | 0.07-0.14 | 0.9993 | 0.13 |
| encoder_ffn_out | 16 | 0.0168 | 0.0056-0.0239 | 0.08 | 0.03-0.12 | 0.9997 | 0.12 |
| encoder_norm_bias | 48 | 0.3743 | 0.0063-1.0221 | 1.87 | 0.03-5.11 | 0.6546 | 3.22 |
| encoder_other | 16 | 0.0655 | 0.0560-0.0851 | 0.33 | 0.28-0.43 | 0.9853 | 0.35 |
| mlm_head | 5 | 0.0405 | 0.0184-0.7568 | 0.20 | 0.09-3.78 | 0.4671 | 0.30 |
| other | 1 | 0.0242 | 0.0242-0.0242 | 0.12 | 0.12-0.12 | 1.0000 | 0.12 |
| other_norm_bias | 2 | 0.5018 | 0.1043-0.8993 | 2.51 | 0.52-4.50 | 0.7093 | 3.52 |
| relative_position | 1 | 0.0613 | 0.0613-0.0613 | 0.31 | 0.31-0.31 | 1.0000 | 0.31 |
| word_embeddings | 1 | 0.0464 | 0.0464-0.0464 | 0.23 | 0.23-0.23 | 1.0000 | 0.23 |

### lamb005_14M

| family | tensors | trust median | trust p10-p90 | multiplier median | multiplier p10-p90 | full-vector cos | full stepnorm ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| encoder_attention_out | 16 | 0.0108 | 0.0002-0.0220 | 0.05 | 0.00-0.11 | 0.9988 | 0.11 |
| encoder_attention_qkv | 48 | 0.0108 | 0.0000-0.0251 | 0.05 | 0.00-0.13 | 0.9969 | 0.12 |
| encoder_ffn_in | 16 | 0.0109 | 0.0002-0.0222 | 0.05 | 0.00-0.11 | 0.9989 | 0.11 |
| encoder_ffn_out | 16 | 0.0108 | 0.0002-0.0221 | 0.05 | 0.00-0.11 | 0.9997 | 0.11 |
| encoder_norm_bias | 48 | 0.0002 | 0.0000-1.0819 | 0.00 | 0.00-5.41 | 0.6431 | 3.39 |
| encoder_other | 16 | 0.0686 | 0.0503-0.1075 | 0.34 | 0.25-0.54 | 0.9613 | 0.33 |
| mlm_head | 5 | 0.0003 | 0.0002-1.4317 | 0.00 | 0.00-7.16 | 0.2292 | 0.53 |
| other | 1 | 0.0263 | 0.0263-0.0263 | 0.13 | 0.13-0.13 | 1.0000 | 0.13 |
| other_norm_bias | 2 | 0.5139 | 0.1029-0.9248 | 2.57 | 0.51-4.62 | 0.7054 | 3.62 |
| relative_position | 1 | 0.0337 | 0.0337-0.0337 | 0.17 | 0.17-0.17 | 1.0000 | 0.17 |
| word_embeddings | 1 | 0.0293 | 0.0293-0.0293 | 0.15 | 0.15-0.15 | 1.0000 | 0.15 |

### lamb007_20M

| family | tensors | trust median | trust p10-p90 | multiplier median | multiplier p10-p90 | full-vector cos | full stepnorm ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| encoder_attention_out | 16 | 10.0000 | 0.3161-10.0000 | 70.00 | 2.21-70.00 | 0.8000 | 1.11 |
| encoder_attention_qkv | 48 | 10.0000 | 6.9927-10.0000 | 70.00 | 48.95-70.00 | 0.7730 | 5.90 |
| encoder_ffn_in | 16 | 10.0000 | 0.2254-10.0000 | 70.00 | 1.58-70.00 | 0.8496 | 1.38 |
| encoder_ffn_out | 16 | 10.0000 | 0.0720-10.0000 | 70.00 | 0.50-70.00 | 0.7194 | 0.58 |
| encoder_norm_bias | 48 | 10.0000 | 0.2555-10.0000 | 70.00 | 1.79-70.00 | 0.5287 | 7.27 |
| encoder_other | 16 | 10.0000 | 10.0000-10.0000 | 70.00 | 70.00-70.00 | 1.0000 | 70.00 |
| mlm_head | 5 | 0.0004 | 0.0003-0.4926 | 0.00 | 0.00-3.45 | 0.5379 | 0.37 |
| other | 1 | 10.0000 | 10.0000-10.0000 | 70.00 | 70.00-70.00 | 1.0000 | 70.00 |
| other_norm_bias | 2 | 10.0000 | 10.0000-10.0000 | 70.00 | 70.00-70.00 | 1.0000 | 70.00 |
| relative_position | 1 | 10.0000 | 10.0000-10.0000 | 70.00 | 70.00-70.00 | 1.0000 | 70.00 |
| word_embeddings | 1 | 0.0278 | 0.0278-0.0278 | 0.19 | 0.19-0.19 | 1.0000 | 0.19 |

## Core hidden spectra

| model | stable rank | entropy rank | top8 energy |
|---|---:|---:|---:|
| exact_init | 152.97 | 335.13 | 0.0535 |
| reference_20M | 43.96 | 274.82 | 0.1332 |
| lamb005_14M | 61.18 | 318.05 | 0.0845 |
| lamb007_20M | 7.20 | 192.32 | 0.2468 |

## Interpretation notes

A high per-tensor direction cosine with a low full-vector cosine means LAMB preserves local Adam-like directions while reallocating update magnitude across tensors/layers. Extreme trust-ratio clipping or very large interface multipliers would weaken the intended mechanism and make a mature continuation less scientifically clean.
