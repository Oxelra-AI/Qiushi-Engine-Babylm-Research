# dual mechanism 20m overlap — scale1.75 endpoint mechanism readout

Status: `SCALE1P75_ENDPOINT_MECHANISM`

## Stock displacement vs matched spatial repair route status

| exposure | cosine | rel L2 | top diff group | top group fraction |
|---|---:|---:|---|---:|
| 80M | 0.735855 | 0.726465 | embeddings | 0.1685 |
| 90M | 0.735368 | 0.727123 | embeddings | 0.1681 |
| 100M | 0.735329 | 0.727179 | embeddings | 0.1680 |

## Late stock update alignment

| interval | update cosine | adapter/spatial repair route status update norm |
|---|---:|---:|
| stock_update_80M_to_90M | 0.180564 | 0.999138 |
| stock_update_90M_to_100M | 0.186034 | 0.998145 |
| stock_update_80M_to_100M | 0.183354 | 0.999354 |

## Adapter parameter RMS

| exposure | all RMS | up RMS | down RMS | layer-norm RMS |
|---|---:|---:|---:|---:|
| 80M | 0.066164 | 0.019547 | 0.036559 | 0.676118 |
| 90M | 0.066151 | 0.019564 | 0.036557 | 0.675926 |
| 100M | 0.066149 | 0.019565 | 0.036555 | 0.675897 |

## Core matrix spectra

| checkpoint | stable rank mean | top8 energy mean |
|---|---:|---:|
| reference_80M | 32.919 | 0.2941 |
| reference_90M | 32.959 | 0.2940 |
| reference_100M | 32.963 | 0.2940 |
| scale1p75_80M | 32.952 | 0.2932 |
| scale1p75_90M | 32.954 | 0.2931 |
| scale1p75_100M | 32.963 | 0.2931 |

## Scientific reading

At 100M the stock-parameter displacement remains large but smooth (cosine 0.7353, rel-L2 0.7272), while the final 90M→100M stock update still has low alignment with spatial repair route status (cosine 0.1860) and near-equal norm ratio 0.9981. Adapter RMS remains stable (0.066149). This means the endpoint is still in the same separately routed trajectory-redirection regime seen at 80M, not a late collapse or a hidden spectral-breadth transition. The official score will decide whether this redirected endpoint is broadly useful or a localized redistribution.

JSON: `experiments/archive/frontier_consolidation/data/scale1p75_endpoint_mechanism/scale1p75_endpoint_mechanism.json`
