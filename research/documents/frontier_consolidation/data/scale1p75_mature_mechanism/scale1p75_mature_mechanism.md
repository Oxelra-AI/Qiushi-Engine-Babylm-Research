# scale1p75 pre80m state scale1.75 mature mechanism

Status: `SCALE1P75_MATURE_MECHANISM`

## Stock displacement vs matched spatial repair route status

| exposure | cosine | rel L2 | top diff group | top group fraction |
|---|---:|---:|---|---:|
| 50M | 0.755852 | 0.698160 | embeddings | 0.1713 |
| 70M | 0.737882 | 0.723647 | embeddings | 0.1696 |
| 80M | 0.735855 | 0.726465 | embeddings | 0.1685 |

## Stock update alignment

| interval | update cosine | adapter/spatial repair route status update norm |
|---|---:|---:|
| stock_update_50M_to_70M | 0.172684 | 1.004788 |
| stock_update_70M_to_80M | 0.171162 | 1.000553 |
| stock_update_50M_to_80M | 0.180663 | 1.005928 |

## Adapter parameter RMS

| exposure | all RMS | up RMS | down RMS | layer-norm RMS |
|---|---:|---:|---:|---:|
| 50M | 0.066315 | 0.018637 | 0.035987 | 0.681636 |
| 70M | 0.066200 | 0.019462 | 0.036530 | 0.676831 |
| 80M | 0.066164 | 0.019547 | 0.036559 | 0.676118 |

## Core matrix spectra

| checkpoint | stable rank mean | top8 energy mean |
|---|---:|---:|
| reference_50M | 32.754 | 0.2946 |
| reference_70M | 32.943 | 0.2942 |
| reference_80M | 32.919 | 0.2941 |
| scale1p75_50M | 32.963 | 0.2935 |
| scale1p75_70M | 32.915 | 0.2933 |
| scale1p75_80M | 32.952 | 0.2932 |
