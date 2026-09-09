# u256 endpoint mechanism — U256 endpoint mechanism readout

Status: `U256_ENDPOINT_MECHANISM`

## Metric core
- spatial repair route status: words=100000000, steps=2529, loss=9.837543487548828→2.5525617599487305, params=34467424, checkpoints=100
- u256: words=100000000, steps=2530, loss=9.826857208144903→2.516624725910071, params=34467424, checkpoints=100

## Weight displacement versus spatial repair route status
| exposure | cosine | rel L2 | top diff group | top diff fraction |
|---|---:|---:|---|---:|
| 20M | 0.797325 | 0.637042 | embeddings | 0.2284 |
| 80M | 0.645656 | 0.843406 | embeddings | 0.2236 |
| 90M | 0.645322 | 0.843782 | embeddings | 0.2228 |
| 100M | 0.645323 | 0.843789 | embeddings | 0.2227 |

## Update alignment
| interval | update cosine | U256/spatial repair route status update norm |
|---|---:|---:|
| 20M_to_80M | 0.183527 | 1.011383 |
| 80M_to_90M | 0.019275 | 1.004106 |
| 90M_to_100M | 0.017136 | 1.008255 |
| 80M_to_100M | 0.023062 | 1.004651 |

## Core matrix spectra
| checkpoint | stable-rank mean | top8-energy mean |
|---|---:|---:|
| reference_100M | 32.963 | 0.2940 |
| reference_80M | 32.919 | 0.2941 |
| u256_100M | 33.315 | 0.2928 |
| u256_80M | 33.296 | 0.2929 |

## Scientific reading
U256 began as a strong 20M behavioral screen (+0.9179 cheap7) from only a 2.6% active-token visibility repair. The endpoint tensor geometry is a large same-architecture trajectory redirection: 20M cosine 0.7973, rel-L2 0.6370; 100M cosine 0.6453, rel-L2 0.8438. The final 90M→100M update has cosine 0.0171 and norm ratio 1.0083 versus spatial repair route status. Core matrix spectra do not show a Muon/LAMB-like collapse or broadening: U256 stable-rank mean changes 33.296→33.315 from 80M to 100M. Thus the pending official score should be interpreted as the behavioral effect of making row-tail experience visible under the same architecture/objective, not as a new parameterization or optimizer artifact.

JSON: `experiments/archive/frontier_consolidation/data/u256_endpoint_mechanism/u256_endpoint_mechanism.json`
