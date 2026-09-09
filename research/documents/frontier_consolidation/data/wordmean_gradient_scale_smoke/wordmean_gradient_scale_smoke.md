# wordmean screen and substrate constraints word-mean actual-gradient scale smoke

No optimizer step and no training; same initialized model, same actual training rows, same masks and dropout seeds for token-mean and word-mean backward passes.

Device: `cpu`; microbatches: `2`; micro batch size: `4`

- Mean parameter gradient norm ratio wordmean/tokenmean: `1.0631` (p10 `1.0623`, p90 `1.0639`)
- Mean parameter gradient cosine: `0.9102`
- Mean uncorrelated logit-weight RMS scale: `1.0891`

| batch idx | rows | words | selected tokens | selected groups | mean k | token loss | word loss | grad norm ratio | grad cosine | logit RMS scale |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 550 | 115 | 80 | 1.4375 | 9.8486 | 9.8473 | 1.0621 | 0.8915 | 1.1020 |
| 1024 | 4 | 633 | 126 | 93 | 1.3548 | 9.8732 | 9.8495 | 1.0640 | 0.9288 | 1.0762 |

## Interpretation

If word-mean and token-mean parameter gradients have norm ratio far from 1 or cosine materially below 1, the earlier analysis unchanged-LR training result combines relative credit reweighting with update-scale/direction changes. This does not invalidate the screen, but it constrains mechanism attribution and motivates scale-aware follow-up only if scores improve.

Full JSON: `experiments/archive/frontier_consolidation/data/wordmean_gradient_scale_smoke/wordmean_gradient_scale_smoke.json`
