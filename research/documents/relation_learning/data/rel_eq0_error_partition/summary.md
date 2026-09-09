# error partition and paired contrastive rel_eq0 error-mass partition

## Flip analysis: chck82 correct -> binding wrong on rel_eq0

### binding_ep25_alpha0p50
- rel_eq0 items: 1541
- chck82 correct: 577
- flips to wrong: 199
- reverse flips to correct: 90
- net loss: 109
- nothing picks: 0 (0.0%)
- matches initial state: 1 (0.5%)
- matches recency: 19 (9.5%)
- longer than gold: 0 (0.0%)
- shorter than gold: 0 (0.0%)
- more items than gold: 0
- fewer items than gold: 0
- mean pred word count: 6.05 vs gold: 6.05
- mean pred item count: 2.17 vs gold: 2.17
- by irrelevant ops stratum:
  - 1-3: n=89, nothing=0, recency=5, longer=0, shorter=0
  - 4-6: n=62, nothing=0, recency=5, longer=0, shorter=0
  - 7+: n=48, nothing=0, recency=9, longer=0, shorter=0

### binding_ep25_alpha0p75
- rel_eq0 items: 1541
- chck82 correct: 577
- flips to wrong: 292
- reverse flips to correct: 119
- net loss: 173
- nothing picks: 0 (0.0%)
- matches initial state: 1 (0.3%)
- matches recency: 28 (9.6%)
- longer than gold: 0 (0.0%)
- shorter than gold: 0 (0.0%)
- more items than gold: 0
- fewer items than gold: 0
- mean pred word count: 6.33 vs gold: 6.33
- mean pred item count: 2.28 vs gold: 2.28
- by irrelevant ops stratum:
  - 1-3: n=147, nothing=0, recency=9, longer=0, shorter=0
  - 4-6: n=85, nothing=0, recency=11, longer=0, shorter=0
  - 7+: n=60, nothing=0, recency=8, longer=0, shorter=0

### binding_ep25_alpha1p00
- rel_eq0 items: 1541
- chck82 correct: 577
- flips to wrong: 386
- reverse flips to correct: 136
- net loss: 250
- nothing picks: 0 (0.0%)
- matches initial state: 1 (0.3%)
- matches recency: 27 (7.0%)
- longer than gold: 0 (0.0%)
- shorter than gold: 0 (0.0%)
- more items than gold: 0
- fewer items than gold: 0
- mean pred word count: 6.54 vs gold: 6.54
- mean pred item count: 2.34 vs gold: 2.34
- by irrelevant ops stratum:
  - 1-3: n=213, nothing=0, recency=10, longer=0, shorter=0
  - 4-6: n=111, nothing=0, recency=9, longer=0, shorter=0
  - 7+: n=62, nothing=0, recency=8, longer=0, shorter=0

### coherent86
- rel_eq0 items: 1541
- chck82 correct: 577
- flips to wrong: 15
- reverse flips to correct: 11
- net loss: 4
- nothing picks: 0 (0.0%)
- matches initial state: 0 (0.0%)
- matches recency: 1 (6.7%)
- longer than gold: 0 (0.0%)
- shorter than gold: 0 (0.0%)
- more items than gold: 0
- fewer items than gold: 0
- mean pred word count: 7.93 vs gold: 7.93
- mean pred item count: 2.53 vs gold: 2.53
- by irrelevant ops stratum:
  - 1-3: n=7, nothing=0, recency=0, longer=0, shorter=0
  - 4-6: n=6, nothing=0, recency=0, longer=0, shorter=0
  - 7+: n=2, nothing=0, recency=1, longer=0, shorter=0

## Global prediction characteristics on rel_eq0

| label | n | mean_pred_wc | mean_pred_ic | nothing_picks | nothing_pct |
|---|---:|---:|---:|---:|---:|
| chck82 | 1541 | 6.38 | 2.21 | 0 | 0.0 |
| binding_ep25_alpha0p50 | 1541 | 6.38 | 2.21 | 0 | 0.0 |
| binding_ep25_alpha0p75 | 1541 | 6.38 | 2.21 | 0 | 0.0 |
| binding_ep25_alpha1p00 | 1541 | 6.38 | 2.21 | 0 | 0.0 |
| coherent86 | 1541 | 6.38 | 2.21 | 0 | 0.0 |
| GOLD | 1541 | 6.38 | 2.21 | 0 | 0.0 |

