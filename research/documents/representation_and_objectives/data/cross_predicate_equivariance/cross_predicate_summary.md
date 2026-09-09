# equivariance protocol Cross-Predicate Equivariance Comparison

- Train preds: ['defeated', 'beat', 'lost_to', 'was_beaten']  |  Held preds: ['overcame', 'prevailed', 'fell_to', 'was_defeated']
- 80 train fams, 20 held fams, 7680 equiv pairs
- GRU(embed=48, hidden=96), 15 epochs, lr=0.002, λ=1.0

## Transfer comparison (mean ± std over 3 seeds)

| Condition | single_defeated | standard_multi | equivariant |
|---|---:|---:|---:|
| Train | 0.4997±0.0006 | 0.4992±0.0012 | 0.4999±0.0028 |
| Held fam, train pred | 0.5005±0.0007 | 0.5008±0.0028 | 0.5010±0.0007 |
| **Cross-pred (train fam, held hyp)** | 0.5001±0.0015 | 0.4988±0.0008 | 0.4998±0.0017 |
| **Full held** | 0.4977±0.0033 | 0.5005±0.0010 | 0.5052±0.0016 |
| Held ctx, train hyp | 0.4961±0.0032 | 0.5005±0.0019 | 0.5000±0.0013 |

## Per-held-predicate breakdown (seed 42)

| Predicate | standard | equivariant |
|---|---:|---:|
| overcame (wf) | 0.5344 | 0.4875 |
| prevailed (wf) | 0.5000 | 0.5281 |
| fell_to (lf) | 0.5000 | 0.5094 |
| was_defeated (lf) | 0.5000 | 0.5000 |

## Key deltas
- Cross-pred (equiv − std): +0.0010
- Full held (equiv − std): +0.0047
