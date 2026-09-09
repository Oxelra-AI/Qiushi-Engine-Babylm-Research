# decomposition and paired context paired-context contrastive objective

## Scientific question

Does a paired-context loss that directly targets (U+R)/2 (the recipient-dependent
component) improve held-out generalization compared to standard CE on both contexts?

## Decomposition baseline

- Train: (U+R)/2 = +0.0122, (U-R)/2 = -0.2659
- Held: (U+R)/2 = +0.0047, (U-R)/2 = +0.0606

## Arms

| Arm | Train flip | Held flip | Train (U+R)/2 | Held (U+R)/2 | Train (U-R)/2 | Held (U-R)/2 |
|---|---:|---:|---:|---:|---:|---:|
| ce_only | 36/36 | 3/12 | +8.5357 | +1.9774 | -0.1327 | +6.7202 |
| paired_only | 36/36 | 3/12 | +6.0054 | +0.8278 | +0.6785 | +3.4750 |
| combined | 36/36 | 3/12 | +8.6433 | +2.0986 | -0.1401 | +6.6141 |

## Interpretation

If the combined arm achieves higher held (U+R)/2 than ce_only, the paired loss
successfully directed optimization toward the context-dependent circuit rather than
the degenerate shared-preference solution. This would be a specific mechanism for
data-efficient learning: explicit paired-context supervision can install recipient-
dependent computation that standard answer CE (even on both contexts) fails to
generalize reliably.
