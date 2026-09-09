# shuf ordinary heldout: SHUF seed43122 ordinary-heldout MLM loss

Deterministic whole-word masking at p=0.15 on 6,992 held-out rows.

## Results

| Checkpoint | OFF loss | SHUF loss | Δ (SHUF−OFF) |
|---|---|---|---|
| chck_80M | 2.971226 | 2.985288 | +0.014062 |
| chck_90M | 2.941856 | 2.954037 | +0.012181 |
| chck_100M | 2.933582 | 2.947947 | +0.014365 |

SHUF incurs a small ordinary-heldout cost (~+0.01-0.02 nats), consistent with the
pre-stated criterion of small ordinary-heldout change. This is much smaller than
the compact-family source-conditioned effects (ΔA_T ~ -0.72) and comparable to
the split-control ordinary-heldout differences (~+0.02 nats).

Data: `experiments/archive/relation_learning/data/shuf_ordinary_heldout`
