# shuf ordinary heldout: SHUF seed43122 replication — complete evidence summary

## Ordinary-heldout MLM loss (completed this step)

| Checkpoint | OFF loss | SHUF loss | Δ (SHUF−OFF) |
|---|---|---|---|
| chck_80M | 2.971226 | 2.985288 | +0.014062 |
| chck_90M | 2.941856 | 2.954037 | +0.012181 |
| chck_100M | 2.933582 | 2.947947 | +0.014365 |

SHUF incurs a small ordinary-heldout cost (~+0.012-0.015 nats), consistent with
the pre-stated criterion. This is comparable to the split-control ordinary-heldout
differences (~+0.02 nats) and much smaller than compact-family source-conditioned
effects (ΔA_T ~ -0.72).

## Pre-stated replication criteria vs results (ref: notes/shuf_dup_prestate.md)

| Criterion | Required | Result | Status |
|---|---|---|---|
| Source-recurring ΔA_T < 0 (compact) | Below OFF | −0.7216 | ✅ |
| Source-recurring ΔA_T < 0 (Wikipedia overlap) | Below OFF beyond pair SE | −0.7077 ± 0.0589 (high: −0.8917 ± 0.1021) | ✅ |
| Ordinary-heldout change | Small | +0.012-0.015 nats | ✅ |
| Unrelated-source ΔA_U | Flat or positive | +0.0207 ± 0.0337 overall | ✅ |

**All four pre-stated source-recurring discounting criteria satisfied.**

## Non-replicating face: natural-copy gain

- seed43022: positive (copy gain > 0)
- seed43122: **−0.2216** (negative)

Natural-copy behavior does NOT replicate and must be reported as a non-replicating
face of the SHUF intervention, rather than dropped. The primary replicated claim is
source-recurring wrong-correspondence discounting; copy behavior is arm-specific.

## Entity (replicated pattern: no improvement)

- SHUF seed43122 official Entity: 20.78
- SHUF−OFF stratified: ALL −1.47, rel_eq0 +0.26, rel_ge3 −2.00, rel_updates_5 −5.52

Entity is not improved by SHUF, consistent with the principle: wrong correspondence
does not teach useful state-update discrimination.

## Conclusion

SHUF is now **two-seed evidence** for source-recurring wrong-correspondence discounting:
wrong local correspondence (shuffled rewrite) depresses true-source-conditioned readout
on recurring content while leaving unrelated-source behavior approximately flat, with
small ordinary-heldout cost. Entity is not helped and copy behavior is non-replicating.
The report can upgrade SHUF from one-seed to two-seed for the source-recurring readout
claim only.

## Data sources
- Mechanism: `data/shuf_seed43122_probe/`
- Ordinary-heldout: `data/shuf_ordinary_heldout/`
- Pre-stated criteria: `notes/shuf_dup_prestate.md`
