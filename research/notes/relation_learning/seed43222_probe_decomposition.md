# research synthesis seed43222 V-R probe decomposition

## Nonoverlap rewrite-token T/U decomposition (late mean 80M/90M/100M)

| arm | gain (G) | true-source NLL (T) | unrelated-source NLL (U) | n |
|---|---:|---:|---:|---:|
| VIEW (43222) | 1.9805 | 5.7064 | 7.6870 | 2732 |
| REPEAT (43222) | 0.2910 | 7.4900 | 7.7810 | 2732 |
| **V−R** | **+1.6896** | **−1.7836** | **−0.0940** | — |

The entire V−R gain difference is in the true-source term. VIEW and REPEAT have nearly
identical unrelated-source NLL (7.687 vs 7.781), so the gap is content-specific, not
model quality. This is stable across all three checkpoints (dG: +1.74, +1.65, +1.68).

## Three-seed stability of V-R nonoverlap rewrite gain

| seed | V−R gain delta | V−R T delta | V−R U delta |
|---|---:|---:|---:|
| 43022 | +1.4354 | — | — |
| 43122 | +1.9371 | — | — |
| 43222 | +1.6896 | −1.7836 | −0.0940 |
| **mean** | **+1.687** | — | — |
| **sd** | **0.251** | — | — |

## Three-seed stability of copy gain

| seed | R−V copy gain |
|---|---:|
| 43022 | +0.1725 |
| 43122 | +0.3575 |
| 43222 | +0.2447 |
| **mean** | **+0.258** |
| **sd** | **0.094** |

## Entity cue ablation (seed43222)

| ablation | V−R delta | prior mean (2-seed) | direction match |
|---|---:|---:|---|
| remove initial clause | −0.877 | −0.618 | ✓ |
| remove last update | −0.258 | −0.282 | ✓ |
| remove all updates | +0.781 | — | — |

## Scientific reading

The held-out probes replicate much more stably than Entity magnitude. Across three
independent DeBERTa seeds:
- V−R nonoverlap rewrite gain: mean +1.687, sd 0.251 (CV 0.15)
- R−V held-out copy gain: mean +0.258, sd 0.094 (CV 0.36)
- Entity cue ablation directions: all replicate

The T/U decomposition at seed43222 confirms that the V−R gap is entirely content-specific:
the unrelated-source baseline is matched (delta only −0.094 nats), while the true-source
improvement drives the full +1.69 gain difference.

Without CLEAN at seed43222, we cannot directly confirm the R−C active cost (REPEAT below
baseline). But the V−R decomposition structure — content-specific gap, not register fit —
is consistent with the active-cost account from seeds 43022/43122.

## Files
- Probe data: `experiments/archive/relation_learning/data/seed43222_probes`
- Prior decomposition: `experiments/archive/relation_learning/data/relation_decomposition`
