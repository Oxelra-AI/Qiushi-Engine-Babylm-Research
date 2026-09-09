# Two-Seed N-Anchored Locality Summary

This note integrates the seed43022 neutral-anchor data (anchor corrected price interpretation) with the
newly completed seed43122 neutral-anchor data (argument map relation typed composition). Together they provide
the complete T/U/N decomposition for both original and split arms at two seeds,
which is the strongest form of the causal locality evidence.

## The Central Table: Compact Nonoverlap Δ(T−N) and Δ(U−N)

Δ(T−N) = (T − N) for arm minus (T − N) for CLEAN. Positive means the true
related source helps *less* than in CLEAN; negative means it helps *more*.
Δ(U−N) = (U − N) for arm minus (U − N) for CLEAN. Measures whether unrelated
compact source sensitivity differs from CLEAN's.

### Original arms (three-seed means from anchor corrected price interpretation):
| contrast | Δ(T−N) | Δ(U−N) |
|---|---|---|
| R−C | **+0.814 ± 0.173** | −0.068 ± 0.034 |
| V−C | **−0.797 ± 0.076** | +0.009 ± 0.054 |

### Two-seed N-anchored split locality:
| Seed | R−C Δ(T−N) | RS−C Δ(T−N) | V−C Δ(T−N) | VS−C Δ(T−N) |
|---|---|---|---|---|
| 43022 | +0.683 | **−0.009** | −0.712 | **−0.116** |
| 43122 | +1.010 | **+0.134** | −0.824 | **−0.028** |

### Reduction by splitting:
| Seed | R-side collapse | V-side collapse |
|---|---|---|
| 43022 | +0.683 → −0.009 (101% reduction) | −0.712 → −0.116 (84% reduction) |
| 43122 | +1.010 → +0.134 (87% reduction) | −0.824 → −0.028 (97% reduction) |

### Two-seed Δ(U−N) for split arms:
| Seed | RS−C Δ(U−N) | VS−C Δ(U−N) |
|---|---|---|
| 43022 | −0.041 | −0.086 |
| 43122 | −0.056 | −0.064 |

All Δ(U−N) terms are small and negative, consistent with split arms being
slightly better than CLEAN on unrelated compact source handling but without
the defining source-specific structure of the local arms.

## Scientific Interpretation

1. **The sign reversal is robust.** Across three original DeBERTa seeds,
   exact local recurrence makes the true related source help *less*
   (Δ(T−N) ≈ +0.81) while local restatement makes it help *more*
   (Δ(T−N) ≈ −0.80). The unrelated-source terms are near zero in both cases.

2. **The causal locality is replicated with N anchoring.** At both seeds
   43022 and 43122, splitting source/companion across rows while preserving
   all content and budget collapses the large Δ(T−N) toward zero. The
   reduction ranges from 84% to 101% across the four arm-seed combinations.

3. **The N anchor confirms source-specificity.** Δ(U−N) is uniformly small
   for both original and split arms, so the measured effects are specifically
   about the true related source, not about having *any* compact neighbor.

4. **The compact-family price is replicated.** Local-vs-split ΔN is
   +0.20–0.30 nats at both seeds, while ordinary held-out local-vs-split
   loss is only +0.02 nats (ordinary heldout price probe). The reallocation is within the
   companion target family.

## Connection to ICLM (Shi et al. 2023)

ICLM \cite{shi2023context} groups related documents in windows and explicitly
filters near-duplicates, noting that "near duplicate documents in the same
context" can encourage copying and reports broader perplexity harm when filtering
is removed. The present T/U/N decomposition should not be stated as the measured
cause of ICLM's broad loss, because ICLM changes corpus-level duplicate presence,
window composition, and training stability together. The sharper contribution is
methodological and mechanistic: our matched row-split intervention isolates the
window-adjacency component that ICLM did not separate. At this BabyLM dose that
component is source-specific in the compact companion family, with only small
ordinary-heldout local-minus-split loss, and exact recurrence and nonidentical
restatement have opposite true-source effects.

## Data Sources

- Seed43022 original C/R/V N: `data/neutral_anchor_rewrite_probe/`
- Seed43022 split RS/VS N: same directory (scored in anchor corrected price interpretation)
- Seed43122 all five arms N: `data/split_seed43122_neutral_anchor/`
- Three-seed original means: `data/original_threeseed_neutral_anchor/`
- Two-seed T/U split replication: `data/split_seed_replication/`
- Ordinary held-out price: `data/ordinary_heldout_price_probe/`
