# anchor corrected price interpretation corrected price reading with neutral-source anchor

The price tradeoff price-and-purchase table used the unrelated compact-source term U as the broad target-fit term. A rival interpretation was considered: a local arm might not be broadly worse; it might instead be especially sensitive to an unrelated compact neighbor because it learned to expect a related neighbor. anchor corrected price interpretation added a neutral ordinary-text source slot N on the same 1,626 compact rewrite pairs, 5,892 masked targets, five seed43022 arms, and the late 80M/90M/100M checkpoints.

Definitions:

- T: target NLL with the true related compact source in the source slot.
- U: target NLL with an unrelated compact source in the source slot.
- N: target NLL with length-matched ordinary held-out text in the source slot.
- T-N: related-source use relative to ordinary text; more negative means more help from the true source.
- U-N: unrelated compact neighbor effect relative to ordinary text; positive would mean the unrelated compact source hurts relative to ordinary text.

## Main token-nonoverlap result

| comparison | ΔN | ΔU | ΔT | Δ(U-T) | Δ(T-N) | Δ(U-N) | pair ΔN | pair Δ(U-N) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R−RS | +0.3027 | +0.2745 | +0.9949 | −0.7204 | +0.6922 | −0.0282 | +0.3033 | −0.0291 |
| V−VS | +0.2688 | +0.3269 | −0.3270 | +0.6539 | −0.5958 | +0.0581 | +0.2837 | +0.0533 |

The local-minus-split broad price is visible under N itself: original REPEAT is +0.303 nats worse than REPEAT_SPLIT, and original VIEW is +0.269 nats worse than VIEW_SPLIT, with pair-averaged values +0.303 and +0.284. The U gaps are not primarily created by unrelated compact-neighbor damage. U-N changes are small by comparison: −0.028 for R−RS and +0.058 for V−VS at token level, with pair means −0.029 and +0.053. Thus the price tradeoff price term is now anchored as a neutral-context target-fit difference, not merely inferred from U.

## Relation-specific purchase after anchoring

For exact recurrence, local REPEAT pays the broad N price and additionally becomes much worse when the true related source is present: R−RS has Δ(T-N)=+0.692 and Δ(U-T)=−0.720. This is the identity-cost side: same-window exact recurrence creates a source-triggered readout that hurts nonidentical targets.

For restatement, local VIEW also pays a broad N price, but the true related source more than compensates: V−VS has Δ(T-N)=−0.596 and Δ(U-T)=+0.654. This is the content-use side: same-window restatement creates source-conditioned target support.

The two relation types therefore share a measured local-co-occurrence price in neutral target fit but buy opposite source-specific computations.

## Arm-versus-CLEAN anchor terms

| comparison | ΔN | ΔT | ΔU | Δ(U-T) | Δ(T-N) | Δ(U-N) |
|---|---:|---:|---:|---:|---:|---:|
| R−C | −0.2348 | +0.4480 | −0.3037 | −0.7517 | +0.6828 | −0.0689 |
| RS−C | −0.5375 | −0.5469 | −0.5782 | −0.0313 | −0.0094 | −0.0407 |
| V−C | −0.4672 | −1.1788 | −0.4951 | +0.6837 | −0.7115 | −0.0278 |
| VS−C | −0.7360 | −0.8518 | −0.8220 | +0.0298 | −0.1157 | −0.0859 |

Split arms have the strongest neutral target fit relative to CLEAN (RS−C −0.538, VS−C −0.736). Local arms still improve N relative to CLEAN, but less than their split twins. The local relation does not simply make the model globally bad; it reallocates finite prediction work away from broad compact-target fit and into a relation-specific source-conditioned computation.

## Consequence for the working principle

The stronger wording is now:

Under a fixed token/update budget, putting a source and companion in the same training window spends some ordinary target-fitting value relative to spacing the same material across rows. The spent value is measured by N, a neutral-context target NLL. The relation made locally available determines what computation is acquired: exact recurrence acquires source-triggered identity use that harms related nonidentical targets, while restatement acquires source-conditioned content use that helps them. Unrelated-neighbor sensitivity is present at most as a small correction in this probe, not the main source of the local-minus-split price.

Primary files: `research/notes/relation_learning/neutral_anchor_rewrite_probe.md` and `experiments/archive/relation_learning/data/neutral_anchor_rewrite_probe`.
