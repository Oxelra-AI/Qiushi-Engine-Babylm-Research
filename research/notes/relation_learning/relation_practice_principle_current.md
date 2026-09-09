# anchor corrected price interpretation current Relation-Practice Principle

## Measured principle

Under a fixed token/update budget, the learner does not only count how often a piece of text is seen. It practices the relation that is locally available between spans inside the training window. The same source content and the same budget can install different later computations depending on whether the companion span is an exact recurrence, a nonidentical restatement, or separated into another row.

The current strongest form is measured through compact-rewrite held-out probes with three contexts:

- T: true related source in the source slot.
- U: unrelated compact source in the source slot.
- N: length-matched ordinary held-out text in the source slot.

The target quantity is not a single benchmark score. It is the vector of T, U, N, natural-copy behavior, source-content probability mass, and Entity relevant-update behavior.

## Three-seed original-arm T/U/N result

On token-nonoverlap compact-rewrite targets, where the masked target token does not occur in the source token set, the original DeBERTa arms show the same anchored structure across seeds 43022, 43122, and 43222.

| contrast | ΔT | ΔU | ΔN | Δ(U-T) | Δ(T-N) | Δ(U-N) |
|---|---:|---:|---:|---:|---:|---:|
| R−C | +0.5189±0.1517 | −0.3629±0.0664 | −0.2949±0.0527 | −0.8818±0.1483 | +0.8138±0.1729 | −0.0679±0.0340 |
| V−C | −1.3004±0.1056 | −0.4948±0.0341 | −0.5033±0.0326 | +0.8056±0.1090 | −0.7970±0.0758 | +0.0085±0.0536 |
| V−R | −1.8193±0.2126 | −0.1319±0.0521 | −0.2084±0.0273 | +1.6874±0.2509 | −1.6109±0.2196 | +0.0765±0.0321 |

This anchors the original compact-probe interpretation. REPEAT is worse than CLEAN specifically when the true related source is present: R−C has positive Δ(T−N) in every seed, while Δ(U−N) is small. VIEW is better than CLEAN because the true source helps more: V−C has negative Δ(T−N) in every seed, while Δ(U−N) is small. Therefore the T/U sign reversal is not an artifact of an unrelated compact neighbor; it is a true-source effect relative to a neutral source slot.

Files: `research/notes/relation_learning/original_threeseed_neutral_anchor.md` and `experiments/archive/relation_learning/data/original_threeseed_neutral_anchor`.

## Locality and price at seed43022

The split arms preserve selected source/companion content and the 100M-word budget while moving source and companion into different rows. At seed43022:

- REPEAT_SPLIT removes the exact-recurrence cost: RS−C token-nonoverlap gain is −0.0313 rather than original R−C around −0.75 to −1.04. Pair aggregation gives −0.0470, strict-word filtering gives −0.0964, and the pair-bootstrap interval for excess true-source cost is [−0.0314, +0.1238].
- VIEW_SPLIT removes the restatement benefit: VS−C token-nonoverlap gain is +0.0298 rather than original V−C +0.6837 at seed43022. Pair aggregation gives +0.0040 with interval [−0.0779,+0.0865], and strict-word filtering gives −0.0809.

anchor corrected price interpretation added the neutral-source anchor N to the local-versus-split reading:

| comparison | ΔN | ΔU | ΔT | Δ(U-T) | Δ(T-N) | Δ(U-N) |
|---|---:|---:|---:|---:|---:|---:|
| R−RS | +0.3027 | +0.2745 | +0.9949 | −0.7204 | +0.6922 | −0.0282 |
| V−VS | +0.2688 | +0.3269 | −0.3270 | +0.6539 | −0.5958 | +0.0581 |

The broad local price is visible under N itself: local REPEAT and local VIEW are about 0.27–0.30 nats worse than their split twins in a neutral ordinary-text context. U−N moves only slightly. Thus the price term is not mainly unrelated-neighbor interference; same-window pairing reallocates finite prediction work away from broad neutral target fit. The relation determines what is acquired with that reallocation: exact recurrence acquires a true-source-triggered identity cost, while restatement acquires true-source-conditioned target support.

Files: `research/notes/relation_learning/repeat_split_robustness.md`, `research/notes/relation_learning/view_split_integration.md`, `research/notes/relation_learning/anchor_corrected_price_interpretation.md`.

## Output-level mechanism

The source-content mass readout separates recognized-source effects from frequency hedging. Across three DeBERTa seeds, REPEAT−CLEAN raises content-token mass from the true related source by about +0.055 to +0.097 under T, but only +0.005 to +0.008 under U. Target probability falls only under T. VIEW also raises true-source content mass but raises the correct target probability. The readout-level distinction is therefore:

- exact recurrence: recognized source triggers identity-biased output mass, harmful when the target is a related nonidentical token;
- restatement: recognized source supports the transformed or nonidentical target.

This is output-level evidence. It does not by itself identify a unique internal pathway.

Files: `experiments/archive/relation_learning/data/source_specificity_misfire` and `research/notes/relation_learning/relation_practice_principle.md`.

## Behavioral face

Entity Tracking is no longer the primary quantitative object, but it remains behaviorally aligned with the exact-recurrence side. Across three DeBERTa seeds, REPEAT−CLEAN is positive when the queried entity has zero relevant updates (+8.78, +9.09, +3.11 points) and negative for deeper relevant-update groups (weighted rel≥3: −3.61, −1.30, −4.50). This matches identity practice helping unchanged retrieval and hurting after supersession. The official split-Entity readout is running at this stage; it will test whether this behavioral pattern is also localized by splitting source and companion rows.

## Natural CHILDES variation-set result

The research-built CHILDES instrument is validated and surface-held-out, with 732 adjacent non-CHI utterance pairs and 4,222 masked records. It is valuable because it directly checks a natural adjacent-utterance domain, but it does not reproduce the compact source/rewrite pattern.

The strong natural signal is lexical reoccurrence: overlap targets gain about +3.07 to +3.13 nats from the adjacent utterance across arms, while nonoverlap content words have negative mean context benefit in all arms. Across three DeBERTa seeds, all-bin nonoverlap contrasts are small: V−C +0.012, R−C −0.020, V−R +0.033, with seed movement larger than the means. The high-overlap nonoverlap subset has only 16 pairs. The natural result therefore says that adjacent child-directed utterances often help repeated words, but they are not automatically compact restatements that support new content tokens.

Files: `experiments/archive/relation_learning/analysis/childes_variation_set_probe`, `research/notes/relation_learning/childes_variation_probe_score.md`, and `experiments/archive/relation_learning/data/childes_variation_probe_score`.

## Hash-mixed in-window arm

The hash-mixed arm has been materialized and preflighted but not trained. It keeps every selected source and the full 100M budget; a deterministic hash assigns about half of paired companions to exact recurrence and half to restatement. The four possible readings require the joint compact-rewrite and natural-copy readouts:

1. Proportional mixture: compact-rewrite nonoverlap gain near −0.034 nats versus CLEAN, with copy gain also intermediate.
2. Identity-dominant capture: both rewrite and copy behavior move near original REPEAT.
3. Restatement-robust content use: compact-rewrite behavior stays near VIEW without REPEAT-like copy gain.
4. Context-selected dual competence: compact-rewrite behavior is VIEW-like while natural-copy behavior is REPEAT-like, showing that the learner can use the local relation format to choose between content and identity readouts.

Files: `experiments/archive/relation_learning/data/hash_mixed_inwindow_pools`, `experiments/archive/relation_learning/scripts/train_hash_mixed_inwindow_arm.py`, and `experiments/archive/relation_learning/scripts/score_hash_mixed_probes.py`.

## Current open work

- Collect and score REPEAT_SPLIT seed43122 and VIEW_SPLIT seed43122 when delivered. These determine whether the locality pattern repeats across training seeds.
- Collect official split Entity scoring and run `experiments/archive/relation_learning/scripts/integrate_split_entity_official.py`.
- After split replications, the proposed next test is to train the hash-mixed in-window arm and score copy/rewrite/Entity plus the neutral N anchor.
- Test objective generality. RoBERTa only weakly resolves the phenomenon. A minimal causal-LM version of CLEAN/REPEAT/REPEAT_SPLIT with T/U/N readout would show whether the identity cost is tied to MLM masking or also appears in next-token learners.
- Treat the synthetic binding study as a conceptual comparison rather than direct support: its rebinding analysis shows weak durable rebinding on the current substrate, so it should not be used as natural-language mechanism evidence.
