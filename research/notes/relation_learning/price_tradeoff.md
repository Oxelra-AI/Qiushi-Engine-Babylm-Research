# price tradeoff price-and-purchase decomposition

This note rewrites the seed43022 relation-practice 2×2 in the fixed-budget language inherited from frontier_consolidation. The split arms preserve the same selected text and 100M budget while removing paired same-window co-occurrence. Therefore local-minus-split isolates what same-window pairing spends and buys on the held-out compact-rewrite probe.

For token-nonoverlap targets, the terms are:

| local relation | local vs split gain Δ | true-source NLL Δ | unrelated-source NLL Δ | reading |
|---|---:|---:|---:|---|
| exact_recurrence (R−RS) | -0.7204 | +0.9949 | +0.2745 | local exact recurrence spends broad fit and buys an identity readout that is harmful for nonidentical targets |
| restatement (V−VS) | +0.6539 | -0.3270 | +0.3269 | local restatement spends broad fit and buys content-conditioned use of the related source |

anchor corrected price interpretation added a neutral ordinary-text source slot N and showed that the broad target-fit price is visible under N itself, not merely inferred from U: original REPEAT is worse than REPEAT_SPLIT by +0.3027 nats on N, and original VIEW is worse than VIEW_SPLIT by +0.2688 nats on N. The corresponding U-N shifts are small (-0.0282 and +0.0581), so unrelated compact-neighbor sensitivity is not the main source of the local-minus-split price in this probe. The symmetry matters: same-window pairing is not simply bad repetition or good restatement. It spends finite prediction work in both cases because part of the target-relevant answer is locally visible during training. See `research/notes/relation_learning/anchor_corrected_price_interpretation.md`.

What differs is the computation bought by that spending. Exact recurrence changes gain by −0.7204 nats relative to its split form; restatement changes gain by +0.6539 nats relative to its split form. Thus the local relation determines the sign of the source-specific competence purchased with roughly the same broad-fit price.

## Hash-mixture arm prediction

A 50/50 hash mixture of exact local recurrence and local restatement, with every source retained and companion type assigned per pair, separates four possible forms when both compact-rewrite and natural-copy probes are read together:

- Proportional mixture: token-nonoverlap gain versus CLEAN should be near the average of original R−C and V−C, -0.0340 nats, i.e. close to zero compared with the original endpoints, with copy gain also intermediate.
- Identity-dominant capture: a half admixture of exact local copies pulls both rewrite and copy readouts near original REPEAT, implying that near-duplicates inside a context window can disproportionately set the readout.
- Restatement-robust content use: compact-rewrite T and gain stay near original VIEW while copy gain does not become REPEAT-like, implying that content-conditioned reading survives substantial identity admixture.
- Context-selected dual competence: compact-rewrite T/gain are VIEW-like while natural-copy gain is REPEAT-like, implying that the learner can use local relation format to choose between content and identity readouts rather than averaging them.

This graded test keeps the source set, budget, and row-local availability fixed while changing the per-pair relation composition by hash.

Data table: `experiments/archive/relation_learning/data/price_tradeoff/price_tradeoff_seed43022.csv`
