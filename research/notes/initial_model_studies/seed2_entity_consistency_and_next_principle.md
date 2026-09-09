# entity consistency closed — seed2 interpretation and next learning-principle direction

## Entity Mention Consistency after seed2

Entity Mention Consistency / repeated same-form content-word InfoNCE is **not** a 100M scaling route now.

Training-side mechanism replicated across two seeds: matched manifests, matched pair/candidate telemetry, distinct checkpoints, lower structured auxiliary loss than shuffled, and nearly unchanged MLM loss. However, ability transfer did **not** replicate.

### 20M deltas, consistency minus shuffled_pair

| quantity | seed1 | seed2 |
|---|---:|---:|
| BLiMP | +2.04 | **-0.51** |
| Supplement | -0.98 | -0.17 |
| Entity | +0.39 | +0.21 |
| COMPS | -0.03 | +0.43 |
| GlobalPIQA mean | +3.92 | +1.97 |
| Reading mean | -0.21 | **-1.03** |
| six-column sum | +5.135 | **+0.905** |
| guard Supp+BLiMP+Reading | +0.85 | **-1.705** |

Seed2 fails the predeclared replication pattern: BLiMP reverses, Reading worsens strongly, guard columns become negative, and net available-score gain collapses. The route is therefore best treated as a controlled negative for full-budget scaling. The evidence supports only that true repeated-form pairs are easier to learn than wrong pairs; it does not establish a stable data-efficient learning principle for Entity+EWoK+GlobalPIQA.

## Mechanism constraint after entity-consistency tests

The missing abstraction is: an auxiliary that can be minimized using single-token or surface-form features will not move the integrated hard columns. The next mechanism must require **cross-sentence binding** while preserving lexical/frequency statistics in the control.

Candidate principles to construct next:

1. **Next-Sentence Masked-State Prediction**: recover content words in a true next sentence conditioned on its previous sentence, with random-document and bag-of-words previous-sentence controls. Must count exposure correctly and not alter ordinary WWM mask distribution.
2. **Hard-Negative Discourse Coherence**: distinguish true adjacent 3–4 sentence windows from order-shuffled and TF-IDF hard-spliced windows with high lexical overlap. This destroys temporal/causal binding while preserving surface content.
3. **Downstream Counterfactual State Propagation**: edit a content word in sentence 1 but predict perturbation from sentence 2 representations only, with a local-edit detection control. This is most isomorphic to Entity Tracking.

Both A and C are the best first parallel mechanisms if resources permit: A is generative and cheap; C directly tests downstream state propagation. B is a strong hard-negative coherence route after document/sentence boundary recovery.

## Immediate next evidence

- Absolute attribution: evaluate ordinary protected WWM baseline at `chck_10M`/`chck_20M` so seed1/seed2 consistency can be compared to WWM, not only shuffled_pair.
- Per-item analysis: GlobalPIQA wrong→right/right→wrong flips for seed1 and seed2, because aggregate GlobalPIQA is small and seed-sensitive.
- Construction prerequisite: recover true document/sentence adjacency from official corpus before building any cross-sentence auxiliary; packed 256-token chunks are not sufficient evidence of adjacency.

Do not launch 100M Entity Mention Consistency. Do not final-package. Continue Execute/Explore toward a cross-sentence binding auxiliary with strong controls.

Evidence files: `data/entity_consistency_available_coordinate_comparison.json`, `data/seed2_entity_consistency_available_coordinate_comparison.json`, .
