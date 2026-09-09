# earlier analysis — mechanism plan after first complete 9/9 coordinate

## Current scientific state

The protected baseline16k DeBERTa-v2 8x480 WWM model now has a complete local 9/9 coordinate:

| column | score |
|---|---:|
| BLiMP | 66.76 |
| BLiMP Supplement | 59.88 |
| EWoK | 52.19 |
| Entity Tracking | 22.62 |
| COMPS | 52.19 |
| (Super)GLUE | 68.0218 |
| GlobalPIQA | 35.635 |
| Reading | 7.62 |
| AoA | -0.1745 |

Aggregates: NLP Average 51.0424, Human-like Average 3.7228, Overall Average 40.5269.

Gaps vs visible strict-small reference `wwm_curriculum_simplification_40k` (Overall 41.80): EWoK -3.88, Entity -5.83, GlobalPIQA -4.035, COMPS -1.38, SuperGLUE -1.77, BLiMP -0.44; advantages Supplement +3.87, Reading +2.20.

The lexicon-weighted relation-WWM screen at 20M is mixed-to-negative: Entity +1.45 and COMPS +1.03 vs shuffled, but Supplement -3.90, GlobalPIQA mean -2.44, BLiMP -0.44, Reading mean -0.335. Do not continue tuning lexical mask weights.

## Next mechanism comparison

Keep ordinary WWM completely unchanged. Add a small auxiliary loss on top of the same backbone, data, tokenizer, optimizer, and exposure schedule. Compare at 10M/20M before any 100M run.

### Ranked candidate mechanisms

1. **Entity Mention Consistency** — contrastive loss over repeated content-word mentions in the same window, using mean-pooled whole-word group representations. Directly targets Entity Tracking's cross-position identity binding without changing mask distribution.

2. **Shared-Mask View Consistency** — two WWM views with partially shared masked groups; symmetric stop-gradient KL on shared positions. Tests whether the model learns context-invariant predictions.

3. **Counterfactual Entity-Swap Detection** — minimal entity-swap interventions on repeated mentions; token-level replaced-token detection head. Tests whether the model tracks entity identity under perturbation.

4. **Adjacent-Span State Contrast** — document-adjacent span contrastive + order classification. Requires recovering document boundaries from official corpus.

### Unified experiment design

All arms share: baseline16k DeBERTa-v2 8x480, ordinary WWM 15%, official corpus only, same seeds/init/order, same LR schedule prefix from the 100M trajectory, checkpoints at 10M/20M.

Each mechanism needs:
- WWM baseline (existing protected trajectory)
- structured auxiliary arm
- structure-destroyed control (matched extra heads, forward count, exposure)

### Signal that justifies 100M scaling

1. Target columns improve in correct direction at 10M and widen at 20M
2. Improvement beats both WWM baseline and structure-destroyed control
3. Gains come from multiple subsets and real wrong→correct flips
4. Supplement, BLiMP, Reading do not systematically degrade
5. Preferably net gains on at least two complementary target columns

### Measurement hygiene

- Pass checkpoint directories directly (`hf_model/chck_*M`), not via `revision=...`
- Save per-item predictions for Entity, GlobalPIQA, EWoK
- Verify checkpoint hashes and fixed-prompt log-probs differ before interpreting score trajectories

## Construction priority

Start with Entity Mention Consistency (mechanism 1): it has the clearest task correspondence, minimal code change, and no extra data construction. The design requires an exact loss, pair-construction rules, a projection head and matched controls.

Evidence files: `experiments/archive/initial_model_studies/data/debertav2_b256_true_9of9_coordinate.json`, `experiments/archive/initial_model_studies/data/relation_wwm_available_coordinate_comparison.json`.
