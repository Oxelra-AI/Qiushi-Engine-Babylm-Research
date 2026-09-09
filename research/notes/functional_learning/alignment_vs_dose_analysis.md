# alignment vs dose analysis: relation-compatible answer credit is necessary beyond reduced context dose

## Purpose

Step021b showed that a static objective with context weight \(w=1/17\) matches the nominal answer/context coefficients of answer/full interleaving and preserves most held-symbol transfer while learning context prediction. answer credit alignment showed that generic answer/RWT pressure is insufficient when the answer target is random bag-independent. alignment vs dose resolves two remaining alternatives:

1. **Reduced context dose alone:** perhaps lowering the effective context update is enough, with no answer-gradient maintenance.
2. **Learnable but query-misaligned answer pressure:** perhaps any deterministic answer-position RWT target at the matched coefficient is enough, even if it does not reward the query-conditioned relation.

## Design

Both arms start from the same query-first bound answer-only preparation checkpoints used in accumulated mechanism synthesis--022. Held entity tokens are absent from continuation rows.

- `context_only_lr_half`: trains only context positions with no answer loss and learning rate \(1.5\times 10^{-4}\), half of the direct-full learning rate. This roughly matches the reduction of normalized context coefficient from direct full \(15/16\) to the matched static/interleaved coefficient \(15/32\). It learns local context prediction without relation-aligned answer credit.
- `slot0_static_1over17`: uses the same static coefficient \(w=1/17\) as the successful bound arm, but the answer target is the first visible context attribute rather than the queried entity's attribute. This is deterministic and quickly learnable from the sequence, uses the same answer position and RWT family, but conflicts with the query-conditioned relation for most rows.

Comparators from Step021b:

| comparator | final held top-4 | final held B | final held selectivity | final context CE |
|---|---:|---:|---:|---:|
| bound static \(w=1/17\) | 0.694 | +7.548 | +0.574 | 1.238 |
| direct full | 0.418 | +3.111 | +0.256 | 1.233 |
| interleaved bound/full | 0.746 | +7.765 | +0.665 | 1.316 |

## Results

Artifacts:

- script: `experiments/archive/functional_learning/scripts/alignment_vs_dose.py`
- data: `experiments/archive/functional_learning/data/alignment_vs_dose/results.json`
- auto-note: `research/notes/functional_learning/alignment_vs_dose.md`

Final means across seeds 42/43/100:

| arm | final held top-4 | final held B | final held selectivity | final train top-4 | final blocked top-4 | final context CE | final answer CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `context_only_lr_half` | 0.272 | -0.144 | +0.015 | 0.280 | 0.257 | 1.237 | 3.964 |
| `slot0_static_1over17` | 0.238 | +0.578 | -0.035 | 0.221 | 0.221 | 1.232 | 0.000 |

Per-seed final held top-4 / held-B:

| seed | context-only half-lr | slot0 static \(w=1/17\) | bound static \(w=1/17\) |
|---:|---:|---:|---:|
| 42 | 0.281 / -0.38 | 0.238 / +0.52 | 0.547 / +5.49 |
| 43 | 0.258 / +0.04 | 0.238 / +0.68 | 0.758 / +8.24 |
| 100 | 0.277 / -0.09 | 0.238 / +0.54 | 0.777 / +8.92 |

The context-only arm learns the context positions to the same CE as the successful bound static arm (1.237 vs 1.238) but loses both trained and held query-conditioned binding. Lowering context-update scale without answer credit is therefore not sufficient.

The slot0 arm learns its deterministic answer target essentially perfectly (answer CE \(3.6\times 10^{-5}\)) and learns context prediction to the same CE as direct/static arms, but it drives query-conditioned train and held binding to chance. Deterministic answer-position RWT learning at the matched coefficient is therefore not sufficient unless it is compatible with the queried-entity relation.

## Mechanistic inference

alignment vs dose substantially strengthens the effective-credit interpretation. The preservation in bound static \(w=1/17\) is not explained by:

- merely reducing the magnitude/frequency of context updates;
- merely emphasizing answer positions;
- merely rehearsing the RWT output family;
- merely training a learnable answer target under the same nominal coefficients.

The successful objective must keep answer-position credit aligned with the same query-conditioned variable-binding computation that supports held-symbol transfer. The evidence still does not directly identify the activation-level object, but it establishes a stricter causal statement at the behavioral/training-objective level:

**Cross-symbol transfer survives broader context learning when the objective both limits competing local-prediction pressure and continues to reward the reusable relation itself. Neither condition alone suffices in this controlled task.**

This is a better candidate principle than the earlier temporal-repair story. Alternation is useful but not primary; its main role can be understood as one way of implementing a favorable relation-aligned credit ratio. Static coefficient matching achieves most of the benefit. The small residual interleaving advantage may involve optimizer-state dynamics or path dependence, but it is not the core preservation mechanism.

## Caveats

- `context_only_lr_half` matches the direct-to-static reduction in normalized context coefficient only approximately through learning rate, and AdamW preconditioning makes exact update equivalence impossible. Its collapse is nevertheless informative because it tests the strongest form of “context dose alone can preserve transfer”: even with a smaller context update and normal context CE by epoch 500, transfer falls to chance without answer credit.
- `slot0_static_1over17` is a deterministic visible-slot target, not a neutral unrelated relation; it actively competes with the query-conditioned relation. This is precisely why it is useful as a negative control against generic learnable answer pressure, but a future experiment could add a non-conflicting auxiliary target to test whether any compatible non-query answer objective helps.
- All current conclusions remain within the query-first orbit-binding task. They guide a principle for BabyLM-scale data-efficient learning but do not yet prove it in natural language.

## Prediction for next work

The next mechanistic layer should find the internal transfer-bearing variable and show that bound static \(w=1/17\) preserves it while direct full, context-only, bag-independent, and slot0 objectives suppress it. Useful tests:

1. Layerwise query-match probes at the four context attribute positions: can a linear readout identify which attribute belongs to the query from hidden states? The signal should persist in bound static/interleaved and vanish in direct/context-only/slot0 arms.
2. Behavioral interventions on the candidate component: transplant or ablate the query-match subspace in held probes and measure held top-4/B, instead of relying on separability alone.
3. Scaling of the matched coefficient with the number of non-answer positions: if the effective-credit account is correct, the preserving static weight should move roughly as \(w^* = 1/(K+2)\) for equal answer-only/full alternation or more generally with the desired answer/context coefficient ratio.
4. Natural-language bridge: construct a small LM continuation where held lexical symbols are excluded, relation-aligned examples are contrasted with local context-prediction or visible-slot-like auxiliary targets, and transfer to held symbols is measured on the same relation.
