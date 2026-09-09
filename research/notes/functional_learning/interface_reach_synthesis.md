# Interface reach synthesis after causal interface trajectory

Created: 2026-09-06

## Research question

The orbit-binding line asks how finite experience can create a reusable computation and why broader continuation can preserve familiar competence while losing transfer to held symbols. causal intervention gave a donor-query causal handle at layer 1; the interpretation was qualified because perfect familiar-symbol redirection must not replace held transfer as the target. causal interface trajectory therefore asked what actually changes during continuation: signal formation, downstream sensitivity, or symbol reach.

## causal interface trajectory: trajectory of the causal interface

causal interface trajectory followed the layer-1 donor-query d-component across continuation for seeds 43 and 100, arms `direct_full`, `static_1over17`, and `interleaved_ans_full`. It compared:

- `self_current`: current donor activation patched into the current recipient;
- `prep_to_current`: preparation donor signal patched into the current recipient;
- `current_to_prep`: current donor signal patched into the preparation recipient;
- train-only pairs, held-donor pairs, and train-donor pairs in held-containing contexts.

Key files:

- script: `scripts/causal_interface_trajectory.py`
- data: `data/causal_interface_trajectory/results.json`
- note: `notes/causal_interface_trajectory_analysis.md`
- figure: `figures/causal_interface_trajectory.png`

### Main causal interface trajectory findings

1. **Early direct-full damage primarily weakens formation of the preparation-readable signal.** At epoch 5, mean direct-full held `self_current` d-only redirection is 0.222, but held `prep_to_current` d-only redirection remains 0.619. Train pairs show the same separation: self 0.283 versus prep→current 0.807. Thus the current model can still respond to a supplied signal much better than it can form the signal itself.

2. **Direct-full later recovers familiar-symbol use without held-symbol transfer.** At epoch 500, direct-full mean train top-4 is 0.889 and train self d-only redirection is 0.838, but held top-4 is 0.359 and held self d-only redirection is 0.391. In seed100 the contrast is sharper: at epoch 500 train self d-only is 0.996 and train-in-held-context self d-only is 0.956, while held-donor self d-only is 0.356. The failure is therefore not generic intolerance of held-containing contexts; it is query-token reach.

3. **Aligned credit preserves a more transferable interface, but not monotonically.** Static `w=1/17` and interleaved training keep train self-redirection saturated and maintain much higher held self-redirection than direct full. However, seed100 static at epoch 100 has train self d-only 1.000 but held self d-only only 0.500; interleaved seed100 is similarly low at epoch 250 (0.487) before recovery by epoch 500 (0.787). Familiar-token perfection does not guarantee held transfer.

4. **Downstream sensitivity is graded, not all-or-none.** Direct-full recipients still use preparation signals above chance, but less than aligned-credit recipients. At epoch 500, mean held `prep_to_current` d-only redirection is 0.522 for direct full versus 0.844 for static and 0.809 for interleaved. Cross-model rescue is therefore compatibility evidence, not proof that all damage lies exclusively at signal formation.

5. **Direction classification and causal redirection diverge.** Seed100 direct-full can have low slot-classification accuracy while d-only patching still redirects answers. The scalar component can have a causal logit effect even when absolute slot ranking is contaminated by offsets or reshaped geometry.

## independent_review verification and corrected language

independent_review verification () supported the causal interface trajectory interpretation but narrowed several earlier statements:

- causal intervention donor replacement gives strong sufficiency evidence; necessity requires clean-run ablation or scrambling.
- Cross-model patching is not an additive decomposition of formation versus downstream loss.
- Held failure could still reflect embedding/trunk drift or a held-specific rotated code unless directly tested.
- The coefficient `1/(3K+5)` is only algebraic matching of answer-only/full mixture coefficients for the current sequence layout, not a preservation threshold or optimum.

## clean d component ablation: clean-run necessity-style intervention

clean d component ablation edited the ordinary clean forward pass at L1 attribute positions using the preparation-learned d direction:

- `center_d`: set every attribute slot's d projection to the per-example mean;
- `zero_d`: remove all d projection;
- `rotate_d`: cyclically rotate the d projection pattern across slots.

Key files:

- script: `scripts/clean_d_component_ablation.py`
- data: `data/clean_d_component_ablation/results.json`
- note: `notes/clean_d_component_ablation.md`

### clean d component ablation endpoint results

Mean over seeds:

| arm | bank | clean | center | zero | rotated target |
|---|---|---:|---:|---:|---:|
| prep | train | 1.000 | 0.283 | 0.264 | 0.923 |
| prep | held | 0.875 | 0.266 | 0.250 | 0.676 |
| direct_full | train | 0.872 | 0.348 | 0.345 | 0.478 |
| direct_full | held | 0.402 | 0.251 | 0.311 | 0.314 |
| static 1/17 | train | 1.000 | 0.367 | 0.404 | 0.835 |
| static 1/17 | held | 0.755 | 0.312 | 0.336 | 0.482 |
| interleaved | train | 1.000 | 0.352 | 0.449 | 0.992 |
| interleaved | held | 0.833 | 0.329 | 0.387 | 0.639 |

The clean d-pattern is not merely sufficient: ordinary answers depend on it. Centering or zeroing the slot-specific component generally drives accuracy toward chance, and rotating the component sends the answer toward the moved marker. For direct-full seed100 at epoch 500, the recovered familiar behavior is also d-dependent: train clean 1.000, center 0.451, zero 0.234, but held clean is only 0.411 and rotated-target only 0.250. Direct full recovers the interface for trained query tokens while leaving held query tokens weak.

## held fitted direction test: held-fitted directions and the rotated-code alternative

held fitted direction test tested whether held failure is only a rotation outside the preparation/train direction. For each endpoint model, L1 directions were fitted from train-query states, one-held-query states, and two-held-query states. They were then tested on train, one-held, and two-held classification and donor redirection. The two-held case distinguishes true query selection between held symbols from detecting the unique held slot.

Key files:

- script: `scripts/held_fitted_direction_test.py`
- data: `data/held_fitted_direction_test/results.json`
- note: `notes/held_fitted_direction_test.md`
- figure combining clean d component ablation/028: `figures/028_endpoint_interface.png`

### held fitted direction test mean results

| arm | fitted direction | redir train | redir one-held | redir two-held |
|---|---|---:|---:|---:|
| prep | train | 1.000 | 0.875 | 0.686 |
| prep | held_one | 1.000 | 0.869 | 0.664 |
| prep | two_held | 1.000 | 0.873 | 0.666 |
| direct_full | train | 0.879 | 0.408 | 0.451 |
| direct_full | held_one | 0.395 | 0.297 | 0.338 |
| direct_full | two_held | 0.623 | 0.377 | 0.338 |
| static 1/17 | train | 1.000 | 0.719 | 0.691 |
| static 1/17 | held_one | 0.979 | 0.639 | 0.598 |
| static 1/17 | two_held | 0.988 | 0.656 | 0.621 |
| interleaved | train | 1.000 | 0.834 | 0.723 |
| interleaved | held_one | 1.000 | 0.826 | 0.705 |
| interleaved | two_held | 1.000 | 0.824 | 0.707 |

Held-fitted directions do not rescue direct-full held transfer. In direct full, a train-fitted direction still gives high train redirection (0.879) but only 0.408 one-held and 0.451 two-held redirection. A held-one fitted direction is worse, not better: one-held 0.297 and two-held 0.338. A two-held fitted direction also fails to recover two-held redirection (0.338). This argues against a hidden rotated held selector. The endpoint has narrowed the functional interface toward practiced query symbols.

## Updated controlled mechanism

The strongest synthetic mechanism after causal interface trajectory is:

1. Query-first finite experience installs a layer-1 attribute-slot d-pattern that causally controls selection. It is sufficient to redirect answers by donor patching, and clean-run centering/zeroing/rotation show that ordinary behavior depends on it.
2. Abrupt full next-token continuation first weakens formation of a preparation-readable signal. A supplied preparation signal can still drive the current recipient above chance, so early loss is not simply output-readout destruction.
3. Later direct-full continuation can recover the d-dependent interface for trained query tokens. This recovery does not generalize to held query tokens, even when the context contains held entities and train queries still work. Held failure is not rescued by directions fitted on held or two-held states, so it is not merely a rotated held code.
4. Relation-aligned answer credit under restrained non-answer pressure preserves downstream sensitivity and keeps the interface reachable by held queries much better than direct full. It may still fluctuate; preservation is a trajectory and reach property, not a guarantee from familiar-symbol performance.
5. The data-efficient learning principle suggested by this substrate is interface credit: reusable knowledge is a causal interface between forming a relation-bearing signal and later using it. Efficient learning requires continued credit to the same reusable interface across broader objectives; otherwise training can retain or reacquire familiar readout while narrowing the interface's symbol reach.

## Boundaries

- The result remains a synthetic orbit-binding mechanism, not yet a BabyLM-scale or natural-language law.
- The current tests use two strong-transfer seeds. More seeds and item bootstraps are needed before treating oscillatory timing or exact numerical gaps as stable.
- Held entity embeddings remain absent from continuation. Prior Step019b ruled out held input row drift as a sufficient/necessary explanation for the earlier collapse, but causal interface trajectory/028 do not completely remove possible query-embedding/trunk compatibility contributions.
- The task has a very clean answer slot and small vocabulary. Natural-language analogues must show relation-aligned credit and reach loss without relying on engineered markers.

## Next discriminating work

Do not treat `w=1/(3K+5)` as a scaling law. It is only the coefficient that matches an answer-only/full mixture for the current layout. The next controlled experiment should independently vary:

- total non-answer coefficient mass \(\beta\) in \(L=\alpha A+\beta C\),
- number of supervised non-answer positions \(q\) at fixed \(\beta\),
- binding load \(K\),
- non-answer supervision semantics: local structural tokens, attribute/value tokens, unrelated filler, or relation-relevant context.

The measured outcome should be the causal interface reach: held `self_current`, `current_to_prep`, clean d ablation loss, and rotated-marker control, not only held top-4. A natural/semi-natural bridge with relation_learning should ask whether exact recurrence or wrong correspondence creates a train/familiar readout that narrows transfer reach, while aligned restatement preserves a relation-specific interface under fixed budget.
