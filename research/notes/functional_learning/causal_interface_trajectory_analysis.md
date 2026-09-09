# causal interface trajectory analysis: continuation changes the query-conditioned interface and its reach

Created: 2026-09-06

## Why this experiment was needed

causal intervention gave a causal interface: at layer 1, replacing the donor-query scalar component along the preparation-learned query-match direction can redirect the answer. The next question was not whether familiar-token redirection can be perfect. The retention problem was held-symbol transfer: after broader next-token learning, a model may still answer trained symbols or use a supplied signal while failing to let a held query instantiate the same selector.

causal interface trajectory therefore followed the interface through continuation for seeds 43 and 100, arms `direct_full`, `static_1over17`, and `interleaved_ans_full`, at continuation epochs 0, 1, 5, 25, 100, 150, 250, and 500. The script is `scripts/causal_interface_trajectory.py`; full data are in `data/causal_interface_trajectory/results.json`; the trajectory figure is `figures/causal_interface_trajectory.png`.

## Measurements

All causal patching is at layer 1 attribute positions along the preparation-learned direction unless otherwise stated.

- `self_current`: donor activation and recipient are from the same current model. This measures whether the current forward pass forms a signal that the current downstream computation can use.
- `prep_to_current`: donor activation is from the preparation model and recipient is the current model. This measures whether the current downstream computation can still use a well-formed preparation-era signal.
- `current_to_prep`: donor activation is from the current model and recipient is the preparation model. This measures whether the current activation still carries the preparation-readable selection component.
- `held_donor`: the donor query is a held entity in a context containing one held entity and three train entities; the recipient query is a train entity from the same context.
- `train_in_held_ctx`: the donor query is a train entity in the same held-containing context. This separates failure on a held query token from mere presence of a held entity in the context.

## Main results

### 1. The early full switch weakens signal formation more than it destroys downstream use

Averaged over seeds, at `direct_full` continuation epoch 5, behavioral held top-4 was 0.270 and held `self_current` d-only redirection was 0.222. Yet `prep_to_current` held redirection was still 0.619. On train-only pairs, `self_current` d-only was 0.283 while `prep_to_current` was 0.807. Thus the current model can still use a supplied preparation-like selection signal much better than chance even when its own forward pass no longer forms that signal reliably.

The same pattern appears in the seed-level records:

| seed | arm/be | case | self d-only | prep→current d-only | current→prep d-only |
|---:|---|---|---:|---:|---:|
| 43 | direct 5 | train | 0.316 | 0.934 | 0.273 |
| 43 | direct 5 | held donor | 0.237 | 0.744 | 0.281 |
| 100 | direct 5 | train | 0.250 | 0.680 | 0.188 |
| 100 | direct 5 | held donor | 0.206 | 0.494 | 0.200 |

This does not support a story in which continuation immediately makes the downstream readout unable to use selection information. The first damage is mainly that the current forward pass no longer produces a preparation-readable, answer-controlling signal.

### 2. Familiar-symbol recovery is not the same phenomenon as held-symbol transfer

By epoch 500, `direct_full` has recovered much familiar behavior, but held behavior remains weak. Mean train top-4 is 0.889 and train `self_current` d-only redirection is 0.838, while held top-4 is 0.359 and held `self_current` d-only redirection is 0.391.

Seed 100 shows the separation most sharply:

| seed100 direct | train top-4 | held top-4 | self d train | self d held | self d train-in-held-context |
|---|---:|---:|---:|---:|---:|
| be=250 | 1.000 | 0.398 | 0.965 | 0.375 | 0.906 |
| be=500 | 1.000 | 0.344 | 0.996 | 0.356 | 0.956 |

The train-in-held-context column is high, so the model is not merely disrupted by contexts that contain a held entity. Rather, broader continuation recovers a selector for practiced query symbols while held query symbols no longer instantiate the same interface.

### 3. Compatible credit preserves downstream sensitivity and often restores held formation, but held transfer still fluctuates

For `static_1over17` and `interleaved_ans_full`, train `self_current` d-only redirection is essentially perfect after the first few epochs. Held redirection is much higher than in `direct_full`, but not monotonic and not guaranteed by familiar redirection.

Seed 100 illustrates this distinction:

| seed100 arm/be | train self d | held self d | held prep→current d | held current→prep d | held top-4 |
|---|---:|---:|---:|---:|---:|
| static be=100 | 1.000 | 0.500 | 0.838 | 0.481 | 0.555 |
| static be=250 | 1.000 | 0.825 | 0.887 | 0.831 | 0.859 |
| static be=500 | 1.000 | 0.731 | 0.925 | 0.694 | 0.797 |
| interleaved be=100 | 1.000 | 0.613 | 0.831 | 0.569 | 0.648 |
| interleaved be=250 | 1.000 | 0.487 | 0.831 | 0.469 | 0.500 |
| interleaved be=500 | 1.000 | 0.787 | 0.844 | 0.750 | 0.867 |

The held `prep_to_current` values remain high even when held `self_current` is low, especially at static be=100 and interleaved be=250. This says the recipient pathway can still use a well-formed held query signal, while the current model temporarily fails to form it. Later recovery is possible under aligned answer credit.

### 4. Direct full also narrows downstream sensitivity to held-like signals

The supplied-signal path is not perfectly preserved under `direct_full`. At epoch 500, mean held `prep_to_current` d-only redirection is 0.522, far below the preparation baseline 0.856 and below the aligned-credit arms (`static`: 0.844, `interleaved`: 0.809). The corresponding train `prep_to_current` value under direct is 0.746. Thus full continuation narrows both signal formation and downstream sensitivity, with the strongest loss for held-donor signals.

This matters for the causal intervention cross-model rescue: rescue proves that a supplied signal can be used, but it does not make all damage exclusively a formation-layer phenomenon. The current measurements show a graded interface: supplied preparation signals help, but direct-full recipients become less sensitive to held-like signals than aligned-credit recipients.

### 5. Direction separability and causal functionality can diverge

In seed100 direct-full, direction classification can be near chance while d-only patching still redirects answers. For example, at be=100 the preparation-direction train/held slot accuracies are 0.253/0.233, but self d-only redirection is 0.492 on train and 0.425 on held; at be=250, train direction accuracy is 0.580 while train d-only redirection is 0.965. Slot classification over four attributes and answer-logit redirection are different measurements. The scalar component can still have causal effect on the answer even when it is not the top-scoring slot feature under the linear classifier.

Therefore the mechanistic language should be: continuation weakens, reshapes, and narrows the query-conditioned interface. It should not say that the marker is simply and completely erased whenever direction classification falls to chance.

## Updated mechanism

The controlled orbit-binding mechanism is now more precise:

1. Query-first preparation forms an intermediate query-conditioned selector that can be causally read through layer-1 attribute-position patches.
2. Abrupt full next-token continuation first disrupts the model's ability to form the preparation-readable signal. Downstream use of a supplied signal remains substantially above chance early, especially for train contexts.
3. Later direct-full training can recover a familiar-symbol interface while held-symbol transfer remains weak. The recovery is symbol-narrow: train queries work even in held-containing contexts, but held queries do not instantiate the same selector.
4. Relation-aligned answer credit with limited competing context pressure preserves downstream sensitivity to held-like signals and allows held signal formation to recover, but familiar-token perfection does not ensure held-token transfer at every time point.
5. The data-efficient learning object is not just a learned representation, but an interface between signal formation and downstream use whose reach over new symbols depends on continuing credit. Efficient learning fails when the training objective rewards a practiced readout while allowing the reusable part of the interface to narrow.

## Implication for the next experiment

A K-scaling run should not be framed as confirming an algebraic coefficient. The value `1/(3K+5)` only matches the nominal answer/full mixture; it is not a predicted optimum. The next stronger controlled experiment should vary two quantities independently:

- binding load: number of entity-attribute slots or the number of query alternatives that the answer must distinguish;
- non-answer supervision: amount and type of local context prediction pressure, including context tokens that are relevant or irrelevant to the binding relation.

The measurable prediction from the current mechanism is not a single coefficient. It is that held transfer follows preservation of the query-conditioned interface: held `self_current` and `current_to_prep` should remain high when aligned answer credit keeps the held signal formation active and should fall when non-answer supervision permits a train-symbol-only shortcut. `prep_to_current` distinguishes whether failures come from formation or downstream sensitivity.

## Quantitative association across recorded checkpoints

Across all nonzero checkpoints in the two-seed trajectory, held behavioral top-4 is tightly tied to held self-redirection at the causal interface: correlation between held top-4 and held `self_current` d-only redirect is 0.992, with mean absolute difference 0.038. The correlation with held direction-slot accuracy is also high (0.949) but less direct, and the correlation with familiar train self-redirection is lower (0.777 overall and near zero inside aligned-credit arms where train redirection is saturated). This confirms that the relevant measured variable for retention is held instantiation of the causal interface, not familiar-symbol redirection or a scalar context CE.

The supplied-signal measurement has a different meaning: held `prep→current` stays high in many checkpoints where held self-redirection and behavior are low, so it is not a behavioral predictor by itself. It identifies downstream sensitivity to a well-formed signal and thereby localizes whether the active failure is formation/narrowing or downstream loss.
