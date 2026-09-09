# role coordinate anchor and state probe — Role-coordinate anchoring and repaired state-substrate probes

## Scientific Motivation existed

equivariance control and state substrate repair left one promising but unproven mechanism: a predicate might become composable after cheap flat role-word supervision, rather than needing direct combinatorial coverage. Independent review found leakage in the previous held-predicate experiment, making a corrected causal test necessary before any natural or BabyLM-scale training.

This analysis ran only cheap CPU measurements. No TinyMLM, DeBERTa, RoBERTa, BabyLM, official evaluation, upload, or large GPU training was started.

## 1. Corrected atomic role-anchor script did not fit, so it does not decide the mechanism

Script: `scripts/causal_role_anchor_corrected.py`
Output: `data/causal_role_anchor_corrected/causal_role_anchor_summary.json`

Design repairs relative to the flawed equivariance control and state substrate repair positive:

- identical atomic relation syntax for every predicate;
- no world/domain markers;
- labels require both context-predicate and hypothesis-predicate role maps;
- held predicates absent from composition training;
- matched arms: `noheld_filler`, `exposure_only`, `true_anchor`, `shuffled_anchor`, `coverage_only`.

Result: no arm reached train fit. All 5 arms had 0/5 converged seeds under the run's train-fit definition. Raw held surfaces stayed at chance. The control is useful as an operational negative on this exact raw sequence setup, but it **cannot** refute role anchoring because the model did not learn the seen composition/support distribution either.

A direct read of the generated rows confirmed that the labels are coherent, but the fixed ENTITY_A/ENTITY_B coordinate and balanced equality task leave the generic BiGRU in the same uninformative solution family seen earlier.

## 2. Naturalized raw sequence repair also did not fit

Script: `scripts/role_anchor_naturalized.py`
Output: `data/role_anchor_naturalized/role_anchor_naturalized_summary.json`

This repair removed the artificial fixed entity tokens:

- naturalized reusable names such as `alisa`, `bren`, ...;
- held pair combinations with individual names reused;
- identical relation syntax `name predicate name`;
- no world/domain/case markers;
- true-anchor, shuffled-anchor, exposure-only, no-held, and sparse-coverage arms.

Result: still no arm converged. Train accuracies stayed near 0.50 for true/shuffled anchors and near 0.58 or 0.56 only when trivial filler rows were present. By-kind readout showed the sequence model had not learned the seen composition or anchor rows:

- `true_anchor`: train mean 0.501; seen-composition mean 0.502; held_true_anchor mean 0.500.
- `shuffled_anchor`: train mean 0.500; seen-composition mean 0.500; held_shuffled_anchor mean 0.500.
- `noheld_filler`: train mean 0.584 only because filler rows were solved at 1.000 while seen composition stayed ~0.500.

Therefore the raw BiGRU/attention-BiGRU experiments remain unmeasured mechanism surfaces. They reveal that ordinary sequence learners do not spontaneously find the relevant factorization in these highly symmetric artificial rows, but they do not decide whether a better representation or objective can use anchors.

## 3. Factorized role-coordinate learner gives a clean positive and explains the right surface

Script: `scripts/factorized_role_anchor.py`
Output: `data/factorized_role_anchor/factorized_role_anchor_summary.json`

The model explicitly parameterizes the hypothesized reusable coordinate:

- each predicate has one signed latent parameter `theta_p`;
- `p_A_positive = sigmoid(theta_p * orient_factor)`;
- a composition row predicts whether the positive-role entity matches across context and hypothesis: `p_ctx*p_hyp + (1-p_ctx)*(1-p_hyp)`;
- an anchor row directly supervises the queried entity's role under that predicate.

This is not a large LM result. It is a mechanistic algorithm test of the proposed representation.

Key results over 5 seeds:

| arm | fit | held sign | seen comp | held hyp only | held ctx only | held both |
|---|---:|---:|---:|---:|---:|---:|
| noheld_filler | relevant seen rows fit, dummy rows unresolved | n/a | raw 1.000 | raw 0.550 | raw 0.550 | raw 0.725 |
| exposure_only | relevant seen rows fit, dummy rows unresolved | n/a | raw 1.000 | raw 0.550 | raw 0.550 | raw 0.725 |
| true_anchor | 5/5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| shuffled_anchor | 5/5 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| sparse_ctx_coverage | 5/5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sparse_hyp_coverage | 5/5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Two interpretations matter:

1. **True anchors pin the absolute role coordinate.** Held predicates become composable in both context and hypothesis positions after only flat role supervision.
2. **Held-held composition is not load-bearing.** The shuffled-anchor arm learns every held predicate with the wrong sign, so mixed held-seen surfaces are exactly wrong (0.000) while held-held composition remains perfect (1.000). This explains why earlier held-held or symmetric surfaces could look strong while not transferring the operation.

Sparse held-seen composition coverage also pins the coordinate. Thus the principle is not “role words are magical”; it is: **data-efficient composition requires small pieces of supervision that connect new symbols/events/states to an already anchored latent role coordinate.** Flat role words are one cheap connector; a few direct compositions with anchored predicates are another.

The noheld/exposure arms have no such connector for held predicates, so their held signs default arbitrarily and mixed held-seen transfer remains near chance despite perfect seen-predicate composition.

## 4. Identifiability simulation generalizes the same mechanism beyond binary winner/loser

Script: `scripts/anchor_coordinate_identifiability.py`
Output: `data/anchor_coordinate_identifiability/anchor_coordinate_identifiability_summary.json`

The simulation treats predicates as carrying one of R abstract roles and asks what different supervision types identify.

Main finding: relative constraints among held predicates can make held-held equality nearly perfect while leaving mixed held-seen accuracy at the no-anchor level. Absolute anchors or comparisons to already anchored seen predicates remove the permutation/inversion ambiguity.

Examples:

- R=2, no held information: mixed 0.500. True anchors: mixed 1.000. Shuffled anchors: mixed 0.000 but held-held 1.000. Held-held-only with m=10: held-held 1.000 but mixed 0.502.
- R=4, held-seen coverage m=4: mixed 0.955 and held-held 0.918; held-held-only m=4: held-held 0.816 but mixed only 0.628, near its no-anchor baseline 0.625.
- R=6, held-seen coverage m=10: mixed 0.987; held-held-only m=10: held-held 0.942 but mixed 0.720, near the no-anchor baseline 0.722.

This makes the emerging principle precise: **relative consistency is insufficient; the learner must attach each new relation/state/operator to a shared absolute role coordinate.** Once that coordinate is attached, combinatorial reuse is cheap.

## 5. Repaired upset-balanced state substrate passes cheap contemporaneous shortcut probes

Script: `scripts/upset_state_binding_probe.py`
Output: `data/upset_state_binding_probe/upset_state_binding_probe_summary.json`

Input substrate from equivariance control and state substrate repair:
`data/upset_balanced_state_substrate/`

The probe measured transparent parsers and TF-IDF held-family baselines.

Key results:

- Event-winner parser on `state_at_time`: **0.5003**, so the contemporaneous ranking state is dissociated from the match outcome.
- Numeric-rank parser on `state_at_time`: **0.9997**, as expected because the state sentence explicitly states ranks.
- State-at-time query inside event+state context:
  - natural text TF-IDF: 0.583, showing some name/rank wording residue;
  - canonical names: 0.491;
  - canonical names + rank numbers ablated: 0.491.
- Event-only TF-IDF stayed near chance on held families (0.495 natural, 0.468 canonical), despite a transparent event parser being able to solve event role. This means sparse lexical models are not reading the event syntax reliably.
- Later-state remains less clean: event-winner parser on `state_later` is 0.5628 and TF-IDF canonical/no-number later-state is 0.557, so later-state overwrite should be separated or rebalanced before use.

Thus the repaired substrate is suitable as a **contemporaneous state-binding scaffold** for the next controlled test: a learner must bind an explicitly stated independent state relation to the queried entity under event interference, while not using event outcome. It is not yet a clean temporal-overwrite substrate.

## Updated scientific state

The route has sharpened from “equivariance objective” or “more paired-world rows” to a more exact candidate data-efficient learning principle:

> A small learner can reuse knowledge compositionally when each new predicate, event, or state is attached to a shared latent role coordinate. Supervision that only builds relative consistency inside a new cluster is not enough; it can leave the whole cluster permuted or inverted with respect to previously grounded roles. Cheap anchor facts or a few comparisons to already anchored items resolve the coordinate ambiguity and make later compositions cheap.

What is already supported:

- the factorized learner and identifiability simulation establish the mathematical/algorithmic core;
- wrong anchors give perfect held-held but zero mixed transfer, explaining a common false positive;
- no/exposure-only arms fail on mixed held-seen surfaces despite seen composition being learned;
- the upset-balanced natural ATP substrate removes event-winner shortcut for contemporaneous ranking state.

What is not yet supported:

- a generic raw sequence learner has not learned the controlled role-anchor rows; this is a representation/optimization limitation, not a refutation of the principle;
- the principle has not yet been shown in TinyMLM/DeBERTa/RoBERTa/BabyLM or on EWoK;
- later-state temporal overwrite remains contaminated by outcome correlation and must be rebalanced or analyzed only on changed/rebalanced worlds;
- natural language relation diversity remains mostly sports/ranking-derived.

## Next work

1. Build a small **coordinate-anchored state-binding model** on the equivariance control and state substrate repair upset-balanced substrate. The minimal useful comparison is not a raw BiGRU alone; it should compare:
   - plain sequence classifier on event+state context;
   - an extractor/factorized model that explicitly attaches entities to event roles and ranking-state roles;
   - ablations removing state text, event text, rank numbers, or entity names;
   - evaluation separated for `state_at_time` and `state_later`, with `state_later` restricted to changed/rebalanced worlds.
2. The central measured surface should be mixed-role transfer: querying entity A/B against the independent state relation under event interference. Do not use held-held-only or within-cluster consistency as evidence.
3. If the anchored state-binding model succeeds while plain sequence baselines fail or require much more direct coverage, assess whether this supports a general data-efficient principle and which non-sports relation family or EWoK bridge subset should be the next pressure test.
4. Do not start GPU BabyLM-scale training, official evaluation, upload, or endpoint work from this route yet.

## Artifacts

- `scripts/causal_role_anchor_corrected.py`
- `data/causal_role_anchor_corrected/`
- `scripts/role_anchor_naturalized.py`
- `data/role_anchor_naturalized/`
- `scripts/factorized_role_anchor.py`
- `data/factorized_role_anchor/`
- `scripts/anchor_coordinate_identifiability.py`
- `data/anchor_coordinate_identifiability/`
- `scripts/upset_state_binding_probe.py`
- `data/upset_state_binding_probe/`
