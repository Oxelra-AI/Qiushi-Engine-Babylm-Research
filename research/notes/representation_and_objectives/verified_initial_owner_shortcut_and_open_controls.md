# matched state and initial owner results — Verified initial-owner shortcut and remaining controls

This note supersedes the stronger wording in `matched_state_and_initial_owner_results.md` where it suggested that the whole complete orientation probe results/281 state effect is simply one transition heuristic and that generic state-format calibration is fully excluded. independent_review verification (`data/external/independent_review01_verifier1_integration.md`) supports the core counterfactual finding but narrows the claim.

## Verified core finding

In equivariant symmetry repair and macro context `state_rows`, changed-object state training always sets the initial owner to the complement of the final owner under the assignment used for labels:

```python
new = final_owner(rel_key, a, b, assignment_for_labels)
old = owner_from_slot(1 - slot(rel_key, assignment_for_labels), a, b)
```

Therefore, within the state rows, the intended final-role rule is observationally equivalent to a relation-independent rule:

> changed object: choose the participant who did **not** initially own the changed object.

This alternative lies outside the equivariant symmetry repair and macro context `satisfying_assignments` enumeration, which only ranges over relation-to-slot assignments. The formal Z2 assignment result remains valid **inside that restricted role-assignment hypothesis class**, but the language corpus does not force a learner to use that class.

The matched state and initial owner results counterfactual directly breaks this equivalence. When the changed object initially belongs to the true final owner, the non-initial rule becomes false. Changed-object accuracy then collapses while unchanged/static ownership remains high:

| arm | ordinary/opposite-initial changed | true-final-initial changed | true-final-initial unchanged | mixed held-seen relation |
|---|---:|---:|---:|---:|
| aligned matched | 0.805 | 0.203 | 0.859 | 0.507 |
| inverted matched | 0.768 | 0.250 | 0.932 | 0.510 |
| neutral matched | 0.609 | 0.471 | 0.529 | 0.521 |
| heldheld only | 0.643 | 0.354 | 0.513 | 0.506 |

The deterministic heuristic analysis shows aligned/inverted model choices agree with the non-initial-owner rule on changed rows at about 0.80/0.75 even when that rule is false. With two candidates this agreement is mathematically tied to low true accuracy in the false-heuristic condition, but it is still the right compact description of the behavior.

## What the matched state-format run really says

The matched run equalized state row counts and update counts across the polarity arms, neutral seen-state arm, random-label arm, and relation-only repeat control.

Core means over three seeds:

| arm | mixed true stmt | paired state true choice | changed true | unchanged true | pair-both true | exact train state row-label choice |
|---|---:|---:|---:|---:|---:|---:|
| heldheld only | 0.498 | 0.543 | 0.693 | 0.393 | 0.292 | n/a |
| heldheld repeat | 0.473 | 0.544 | 0.690 | 0.399 | 0.289 | n/a |
| aligned matched | 0.490 | 0.776 | 0.682 | 0.870 | 0.552 | 0.885 |
| inverted matched | 0.490 | 0.842 | 0.818 | 0.867 | 0.685 | 0.859 |
| neutral matched | 0.516 | 0.509 | 0.526 | 0.492 | 0.271 | 0.979 |
| random-label matched | 0.490 | 0.552 | 0.516 | 0.589 | 0.302 | 0.698 |

Important corrections:

1. The aggregate state score is a mixture of two separable behaviors:
   - changed object: often choose the non-initial participant;
   - unchanged object: preserve/copy the static owner.
2. Aligned matched does not clearly improve changed-object accuracy over the no-state/repeat baselines in the matched run; its aggregate gain is mostly unchanged/static-owner preservation. Inverted matched improves changed and unchanged in this run, but its counterfactual behavior still follows initial ownership rather than a signed role coordinate.
3. Pair-both is therefore not pure role-event semantics; it can rise by combining non-initial changed-object responses with easier static-owner preservation.
4. Mixed held-seen relation comparisons stay chance in all learned polarity arms. No stable signed role coordinate is expressed in the reported behavioral decisions.

## Generic state-format calibration: what is and is not ruled out

The matched controls disfavor the simplest calibration stories:

- adding more relation-only updates did not reproduce state transfer;
- decoupled seen-state examples with a held distractor did not reproduce held-state transfer;
- partially fitted random labels did not reproduce the aligned/inverted aggregate gains;
- neutral exact training-state fit was high (0.979) but held-state transfer stayed near chance.

However, neutral is not a strict causal-held-event control. In equivariant symmetry repair and macro context `decoupled_orbit`, the held event appears as an explicitly unrelated note, and the tested event is a seen relation with different wording. Random-label rows also did not fit perfectly. The supported wording is:

> Extra updates, causally decoupled seen-state examples containing held distractors, and partially fitted random held-state labels do not explain the aligned/inverted behavior. A fully matched causal-held-event state-format control remains unresolved.

A good next control would keep the same held event in the tested causal position but assign labels from a relation-independent pattern that is not the non-initial rule, or balance initial-owner conditions so the non-initial rule is insufficient.

## Current scientific status

matched state and initial owner results does not establish role-coordinate induction. It strengthens a different data-efficient learning insight:

> Limited structured supervision can make a model look competent out of distribution because the experience leaves a lower-complexity non-role rule observationally equivalent to the intended compositional rule. When a counterfactual intervention breaks the equivalence, changed-event semantics fails while easier static facts remain preserved.

This is promising as a general principle about underspecified finite experience and hypothesis selection, but the current evidence is one DeBERTa/substrate family. It must not be generalized yet beyond the tested construction.

## Next research work implied by matched state and initial owner results

The next construction should not run more equivariant symmetry repair and macro context state rows unchanged. It should repair the substrate so event role and initial ownership are independently varied during training, while preserving the symmetry-identification algebra:

1. **Initial-owner-balanced state substrate.** For each held relation and bridge arm, include both initial-opposite and initial-true-final worlds. The label should always depend on the relation role, not on whether ownership changes hands. This breaks the non-initial rule during training.
2. **Matched causal-held controls.** Keep the same causal held-event position and surface template, but use label patterns that train query format without providing role-coordinate information; compare with aligned/inverted role labels.
3. **Signed changed-only and unchanged-only reporting.** Do not treat aggregate state accuracy or pair-both as role semantics unless changed-object accuracy survives initial-owner counterfactuals and unchanged/static facts are preserved.
4. **Only then** test whether aligned/inverted evidence can propagate to mixed held-seen relations or whether another lower-complexity rule appears.

The branch remains valuable, but the immediate object is now substrate repair and hypothesis-class separation, not more training on the old shortcut-bearing state rows.
