# matched state and initial owner results — Matched state-format and initial-owner counterfactual results

## Purpose

corrected orientation route and next experiments corrected the complete orientation probe results interpretation: the central role-coordinate prediction failed because aligned and inverted bridge evidence did not produce opposite orientations on unbridged mixed held-seen relation comparisons. matched state and initial owner results executed the lowest-cost learned discriminators before any larger training:

1. a matched state-format run with equal state row counts and update counts across aligned, inverted, neutral, and random-label arms;
2. a counterfactual initial-owner readout to test whether the state-query improvement is a true final-role coordinate or a weaker transition-away-from-initial-owner rule.

No BabyLM-scale training, official evaluation, upload, or final-facing expression occurred.

## Files

- Runner: `experiments/archive/representation_and_objectives/training/scripts/matched_state_probe.py`
- Main matched output: `experiments/archive/representation_and_objectives/data/matched_state_probe`
  - summary: `matched_state_probe_summary.md`
  - aggregate: `aggregate_summary.json`
  - per-row logits: `per_row_predictions.jsonl`
  - train construction: `matched_train_construction.json`
- Counterfactual runner: `experiments/archive/representation_and_objectives/training/scripts/initial_owner_counterfactual_probe.py`
- Counterfactual output: `experiments/archive/representation_and_objectives/data/initial_owner_counterfactual_probe`
  - summary: `initial_owner_counterfactual_summary.md`
  - aggregate: `counterfactual_aggregate.json`
  - per-row logits: `counterfactual_per_row_predictions.jsonl`
- Deterministic heuristic analysis: `experiments/archive/representation_and_objectives/scripts/noninitial_heuristic_analysis.py`
- Heuristic output: `experiments/archive/representation_and_objectives/data/noninitial_heuristic_analysis/noninitial_heuristic_summary.md/json/csv`

## Matched state-format result

All state-format arms used 192 held-held comparison rows + 128 state-query rows (320 supervised rows total), equal epochs and seeds.

Core mean over 3 seeds:

| arm | mixed true stmt | paired state true choice | paired state inverted choice | pair_both true | exact train state row-label choice |
|---|---:|---:|---:|---:|---:|
| heldheld_only | 0.498 | 0.543 | 0.350 | 0.292 | n/a |
| heldheld_repeat_control | 0.473 | 0.544 | 0.354 | 0.289 | n/a |
| aligned_matched | 0.490 | 0.776 | 0.594 | 0.552 | 0.885 |
| inverted_matched | 0.490 | 0.842 | 0.525 | 0.685 | 0.859 |
| neutral_matched | 0.516 | 0.509 | 0.483 | 0.271 | 0.979 |
| random_label_matched | 0.490 | 0.552 | 0.536 | 0.302 | 0.698 |

Interpretation:

- Mixed held-seen relation orientation remained chance in aligned and inverted arms. State evidence still did not become a reusable relation coordinate.
- Matched neutral state-format evidence did **not** reproduce the held-state transfer (paired true choice 0.509), so the state effect is not generic exposure to the NLI state-query format.
- Random-label state rows did not reproduce it either (paired true choice 0.552, pair_both 0.302), so the result is not just arbitrary state-label fitting.
- Aligned and inverted both improved true-labeled state choices, with inverted actually higher on true labels than inverted labels. This already argued against polarity-controlled coordinate induction.

## Initial-owner counterfactual result

The decisive substrate flaw is now clear: in the original equivariant symmetry repair and macro context `state_rows`, the changed object always starts with `old_owner = complement(final_owner)` under the assignment used for labels. Therefore a learner can solve changed-object state queries by the weaker rule:

> choose the participant who did **not** initially own the changed object.

This rule is not a role coordinate. It does not identify whether the relation's final slot is agent-like or patient-like; it only predicts that a transfer event changes ownership away from the initial holder.

The counterfactual readout varied the changed object's initial owner while keeping the true final role fixed. Mean over 3 seeds:

| arm | ordinary changed true | cf true-final-initial changed | cf opposite-initial changed | ordinary unchanged true | cf true-final-initial unchanged | mixed true stmt |
|---|---:|---:|---:|---:|---:|---:|
| heldheld_only | 0.609 | 0.354 | 0.643 | 0.521 | 0.513 | 0.506 |
| aligned_matched | 0.818 | 0.203 | 0.805 | 0.849 | 0.859 | 0.507 |
| inverted_matched | 0.753 | 0.250 | 0.768 | 0.927 | 0.932 | 0.510 |
| neutral_matched | 0.557 | 0.471 | 0.609 | 0.549 | 0.529 | 0.521 |

The deterministic non-initial-owner analysis makes the interpretation explicit:

| arm | ordinary changed: model true / heuristic agreement | true-final-initial changed: model true / heuristic agreement | opposite-initial changed: model true / heuristic agreement |
|---|---:|---:|---:|
| aligned_matched | 0.818 / 0.818 | 0.203 / 0.797 | 0.805 / 0.805 |
| inverted_matched | 0.753 / 0.753 | 0.250 / 0.750 | 0.768 / 0.768 |
| neutral_matched | 0.557 / 0.557 | 0.471 / 0.529 | 0.609 / 0.609 |

Aligned and inverted state-trained models agree with the non-initial-owner heuristic at about 0.75–0.80 on changed rows even when that heuristic is **false**. Their apparent state transfer collapses from ~0.75–0.82 to ~0.20–0.25 when the changed object initially belongs to the true final owner. Unchanged facts remain high (aligned 0.859, inverted 0.932 in the same counterfactual), so the failure is specifically the changed-event semantics, not general state-query breakdown.

## Scientific update

matched state and initial owner results changes the mechanism interpretation substantially:

1. The state-query improvement in Steps279/281 is real but mostly explained by a **transition-away-from-initial-owner heuristic** on this corpus.
2. It is not generic state-format calibration: matched neutral and random-label state-format controls do not reproduce it.
3. It is not sparse absolute role-coordinate induction: mixed held-seen relation comparisons stay at chance and inverted evidence does not flip the signed orientation.
4. It is not conserved entity-event-state binding: when initial ownership is counterfactually decoupled from the expected transfer direction, changed-state accuracy collapses while unchanged facts remain easy.
5. The formal Z2 identifiability law remains valid in the slot-orbit/supplied-coordinate controls, but the repaired language substrate still does not force a neural learner to identify the relation coordinate.

This is valuable negative and mechanistic evidence for the general BabyLM goal. It shows a precise way in which limited supervision can produce out-of-distribution-looking success without compositional role knowledge: the learner adopts a lower-complexity event-transition invariant that is valid on natural transfer episodes but fails under counterfactual initial-state interventions.

## What remains unresolved

The data-efficient learning principle is not yet established. A stronger route would need a substrate or natural data regime where:

- initial state and event role are independently varied so a non-initial-owner rule is insufficient;
- changed and unchanged facts are both conserved under counterfactual initial ownership;
- aligned/inverted sparse evidence can be tested through signed orientation on unbridged relations;
- or the negative finding is developed into a general principle about **underspecified experience selecting low-complexity transition heuristics instead of reusable role coordinates**, supported across architecture/init/data variations.

The immediate next construction should not run more of the same equivariant symmetry repair and macro context state rows. The scientific bottleneck is now a better substrate/algorithmic control that explicitly breaks the initial-owner shortcut while preserving the symmetry-identification algebra and surface balance.
