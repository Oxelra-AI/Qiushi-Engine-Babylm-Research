# independent orientation cross and context interface — independent event/ranking orientation and context-interface correction

## Scientific Motivation changed the interpretation

contrastive wording cross and grounding's full three-seed run was collected and compared with the pilot. It replicated a narrow phenomenon: in familiar ATP contexts, sparse aligned evidence supports held-hypothesis readout, including the held ranking hypothesis wording. It did **not** establish a universal representation-similarity or corpus-frequency law. The decisive correction is that the held state hypothesis can transfer despite low isolated state-hypothesis cosine, while held state contexts remain weak and seed-variable. Thus the active object is context-side directional extraction and routing, not a global cosine threshold.

The contrastive wording cross and grounding jointly flipped arm also could not identify a shared relation coordinate or untouched-entity conservation, because event, focal ranking, and secondary ranking labels were reversed together. A whole-head or broad label-convention explanation was still compatible with that result.

## What independent orientation cross and context interface built

Script: `experiments/archive/representation_and_objectives/training/scripts/independent_orientation_cross_probe.py`.

The new experiment varies sparse event and focal-ranking labels independently while leaving the secondary ranking label true:

- `exposure`: no sparse relation labels beyond the shared base task
- `Etrue_Rtrue`: event true, focal ranking true, secondary ranking true
- `Eflip_Rtrue`: event flipped, focal ranking true, secondary ranking true
- `Etrue_Rflip`: event true, focal ranking flipped, secondary ranking true
- `Eflip_Rflip`: event flipped, focal ranking flipped, secondary ranking true

Evaluation labels are always the real independently sourced ATP facts. The same AB-vs-BA contrastive scorer is used. The run adds context variants:

- train event + train state
- held event + held state
- directional event paraphrase + directional state paraphrase
- train event + directional state
- directional event + train state
- train event + relation-erased state text
- relation-erased event text + train state

A dry construction pass verified matched rows and exactly balanced labels for all arms. The first tiny pilot had two underfit arms, so it was used only to show the experiment structure. The fit-repaired one-seed run raised rows and epochs modestly; all five arms reached training accuracy 1.0.

Primary results:

- Fit-repaired summary: `research/documents/representation_and_objectives/data/independent_orientation_cross_fitrepair1/independent_orientation_cross_summary.md`
- Analysis: `research/documents/representation_and_objectives/data/fitrepair_analysis/fitrepair_analysis.md`
- Balanced factorial effects: `research/documents/representation_and_objectives/data/fitrepair_factorial_effects/factorial_effects.md`
- contrastive wording cross and grounding/independent orientation cross and context interface combined analysis: `research/documents/representation_and_objectives/data/orientation_result_analysis/orientation_result_analysis.md`
- Independent independent_review reading: `data/external/independent_review01_verifier1_integration.md`

A three-seed replication of the same fit-repaired design is running; it writes to `experiments/archive/representation_and_objectives/data/independent_orientation_cross_fitrepair3`.

## Fit-repaired one-seed result

On familiar train-context/train-hypothesis ATP compound rows:

| arm | event | focal ranking | secondary ranking |
|---|---:|---:|---:|
| exposure | 0.537 | 0.283 | 0.221 |
| Etrue_Rtrue | 1.000 | 1.000 | 1.000 |
| Eflip_Rtrue | 0.004 | 1.000 | 0.992 |
| Etrue_Rflip | 1.000 | 0.196 | 0.988 |
| Eflip_Rflip | 0.004 | 0.113 | 0.967 |

The balanced factorial effects on this surface are:

| readout | event-orientation effect | ranking-orientation effect | interaction |
|---|---:|---:|---:|
| event | 0.996 | 0.000 | 0.000 |
| focal ranking | 0.042 | 0.846 | -0.083 |
| secondary ranking | 0.015 | 0.019 | -0.013 |

This is the cleanest current result. It rejects a single head-wide output-polarity explanation on the familiar interface. Sparse event orientation can be inverted without moving ranking readouts; sparse focal-ranking orientation can be inverted while event stays true and the secondary ranking record mostly remains true. Because focal and secondary use the same ranking-hypothesis family, the learner must bind the query aliases to different records at least on this familiar surface.

## Context-side boundary

The context side remains the central hard part. In the same fit-repaired run:

- `dirEvent_trainState_trainHyp`: event 0.967, focal ranking 1.000, secondary ranking 1.000.
- `trainEvent_dirState_trainHyp`: event 1.000, focal ranking 0.021, secondary ranking 0.013.

Changing only event wording leaves the three readouts strong; changing only ranking context wording makes the ranking readouts systematically wrong while event stays strong. Held state contexts are also weak. Therefore the current principle cannot be stated as sparse orientation plus a generic representation-similarity threshold. It should be stated around a usable directional context interface: sparse oriented evidence composes only when the learner can extract the signed relation from the context surface.

The relation-erased context controls support this reading:

- `trainEvent_nonState_trainHyp`: event remains controlled by event orientation; focal and secondary ranking are near chance.
- `nonEvent_trainState_trainHyp`: event is near chance; focal and secondary ranking remain strong when ranking context is present.

Mean-pooled pretrained cosine is not a reliable measure of directional grounding: the state relation-erased context had cosine about 0.742 to train state contexts, close to the directional state paraphrase cosine about 0.749, despite lacking higher/lower information. Behavior under relation-erased contexts is more informative than whole-sentence cosine for this object.

## What should not be claimed

The secondary ranking readout should not be called untouched-state conservation yet. In the current design the secondary ranking labels are directly present in base training and in every sparse compound arm with true labels. The result shows supervised coexistence of a true secondary record and an intentionally flipped focal record, not unsupervised retention of an unqueried fact. A genuine conservation test needs a learned secondary record before intervention, then event/focal updates without secondary query labels, with secondary signed margins measured before and after.

The fit-repaired result is one seed. It is strong enough to justify the small three-seed replication already launched, but not a Strict-Small 100M language trajectory. If `s263_t17_tool1` replicates, the next scientific object is not leaderboard training but the unit to which sparse orientation attaches: semantic relation, predicate surface, query template, record position, or entity-bound record.

## Next discriminating work

After the three-seed fit-repaired replication completes:

1. Compute balanced factorial effects by seed for event, focal ranking, and secondary ranking on familiar, held, directional, and relation-erased contexts.
2. Inspect whether selective event/ranking movement survives on conflict ATP worlds, where event winner and higher-ranked player differ.
3. Build a position-counterbalanced variant: randomize record order, whether the focal pair also appears in the event clause, which ranking record receives flipped labels, and the wording assigned to each ranking record.
4. Build a no-secondary-rehearsal retention test: first learn event, focal ranking, and secondary ranking; then update event or focal ranking without secondary labels and compare secondary signed margins.
5. Add state-only and event-only context ablations for held/directional ranking surfaces to separate predicate-specific sign inversion from event-sign import.

No BabyLM-scale intervention should be launched until the small bridge shows stable, selective, context-aware orientation beyond familiar templates and the conservation question is separated from direct secondary supervision.
