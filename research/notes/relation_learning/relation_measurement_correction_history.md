# Measurement Corrections in the Relation Studies

This historical record preserves the sequence of methodological revisions in the relation experiments. It distinguishes a measured value from the interpretation that a later control invalidated. No training, evaluation or new numerical analysis was performed to prepare this record. Old statements that an experiment was pending describe only the evidence available when those statements were written.

## From a General Pairing Component to Separable Readouts

The mixed-relation experiment compared two arms with the same **16,731** local source-rewrite pairs. For the remaining approximately **16,560** sources, HALF_VIEW (HV) supplied ordinary neutral text, whereas HASH_MIXED (HM) supplied exact source copies. Thus HV was not simply half the dose of VIEW: it also practiced a different source-neighbor distribution.

An early compact-probe interpretation attributed HM's larger neutral-referenced advantage to cooperation between copy and rewrite practice. The decomposition did not support that attribution. HM-minus-HV nonoverlap `Delta(N-T)` was approximately **+0.33**, but the neutral and unrelated losses moved by **+0.37** and **+0.14**. True-source NLL was **6.276** for HM and **6.233** for HV, slightly better for HV. HV's neutral NLL of about **7.78** reflected practice on a neutral-like input format. A larger difference from that anchor was not an absolute true-source improvement.

The natural-domain comparison was different: HM-minus-HV overlap `Delta(N-T)` was **+0.292**, pair SE **0.047**, while `Delta(N-U)` was **+0.005**. That effect was source-specific rather than an anchor shift. On source-absent substitutions, HM-minus-CLEAN was **-0.214**, close to REPEAT-minus-CLEAN **-0.207**. In this one-seed comparison, half-dose recurrence reproduced nearly the full substitution liability despite co-practiced rewrites. It did not establish a general dose law or a single latent pairing mechanism.

The subsequent Entity integration further separated the readouts: HM was below HV at zero relevant updates (**39.78% versus 41.73%**) but above it at three or more relevant updates (**29.69% versus 23.79%**). These measurements do not support interpreting every HM advantage as a generic unchanged-state retrieval increment. See the [composition correction](corrected_composition_and_trust.md) and [completed Entity integration](entity_hm_hv_partial_integration.md); the former's incomplete-checkpoint wording is historical, not the final integration status.

## From Source-Only Margins to Format-Matched State Contrasts

The early state-phrase analysis emphasized a true-update versus source-only margin. The source-only condition lacked the update and full query/use format, so its movement could reflect familiarity with the input format, particularly for split-trained arms.

The repaired comparison kept source, update and use-frame structure in both conditions and changed whether the update belonged to the queried entity. On **1,125** extended nontraining packets, VIEW-minus-CLEAN true-versus-swapped-update discrimination was **+0.3656** and **+0.4036** across the two seeds; REPEAT-minus-CLEAN was **+0.2083** and **+0.3318**. The paired standard errors were **0.0374/0.0376** for VIEW and **0.0354/0.0372** for REPEAT. This supported a recurring-state-phrase readout under a changed frame, not entity-gated assignment by itself.

The separate replacement-training experiment did not inherit that conclusion as a success claim. Its updated-state uptake was flat or negative and its Entity changes were not consistently positive across seeds. Probe discrimination, retention of an unchanged source state and correct assignment of an update to a queried entity were distinct outcomes. The [format-matched integration](relation_arm_state_margin_tu_integration.md) and [replacement-route synthesis](state_update_route_synthesis.md) retain those separate evidence boundaries.

## Correcting the Null for Query-Flipped Binding Pairs

The first natural recombination corpus contained **4,998** training rows: **1,666** unchanged-source halves, **1,666** updated-state halves and **1,666** extra updated-state singles. This produced a 1:2 source/new-answer imbalance. A balanced rerun removed the singles, retained **1,666** complete training pairs and placed both halves in the same optimizer update, with **4,180** updates per arm.

The original analysis compared joint-correct count `J` with an independent-marginal expectation `A*B/n`. Its recorded epoch-20 quantities for the balanced rerun were:

| Arm | J / 200 | A | B | Historical A*B/n | Historical J-A*B/n |
|---|---:|---:|---:|---:|---:|
| Answer-clean | 44 | 83 | 161 | 66.815 | -22.815 |
| Uniform WWM | 7 | 64 | 142 | 45.440 | -38.440 |
| Corrupt-update answer | 30 | 116 | 113 | 65.540 | -35.540 |

Those arithmetic values are retained unchanged, but **the failure conclusion based on their negative excess is superseded**. Each pair keeps the context fixed and changes only the queried entity. A deterministic policy that ignores that query change gives the same answer to both halves and cannot solve both. The relevant no-gating null for that design is therefore **joint = 0**, not independent marginal accuracy. The later interpretation recognized answer-clean's **44/200 = 22.0%** joint success, versus **7/200 = 3.5%** for WWM and **6/200 = 3.0%** at the starting checkpoint, as a bounded query-conditioned signal.

This correction does not establish general binding or transfer. The later position audit found answer-clean successes of **38/160** with the unchanged entity first, **5/29** with the updated entity first and **1/11** in ambiguous order. Position enrichment, no-context solvability, candidate priors and expression changes still require separate controls. In particular, negative independence excess must not be reused to veto a query-conditioned effect, and positive joint accuracy must not be promoted automatically to a broad capability. See the [original balanced measurements](../../documents/relation_learning/data/binding_factorial_balanced_pairbatch/summary.md) and [corrected interpretation](corrected_binding_and_practical_route.md).

## Correcting the Population Behind Ordinary-Loss Comparisons

The exposure correction was independent of the binding-metric correction. An evaluation row could be absent as a constructed record while its source text had been deliberately trained; a corpus-drawn row could also re-enter one comparison arm through a selected source or rewrite. These relationships support different claims.

The historical re-admission screen removed **2,691 of 20,859** candidate pairs (**12.90%**); the two shard rates were **12.86%** and **12.95%**. This was a rate for those candidates under the specified screen, not a universal contamination rate. In the inherited aligned block, **3,395 of 37,594** pairs had at least one blocking reference hit. Later corpus-row sets contained inherited pair text in **2,354 of 6,992** rows and **904 of 2,647** rows.

The aligned/off/shuffled/duplicated/separated family was differentially exposed to those corpus-row sets, so its ordinary-loss differences were trained-text or differential-exposure comparisons rather than clean broad-generalization evidence. The designed compact and matched-split families had zero row-identity hits on those same sets; that earned a family-specific row-identity holdout claim, not a guarantee of semantic independence.

For the later base/dose family, **4,345** of the **6,992** rows were shared trained row identities. The **2,647**-row axis had no row-identity hits but separated into **1,743** rows without inherited-pair-text hits and **904** harder rows with such hits. The full axis and both subsets were therefore needed to separate costs outside the selectable register from movement on inherited text. Row absence, source-text absence and semantic independence must not be used interchangeably.

This narrowed the ordinary-fit interpretation without erasing separately defined source-conditioned contrasts. Construction of screened dose streams was not itself an experimental outcome; later dose measurements must be read at their own seed, checkpoint, exposure and model-loading identity. The [row-exposure audit](../../../experiments/archive/relation_learning/data/current_dose_stream_heldout_exposure/audit_summary.json), [2,647-row audit](../../../experiments/archive/relation_learning/data/current_dose_stream_heldout2647_exposure/audit_summary.json) and [existing cost-axis measurements](../../../experiments/archive/relation_learning/data/compactview_ordinary_fit_subsets/ordinary_fit_late_summary.csv) retain the corresponding records. No earlier pending-result statement is asserted here as current status.
