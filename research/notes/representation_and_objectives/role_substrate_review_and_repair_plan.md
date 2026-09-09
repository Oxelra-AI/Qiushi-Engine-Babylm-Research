# role substrate review and repair plan — Role-substrate review and repair plan

## Scientific purpose

earlier analysis showed a strong hand-built role-label signal on 12 curated source-attested transformation cases, but the automatic 53-case pipeline was mixed. This review asks whether that mixed result is a harmless parser problem, a candidate-quality problem, a generated-label problem, a teacher-disagreement problem, or a deeper reason to stop the teacher-role route before training.

## What survives

- Manual panel: overall accuracy 0.993, source-attested bridge accuracy 1.000, false role-swap rejection 1.000, source↔bridge invariance 1.000.
- Automatic panel original: 1021/1032 parsed; expected-label consistency 0.902; cross-teacher agreement 0.850; both-teachers-expected 0.826; source↔bridge expected invariance 0.861; bridge false-role rejection 0.849; all-row stable cases 11/43.
- Benign parser repair maps 11 `ENTAILLED` outputs to `ENTAILED`: 1032/1032 rows valid, expected-label consistency 0.903, cross-teacher agreement 0.853, both-teachers-expected 0.829, strict retained cases 13/43.

The hand-built result is real: approved teachers can judge role-preserving natural/source-attested transformations sentence-by-sentence. The automatic substrate is not yet reliable enough to train from.

## Failure-mode separation

1. **Parser artifact is small and repairable.** All invalid labels are harmless Qwen spellings such as `ENTAILLED`. Repairing them removes invalid rows but does not lift cross-teacher agreement or negative rejection to the level needed for a learning substrate.
2. **Bridge/reference semantic loss is common enough to matter.** Some source-attested bridges drop source facts while the fact-generation prompt required support from all three sentences. Example: B018 generated `The American Bulldog is brave and protective`; the source supports it, but the source-attested bridge says only that the dog is best when trained young. Training on this would punish a student for correctly noticing missing information.
3. **Generated negatives are sometimes not actually false.** B045 labels `Children learn the material because they want to be competitive in the game` as NOT_ENTAILED, but the source says the game lets children be competitive, want to win, and therefore want to learn the material. B059 similarly labels a same-token paraphrase as an agent-patient swap when the context supports it. These are not role-learning examples; they are wrong targets.
4. **Teachers differ systematically.** Qwen is stricter on some positive hypotheses in compressed or altered contexts; Llama over-accepts many false entity/role swaps. The automatic substrate cannot use a single-teacher label stream. Retention must require agreement on the exact one-sentence prompt, not aggregate accuracy.
5. **The usable core is real but small.** Strict retained cases after benign parser repair are: B010, B012, B017, B019, B022, B024, B051, B060, B061, B072, B075, B096, B098. These are the seed cases for v2 construction, not a training corpus by themselves.

## Consequence for the BabyLM research route

This review strengthens the case against returning to compact-view tuning, score patches, or BabyLM training. The scientific object is a natural, source-attested role-assignment signal that can teach mention-to-entity, event-to-affected-entity, and query-to-entity roles without hard coordinates. Earlier analysis proves existence in curated cases, but the automatic version currently mixes clean role facts with missing facts, wrong false hypotheses, and teacher-specific biases. Training now would likely reproduce label noise rather than address the route2 factorial causal review/244/245 binding failure.

## Smallest Useful Repair

- Build `auto_role_fact_substrate_v2.py` from earlier analysis, not a BabyLM trainer.
- Generate exactly two positives and two negatives, but make each positive extractive or near-extractive from the intersection of SOURCE, SOURCE_ATTESTED_BRIDGE, and NATURAL_COMPACT_REFERENCE; reject facts mentioning tokens absent from the target context unless an exact alias table is present.
- Generate negatives only by one controlled swap at a time: actor/patient, cause/effect, entity/state, location/object, or outcome-to-wrong-entity. Reject negatives that either teacher accepts in any context.
- Normalize harmless output variants such as `ENTAILLED` before scoring.
- Keep a case only if Qwen and Llama both give the intended label for every retained fact in SOURCE and SOURCE_ATTESTED_BRIDGE, and preferably NATURAL_COMPACT_REFERENCE; if natural compact fails but source↔bridge is stable, save it as a source-bridge-only split rather than discarding the whole case.
- Preserve row-level reasons: parser spelling, bridge lost positive fact, natural reference lost positive fact, generated negative accepted, generated positive unsupported by source, Qwen-only miss, Llama-only miss.
- Connect retained v2 facts to the rawtoken bridge screen and route judgment EWoK bridge panel by role type and conditional-reversal family before any student distillation. A tiny student test is only scientifically useful if it predicts held source-attested role facts and moves the same EWoK/Entity-like item groups, not just teacher labels on easy paraphrases.

## Files produced by this review

- summary JSON: `experiments/archive/representation_and_objectives/data/role_substrate_review/review_summary.json`
- fact-level categories: `experiments/archive/representation_and_objectives/data/role_substrate_review/fact_level_failure_categories.jsonl`
- tolerant teacher-pair rows: `experiments/archive/representation_and_objectives/data/role_substrate_review/prompt_pair_teacher_comparison_tolerant.jsonl`
- case retention table: `experiments/archive/representation_and_objectives/data/role_substrate_review/case_retention_after_parser_repair.jsonl`
