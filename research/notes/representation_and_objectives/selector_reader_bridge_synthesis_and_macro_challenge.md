# selector reader bridge synthesis and macro challenge — Selector-reader bridge result and macro allocation challenge

## What changed

prephase alignment design and bias analysis showed that the direct-tag warm-start rescue was generic supervised calibration, not tag-coordinate reuse: aligned, permuted, and disjoint prephases all rescued the same inline direct-tag continuation. The strategist note for selector reader bridge synthesis and macro challenge therefore asked for one final high-information bridge test that separates the role selector from the state reader instead of treating the previous role-query failure as an architecture-wide statement.

The analysis built and ran that decomposed bridge.

Main files:

- Plan: `notes/selector_reader_final_bridge_plan.md`
- First decomposed script: `training/scripts/selector_reader_bridge.py`
- First decomposed result: `data/selector_reader_seed27000/selector_reader_summary.md`
- Group-softmax repair: `training/scripts/selector_reader_groupsoftmax.py`
- Group-softmax result: `data/selector_reader_groupsoftmax_seed27000/selector_reader_summary.md`
- Exact surface matcher: `scripts/lexical_selector_baseline.py`
- Matcher output: `data/lexical_selector_baseline/lexical_selector_baseline.md`
- independent_review scientific reading: 

## Construction and cost discipline

The dry build passed before GPU use:

- reader tag-only rows: 5,632
- reader inline direct-tag rows: 5,632
- selector rows: 11,264
- selector groups: 2,816 groups of exactly four candidate tags, exactly one positive tag per group
- held selector suites: five suites × 2,560 rows each
- token length: max 110 at max length 200, no truncation

This was one small bridge seed at the proven fit-capable seed 27000, not Strict-Small pretraining or official evaluation.

## First decomposed run: binary selector objective failed by all-negative selection

The first version trained `M(role, tag)` as four binary candidate rows with a 3:1 negative:positive ratio and class weight 3.0. It did not learn the selector:

- selector train top-1: 0.253
- row accuracy: 0.75
- predicted positive rate: 0.0, true positive rate: 0.25
- held exact top-1: about 0.25 across suites

Thus the first composition scores are not evidence about role composition. They reflect the all-negative shortcut of the binary candidate objective. This run was useful because it localized a training-objective failure before overreading the result.

## Repaired selector objective: four-way group softmax over candidate tags

The repair changed only the selector objective: for each role query and context, train over the four candidate tags with a group softmax, directly optimizing top-1 role-to-tag selection. `R` remains a separately trained and fixed reader. `M` still sees no state labels.

### Reader `R(k, state)`

The fixed reader was established cleanly:

- tag-only direct reader: best train accuracy 0.999645; sparse changed focal 0.996094
- inline direct reader: best train accuracy 1.0; all train families 1.0 including sparse changed focal
- oracle-reader held composition: 0.994–0.995 across exact, swapped-role, and paraphrase suites

The oracle-reader ceiling shows that, when the correct tag is supplied, the reader can recover before/after states on held worlds and fresh tag namespaces.

### Selector `M(role, tag)` under exact role wording

The group-softmax selector learned exact role-to-tag selection:

- train top-1: 1.0
- held exact namespace A: 0.996875
- held exact namespace B: 1.0
- held train-changed namespace A: 0.9984375
- held role-swap namespace A: 1.0

This establishes that an explicitly supervised selector can recover an in-context tag under fresh tag assignments and role-label swaps. It corrects the previous monolithic role-query failure: role→tag→state composition is not impossible when selector and reader are separated and the selector is trained with the right objective.

### Composition `M∘R`

Using the selector-chosen tag rather than the true tag:

- held exact namespace A: selector 0.996875, composed state 0.99375, oracle 0.99375
- held exact namespace B: selector 1.0, composed state 0.9953125, oracle 0.9953125
- held train-changed namespace A: selector 0.9984375, composed state 0.9921875, oracle 0.9921875
- held role-swap namespace A: selector 1.0, composed state 0.99375, oracle 0.99375

So the selected address passes cleanly into the frozen reader. There is no measurable interface loss when exact role strings are shared between context and query.

## What fails: semantic temporal paraphrase

The held paraphrase selector does not learn a bidirectional before/after map:

- overall held paraphrase top-1: 0.5109375
- focal_before: 0.0
- focal_after: 1.0
- secondary_before: 0.04375
- secondary_after: 1.0

The composed paraphrase score is 0.75 only because stable secondary records have identical before/after state labels. On the address-identifying changed focal rows, the composed result is effectively chance: focal_before 0.006 and focal_after 0.994 average to about 0.5, while oracle routing remains near 0.9875 on the same focal before/after pair.

Thus exact role composition works, but semantic transfer from `background/update` to `prior/revised` is still absent in the important before/after axis. The surviving result is exact surface-correspondence selection, not temporal/discourse meaning.

## Exact surface matcher baseline

The CPU-only matcher extracts the tag from the context line whose role words exactly match the query role:

| suite | exact matcher | synonym-map matcher |
|---|---:|---:|
| held exact nsA | 1.0 | 1.0 |
| held exact nsB | 1.0 | 1.0 |
| held role swap nsA | 1.0 | 1.0 |
| trainChanged exact nsA | 1.0 | 1.0 |
| held paraphrase nsA | 0.0 with no matching line | 1.0 with hand-supplied prior→background and revised→update |

This matters for interpretation: exact selector reader bridge synthesis and macro challenge success is compatible with a local role-string/tag extraction rule. It is compositional as an engineered pipeline, but not evidence that the model has acquired temporal-role synonymy.

## Current bridge-level result

The supplied-address loop now has a clean ending point:

1. Direct address-to-state readers can be trained and then fixed.
2. Exact role-to-address selectors can be trained with explicit candidate selection.
3. A selector-chosen address composes with the fixed reader at the oracle-reader ceiling.
4. The previous monolithic role-query failure was a bad interface/objective, not a proof of incapacity.
5. Semantic role paraphrase remains unsolved; exact-role success is explainable by surface correspondence.

This is a bounded constructive result, not a complete general data-efficient learning principle. It is still valuable because it cleanly separates reader formation, selector formation, and composition, and because it gives a concrete failure mode for small-data learning: without the semantic role map, a learner may have a usable record reader and still fail to choose the right record under a paraphrased role.

## Implication for frontier_consolidation's fixed-budget allocation question

The micro result does **not** predict a uniform compact-view-over-breadth advantage across broad ex-Entity BabyLM families. It predicts advantages only where an own-source companion supplies useful correspondence for selecting, aligning, or preserving a record/version/state that an unrelated breadth item cannot supply.

Therefore:

- A broad ex-Entity `V-B` near zero would be consistent with the micro result: most non-Entity broad gain can sit in `B-C` as finite-experience stream composition or generic diversity, without requiring own-source correspondence.
- A broad ex-Entity `V-B` strongly positive across several families would require an additional mechanism beyond this exact role-address bridge; it would mean source-conditioned compact companions help even where no obvious state/update selector exists.
- Entity aggregate `V-B` must be split by zero/nonzero operations and by affected/unaffected binding panels. A positive mean can be an update-side allocation shift rather than better record correspondence.

Reference-arm logic:

- `R` is not the best broad mechanism reference because exact repeat fuses duplicate recurrence with relatedness and can hide or mimic source-related effects.
- `C` is useful for placing the full stream against clean finite experience, but a positive `V-C` alone does not say the companion must belong to its own source.
- `B` is the right first reference for the question “does the compact companion need to be about its own source rather than merely adding more same-population text?”, but it is still not the final correspondence reference because it does not match the compact rewrite multiset and row-pair geometry as tightly as the permuted-companion arm.
- If broad or balanced Entity `V-B` survives after the pending scorer output, the next high-value macro asymmetry is aligned view versus permuted companion, preferably in the second basin and split by operation/state families. More clean totals alone would not isolate correspondence.

## Cheap next evidence

No more small supplied-address training is justified unless a new premise appears. The useful immediate work is file-level and already-running macro readout:

1. Complete `V-C=(V-B)+(B-C)` and `V-R=(V-B)+(B-R)` by family and checkpoint after scorer outputs land.
2. For Entity, compute `V-B`, `B-C`, and `B-R` separately on zero-operation/nonzero-operation rows and affected/unaffected binding panels.
3. If broad ex-Entity `V-B` is small while `B-C` is positive, treat broad gain as stream/breadth/geometry rather than source-correspondence.
4. If broad ex-Entity `V-B` survives, prioritize the permuted companion comparison over additional clean seeds, because it keeps compact companion fertility and token geometry while breaking own-source pairing.
