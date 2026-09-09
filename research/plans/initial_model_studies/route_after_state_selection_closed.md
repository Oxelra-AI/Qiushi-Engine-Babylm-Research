# earlier analysis — Route after the state-selection route closed

## Fixed target and current gap

The public Strict-Small leader is not merely a card artifact. public leader available coordinate locally loaded and re-scored `go76dof/wwm_curriculum_simplification_40k` with the retained evaluator:

| column | public leader local |
|---|---:|
| BLiMP | 67.20 |
| Supplement | 56.04 |
| EWoK | 56.07 |
| Entity | 28.45 |
| COMPS | 53.57 |
| GlobalPIQA | 39.665 |
| Reading | 5.425 |

The current best internal coordinate is `wwm_seed43 chck_80M`, Overall about 40.703, with the largest deficits:

| column | leader - current internal |
|---|---:|
| Entity | +6.25 |
| EWoK | +5.63 |
| GlobalPIQA | +2.07 |
| SuperGLUE | +1.53 |

Entity alone cannot solve the Overall gap; fully matching Entity would add only about 0.69 Overall. The route must improve several columns while preserving our advantages in Supplement and Reading.

## What recent experiments actually show

### 1. Static relation FineWeb is an EWoK lever, not an Entity mechanism

fineweb relation vs random 1m profile compared same-source FineWeb-Edu random-quality text against relation-explicit FineWeb text at matched 1M words, same model/init/training geometry. Relation-explicit minus random-quality:

| column | delta |
|---|---:|
| BLiMP fast | +0.71 |
| Supplement fast | +0.00 |
| EWoK fast | +3.27 |
| Entity fast | −0.12 |
| COMPS | −0.29 |
| Reading mean | −0.41 |

fineweb experience characterization characterized the examples. The relation filter mostly increased static/factual relation density and only weakly increased repeated-entity transition threads. This explains why EWoK moved and Entity did not.

### 2. Model-selected natural state passages are not ready for training

validated state stability interpretation added a filler-only repetition control and cross-model stability to the earlier analysis proposed state selection. In 5k FineWeb-Edu documents:

- 19 cases were extracted;
- 18 were common across the scored models;
- all-model `R_extra > 0`: 10/18;
- all-model `relation_over_filler > 0`: 7/18;
- many stable-looking rows were static templates or false entity/state selections such as `english/letters`, `parts/causing`, `standing/rail`, `streets/center`, `nasa/northern`, and `cockroaches/head`.

The current state-selection extractor should not be scaled to 1M. It would mostly amplify lexical/template preferences already present in old checkpoints.

### 3. WikiAuto pair adjacency was real but too weak and domain-mismatched

Earlier WikiAuto controls matter because they separate pair adjacency from source distribution:

- Pure pair-adjacent vs pair-shuffled had seed-consistent but small gains: EWoK mean +1.275, Entity mean +0.685.
- Pure WikiAuto pairs were poor in absolute terms versus official WWM: BLiMP −2.17, Supplement −2.40, Entity −1.03, Reading eye −5.36.
- A 50/50 official+WikiAuto mixture did not preserve the EWoK signal and damaged Supplement/Reading.

Thus paired semantic restatement is not closed; WikiAuto as the substrate is weak. The leader’s key difference is not generic Wikipedia simplification but FineWeb simplification pairs under a leader-like masked encoder recipe.

## Next central question

Does aligned FineWeb simplification pairing itself create a broad multi-column signal beyond source distribution and static relation density?

This is the next question because the leader bundle combines:

1. FineWeb source distribution;
2. simplification/restatement pairs;
3. static relation explicitness;
4. 40k SentencePiece tokenizer;
5. 12×384 DeBERTa-v2 shape;
6. LAMB/cosine training;
7. length and WWM-to-token masking schedules.

Architecture shape, 40k in the 8×480 setting, static relation filtering, official-corpus curricula, and official-corpus objectives have not reproduced the Entity/EWoK package. The unisolated factor with the best source support is still aligned FineWeb restatement.

## Proposed next experiment: FineWeb paired-restatement decomposition

Create a controlled 1M-word FineWeb-Edu paired-restatement experiment. It should use deterministic, reproducible transformations, not the gated leader data and not an external LLM.

### Source pool

Use FineWeb-Edu `sample-10BT`, basic quality filters from earlier analysis, plus stricter noise removal from fineweb experience characterization:

- remove lists/catalogs/archive metadata/code/manual pages;
- prefer sentences with named entities, numbers, predicates, appositives, relative clauses, passives, coordination, or explicit relation clauses;
- record all source document ids and file hashes.

### Deterministic simplification rules

Transform only when a high-confidence rule applies:

1. appositive extraction: `X, a Y, ...` → `X is a Y.` plus the original main clause;
2. relative clause extraction: `X, who/which/that V..., ...` → `X V...`;
3. coordination split: `X V1 ... and V2 ...` → `X V1 ... . X V2 ... .` when the subject is recoverable;
4. passive-to-active only when an explicit `by X` agent is present;
5. parenthetical definition split: `X (Y)` → `X is Y.` when the parenthetical is a noun phrase;
6. number/date/property restatement: preserve the same entity and value in a shorter sentence.

Each transformed item must record:

- original sentence;
- simplified/restated sentence(s);
- shared entities;
- shared numbers/content words;
- rule type;
- token and word counts;
- whether the simplification preserves at least one entity-value or entity-predicate binding.

Reject transformed items with low shared content, missing entity overlap, extreme length ratio, or target text that is mostly deletion.

### Four-arm decomposition

All arms should be exact 1M whitespace words if enough material is available, token-aware packed to reduce truncation.

| arm | content | purpose |
|---|---|---|
| `orig_only` | original FineWeb sentences/windows only | FineWeb source and complex form |
| `simp_only` | deterministic simplified/restated sentences only | atomic explicitness without adjacent repetition |
| `true_pair_adjacent` | original immediately followed by its own simplification | same binding in two adjacent views |
| `shuffled_pair_adjacent` | original followed by simplification from another item, same target multiset | same source/target distribution with broken binding match |

The key contrast is `true_pair_adjacent - shuffled_pair_adjacent`. This tests whether paired restatement of the same entity/value binding matters beyond style, token statistics, and seeing two sentences.

### Optional fifth arm if material is abundant

`distant_true_pair`: original and its simplification both included but separated by many examples. This distinguishes binding repetition from strict adjacency.

## Pretraining-free measurements before 1M training

Before launching training, materialize a small sample and compute:

1. yield: transformed items per streamed FineWeb document;
2. shared-binding quality: entity overlap, number overlap, predicate/content overlap;
3. true-vs-shuffled pair contrast under current frozen models: mask shared content in the simplification and compare its log-probability with the true original context versus shuffled original context;
4. manual reading sample: 50 true pairs and 30 shuffled controls.

This measurement is not used to select examples by the current model. It only tests whether the transformed pairs have real shared content and whether the local context can in principle expose a pairing effect.

Proceed to 1M training only if the pair material is plentiful, shared content is high, and manual samples show genuine meaning-preserving restatements rather than deletion or template artifacts.

## 1M training and scoring

Use the same strong short-run configuration as fineweb relation vs random 1m profile unless memory requires the b128 repair:

- DeBERTa-v2 8×480;
- baseline16k tokenizer first;
- WWM p=0.15;
- seed/init held fixed;
- exact 1M word exposure;
- batch 128 if necessary;
- same fast profile: BLiMP, Supplement, EWoK, Entity, COMPS, Reading.

If the four-arm test is too large for one step, run the most informative pair first:

- `true_pair_adjacent` vs `shuffled_pair_adjacent`, plus reuse fineweb relation vs random 1m profile random/static results only as external context, not as matched controls.

## How to interpret outcomes

| result | interpretation | next route |
|---|---|---|
| true_pair > shuffled on Entity and EWoK | aligned FineWeb restatement is a real leader-like lever | scale pair construction and add tokenizer/optimizer factors |
| true_pair ≈ shuffled, both > orig | simplification style/source matters, not binding match | use simplified corpus as source layer, not pair objective |
| simp_only > orig but true_pair not better | atomic explicitness helps; adjacency unnecessary | build high-quality simplified FineWeb base |
| no arm improves Entity/EWoK | deterministic simplification is not enough | move to controlled counted microstories or model-side factors |
| EWoK improves but Entity stays flat | same pattern as static relation filter | mix for EWoK only, seek separate Entity mechanism |

## Parallel route to keep in view

A controlled state-transition microstory layer remains the strongest explicit Entity route if FineWeb pairs fail. It should be executor-verifiable with wide surface variation and mixed conservatively with FineWeb static-relation and official linguistic anchors. It should not be the next training route until the FineWeb pairing factor is tested, because the public leader’s actual success points more directly to simplification-pair data than to synthetic state stories.

## Next experiment

The proposed materializer is `fineweb_pair_decomposition_materialize.py`:

- define exact simplification rules;
- write the metadata schema;
- specify token-aware packing;
- specify true/shuffled pairing construction;
- specify small materialization and frozen-model pair-contrast measurements;
- preserve exact 1M arms for subsequent matched training.
