# stage3 execution synthesis Stage-III execution synthesis for the legal v5 search

This note records the evidence after the AutoModel SuperGLUE repair and before the corrected format arms finished. Their performance remained unresolved at this stage.

## Current highest question

The Stage-III task is not simply to find any scalar movement above the v4 number. The target is a legal Strict-Small model improvement that inherits the Stage-II finite-experience principle and is measured through the same model computation on every official column. The current strongest practical levers are now separated into four objects:

1. model-file fidelity for downstream classification, so SuperGLUE sees the same adapter/private computation that zero-shot MLM and Reading see;
2. special-token format exposure, because official inputs include `[CLS]/[SEP]` while v4 pretraining and coherent private replay did not;
3. row-format exposure, especially whether isolated or half-isolated streams teach the private branch to behave better on short evaluation rows without damaging context-supported competence;
4. dense unchanged-Qwen target coverage, which gives a replicated Entity/COMPS/GlobalPIQA-positive and BLiMP/Supplement-negative profile on the completed zero-shot/Reading columns.

These objects are not interchangeable. Relation composition redistributes finite prediction practice across columns; a model-file or input-format mismatch can remove dead weight and may therefore move the surface differently from relation redistribution. The next v5 candidate should be a measured composition of surviving levers in one private phase plus one faithful model file, not an arithmetic assumption that separate improvements add.

## AutoModel SuperGLUE repair changes the reference system

earlier analysis showed that the official SuperGLUE finetuning path calls `AutoModel.from_pretrained`, while the Qiushi custom checkpoints had registered only `AutoModelForMaskedLM`. stage3 execution synthesis independently repaired this for both the slow-adapter `chck_82M` reference and the coherent86 alpha0.75 private endpoint.

The repair script is `experiments/archive/relation_learning/scripts/automodel_bridge_and_validate.py`. It creates isolated repaired copies under `experiments/archive/relation_learning/data/automodel_bridge` and validates hidden states against the trusted MLM encoder. The output `automodel_bridge_validation.md/json` establishes:

| checkpoint | repaired AutoModel computation | source AutoModel without bridge |
|---|---|---|
| `chck82_scale1p75` | matches trusted `AdapterDebertaV2ForMaskedLM` encoder with max diff `0`; exposes `995584` adapter parameters at scale `1.75` | stock `DebertaV2Model`, `0` adapter params, max hidden-state diff `1.512444`, mean `0.249628` |
| `coherent86_alpha075` | matches trusted `FrozenSlowPrivateDebertaV2ForMaskedLM` encoder with max diff `0`; exposes `995584` slow-adapter and `995584` private-adapter params at scales `1.75` and `0.75` | stock `DebertaV2Model`, `0` adapter/private params, max hidden-state diff `1.484657`, mean `0.251423` |

This means the old SuperGLUE value inside coherent86/v4 was produced from a stripped encoder. The cheap7 and Reading measurements remain valid because they use the MLM path. AoA is also unaffected by AutoModel because it uses `AutoModelForMaskedLM`, and the truthful alpha0.75 AoA had already been measured as `0.0` with `19 × 8005` finite rows in `experiments/archive/representation_and_objectives/data/alpha075_aoa_minctx0/alpha075_aoa_minctx0_summary.json`.

Faithful SuperGLUE for repaired coherent86 and repaired chck82 was planned after the half-format sequence finished. Its script is `experiments/archive/relation_learning/scripts/faithful_superglue_queue.py`. This will answer whether evaluation fidelity alone changes the v4 reference and whether the slow/private adapters improve or hurt classification fine-tuning when they are actually present. If it raises coherent86 above its old value, the scientific reading is evaluation-fidelity correction, not a new training method. The same AutoModel bridge must be applied to any format or dense endpoint before SuperGLUE is used in Overall arithmetic.

## Corrected format arms now test three distinct input-format levers

The corrected trainer is `experiments/archive/relation_learning/scripts/word_paced_format_replay_trainer_eec5bdf8.py`. Its healthy smoke references are:

- `experiments/archive/relation_learning/data/smoke_coherent_special/summary.json`: coherent suffix rows with special tokens, update 1 uses `39,553` words, `8,450` targets over `56,823` non-special tokens, target ratio `0.148707`, main CE `2.440441`, initial slow/private CE equality, and near-zero coherent KL.
- `experiments/archive/relation_learning/data/smoke_isolated_after_del_fix/summary.json`: isolated rows, update 1 uses `39,553` words, `8,760` targets over `58,306` non-special tokens, target ratio `0.150242`, main CE `3.890465`, initial slow/private CE equality, and near-zero coherent KL.

earlier analysis measured the frozen trunk loss gap caused by adding special tokens and by isolating rows. On the same masking convention, special tokens changed held coherent CE by `+0.0776` on the leash set and `+0.0102` on the readout set, but changed the first isolated macro-batch by `+0.2989`. Therefore a gain from the isolated arms would not be cleanly interpretable unless compared to the coherent-special control.

The active tasks are:

- Training/evaluation of `coherent_unsplit_special` and `isolated_all`, seeds `98097` and `98098`, was in progress.
- Training/evaluation of `half_coherent_half_isolated`, seeds `98097` and `98098`, was in progress.
- Paired item localization was planned after all six format payloads became available.
- Repaired AutoModel SuperGLUE evaluation of coherent86 and chck82 was planned after the half-format sequence.

Read the format training logs before using scores: each run should have 101 macro-updates, `3,992,800` words, target ratio near `0.15`, private-on/off equality at the first zero-learning-rate update, coherent readout KL near zero at initialization, and an alpha0.75 endpoint. Score interpretation should be hierarchical: coherent-special isolates special-token exposure; isolated-all adds short-row/isolation practice; half-format tests whether the private branch can condition on context presence and avoid full context loss.

## The dense route remains a serious comparator, with SuperGLUE still being repaired

The dense unchanged-Qwen target-coverage route has replicated zero-shot/Reading behavior over two private seeds. Current completed columns versus coherent86 are approximately:

| seed | cheap7-sized zero/Reading movement | shape |
|---|---:|---|
| 62064 | `+0.22089` | BLiMP `-0.45045`, Supplement `-0.59788`, EWoK `-0.09714`, Entity `+1.04473`, COMPS `+0.10654`, GlobalPIQA `+1.48544`, Reading `+0.055` |
| 62065 | `+0.20612` | BLiMP `-0.48767`, Supplement `-0.54196`, EWoK `-0.24969`, Entity `+1.09347`, COMPS `+0.09828`, GlobalPIQA `+1.48544`, Reading `+0.045` |

The movement is reproducible and scientifically meaningful: Entity gains concentrate in deeper operation strata while 0-operation Entity worsens. The available-column item table in `research/documents/relation_learning/data/dense_available_item_flips/available_pairwise_item_flips.md` shows where the movement lives but should not be used as a veto by summing all discrete items, because that sum is dominated by BLiMP row count rather than the leaderboard macro objective. The important use of item-level analysis is replication and localization of the movement, not replacing the official macro metric.

The AutoModel path for dense and coherent86 checkpoints was repaired and repaired SuperGLUE started. The file `direct_validation.json` reports all repaired checkpoints valid, while `repair_validation.md/json` still show an earlier read-only-cache failure. Subsequent synthesis should cite the successful validation record or an updated repair record, not the stale failed one.

## Mechanism result: transferred priors follow training-family base rates

The two binding attempts now give a sharper scientific statement than simply saying transfer is value-type dependent.

The descriptor binding family was semantically wrong for literal state assignment: its “states” were LLM-written descriptors, so training made operation presence predictive of not using the source. In official Entity, the alpha0.75 endpoint scored cheap7 `41.1579` and Entity `26.32`, about `-2.0` Entity versus coherent86. Entity strata show the damage concentrated in no-relevant-update retention: `rel_eq0` fell to `26.22` versus chck82 `37.44`, and `rel_eq0_irrelevant_ops_gt0` fell to `25.30` versus `38.00`, while `rel_ge1` was roughly unchanged at `25.35` versus `25.08`. The learned coarse prior was therefore “an operation in context makes the original source less trustworthy,” not a usable update operator.

The mirrored literal 480-map family solved the local object much better, but its family statistics trained a different coarse prior. In many rows the query entity should retain its source while operations happen to another entity; official transfer at scale0.2 scored Entity `23.14`. It lifted no-update retention strongly: `rel_eq0` `45.75` versus chck82 `37.44`, and `rel_eq0_irrelevant_ops_gt0` `44.62` versus `38.00`. At the same time it damaged all relevant-update strata: `rel_ge1` `18.10` versus `25.08`, `rel_ge1_postrel_ops0` `17.57` versus `30.38`, and `rel_ge3_postrel_ops0` `18.67` versus `34.56`. Recency/last-operation option picks also fell sharply. Thus the family transferred the base-rate prior “operations are often about someone else / do not update the queried source,” not the intended latest-assignment relation needed by official Entity.

This mechanism evidence strengthens the finite-experience principle: the private branch can learn a relation when diverse clean instances make it the simplest useful predictor, but under transfer it carries the coarse statistical structure of the practiced family. Training an intended operation description is insufficient if the family’s observable base rates license a cheaper prior. Future relation compositions must control not only labels and candidate presence, but the operation/update base rates that can be exported as a shortcut prior.

## Interpretation criteria fixed before reading the pending numbers

The reading object after the format tasks and faithful SuperGLUE finish is the full official macro surface under faithful loading and real AoA handling. Item localization should answer which examples and columns changed and whether the movement replicates, not act as a row-count-weighted replacement for the leaderboard objective.

earlier analysis fixed the interpretation before the six format endpoints and the faithful SuperGLUE numbers returned. The record is `experiments/archive/relation_learning/data/superglue_old_path_spread/superglue_old_path_spread_and_lever_rules.md/json`. Under the old stripped AutoModel path, chck82, coherent86, and dense checkpoints used the same stock encoder, so their old SuperGLUE values estimate fixed-protocol spread rather than model differences: chck82 `69.766181`, coherent86 `69.819222`, dense62064 `69.813329`, and the current dense62065 file `69.795649`, with macro range `0.053041`. A repaired coherent86 movement within that band should be read as ordinary run variation; a movement outside it is model-file fidelity evidence. This band is not a full fine-tuning-seed distribution, so a selected final endpoint should receive one additional SuperGLUE fine-tuning seed before submission.

The practical target has also moved with the repair. The historical `42.1210` coherent86 number is the old stripped-path reference. A new trained v5 must exceed faithful v4: coherent86 cheap7 plus repaired coherent86 SuperGLUE plus the measured coherent86 AoA. If faithful SuperGLUE lifts v4, a bridged v4 re-upload is model-file fidelity, not new training, and the training contribution must clear that faithful reference.

A training lever may enter the composed candidate only if both private seeds move the same relevant columns in the same direction beyond the coherent two-seed band and no important column falls beyond that band. The composition is then fixed once and trained/evaluated across two private seeds; separate favorable deltas from dense and format screens must not be added after seeing the results. The likely composition space remains:

- AutoModel bridge in the checkpoint files for every endpoint that goes to SuperGLUE;
- the best format lever if coherent-special, isolated-all, or half-format gives a two-seed benefit without broad damage;
- the dense target-coverage lever if its repaired SuperGLUE and AoA preserve the current zero-shot/Reading macro gain;
- a truthful AoA ladder constructed and measured for the final composed endpoint, rather than using missing-ladder arithmetic.

The next composition must be trained and evaluated directly across two private seeds. It should not be selected by adding separate deltas from dense and format experiments. The scientific question remains how finite experience becomes evidence for reusable computation under a strict budget; the practical question is whether these now-separated levers can jointly produce a legal `Qiushi-BabyLM-36M-Strict-Small-v5` that improves the overall competence surface beyond faithful v4.
