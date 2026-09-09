# earlier analysis — Route decision: binding-switch representation objective before another large route

## Evidence used

- S1, CLBH and cadence evidence: `notes/route_review_after_s1_clbh_cadence.md`
- CLBH repaired binding result: `notes/clbh_binding_repair_result.md`
- Cadence closure: `notes/cadence_all_arms_interpretation.md`
- Relation-explicit FineWeb closure: `notes/relation_scaling_closure_and_eval_repair.md`
- Paired-restatement closure: `notes/pair_discriminator_tokenaware_1m_profile.md`
  - 
  - 

## Route comparison

The target remains joint improvement of Entity, EWoK, and GlobalPIQA while preserving protected Supplement and Reading. The strongest recent negatives rule out three tempting shortcuts:

- schedule emphasis: cadence moves Supplement/Entity locally but damages EWoK;
- output copy: CLBH copies context tokens but does not switch with queried entity;
- internal coordinate reshuffle: S1 adds GlobalPIQA but lowers Entity, EWoK, SuperGLUE, and Reading.

The two remaining families are:

1. **Relation representation inside the encoder.** This attacks the clbh binding repair result failure directly: the masked-position hidden state must encode which entity is being queried and which value is bound to it. A pure output-copy path cannot supply this. A full LEBS-style slot architecture is plausible but heavy and risks becoming topic/capacity unless a smaller measurement first proves the variable can be made addressable.

2. **Rebuilt data mechanism.** Data remains important because the public leader likely used simplification-pair material, but the simple forms are negative: naive adjacency, paired-restatement, and relation-explicit FineWeb do not produce durable Entity/EWoK gains. A stronger data route would require high-retention surface-divergent re-encounters with exact text-multiset and binding-swapped controls. It is promising but materialization-heavy, and previous data routes show how easily the model absorbs paired text as topic/copy/redundancy rather than relation binding.

A proposed first comparison before either full route is: a **binding-switch representation objective** with no inference-time parameters. It is a small measurement that can decide whether the current DeBERTa MLM interface can be trained to make the query entity addressable at all. If it fails under clean conditions, building larger output heads or more pair data is likely premature; a true architecture change such as role-sensitive slots becomes necessary. If it succeeds with held-out transfer, it gives the exact mechanism that can later be embedded into official-data pretraining or a more realistic pair-data route.

## Chosen next experiment: Binding-Switch Margin (BSM) pre-scale measurement

### Scientific question

Can an ordinary DeBERTa MLM representation be trained so that, in the same context containing two entity-value bindings, the masked-token preference changes when only the queried entity changes?

The essential pair is:

- Context: `Alice found a ball. Bob found a hat.`
- Query A: `Alice lost the [MASK].` should prefer `ball` over `hat`.
- Query B: `Bob lost the [MASK].` should prefer `hat` over `ball`.

A result is meaningful only when both directions of the pair are correct. A model that always prefers `hat` over `ball` repeats the clbh binding repair result CLBH failure.

### Mechanism

Use the normal MLM head and no added inference parameters. For each generated binding pair `(E1,V1),(E2,V2)` produce two masked queries. Add a margin loss through the existing MLM logits:

- Query E1: enforce `logit(V1) - logit(V2) >= m`;
- Query E2: enforce `logit(V2) - logit(V1) >= m`.

This objective acts through the encoder and existing MLM head. It is not a pointer, not a copied-token scatter, not a new decoder, and not an added memory at inference time.

The small experiment should start from the protected 100M DeBERTa checkpoint and fine-tune for a tiny number of words/examples on generated non-evaluation binding text. This is only a mechanism measurement, not a submission training route. If it works, a later step can port the idea to official-text mining or compliant training.

### Data for the first measurement

Build a synthetic-but-controlled non-evaluation binding set using the retained tokenizer:

- candidates must be verified semantic tokens under the clbh binding repair result rule: `tokenizer(' ' + word)` is `[standalone space marker, single semantic word token]` or a single matching token;
- split entities, values, and templates into train and held-out sets;
- include object, location, state, color, and number-like relations;
- vary entity order, value order, sentence order, distance, and query position;
- include both same-sentence and cross-sentence contexts;
- keep paired examples short enough to avoid truncation.

The first held-out measurement must include:

- held-out entity names with seen value types;
- held-out values with seen entity names;
- held-out templates with seen lexical sets;
- held-out entities, values, and templates together;
- order-flipped contexts where the nearest value is not the correct value.

### Arms

Use the same examples, same updates, and the same base checkpoint:

1. **Frozen/base measurement:** protected 100M checkpoint, no fine-tuning.
2. **MLM-only same-text fine-tune:** same generated texts and masks, no binding margin.
3. **BSM fine-tune:** MLM plus binding-switch margin.
4. **Random-correspondence control:** same BSM loss weight and examples, but entity-value labels randomized within each pair so no stable binding relation exists.

No added inference parameters are allowed. This separates relation learning from capacity. Same texts and masks separate relation learning from content distribution.

### Measurements

The important measurements are pair-level, not only item-level:

- pair-level two-direction success: both `E1→V1` and `E2→V2` margins positive;
- mean margin and bootstrap over pairs;
- same-value preference rate: how often the model prefers the same value in both directions;
- order sensitivity: whether changing sentence/query order changes the answer incorrectly;
- held-out entity/value/template transfer;
- representation read: a small linear readout from the mask-position hidden state should predict which entity-value binding is queried better in BSM than MLM-only and random-correspondence arms;
- module-free causal evidence: since BSM has no added module, disabling is not applicable, but comparing to MLM-only and random-correspondence shows whether the relation loss rather than text exposure drives the effect.

Do not use fast EWoK as the primary small-measurement result; fineweb relation vs random 3m direct checkpoint trajectory showed it is too noisy. Fast official columns become useful only after the mechanism measurement passes and a real 1M run is justified.

### Interpretation

If BSM fails to improve held-out pair-level two-direction success beyond MLM-only and random-correspondence, then the ordinary DeBERTa MLM interface is not readily learning an addressable entity-value variable from this form of signal. The next route should move to an encoder architecture with role-sensitive latent binding slots or relation-factorized attention, not another output-copy or data-redundancy variant.

If BSM succeeds only on training templates or only when the answer token appears verbatim in the nearest context, it is still not enough. That would repeat copy/template behavior.

If BSM succeeds on held-out entities, values, templates, and order flips, then it provides a concrete representation objective to port into official-data pretraining. The next step would be to mine or generate compliant high-precision binding pairs from official text or legal auxiliary sources, with the same-content and random-correspondence controls preserved.

## Secondary route preserved: surface-divergent re-encounter data

Do not erase the data route. A stronger data experiment should use the exact same sentence multiset across arms and compare:

- local consistent paired views;
- dispersed consistent paired views;
- binding-swapped paired views;
- shuffled/random paired views.

The decisive data comparison is not ordinary adjacent versus shuffled. It is consistent binding versus binding-swapped under identical entities, values, topics, token counts, and scheduling. This should be constructed only after the BSM measurement clarifies whether the model can use such relation information at all.

## Next construction task

Build `scripts/binding_switch_margin_pretest.py` and a result note path for the first run. It should:

1. load the protected 100M DeBERTa checkpoint and tokenizer;
2. build verified single-semantic-token entity/value inventories;
3. generate train and held-out binding-switch examples with hashes and splits;
4. run the four arms above for a small number of examples/updates;
5. save JSON containing margins, pair-level two-direction success, same-value preference, transfer splits, and hidden-state readout results;
6. save a short interpretation note.

Only after this small measurement should any 1M official-data run be planned.
