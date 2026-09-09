# earlier analysis — Legal BSM strategy after Entity+/EWoK− transfer

## What the recent evidence says

The BSM line now has a real but incomplete signal.

- bsm interpretation: a mature 100M DeBERTa checkpoint learns entity-conditioned value switching from concentrated synthetic binding examples in under 100 updates. Plain MLM on the binding cloze examples already saturates.
- bsm density persistence: the effect is conditional. From `chck_1M`, binding does not form even at 100% binding exposure over 20k words. From `chck_100M`, 100% and 20% binding exposure produce strong templated transfer and persistence, but 20% gives weak natural-template transfer and 5%/1% largely fail.
- bsm transfer official fast: the over-budget mature BSM adaptation transfers to official fast Entity Tracking (+1.51) while leaving BLiMP and Supplement essentially unchanged, but it damages fast EWoK (-1.54).

The scientific interpretation is not “density alone solves the route.” It is: focused entity-value binding experience can move an Entity-like official measure, but in its current synthetic cloze form it trades against EWoK and has not been shown to form during legal from-scratch training.

## Rule constraints that shape the next experiment

From `notes/official_rules_and_landscape.md` and `plans/2026_strict_small_pilot_contract.md`:

- Training corpus: at most 10M whitespace words.
- Total model exposure: at most 100M words / 10 epochs.
- Custom or swapped data may be used with a datasheet, still within the 10M-word corpus budget.
- Synthetic data must obey the closed-system rule; external-model outputs or distillation are not usable unless their language exposure is counted.
- Required checkpoint sequence for a serious full run: `chck_1M` through `chck_10M`, then `chck_20M` ... `chck_100M` as applicable.

Therefore the earlier analysis/334 BSM model can never be a submission coordinate. It starts from a 100M checkpoint and adds extra exposure. Positive Entity transfer only justifies a legal from-scratch replacement/scheduling experiment.

## Routes not to take next

1. **Do not continue optimizing mature-checkpoint adaptation.** It is over-budget and already shows Entity+/EWoK− rather than joint improvement.
2. **Do not use 100% synthetic binding cloze data as a final route.** It is too distribution-shifted and likely explains the EWoK damage.
3. **Do not repeat broad relation-WWM.** `babylm_masked_train_relation_wwm.py` exists and supports relation-weighted masks, but earlier relation-biased masking and cadence experiments produced Entity/Supplement/EWoK tradeoffs rather than the target joint movement.
4. **Do not reopen CLBH or output-copy mechanisms.** clbh binding repair result showed copy without entity-conditioned binding.

## Legal route families still worth comparing

### Family A — official-text high-precision binding targets

Use only official corpus text and change the masking/targeting policy. Mine examples where two entities or objects are associated with distinct values/properties, then mask the queried value. This avoids synthetic-corpus ambiguity and keeps the corpus unchanged.

Risk: earlier broad relation-WWM failed because its target selection was too diffuse. A useful version must identify high-confidence local binding structures, not merely relation/entity words.

### Family B — rule-generated natural binding replacement corpus

Build a custom corpus of at most 10M words by replacing a fraction of official examples with rule-generated natural binding mini-stories. The templates are rule-based, not produced by an external model. Entity/value inventories should be derived from the official corpus and tokenizer where possible: capitalized names, common concrete nouns/adjectives, locations, and simple property words verified as stable semantic tokens. The generated examples must be documented in a datasheet and counted as corpus words; repeated training exposures count toward the 100M cap.

Risk: bsm transfer official fast already shows templated binding can help Entity while hurting EWoK. The replacement corpus must be more natural and must include controls that separate true binding from lexical/topic effects.

### Family C — hybrid official-mined plus small replacement

Use official-mined binding targets as the main source and a small amount of rule-generated binding as scaffolding. This may reduce EWoK damage while providing more concentrated binding signal than official text alone.

This is attractive later, but the first executable screen should be simpler.

## Chosen next executable screen: legal natural-BSM replacement from scratch

The next construction should build a small from-scratch legal screen, not a mature-checkpoint adaptation.

### Corpus construction

Create a materializer `scripts/materialize_legal_bsm_corpus.py` that outputs corpus variants and metadata:

- `official_control`: same official corpus slice and word budget, no binding replacement.
- `bsm_natural_20pct`: replace 20% of the selected corpus words with rule-generated natural binding examples.
- `bsm_swapped_20pct`: same entities, values, templates, word counts, and surface distribution as `bsm_natural_20pct`, but query-answer bindings are swapped or randomized so stable entity-value correspondence is destroyed.
- Optional after the first smoke: `bsm_natural_5pct` to test a lower-dose version, since bsm density persistence shows 5% is weak in the mature short probe but may behave differently over repeated epochs.

The materializer should record:

- selected official examples removed and retained;
- generated binding templates and word counts;
- entity/value inventory source and tokenizer verification;
- row-level `kind`, `pair_id`, `binding_condition`, and answer metadata;
- corpus unique word count and planned exposure count.

For legality, generated rows must replace official rows rather than add to them. A final candidate corpus must remain at or below 10M unique whitespace words. A screen may use a smaller corpus, but the metadata must still distinguish unique corpus words from repeated exposure words.

### Training implementation

Create or adapt `scripts/train_legal_bsm_screen.py` by extending `babylm_masked_train_fullcycle.py` rather than earlier analysis.

Required behavior:

- From random initialization, same protected DeBERTa-v2 8×480/baseline16k/WWM backbone unless deliberately changed.
- Official rows use ordinary WWM exactly as the protected baseline.
- Binding rows are ordinary unmasked text in the corpus, but the trainer uses row metadata to mask the answer word in the query sentence and predict the correct value through the standard MLM head. This avoids putting literal `[MASK]` text into the corpus.
- The swapped/random arm must have the same lexical rows and target positions but wrong correspondence metadata.
- Save official-style checkpoints at least `chck_1M`, `chck_10M`, and `chck_20M` for the first screen; preserve compatibility for full schedules later.
- Record exact word exposure, unique-corpus words, binding-word fraction, official-word fraction, and target-mask counts.

### First screen budget

Start with a 20M-exposure screen (two epochs over a 10M or smaller legal corpus, depending on construction time) for seed42:

1. official control,
2. coherent natural-BSM 20%,
3. swapped/random natural-BSM 20%.

If a full 10M corpus materialization is too slow for the first build, use a smaller corpus slice but keep the same replacement/exposure accounting and do not interpret it as a submission-scale score.

### Measurements

At `chck_1M`, `chck_10M`, and `chck_20M`:

- synthetic binding-switch probes from bsm density persistence, including natural templates;
- official fast Entity Tracking;
- official fast EWoK with the nested path `evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast`;
- BLiMP fast and Supplement fast;
- Reading fast if affordable;
- GlobalPIQA parallel/nonparallel fast if affordable.

The key comparison is coherent natural-BSM minus swapped/random under the same lexical distribution, and coherent natural-BSM minus official control. A useful result must not reproduce bsm transfer official fast’s pattern of Entity gain purchased by EWoK damage.

### How to read outcomes

- If coherent natural-BSM improves Entity but EWoK drops similarly to bsm transfer official fast, then the current synthetic binding formulation is not the SOTA mechanism. Move toward official-mined high-precision binding targets or a relation-representation mechanism that protects EWoK.
- If coherent natural-BSM beats swapped/random on Entity and does not damage EWoK/Supplement/Reading, run a second seed and a lower-dose 5% arm before scaling.
- If neither coherent nor swapped forms move Entity, then early formation remains the central obstacle; revisit architecture or objective scheduling rather than adding more synthetic data.
- If the swapped/random arm moves similarly to coherent, the effect is lexical/topic exposure, not binding.

## Immediate next work

The proposed first test comprises the materializer and trainer smoke, not launch a full 100M run. The first deliverable from the next step should be inspectable code and a tiny smoke dataset/training run that proves:

- row metadata can target answer masks correctly;
- coherent and swapped rows have matched word counts and lexical inventory;
- the trainer saves loadable HF checkpoints;
- exposure accounting is recorded in the run metadata.

Only after the smoke is verified does the design proceed to the 20M screen on H100.
