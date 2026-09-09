# fineweb experience characterization — Narrative-State Data Route after the FineWeb Matched 1M Result

## What the fineweb relation vs random 1m profile result means

The matched 1M FineWeb-Edu experiment separated same-source random text from relation-explicit text.

Both arms used:

- FineWeb-Edu `sample-10BT` from the same streamed prefix;
- exactly 1,000,000 whitespace words;
- DeBERTa-v2 8×480, baseline16k, WWM p=0.15;
- seed 42, extra_init_seed 456, train_rng_seed 789;
- batch 128 after batch 256 OOM;
- 49 optimizer updates and a saved `chck_1M`.

Result, relation-explicit minus random-quality:

| column | delta |
|---|---:|
| BLiMP fast | +0.71 |
| Supplement fast | +0.00 |
| EWoK fast | +3.27 |
| Entity Tracking fast | −0.12 |
| COMPS | −0.29 |
| Reading mean | −0.41 |

The relation-explicit FineWeb filter moved EWoK strongly and BLiMP modestly, while Entity did not move. This is a real source/selection signal, but not the desired entity-state learning signal.

## What the characterization adds

`scripts/characterize_fineweb_experience.py` compared the training examples themselves.

Relation-explicit minus random-quality, measured over packed 160-word examples:

| feature | delta |
|---|---:|
| unique_entities_mean | +0.178 |
| entity_mentions_mean | +0.281 |
| repeated_entities_mean | +0.064 |
| cross_sentence_repeated_entities_mean | +0.068 |
| has_cross_repeat_frac | +0.029 |
| transition_threads_mean | +0.019 |
| has_transition_thread_frac | +0.014 |
| location_threads_mean | +0.019 |
| possession_threads_mean | +0.012 |
| static_relation_sents_mean | +0.124 |
| has_static_relation_frac | +0.007 |
| has_noise_frac | −0.007 |

The current relation filter mostly selects entity-rich static/factual text and only slightly increases repeated-entity transition threads. The samples confirm this: Kennedy half-dollar catalogs, biographical monarchy passages, archives, astronomy pages, and technical descriptions. These can plausibly help EWoK because they add factual and conceptual relation exposure, but they do not force the model to track a changing entity state through a passage.

## Updated scientific object

The next data object should be **Narrative State Transition FineWeb**: examples where the same entity appears in multiple nearby sentences and a later mention depends on a changed or maintained state introduced earlier.

This is different from static relation density:

| property | current relation filter | needed narrative-state filter |
|---|---|---|
| main signal | entity + relation words in one passage | repeated entity across adjacent sentences |
| relation type | static facts, definitions, biography, catalogs | movement, possession, holder, role, physical state, action result |
| temporal shape | often timeless or encyclopedic | before/after or event progression |
| expected transfer | EWoK/world knowledge | Entity Tracking plus EWoK |
| risk | factual/static gain without Entity | lower yield, more careful cleaning needed |

## Next exact-word data experiment

Use the same source family and short budget. Do not scale to 10M/100M until this experiment gives linked Entity and EWoK movement.

### Arms

Create three exact 1M-word arms from the same FineWeb-Edu stream prefix:

1. **random_quality_tokenpacked**
   - Basic high-quality FineWeb-Edu text.
   - Token-aware packing repair compared with fineweb relation vs random 1m profile.

2. **static_relation_tokenpacked**
   - Reproduce the current relation-explicit filter with token-aware packing.
   - This preserves the EWoK-positive reference arm.

3. **narrative_state_transition_tokenpacked**
   - Require repeated entity mentions across sentence boundaries.
   - Require at least one state-changing event tied to that entity.
   - Prefer short adjacent-sentence windows where the entity is mentioned before and after the event.

### Narrative-state filter

For each candidate document, split into sentences and identify entity strings by capitalized spans plus selected common nouns when repeated with determiners.

A document/window is selected when it contains at least one thread:

`sentence_i`: entity `E` + event/state cue + filler/state `S1`  
`sentence_j`, where `1 <= j-i <= 4`: same entity `E` or clear alias/pronoun + continuation/state cue

Relation families:

- **Movement / location**: `went`, `moved`, `returned`, `arrived`, `left`, `entered`, `crossed`, `visited`, plus `in/at/on/from/to/into/onto/near`.
- **Possession / transfer**: `took`, `gave`, `brought`, `carried`, `held`, `kept`, `lost`, `found`, `received`, `put`, `placed`, `packed`.
- **Physical state / availability**: `opened`, `closed`, `broke`, `repaired`, `filled`, `emptied`, `covered`, `trapped`, `freed`.
- **Role / social state**: `became`, `was elected`, `was appointed`, `married`, `joined`, `left`, `served as`, `worked as`.
- **Action consequence**: `built`, `destroyed`, `saved`, `killed`, `sent`, `dropped`, `raised`, `lowered`, only when a later sentence reuses the affected entity or object.

Reject passages dominated by:

- book/product lists, catalog pages, syntax manuals, archive metadata;
- coin/stamp/database tables and long numeric lists;
- pages with many date/fact fragments but no event continuation;
- bullet-heavy or glossary-style text;
- one-sentence static definitions.

### Token-aware packing

fineweb relation vs random 1m profile had about 24% examples truncated at 256 tokens. The next materializer should pack by tokenizer length, not only whitespace words.

Target packing:

- keep each training row at at most 220 baseline16k tokens before special tokens;
- use variable whitespace length rather than fixed 160 words;
- still make each arm exactly 1,000,000 whitespace words;
- record total untruncated tokens, kept tokens, word groups, and truncated row fraction;
- target truncation below 5% for all arms, and preferably below 2%.

This improves the scientific comparison because relation-rich documents are often longer and could otherwise lose event context at the 256-token cutoff.

## Measurements before training

For each materialized arm, compute:

- exact word count and row count;
- source prefix size and document yield;
- tokenizer-length distribution;
- repeated-entity thread rate;
- transition-thread rate by relation family;
- static-relation sentence rate;
- source-noise rate;
- manual sample file with 30 examples per arm and thread annotations.

The narrative-state arm should have a large increase in transition-thread rate over both random and static-relation arms. If it does not, the filter has not produced the intended experience.

## Short training plan

If the materialized arms look like the intended experience:

- train all three arms at 1M words with the memory-safe batch 128 setting;
- same model/tokenizer/WWM/seed/init as fineweb relation vs random 1m profile;
- evaluate the same fast profile: BLiMP, Supplement, EWoK, Entity Tracking, COMPS, Reading;
- optionally add the state counterfactual ranking smoke results-style state-ranking inventory and a held-out narrative-state thread probe derived from the training source but split by entity/filler.

Continuation value comes from the pattern:

- static_relation reproduces EWoK movement;
- narrative_state improves Entity while keeping EWoK positive or neutral;
- Supplement and Reading do not collapse;
- token truncation is reduced enough that the result reflects experience content, not cropping.

## If narrative_state improves Entity but loses EWoK

Then construct a fourth blend later, not now:

- 60–70% narrative_state;
- 30–40% static_relation;
- same exact word budget;
- compare against the best single arm at 1M or 3M.

This would test whether the EWoK-positive static relation signal and Entity-positive narrative-state signal are complementary.

## If narrative_state does not improve Entity

Then the route should not be scaled by data selection alone. The next route would need data plus an objective or architecture that turns transition threads into predictive computation, such as delayed state contrast using the narrative-state windows, but only after the short data-only comparison is known.

## Files carrying the current evidence

- Matched 1M data: `data/fineweb_relation_matched_1M/`
- fineweb relation vs random 1m profile profile: `data/fineweb_relation_vs_random_1m_profile.json`
- fineweb relation vs random 1m profile note: `notes/fineweb_relation_vs_random_1m_profile.md`
- fineweb experience characterization characterization: `data/fineweb_experience_characterization.json`
- fineweb experience characterization note: `notes/fineweb_experience_characterization.md`

## Immediate construction task

Build:

`scripts/fineweb_narrative_state_materialize.py`

It should create the three exact-word token-aware arms and save:

- `data/fineweb_narrative_state_1M/random_quality_tokenpacked_1000000w.jsonl`
- `data/fineweb_narrative_state_1M/static_relation_tokenpacked_1000000w.jsonl`
- `data/fineweb_narrative_state_1M/narrative_state_transition_tokenpacked_1000000w.jsonl`
- `data/fineweb_narrative_state_1M/materialization_meta.json`
- `data/fineweb_narrative_state_1M/samples_for_manual_reading.json`
