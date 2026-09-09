# route after controlled memory result — Route after controlled prefix-memory result

## What changed

The controlled controlled grid profile interpretation profile changed the interpretation of the memory route. With paired shared-core initialization, fixed schedule, identical example order, pool/exposure separation, and two seeds per pool, the current prefix-average memory adapter has no robust positive effect.

Evidence: `experiments/archive/initial_model_studies/data/controlled_grid_profile.json` and `research/notes/initial_model_studies/controlled_grid_profile_interpretation.md`.

Mean memory-minus-dense deltas:

| metric | pool1M | pool10M |
|---|---:|---:|
| BLiMP | -0.06 | +0.06 |
| Supplement | -1.40 | +0.20 |
| EWoK | -1.04 | -0.46 |
| Entity | -0.16 | -0.12 |
| COMPS | -0.20 | -0.20 |
| Reading eye | -0.15 | -0.40 |
| Reading self-paced | +0.04 | +0.12 |

The earlier apparent standalone Entity lift and 10M-trajectory EWoK/Reading lift were not stable once initialization, schedule, and pool/order were controlled. The current vectorized prefix average should not be scaled or treated as the main Entity mechanism.

## Stronger signal: text-pool and presentation effects

The dense controlled runs show a larger and more stable pool effect than memory. Pool10M minus pool1M in dense controls:

| metric | seed42 | seed43 | mean |
|---|---:|---:|---:|
| BLiMP | -0.47 | -0.66 | -0.57 |
| Supplement | -0.40 | -0.40 | -0.40 |
| EWoK | -1.45 | +1.54 | +0.05 |
| Entity | +1.91 | +1.63 | +1.77 |
| COMPS | +0.14 | +0.75 | +0.45 |
| Reading eye | +1.21 | +1.16 | +1.19 |
| Reading self-paced | +0.65 | +0.44 | +0.55 |

The full 10M-pool mixture improves Entity and Reading at 1M exposure but costs BLiMP/Supplement. This is exactly the Strict-Small tension: source mixture and presentation can move human-like columns and NLP columns in opposite directions. This makes data order and presentation the next route with highest value.

## Relevant outside knowledge read this step

- `Beyond Random Sampling` reports that curriculum learning can accelerate pretraining and that compression ratio, MTLD, number of tokens, and Flesch Reading Ease are useful difficulty signals. It distinguishes strict ordering, pacing, and interleaved curricula; interleaving can preserve diversity while still shaping order.
- The readability curriculum source grounds Flesch Reading Ease as a lightweight, cognitively motivated difficulty measure based on sentence length and syllabic complexity.
- The variation-set BabyLM source shows that consecutive or adjacent rephrasings can affect BLiMP/GLUE, but effects depend strongly on presentation order and benchmark family; this supports testing order and repetition with official text only, not importing synthetic text yet.
- The sequence-length BabyLM source reports that shorter contexts tend to favor BLiMP/Supplement/EWoK, while longer contexts can help Entity/Reading-like tasks. This suggests a staged length schedule or mixed short/long batches, not a single fixed sequence length.

## Next route: developmental mixture/order plus length staging

Do not continue current prefix-memory unless a new entity-state mechanism is built from a different idea. The next main experiment should attack the stronger controlled signal: source mixture/order and context length.

### Candidate experiment family

Use the verified trainer path, but add an order mode that replaces pure shuffle with source/difficulty stages while keeping official corpus only and exact word-exposure accounting.

Start with the dense-untied 4x256 control because it is cheap and already has paired-init infrastructure. Compare every new data route against a dense-untied random full-pool control under the same seed, schedule, and exposure.

Initial 1M exposure experiments:

1. **Full-pool random baseline**: current pool10M dense cells already provide this for seeds 42/43.
2. **Developmental source order**: early CHILDES + BNC/Switchboard/OpenSubtitles; later Gutenberg/SimpleWiki. Purpose: keep the Entity/Reading lift from rich dialogue/full-pool data while reducing early formal-text pressure.
3. **Readability/length interleaved order**: compute simple sentence/chunk scores from official text only: words per chunk, punctuation/sentence count, average word length, approximate Flesch Reading Ease, type-token proxy. Split into difficulty bins and cycle easy-to-hard within several interleaves. Purpose: inherit curriculum findings while avoiding a narrow easy-only prefix.
4. **Length staging**: train early with shorter chunks/sequence length for syntactic learning, then expose longer chunks or packed dialogue/document chunks within the same word budget. Purpose: preserve BLiMP/Supplement while improving Entity/Reading.

The first executable version should not combine all ideas. A good next Execute step is to implement order manifests and run a small 2-seed 1M comparison:

- dense-untied random full-pool, seed42/43: already available;
- dense-untied developmental source order, seed42/43;
- dense-untied readability/length interleaved order, seed42/43.

Profile the same fast/local columns: BLiMP, Supplement, EWoK, Entity, COMPS, Reading. If a data order improves Entity/Reading without the BLiMP/Supplement loss seen in random full-pool, then scale that order to a 10M trajectory and compare against dense6x384 and later objective/tokenizer variants.

## Mechanism route held in reserve

A new entity-state mechanism would need a sharper object than prefix averaging: explicit slots updated by noun/pronoun-like token cues or local repeated-name patterns from official text. It should be attempted only after the data-order experiment, or in parallel only if implemented cleanly with a non-persistent capacity control from the start.

## Why this route serves the SOTA goal

The official Overall averages multiple families. Current dense scaling helps BLiMP/Supplement/EWoK but hurts Entity/Reading; current memory does not fix this. The strongest controlled lever now observed is data mixture/order. A curriculum that keeps pool10M Entity/Reading gains while recovering pool1M BLiMP/Supplement would directly attack the central Strict-Small tradeoff and is more likely to raise Overall than another small architecture perturbation.
