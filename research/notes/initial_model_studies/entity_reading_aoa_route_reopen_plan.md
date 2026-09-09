# entity reading aoa route reopen plan — Reopen innovation routes for Entity, Reading, and AoA while testing dense exposure

The capacity controlled 1m profile and next scale result showed that current sparse routing and fixed morph-side representation do not solve the overall tradeoff. Dense6x384 is now the best immediate baseline to test exposure insufficiency, but ordinary dense scaling must not become the whole research program.

## What the 10M dense curve should answer

Run dense6x384 from 1M to 10M official word exposure with `chck_1M` ... `chck_10M`. This distinguishes:

- **Exposure insufficiency:** BLiMP/Supplement/EWoK/COMPS and perhaps Reading improve smoothly with more words.
- **Mechanism deficit:** Entity remains collapsed and Reading/AoA-like behavior remains flat even when NLP columns improve.
- **Trajectory behavior:** intermediate checkpoints reveal whether the model learns in a developmentally plausible order or only improves final task scores.

The 10M dense curve is a baseline and diagnostic instrument, not the innovation endpoint.

## Parallel innovation routes to reopen after/beside the dense curve

### Route A — compact state memory for Entity Tracking and discourse persistence

Mechanistic target: Entity Tracking failure may reflect insufficient state maintenance, not lack of parameters. Candidate should preserve a causal LM interface but add a small recurrent/gated memory update over token representations.

Concrete designs to compare at 1M and then 10M if promising:

1. **Per-layer gated recurrent summary:** maintain a small hidden state updated from each token block output, feed the state back through a learned gate into the next token representation. Must remain causal and reset per sequence.
2. **Entity-slot proxy memory:** allocate a few learned slots updated by attention from current token hidden states; output hidden state attends to slots. No external entity labels.
3. **Linear-recurrent mixer block:** replace or augment one attention/FFN block with a small gated linear recurrence to bias sequential persistence.

Required evidence: Entity Tracking by operation depth, Reading, BLiMP/Supplement, EWoK, and training loss. A useful memory route should raise Entity without destroying BLiMP/Supplement and without flattening Reading.

### Route B — data curriculum/order targeted at developmental trajectory and AoA

Mechanistic target: 10M corpus is a mixture of CHILDES, dialogue, books, subtitles, SimpleWiki, etc. The current trainer shuffles fixed word chunks after taking corpus order. AoA/Reading may depend strongly on presentation order and source mixture.

Candidate curricula:

1. **Child-directed/dialogue early, broad text later:** CHILDES + BNC/Switchboard/OpenSubtitles early; Gutenberg/SimpleWiki later.
2. **short-to-long / concrete-to-abstract proxy:** sort or stage chunks by length, token rarity, and punctuation/structure proxies.
3. **interleaved repetition schedule:** within ≤100M exposure, repeat high-developmental chunks with increasing difficulty rather than a single pass through all files.

Required evidence: checkpoint trajectory across 1M increments, Reading/AoA once available, and no collapse of NLP columns.

### Route C — learning objective aimed at stable competence rather than single-task gain

Mechanistic target: pure next-token prediction learns local syntax early but may not build robust entity/world state. Prior AMLM-like objectives helped NLP but risked human-like tradeoffs, so any objective change must be low-weight and trajectory-monitored.

Candidate objectives:

1. **Low-weight span/order reconstruction auxiliary loss** on official text only, scheduled down over exposure.
2. **Entity-consistency self-supervision by repeated nouns/pronouns** using heuristic spans from text, no external parser/model.
3. **Difficulty-balanced sampling or loss reweighting** from token-frequency/loss EMA, with safeguards against Reading/AoA collapse.

Required evidence: compare against dense6x384 10M curve on all fast profiles and, when ready, AoA/GlobalPIQA/(Super)GLUE.

## Immediate next construction after dense curve

If dense6x384 10M improves NLP but Entity/Reading stay weak, prioritize Route A (state memory) plus Route B (developmental curriculum) for the next equal-exposure comparison. If dense6x384 improves all profile columns including Reading, keep it as the main baseline but still test curriculum/objective changes before 100M/full official evaluation.
