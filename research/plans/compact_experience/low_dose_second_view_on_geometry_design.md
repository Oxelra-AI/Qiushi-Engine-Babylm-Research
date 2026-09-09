# official geometry exposure and steps low-dose same-window second-view design on an official-only geometry substrate

This is a design note only. Do not launch these trainings until the official-only geometry experiment has identified a stronger substrate and the chosen checkpoint/row geometry is frozen by no-AoA evidence.

## Scientific reason

The clean Qwen route established that same-window original–generated-second-view correspondence is an active component: it beats selected-original duplication and separated coexistence. The cap-120 high-dose contextual arm then showed that simply inserting all 37,594 pairs (16.568% of the 10M words as Qwen pair words) inside official context is not a frontier model: it loses no-AoA aggregate to its own official control and remains below the clean 41.3443 coordinate.

The new clue is that the official cap-120 length-matched control itself produced a strong no-AoA profile (especially BLiMP/equal7), but its effect is confounded with row segmentation, source/coverage changes, visible-token exposure, and 2,809 optimizer updates at batch256. official geometry exposure and steps therefore tests official-only geometry first. If a geometry such as G3 remains strong after matched-update testing, the next second-view experiment should treat generated pairs as a *sparse correspondence signal* on top of the stronger official substrate, not as a large replacement of official experience.

## Legal and measurement constraints

- Use only official BabyLM training text plus locally generated Qwen rewrites already produced under approved teacher-family constraints.
- Count all generated words inside the 10M-word pool and 100M-word exposure.
- Use the Strict-Small tokenizer and random initialization for the submission model; no external weights, hidden states, distributions, or tokenizer leak into the model.
- Do not use official AoA/CDI words, child curves, AoA predictions, or AoA scores for data construction, schedule design, checkpoint selection, or route selection.
- Select checkpoints for expensive full eval from no-AoA trajectory columns only: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading. SuperGLUE and AoA enter only as final measurements on frozen targets.

## Candidate experimental family after geometry is established

Let G* be the best official-only geometry after official geometry exposure and steps/official geometry noaoa interpretation evidence, e.g. original official160 G0, cap120 official geometry G3, or matched-update G3. Build all arms with identical model initialization, tokenizer, WWM, max word exposure, checkpoint schedule, row-length sequence when possible, and batch/LR-time policy chosen from G*.

### Arm L0: official-only G* control

Already produced by the official-geometry experiment if G* is one of G0/G3 variants. This is the substrate baseline.

### Arm L1: low-dose same-window second-view

Replace or insert a small fixed subset of selected original–Qwen pairs into G* rows while preserving the 10M pool word count and matching row-length sequence to L0 as closely as possible.

Suggested initial doses by generated-pair words:

- 5% total words: about 500k pair words, roughly 11.3k average clean pairs.
- 8% total words: about 800k pair words, roughly 18.2k pairs.
- Avoid starting with the full 16.568% cap-120 dose, since it already damaged the no-AoA aggregate.

Select pair subset by fixed pre-declared hash/strata using only training-corpus metadata: source, pair length, entity/number preservation flags from the cleaning step, and row location. Do not use downstream evaluation outputs.

### Arm L2: same-source shuffled rewrite control

Use the same selected original rows and the same generated rewrite text inventory as L1, but pair each original with a different rewrite from the same broad source/length bin. This preserves generated text, row length, source weighting, and local context pressure while destroying the exact same-window correspondence.

### Arm L3: contextual original-repeat control

Replace the rewrite with a second copy or compressed excerpt of the selected original side while keeping row length and context topology as close as possible. This tests whether L1 gains come from true generated second-view information rather than selected-sentence repetition or extra masking opportunities.

### Optional Arm L4: selected-context-only control

Use the same source rows/context positions as L1 but no generated rewrite, filling with official text under the same row-length sequence. This isolates selection of high-value source rows from the correspondence signal.

## Outcome interpretation

A useful second-view effect on G* should satisfy at least the following scientific profile before being promoted:

- L1 beats L0 on no-AoA equal7 or preserves equal7 while strongly improving a load-bearing weak column without large damage elsewhere.
- L1 beats L2, showing local correspondence/coherence rather than just generated-text inventory.
- L1 beats L3 or differs in a way that cannot be explained by exact original redundancy.
- The strongest frozen checkpoint remains competitive after corrected full nine-column evaluation; AoA is interpreted only as a final measurement.

If L1 improves only Supplement/EWoK while damaging BLiMP/COMPS/Entity/GlobalPIQA as cap-120 did, reduce dose or change topology rather than repeating high-dose insertion.

## Implementation guidance

A materializer should write:

- exact 10M pool JSONL for each arm;
- exact 100M training JSONL with at most 10 passes;
- metadata with row counts, word counts, pair-dose words, generated-word fraction, official context/filler words, source composition, row-length distribution, selected pair IDs, no-leakage statement, and SHA256 hashes;
- a trainer-visible tokenizer audit under `add_special_tokens=False`, right truncation, seq256, including visible token counts and pair-side visible fractions.

The safest first implementation is to copy the bidir ranking contextual training and eval plan/038 contextual materializer structure, but make the pair subset and pair dose explicit, and instantiate L0/L1/L2/L3 from the same row-length plan so that model score differences are not silently driven by row count or LR-time.
