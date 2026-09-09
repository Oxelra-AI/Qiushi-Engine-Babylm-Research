# Contextual second-view follow-up controls

The clean-Qwen mechanism comparisons and contextual cap-120 materialization motivate the following controls and interpretation limits.

## Strongest scientific updates

1. The existing same-window evidence remains real but not fully single-factor. clean qwen control interpretation and next mechanism supports that attention-visible original--Qwen second-view correspondence/coherence is active, especially through aligned-vs-separated and aligned-vs-selected-original-duplication contrasts, but packing topology, local density, and pair placement are still part of the causal package.

2. Cap-120 is a good first full-recipe test because it changes the high-risk relation-only topology into one-pair rows embedded in real official context while preserving generated pair word dose. The treatment/control pair asks whether the contextual recipe beats a pure-official row-length-matched recipe under the same seed/model/tokenizer/exposure.

3. If cap-120 improves, the result should not immediately be called a pure native-context effect. The most important follow-up control is same pair and same row length with unrelated same-source official context around the pair. The second important control is same true context and position with original+original instead of original+rewrite.

4. If cap-120 fails, it does not falsify context: it may over-dilute the correspondence signal, alter pair distance/order, or introduce tokenized exposure differences. The right response would be cap/density variants (cap80/cap160 or lower pair-dose deep-context) rather than abandoning the mechanism.

## Predefined readout profile

Let R = {BLiMP, EWoK, COMPS, Reading}; these are the columns we want contextual official language to recover relative to clean qwen compliance and validity clean aligned.
Let P = {Supplement, Entity, SuperGLUE}; these are the columns where same-window correspondence should retain its active benefit relative to the cap-120 official control.

A genuinely stronger result should show:

- R improves for at least most members relative to clean qwen compliance and validity clean aligned seed43022, not just one isolated task.
- P remains positive relative to the cap-120 official length-matched control.
- NLP average and fixed no-AoA aggregates rise; AoA must be reported only as final measurement and never used as selector.
- Overall is not driven by a single SuperGLUE/Entity/AoA jump with flat or negative BLiMP/EWoK/COMPS/Reading.

## Next concrete experimental controls if cap-120 is promising

1. `same_source_unrelated_context`: keep the same pair IDs, original/rewrite order, contextual row lengths, source labels, and approximate left/right context lengths, but draw surrounding official words from unrelated rows of the same source. This isolates true native adjacency/discourse context from source-compatible context and pair dilution.

2. `contextual_original_duplicate`: use the exact native context and row position, but replace rewrite with the original sentence. This isolates generated second-view benefit from original repetition inside context.

3. `cap80/cap160` or sparse deep-context variants: vary context amount and pair density to find whether cap120 over- or under-dilutes the correspondence signal.

4. Seed replication: if cap120 seed43022 is strong, repeat treatment/control or at least treatment seed43122 before treating the effect as robust.

## Implementation audits to keep attached to results

- tokenization/truncation audit already saved at `data/contextual_one_pair/cap120/contextual_cap120_tokenization_audit.json`; pair-side visibility is almost complete, but both treatment and control have long-token rows.
- verify pair inventory against clean qwen compliance and validity if making direct bidir ranking contextual training and eval plan--clean qwen compliance and validity mechanistic claims.
- report source composition and unique official row coverage, because cap-120 changes source weighting and coverage relative to clean qwen compliance and validity.
