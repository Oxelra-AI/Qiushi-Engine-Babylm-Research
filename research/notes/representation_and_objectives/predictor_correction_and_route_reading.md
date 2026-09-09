# family specific tokenizer predictor correction — the interpolation predictor cannot adjudicate the support-floor route

An independent verification (`data/external/independent_review01_verifier1_integration.md`) identified a real structural flaw in how the family specific tokenizer predictor tokenizer predictors should be read. This note records the correction so the next expensive-route decision is not misled.

## The flaw

Both family specific tokenizer predictor predictors (`tokenizer_support_frontier_predictor.py` and `family_specific_tokenizer_predictor.py`) are per-column convex interpolations between two measured anchors: legal16k (all driver frac = 0, Overall 40.8640) and legal40k (all driver frac = 1, Overall 40.7804). Every candidate column is forced onto the segment between the two anchor column values, so the recomposed Overall is bounded inside ≈40.88–41.02. The reported "minfreq25 upper bound vs 41.8 = −0.417" is therefore close to a tautological restatement of "a blend of two ≈40.8 anchors cannot reach 41.8."

The support-floor hypothesis is precisely that flooring *breaks* the measured segmentation/support trade-off: it could recover GlobalPIQA/Entity toward the legal16k values **while** keeping Supplement/EWoK near the legal40k values — a dominating corner that is NOT on the interpolation segment. An interpolation predictor is definitionally blind to that corner. So the negative-looking predicted Overall is not valid evidence against the support-floor route.

Substituting the corner columns (legal40k Supplement 60.14 + legal16k GlobalPIQA 37.36 + Entity toward 27.39) into the same Overall composition yields an Overall above either anchor, and nothing measured rules that out.

## Additional verified issues

1. The two family specific tokenizer predictor predictor notes disagree materially: family-specific gives minfreq25 central 41.0224 (best = minfreq25); the global frontier predictor gives minfreq25 40.8359 and names minfreq50 (40.9854) best. ~0.19 Overall disagreement with a different best-candidate ordering, driven only by driver/column-attribution choice. Neither Overall should be cited downstream as decisive until reconciled.
2. The within-note "predictor_disagreement" (0.0022) is not independent robustness — both variants share the same two anchors and most drivers, so agreement is by construction. The honest uncertainty is the measured legal40k two-seed population SD 0.3602.
3. The flat ±0.36 seed band is centered on the interpolated value and is likely the wrong width for a floored tokenizer, because a support floor is a variance-reduction hypothesis on exactly the seed-unstable columns (GlobalPIQA/Entity).

## What the predictors DO validly establish

They validly rule out a **pure vocabulary-size interpolation** (a tokenizer whose per-column behavior lies on the legal16k↔legal40k line) reaching 41.8. This supports the earlier decision against another vocabulary-axis two-seed pair. It does NOT support the stronger conclusion that a support-floored tokenizer will fail.

## Corrected route reading

- Do not use the family specific tokenizer predictor predicted Overall band as evidence against the support-floor route. The route can only be decided by a trained-and-evaluated floored tokenizer, read qualitatively: does GlobalPIQA recover toward ~37 (16k end) while Supplement stays near 59–60 (40k end)? A single seed answers this directionally.
- The natural shared control is minfreq50 measured vector; if it shows GlobalPIQA/Entity recovery with retained Supplement/EWoK, that is the direct corner-vs-interpolation test the predictor cannot supply, and minfreq25 then localizes the curve rather than being a blind pair.
- The inherited 42.033 point (Supplement 63.28 AND GlobalPIQA 35.62 simultaneously) is indirect evidence that the dominating corner is physically reachable, though it used an out-of-budget 100M-trained tokenizer, so it bounds feasibility, not the floored-10M outcome.

## Sequence-curriculum accounting: confirmed, with two prerequisites before any run

The family specific tokenizer predictor sequence accounting arithmetic is internally consistent and its attribution reading is well supported: the existing `seq_len_schedule` slices max-256 rows to short prefixes while charging full row words, so a result from it cannot be credited to the leader's batch-scaled 64→256 factor. Before any faithful sequence-curriculum run:

1. Replace inspection with a direct one-step training-loop measurement: instrument one L64 step and record how many word-groups actually enter the loss and what word count is debited against the 10M budget. If the loop already chunks/re-charges correctly, the accounting concern dissolves.
2. Explicitly fix the word-budget regime of faithful chunking. The 1.61–1.99× is a target-token ratio, not a word ratio. Compliant re-exposure keeps the same 10M words while exposing later groups within budget; budget-multiplying re-chunking that re-charges words across stages could violate ≤10M words. The design must state which regime it uses.

## Net effect on the next decision

The strongest CPU product of family specific tokenizer predictor is the sequence-curriculum attribution/accounting finding (well supported, with the two prerequisites above). The tokenizer predictor is useful only for excluding a pure vocabulary-size sweep; it must not be read as excluding the support-floor route. The unresolved scientific object remains one measured vector: a trained-and-evaluated support-floored tokenizer, or minfreq50 result used as the shared control.
