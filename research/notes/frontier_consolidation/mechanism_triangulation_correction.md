# extractive selected readout result — mechanism triangulation correction before extractive readout

## Why this correction matters

The joint mechanism triangulation note contained an over-clean interpretation: it described extractive-vs-compact as if it isolated generated re-expression and treated separate experimental comparisons as additive factors of one design. That is too strong and would bias interpretation of the pending selected readout.

The corrected view is now saved directly in `notes/joint_mechanism_triangulation.md`.

## Corrected scientific reading

The two comparisons inform the same compact-view hypothesis but are not one additive factorial.

### extractive pair

pending balanced/wide source-only extractive DeBERTa arms hold own-source IDs, row count, word budget, stream placement, tokenizer, and stock DeBERTa recipe fixed. They do **not** change only the generated surface. Relative to natural compact views, they also change:

- source-absent content: extractive has zero, compact has about 0.173 content/content;
- continuity/fluency: extractive is more telegraphic and fragmented;
- skip-bigram continuity: balanced 0.472 and wide 0.354 versus compact 0.223;
- source-position distribution and coverage: extractive has broader source span (balanced 0.972, wide 0.928) than compact (0.807);
- content density and BPE/token geometry.

Therefore an extractive lag cannot be blamed simply on missing source/tail access, but it also cannot be attributed cleanly to generated re-expression alone. It tests whether source-only approximations of compact word budget and coverage are sufficient in the same DeBERTa coordinate.

### RoBERTa factorial

HS/LS/HD/LD factorial is in a RoBERTa MLM coordinate. Its `(HS-LS)-(HD-LD)` pattern can test whether own-source anchoring survives when density is controlled in that RoBERTa coordinate, but it cannot be added numerically to DeBERTa effect.

## Priority order

1. Resolve matched DeBERTa 80M/100M selected result first when the two earlier analysis tasks deliver.
2. Interpret extractive-minus-compact and balanced-minus-wide on common-checkpoint stable families with paired intervals: cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, EWoK+Entity, Supplement, Entity, COMPS.
3. Use factorial later as cross-coordinate survival evidence: ask which explanations remain plausible after both studies, not which additive component explains how many points.
4. Only if the selected deficit is real and stable should the next same-coordinate separation be designed; candidate separations must be motivated by the actual deficit pattern, not by the earlier over-clean factor table.

## Interpretation map for the pending result

- Balanced near compact while wide lags: compact-like density/token geometry matters more than maximum source coverage.
- Wide near compact while balanced lags: source-span/coverage matters more than compact-like density.
- Both lag compact on stable families: source access is insufficient; natural compact continuity, source-absent generated content, token geometry, or their coupling remain load-bearing.
- Both exceed compact: generated compact views may contain avoidable noise; source-only construction becomes a serious successor route.
- Mixed or interval-crossing deltas: do not spend another full 100M arm immediately; use the family/item pattern to identify a lower-cost same-coordinate separation.

## Boundaries preserved

No leaderboard submission. No source-absent target mask/loss weighting from probe evidence. No interpretation from MLM loss, GlobalPIQA-only movement, local pseudolikelihood, or best-checkpoint selection alone.
