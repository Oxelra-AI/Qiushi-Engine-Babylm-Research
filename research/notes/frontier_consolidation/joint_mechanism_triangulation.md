# joint mechanism triangulation — joint mechanism triangulation: extractive pair × HS/LS/HD/LD factorial

## State of the two decisive experiments (joint mechanism triangulation, both mid-run, unpolled)

- source-only extractive DeBERTa pair (balanced and wide): both progressing normally, through `chck_60M` (6/10 checkpoints) at ~3076s of a 9000s budget. No terminal result. The staged 80M/100M selected readout apparatus is complete and validated (earlier analysis/228/229); the paired-interval analyzer (earlier analysis) reproduces known aggregate deltas on legal_compact 80M→100M with sensible item/cluster intervals.
- RoBERTa factorial (`step241_..._accum_mb16`): HS/LS Wave 1 running, through `chck_40M`; HD/LD Wave 2 not started. No interaction result yet.

Neither experiment can yet answer its question. No CPU tooling is added; the readout paths already exist.

## Why these two experiments are complementary, not redundant

Both inform the same surviving object — the natural DeBERTa compact-view-reinvestment benefit (+1.3464 mean7 at 80M vs clean; earlier analysis no-AoA compact 44.289 vs repeat 41.872 vs adjbreak 43.056) — but they are **not** factors of one additive design. the source-only study is a same-DeBERTa matched realization contrast, whereas the factorial study uses RoBERTa. Use them to ask which explanations survive both studies, not to assign clean additive effect sizes across one coordinate:

1. **extractive (same-coordinate source-only realization contrast).** Holds own-source IDs, row count, word budget, stream placement, tokenizer, and stock DeBERTa recipe fixed, but changes several coupled properties at once: zero source-absent content (compact 0.173), less continuous/fluent text, different content density and token geometry, higher skip-bigram fragmentation (balanced 0.472 / wide 0.354 vs compact 0.223), and *broader* source span (0.972/0.928 vs 0.807) and coverage than compact. Therefore an extractive lag cannot be blamed simply on missing source access, but it also cannot be read as a clean effect of generated re-expression alone. It tests whether source-only approximations of compact word budget and coverage are sufficient in the DeBERTa coordinate.

2. **factorial (RoBERTa anchor-sharing × density study).** Holds surface-form family within its own cells and crosses source-anchor sharing (own vs deranged donor) with density (compact vs repeat). The `(HS-LS)-(HD-LD)` pattern on stable families (Supplement, EWoK, Entity, COMPS) can show whether own-source anchoring still matters after density is controlled in a RoBERTa MLM coordinate. Because positive compact effect and the pending extractive arms are DeBERTa, the interaction is a cross-coordinate survival test of explanations, not a same-model decomposition of effect.

## How to combine the studies without overreading them

Extractive arms are closest to HS corner only in the narrow sense that their second views are drawn from each row's own source and preserve high source overlap. They do **not** differ from HS only in surface. They simultaneously change source-absent content, continuity/fluency, source-position distribution, density, BPE/token geometry, and skip-bigram structure; HS/LS/HD/LD comparison is also RoBERTa rather than DeBERTa. The useful joint question is therefore: which explanation remains plausible after both studies are read on the same stable selected families?

- If source-only arms approach compact at 80M/100M, then generated compact surface is not required in this DeBERTa coordinate; source-word selection with own-source recurrence becomes a serious constructive route. A follow-up test could determine whether own-source anchoring survives outside DeBERTa, but it does not supply the DeBERTa effect size.
- If both source-only arms lag compact despite broader source access, compact's advantage is not explained by tail/source access alone. The missing component could be fluent continuity, generated source-absent content, token geometry, or their coupling; the result would motivate a same-DeBERTa separation only if the stable-family deficit is real.
- If the RoBERTa factorial shows own-source anchoring matters in RoBERTa while extractive lags in DeBERTa, the shared survivor is not "generation alone" but the conjunction of own-source recurrence with natural compact realization.
- If the RoBERTa factorial shows no coherent own-source advantage while extractive/compact differs in DeBERTa, the current mechanism is likely model-coordinate dependent; more RoBERTa spending on the same premise would not answer the DeBERTa source-only sufficiency question.
- Balanced-vs-wide within DeBERTa remains important: balanced closer to compact would favor density/token-mass matching; wide closer to compact would favor source-span/coverage; both lagging compact would point toward natural compact continuity/source-absent/token-geometry coupling; both exceeding compact would reveal that generated compact views may carry avoidable noise in the legal DeBERTa coordinate.

## Joint readout plan (deferred until both deliver)

When extractive terminal results arrive: `earlier analysis` integrity → `earlier analysis --checkpoints chck_80M chck_100M` staged readout → `earlier analysis` paired intervals, interpreted on common-checkpoint stable families.

After all four factorial arms complete, selected evaluation should report `(HS-LS)-(HD-LD)`. Compare the extractive-minus-compact stable-family delta with HS-LS and HD-LD on the same official-compatible columns. Agreement across these orthogonal decompositions supports a shared mechanism; disagreement bounds each account.

## Active constraints respected
- Whole-word-control conclusion: no source-absent target mask/loss weighting from a probe signal.
- Hash-rotated cyclic repeat (correction) — extractive-vs-repeat is a source-only density/coverage tradeoff, not a suffix-recovery test.
- Stable selected families drive interpretation, not loss / GlobalPIQA / local pseudolikelihood / best-checkpoint narratives.
