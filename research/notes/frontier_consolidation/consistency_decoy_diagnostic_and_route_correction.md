# consistency decoy diagnostic and route correction — same-row decoy diagnostic corrects the source-view consistency interpretation

CPU-only. No model update, official evaluation, corpus/tokenizer change, H100 work, endpoint continuation, or final artifact.

The init-matched minfreq50 screen and its dependent evaluation remained unresolved; their active outputs were not inspected.

## Scientific Motivation

The projected consistency calibration projected-consistency calibration made source-view consistency look promising because row-centered residual pair geometry is training-strengthened and not identical to raw semantic similarity. However, its written interpretation contained a dangerous ambiguity: it said same-row-other margin exceeded the true-pair shuffle margin and interpreted that as generic same-row/topic agreement dominating true pair-specific correspondence.

If minfreq50 is weak, this sketch alone does not justify an 80M consistency run; first distinguish true source–rewrite correspondence from generic same-row agreement, and construct a collapse-resistant shared/private or predictor-based objective using matched same-row decoys as a cheap diagnostic.

consistency decoy diagnostic and route correction therefore turned the warning into direct CPU evidence before any training.

## Diagnostic 1: asymmetric predictor probe

Script: `experiments/archive/frontier_consolidation/scripts/pair_specificity_predictor_probe.py`
Output: `experiments/archive/frontier_consolidation/data/pair_specificity_predictor_probe/pair_specificity_predictor_probe.{json,md}`

This initial probe sampled 960 front stream rows and found only 43 changed rows / 171 pair records because changed rows are sparse in front stream sampling. It nevertheless exposed the key arithmetic issue:

- tokenmean_80M row-centered residuals: true cosine 0.6916, same-row-decoy cosine -0.2752, true-minus-decoy +0.9668, within-row top1 1.0000.
- tokenmean_100M: true cosine 0.6951, same-row-decoy cosine -0.2755, true-minus-decoy +0.9705, within-row top1 1.0000.
- A closed-form ridge source->rewrite predictor on row-held-out splits did **not** improve this: predictor true-minus-decoy ~0.805 versus identity ~0.968 at 80M; both top1 1.0.

Interpretation: the source residual itself already identifies its true rewrite over same-row decoys; a small linear predictor is not needed for this separation and actually weakens it in this sample.

## Diagnostic 2: repaired global same-row decoy audit

Script: `experiments/archive/frontier_consolidation/scripts/same_row_decoy_global_audit.py`
Output: `experiments/archive/frontier_consolidation/data/same_row_decoy_global_audit/same_row_decoy_global_audit.{json,md}`

The first version of the global audit repeated the front-sampling sparsity problem by using `row_index_in_pool_1based` as a row number into the materialized 100M stream. That was wrong because the 100M stream is shuffled/repeated relative to the 10M pool, matching stream-order confound finding. I repaired the script to sample by stable `example_id` from the route portfolio and intervention assets span map and read the first changed-source occurrence in the 100M stream.

Repaired audit: 768 changed rows / 3,093 pair records, exact spatial repair route status legal tokenizer SHA `91b775...`, frozen reinvest stream SHA `3dd19f09...`, tokenmean 80M/100M checkpoints, row-centered residuals, projection dims 64/128/480.

Key results:

- tokenmean_80M:
  - d64: true_cos 0.6856, same_row_decoy_cos -0.2664, random_other_cos -0.2162, true-decoy-mean +0.9731, true-decoy-max +0.7542, within-row top1 0.9997.
  - d128: true_cos 0.7039, same_row_decoy_cos -0.2676, random_other_cos -0.2175, true-decoy-mean +0.9929, true-decoy-max +0.7881, within-row top1 0.9997.
  - d480: true_cos 0.7006, same_row_decoy_cos -0.2676, random_other_cos -0.2171, true-decoy-mean +0.9894, true-decoy-max +0.7935, within-row top1 1.0000.
- tokenmean_100M:
  - d64: true_cos 0.6887, same_row_decoy_cos -0.2665, random_other_cos -0.2148, true-decoy-mean +0.9763, true-decoy-max +0.7606, within-row top1 1.0000.
  - d128: true_cos 0.7075, same_row_decoy_cos -0.2677, random_other_cos -0.2161, true-decoy-mean +0.9967, true-decoy-max +0.7939, within-row top1 1.0000.
  - d480: true_cos 0.7039, same_row_decoy_cos -0.2677, random_other_cos -0.2157, true-decoy-mean +0.9929, true-decoy-max +0.7997, within-row top1 1.0000.

## Corrected interpretation

projected consistency calibration's printed phrase `pair-same-row-other margin` was `actual_minus_same_row_other_mean`, not the same-row-other cosine. A larger value means the true source–rewrite pair is **more** separated from same-row decoys, not that same-row decoys dominate the true pair.

Therefore:

1. The collapse story that a positive-only row-centered agreement loss would merely learn generic same-row/topic agreement is **not supported** by current representation geometry. Same-row decoys are hard by source row but easy by residual geometry; the true rewrite is almost always the nearest within-row rewrite.
2. The genuine risk for a consistency objective is different: pair-specific residual geometry is already strong/top1-saturated by 80M, so a positive-only agreement loss may have limited useful headroom and may over-compress private information or reduce the beneficial rate-distortion asymmetry of compact views.
3. A predictor-based source->rewrite map is not justified by the cheap diagnostic as a way to distinguish same-row decoys; identity already separates them better. Predictor/asymmetry may still be useful for collapse resistance or private/shared decomposition, but that must be constructed from the goal of preserving private details while gently sharpening shared structure, not from a false same-row-collapse diagnosis.
4. This is not BabyLM score evidence and does not authorize a GPU run. If minfreq50 is weak, a consistency comparison first requires the following construction: build a clean single-variable objective on the legal16k token-mean compact-view substrate, preserve MLM as primary, and include same-row decoys as a cheap monitor or optional margin term only if the construction explains why additional pressure helps beyond already-saturated within-row retrieval.

## Consequence for route choice

- Compact-view reinvestment remains the validated data-efficient core.
- minfreq50 remains the only active GPU experiment. Its 70M/80M score vector must still decide whether to continue support-floor to 100M/full eval.
- If minfreq50 is flat/redistributive, source-view consistency remains a possible successor, but consistency decoy diagnostic and route correction weakens the naive positive-only justification. The comparison first requires a collapse/private-information-safe shared/private or predictor-style objective and a trainer with decoy monitoring; the projected calibration sketch does not justify an 80M run.
- The route should be tested as a single variable: same corpus, spatial repair route status legal16k tokenizer, architecture, seed, WWM token-mean MLM, optimizer/schedule/exposure, and official-compatible 70M/80M cheap eval. No combination with minfreq50, fractional credit, static masking, or sequence scheduling in the first screen.

## Addendum: broad predictor diagnostic on exact changed rows

After writing the initial note, I ran the broad repaired predictor probe:

- Script: `experiments/archive/frontier_consolidation/scripts/predictor_global_decoy_probe.py`
- Output: `experiments/archive/frontier_consolidation/data/predictor_global_decoy_probe/predictor_global_decoy_probe.{json,md}`
- Sample: 768 exact changed rows / 3,093 pair records, row-centered residuals, dims 64/128/480, tokenmean 80M and 100M.

Results refine but do not overturn the corrected interpretation:

- Identity residuals already separate true rewrites from same-row decoys almost perfectly: 80M d128 true-decoy-mean +0.9929 and top1 0.9997; 100M d128 +0.9967 and top1 1.0000.
- Closed-form ridge source→rewrite prediction on row-held-out splits slightly increases **mean margin** at larger dimensions (80M: +0.0202 at d128, +0.0483 at d480; 100M: +0.0196 at d128, +0.0475 at d480), but top1 is already saturated and predictor top1 is fractionally lower by about 0.0002–0.0006.

Interpretation update: a predictor is not justified as a needed extractor of same-row correspondence, because identity residuals already solve the same-row decoy task. The small margin increase indicates that an asymmetric predictor can smooth/denoise the mapping, but this is weak construction evidence, not score evidence. If source-view consistency is pursued after minfreq50, the design should preserve MLM/private detail while gently sharpening an already-existing shared pair residual, with decoy monitoring to detect harmful compression; it should not be framed as rescuing absent pair-specificity.
