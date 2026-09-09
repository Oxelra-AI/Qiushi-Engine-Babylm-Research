# temperature confidence scale and densemask interpretation: confidence-scale alternative for dense unchanged-Qwen focus

## Purpose

Dense unchanged-Qwen focus has a reproducible useful signal—larger Qwen source-help, stronger common-target source-follow swing, and Entity-depth gains—but also broad likelihood and transfer costs: repaired SuperGLUE deficit, BLiMP/Supplement/EWoK losses, endpoint CDI surprisal worsening, and the earlier analysis context-gain pattern where both row-context and isolated NLL worsen. An important alternative explanation motivated the confidence-scale analysis: some larger source margins may come from a broad change in logit confidence scale rather than a better contextual selector.

I built and ran `experiments/archive/functional_learning/scripts/temperature_source_readout.py` to separate these components without changing official scoring. The script fits a single temperature per frozen endpoint on ordinary non-Qwen legal-tail text beyond the 80-update sparse/dense training prefix, not on CDI, SuperGLUE, Entity, GlobalPIQA, Reading, or the 25-item source-reversal bank. It then re-scores (i) a sampled Qwen correct-source / wrong-source / view-only perturbation probe and (ii) the 25-item common source-reversal bank. It reports both temperature-adjusted NLL margins and rank/order metrics; ranks are invariant to positive temperature scaling and therefore mark changes in ordering rather than confidence scale.

## Files

- Script: `experiments/archive/functional_learning/scripts/temperature_source_readout.py`
- Output: `experiments/archive/functional_learning/data/temperature_source_readout/temperature_source_readout.json`
- Human-readable summary: `research/documents/functional_learning/data/temperature_source_readout/temperature_source_readout.md`
- Compact CSV: `experiments/archive/functional_learning/data/temperature_source_readout/compact_model_comparison.csv`

The calibration set contained 192 ordinary post-prefix legal-tail records, yielding 260 masked token positions. Source counts were bnc_spoken 17, childes 50, open_subtitles 45, cleanqwen_fineweb_compact_view_reinvest 9, simple_wiki 21, gutenberg 49, switchboard 1. The Qwen source readout used 120 sampled pair segments, 480 base target triplets, and no wrong-source target-string overlaps. The common source-reversal readout used the existing 25 hand-built items.

## Main results

All four endpoints selected the same best grid temperature, T=1.1, on independent lawful text:

| endpoint | calibration NLL T=1 | best NLL | mean target rank |
|---|---:|---:|---:|
| coherent86 | 3.9846 | 3.9706 | 207.53 |
| sparse_focus_seed62064 | 3.9516 | 3.9455 | 206.23 |
| dense_focus_seed62064 | 4.0164 | 3.9995 | 233.41 |
| dense_focus_seed62065 | 4.0172 | 4.0001 | 234.01 |

This does not support a simple story where dense is merely more overconfident and can be repaired by a distinct global softening. Dense is worse on calibration NLL and rank, and the same mild T=1.1 softening benefits all endpoints.

On the Qwen correct-source vs wrong-source readout, dense’s advantage shrinks under T=1.1 but remains positive and highly replicated:

| endpoint | Δ specific source advantage vs coherent86 at T=1 | Δ at fitted T | Δ rank advantage |
|---|---:|---:|---:|
| sparse_focus_seed62064 | +0.0017 | +0.0021 | -5.41 |
| dense_focus_seed62064 | +0.1743 | +0.1519 | +69.89 |
| dense_focus_seed62065 | +0.1783 | +0.1554 | +71.86 |

About 10–15% of the NLL-margin increase is compatible with temperature-sensitive scaling, but the rank advantage changes strongly and consistently. This indicates that dense changed the ordering of the target under correct/wrong sources for many probe points, not only the sharpness of an unchanged ordering.

On the 25-item source-reversal bank, dense also retains margin movement after temperature adjustment while rank/order movement is large:

| endpoint | both source conditions correct T=1 | both correct at fitted T | both rank-better | Δ swing T=1 | Δ swing fitted T | Δ rank swing |
|---|---:|---:|---:|---:|---:|---:|
| sparse_focus_seed62064 | 21/25 | 21/25 | 21/25 | -0.0488 | -0.0451 | +2.24 |
| dense_focus_seed62064 | 21/25 | 21/25 | 20/25 | +0.6034 | +0.5385 | +115.20 |
| dense_focus_seed62065 | 21/25 | 21/25 | 20/25 | +0.5961 | +0.5319 | +116.43 |

The dense source-reversal success count is not stronger than sparse on this small bank, so the extra common-bank margin should not be interpreted as many new source-following decisions. But the source-conditioned rank swing does change, especially in the trained-content portion. Thus the current evidence separates two components: (1) dense increases existing or nearly-existing source-conditioned margins, partly temperature-sensitive; (2) dense also changes token/candidate ordering in source-sensitive contexts, which is the component more relevant to a reusable selector.

## Interpretation for the incoming dense-mask/sparse-label result

The dense-mask/sparse-label arm should be read with the confidence-scale alternative explicitly separated:

1. **If dense-mask keeps dense-like Qwen source-help and common-target rank/order improvements while broad losses are smaller**, input-side clue suppression is strengthened as a practical design object, and preservation may involve reducing calibration/likelihood damage while keeping rank-changing source selection.

2. **If dense-mask keeps only larger NLL margins but not rank/order movement**, then much of its apparent source-help is likely confidence or margin scaling. The next method should not optimize enlarged gaps; it should target rank-changing evidence following and Entity-depth decisions.

3. **If dense-mask loses dense’s useful source and Entity movement**, dense target coverage, broad denoising pressure, gradient variability, or dense’s lower per-label focus weighting becomes load-bearing. The evidence would not say that all dense targets encode new knowledge; it would say the sparse-label control did not reproduce the effective optimization geometry.

4. **If dense-mask repeats broad costs without useful rank-changing source movement**, dense second-view corruption itself is suspect and the route should be rebuilt rather than continued through coefficient tuning.

## Related repair

earlier analysis’s no-env AoA smoke failed because Transformers had cached read-only HF module paths before the script changed environment variables. In temperature confidence scale and densemask interpretation, `batched_aoa_extractor.py` was patched to test cache-path writability and synchronize `transformers.utils.hub.HF_MODULES_CACHE` and `transformers.dynamic_module_utils.HF_MODULES_CACHE` after setting output-local paths. A no-env shared smoke then passed: `experiments/archive/functional_learning/data/batched_aoa_cache_selftest2/shared_w2_a1/manifest.json` records `BATCHED_AOA_EXTRACTION_COMPLETE`, 13/13 rows for chck_1M, CPU, with no manual cache environment.

## Boundary

This diagnostic is explanatory only. It must not alter official BabyLM arithmetic. It does not establish v5, broad improvement, or a general data-efficient learning principle. It says that dense’s source-responsive evidence contains a rank/order-changing component beyond simple global confidence scale, while some margin enlargement remains calibration-sensitive and should not be the next method’s target by itself.
