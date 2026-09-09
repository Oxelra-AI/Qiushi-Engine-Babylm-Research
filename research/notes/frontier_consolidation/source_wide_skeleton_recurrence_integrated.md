# source wide skeleton recurrence integrated integrated synthesis: source-wide compact skeleton recurrence

## Experimental Status

No GPU training, no official-compatible model evaluation, no HF upload, no leaderboard submission, and no polling of the managed DeBERTa common-grid tasks occurred in source wide skeleton recurrence integrated. The work was CPU/file-only and aimed to sharpen the compact-view mechanism after the reciprocal/topology explanation weakened.

Pending scientific comparisons:

- protected reference scale1.75 seed43022 common 70M-100M grid scoring.
- scale1.25 seed43022 common 70M-100M grid scoring.

Triangle official-compatible scores remained unavailable.

## Why the mechanism was reopened in this form

Recent evidence made the broad `reciprocal semantic paraphrase` explanation too weak:

- reciprocal multiview mechanism and scaffold: DeBERTa non-copy compact lift was tiny or negative while exact-copy lift was huge; causal lift was one-way and also large for repeat controls.
- causal transfer result synthesis: compact semantic views did not transfer as a broad architecture-general principle to the tested GPT2 causal coordinate; positive cheap7 movement was carried by GlobalPIQA/Reading volatility and relation/state became worse.
- deberta grid tooling and topology packing confound: the topology 2x2 scaffold and deberta pending state causal topology 2x2 scaffold is not clean under the current chunking trainer because pair rows can split across 256-token chunks and the compact arm has larger split rate.

The source wide skeleton recurrence integrated alternative is more precise and less inflated:

> Compact views may act as short, natural, content-dense, source-wide skeletons. They re-expose selected content keys from across the whole source, including late source positions absent from the matched first-N repeat control, while saving words that can be reinvested in more distinct legal source text. Under bidirectional MLM, source and skeleton in the same row create dense cross-view reconstruction opportunities. This is not a generic source-free paraphrase-invariance claim.

## Text-geometry measurement

Script and outputs:

- `scripts/compact_skeleton_recurrence_measure.py`
- `data/compact_skeleton_recurrence/compact_skeleton_recurrence.{json,md}`
- `data/compact_skeleton_recurrence/per_pair_skeleton_metrics.csv`
- `data/compact_skeleton_recurrence/high_tail_skeleton_examples.jsonl`
- `figures/compact_skeleton_source_position_coverage.png`

Input compact pair file: `data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`, SHA `d2a3110c110e216180b632ce7a181acdf348b804ea19f172e16b5afeb0d8c9fc`, 12,155 pairs.

Main result over all pairs:

- Mean source length: 21.54 words; mean compact view length: 13.30 words.
- Compact view covers 66.99% of source content-word positions.
- The first-N repeat prefix covers 59.87% of source content-word positions.
- Compact minus repeat source-content coverage is +7.12 points on average; 54.19% of pairs have positive coverage delta.
- Compact view recovers tail source content at 70.26% mean coverage; 97.66% of pairs have at least one tail source content word recovered by compact.
- Compact views are denser in content words: 65.27% content words vs 49.43% for first-N repeat, +15.84 points.
- Decile structure shows the key asymmetry: in source-position deciles 6/7/8/9, compact covers 63.60/65.56/70.62/76.18% of content positions, while the first-N repeat covers only 31.69/7.46/0.11/0.00%.
- The reciprocal multiview mechanism and scaffold/179 selected 6,071-pair subset is essentially the same: compact coverage 66.83%, repeat coverage 59.68%, compact tail coverage 69.99%, 97.58% of pairs with tail recovery.

This makes the copied-token observation scientifically meaningful: compact views are not merely shorter paraphrases; they systematically bring late-source content keys into the second view.

## Existing model-lift attribution at tokenizer level

Scripts/outputs:

- `scripts/bpe_copy_zone_attribution.py`
- `data/bpe_copy_zone_attribution_chck82/bpe_copy_zone_attribution.{json,md}`
- `data/bpe_copy_zone_attribution_chck100/bpe_copy_zone_attribution.{json,md}`

This re-annotated the existing reciprocal multiview mechanism and scaffold DeBERTa reciprocal-lift records at the actual BPE/tokenizer level using source token offsets. No model was loaded and no new NLL was computed.

For `chck_82M` compact rewrite-side targets (`pair_type=compact`, `target_segment=other`, 512 existing target records):

- Same BPE id appears only in the source tail beyond the matched repeat prefix: 160 records, mean lift 6.084254, positive fraction 0.981250.
- Same BPE id appears only in the prefix: 222 records, mean lift 5.816492, positive fraction 0.950450.
- Same BPE id appears in both prefix and tail: 33 records, mean lift 2.674222.
- BPE id not in source: 97 records, mean lift 0.099026, positive fraction 0.443299.

For content-like compact rewrite-side targets at `chck_82M`:

- Tail-only same-id: 91 records, mean lift 6.330639, positive fraction 0.967033.
- Prefix-only same-id: 104 records, mean lift 6.731483.
- Not-in-source: 46 records, mean lift 0.435714.

For `chck_100M` the pattern persists:

- Compact rewrite-side tail-only same-id: 160 records, mean lift 6.029999.
- Content tail-only same-id: 91 records, mean lift 6.270454.
- Not-in-source: 97 records, mean lift 0.079276.

This means the huge copied-token lift is not only trivial prefix recurrence. A large subset of lifted tokens match source BPEs located beyond the repeated prefix, exactly where the compact skeleton has a structural advantage over the first-N repeat arm. However, the similarity between 82M and 100M means this measurement does not explain the late official-score decline.

## Deterministic source-only skeleton variants

Scripts/outputs:

- `scripts/extractive_skeleton_variant_audit.py`
- `data/extractive_skeleton_variant_audit/extractive_skeleton_variant_audit.{json,md}`
- Variant pair files under `data/extractive_skeleton_variant_audit/`
- `figures/extractive_skeleton_variant_deciles.png`

Variants were exact-length at the compact rewrite word count:

- `compact`: original generated compact rewrite.
- `prefix_repeat`: first-N source words.
- `spread_even`: source-only words spread evenly across source positions.
- `content_spread`: source-only content-heavy spread skeleton.
- `scored_source_skeleton`: source-only scored selector emphasizing content, uniqueness, numbers/capitalized terms, and tail positions.
- `oracle_compact_projection`: source words whose normalized forms appear in compact, filled to length; not source-only, an attribution upper bound.

Aggregate geometry:

| variant | source-content coverage | tail-content coverage | content fraction | Jaccard with compact | selected tail fraction |
|---|---:|---:|---:|---:|---:|
| compact | 66.99% | 70.26% | 65.27% | 1.000 | 40.98% |
| prefix_repeat | 61.73% | 4.85% | 49.43% | 0.359 | 0.00% |
| spread_even | 64.69% | 68.08% | 52.17% | 0.383 | 37.45% |
| content_spread | 92.72% | 92.16% | 75.96% | 0.443 | 30.52% |
| scored_source_skeleton | 98.38% | 99.89% | 81.76% | 0.479 | 46.53% |
| oracle_compact_projection | 70.04% | 67.03% | 57.68% | 0.702 | 35.57% |

Source-only extractive skeletons can therefore realize the source-wide-tail-coverage ingredient without generated text. This is mechanically important because it offers a possible dissection: if source-only skeletons reproduce compact-view gains, the load-bearing principle is algorithmic source-wide skeletal recoding; if natural compact views still beat them, grammatical compression / semantic recoding style is load-bearing.

## Surface/distribution shift audit

Scripts/outputs:

- `scripts/skeleton_surface_audit.py`
- `data/skeleton_surface_audit/skeleton_surface_audit.{json,md}`
- `data/skeleton_surface_audit/surface_gap_examples.jsonl`

The source-only variants are not free wins: they can be telegraphic and distribution-shifted.

Surface summary:

| variant | function-word fraction | punctuation/word | mean source gap | LCS with compact | tokens/word |
|---|---:|---:|---:|---:|---:|
| compact | 29.35% | 0.1685 | 2.451 | 99.99% | 1.544 |
| prefix_repeat | 45.57% | 0.0817 | 1.002 | 44.74% | 1.334 |
| spread_even | 42.66% | 0.1594 | 1.711 | 46.97% | 1.424 |
| content_spread | 21.77% | 0.1753 | 1.670 | 51.62% | 1.545 |
| scored_source_skeleton | 13.73% | 0.1907 | 1.583 | 54.88% | 1.588 |
| oracle_compact_projection | 37.39% | 0.1419 | 1.555 | 71.38% | 1.474 |

Examples show the risk: strings like `Based plans, Home Energy Rater uses energy efficiency software package perform energy analysis home's design.` and `trick MySQL always consider "$sEmail" single value, matter quotes semi-colons insert.` are content-dense but less grammatical. Therefore coverage alone should not authorize H100 training. A short screen, if ever run, must allow the possibility that source-only skeletons damage broad competence through unnatural text distribution.

## Exact 10M pool scaffold

Scripts/outputs:

- `scripts/build_skeleton_reinvest_pools.py`
- `data/skeleton_reinvest_pool_scaffold/skeleton_reinvest_pool_manifest.{json,md}`
- Exact 10M pool JSONLs under `data/skeleton_reinvest_pool_scaffold/`

The scaffold preserves the density cleanqwen overlay medium riskhard layout: 3,006 changed rows, 423,520 changed-block words, 61,734 common filler rows, 9,576,480 filler words, 64,740 rows total, and exactly 10,000,000 words in every variant. The 9-word top-up row with empty `pair_ids` is preserved from the original pool. A check showed the reconstructed `compact` scaffold is text/word/example-id identical to the original density cleanqwen overlay medium riskhard compact pool; only the changed-row `source` labels differ, explaining the different file SHA.

Pool SHAs:

- `compact`: `7c3792fdf4b84b7ab0ea06c6f0c2430a290cd4b3ae17424186da4b67f0be9404`
- `prefix_repeat`: `1028409400d6ee0490b076a01a3c8ee5ef4dd3f9d20534ff70c081c4eeeae387`
- `content_spread`: `54ce8ec0e2e2838ade3d9ef5b30a152cfce63d7ec9ec377f1c036353b6ce4fce`
- `scored_source_skeleton`: `a0539d78c6d5ed33d83b7e2e423dfd07d0d1d909ed9ca418ae989abd679c6101`

## Dry screen plan only

Scripts/outputs:

- `scripts/plan_skeleton_mlm_screen.py`
- `data/skeleton_mlm_screen_plan/skeleton_mlm_screen_plan.{json,md}`

No command was launched. The dry plan specifies a possible 40M short MLM mechanism screen with checkpoints at 20M and 40M for compact, prefix repeat, content_spread, and scored_source_skeleton. It is intentionally not authorized while the DeBERTa common-grid tasks are still running and triangle scores are unavailable.

Expensive-work decision logic, if later reviewed and accepted:

- Continue skeleton route only if a source-only skeleton beats prefix_repeat across cheap6/cheap5 and relation/state without volatile-column-only movement.
- If compact beats source-only skeleton broadly, natural compressed sentence / semantic recoding style remains load-bearing.
- If source-only skeleton harms broad families despite coverage, close extractive skeleton as too distribution-shifted.
- Do not use the screen as endpoint work; it is only a mechanism dissection before any mature exposure.

## Scientific state after source wide skeleton recurrence integrated

The strongest current mechanism statement is now:

**A compact second view works as same-window source-wide skeleton recurrence under bidirectional MLM: it repeats a content-dense subset of source-wide keys, including late source content absent from prefix repetition, while freeing words for source diversity.**

This statement is stronger than the earlier generic paraphrase view and more compatible with accumulated evidence:

- It explains why compactness was load-bearing and spatial preservation failed.
- It explains why pair atomicity and same-window visibility were load-bearing.
- It explains why first-N exact repetition underperformed compact: it repeats many prefix/function words and misses late source content.
- It does not contradict the GPT causal negative, because that coordinate lacks reciprocal access and had volatile-column-only compact gains.
- It does not explain the 82M-to-100M decline, because the BPE tail-lift pattern is similar at both checkpoints.

Whether skeleton dissection is the best post-grid experiment should be evaluated after integrating the available DeBERTa common-grid results and triangle scores.

## Immediate next work

1. When the two common-grid evaluations complete, verify them exactly as deberta grid tooling and topology packing confound specified: integrity check first, no interpretation before integrity passes.
2. When a GPU is genuinely free, launch the remaining scale1.75 seed43122 selected grid scoring if it is still unrun, then use `scripts/interpret_deberta_common_grid.py` only after all three trajectories exist.
3. Integrate the triangle official-compatible scores.
4. Evaluate the source-wide skeleton route against the accumulated evidence before any skeleton H100 screen. The exact scaffold exists, but source-only skeletons are telegraphic; their high coverage is not enough.
