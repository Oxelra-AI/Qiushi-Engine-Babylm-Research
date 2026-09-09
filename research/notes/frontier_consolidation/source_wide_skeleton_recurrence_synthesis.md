# source wide skeleton recurrence integrated source-wide skeleton recurrence evidence

## What was checked

The recent reciprocal/topology line was weakened by reciprocal multiview mechanism and scaffold and deberta grid tooling and topology packing confound: DeBERTa non-copy reciprocal lift was tiny or negative, causal lift was one-way and also strong for repeat controls, and the topology 2x2 scaffold and deberta pending state 2x2 packing would split roughly 19-20% of pair rows across 256-token chunks. No model was trained or evaluated for this analysis. It asks whether the `copy` part of the compact-view effect is scientifically meaningful rather than just a nuisance.

The concrete hypothesis tested by CPU/file measurement is:

> A compact view may act as a short, natural, source-wide content skeleton: it repeats selected content keys from across the whole source, including late source positions that the matched first-N repeat control does not expose, while freeing word budget for source diversity. Under bidirectional MLM, same-window source plus skeleton can create dense cross-view reconstruction pressure. This differs from both ordinary prefix repetition and source-free paraphrase.

## Files produced

- `scripts/compact_skeleton_recurrence_measure.py`
- `data/compact_skeleton_recurrence/compact_skeleton_recurrence.{json,md}`
- `data/compact_skeleton_recurrence/per_pair_skeleton_metrics.csv`
- `data/compact_skeleton_recurrence/high_tail_skeleton_examples.jsonl`
- `figures/compact_skeleton_source_position_coverage.png`
- `scripts/lift_tail_attribution.py` (first whitespace-word attribution; useful but superseded by BPE attribution)
- `data/lift_tail_attribution_chck82/lift_tail_attribution.{json,md}`
- `scripts/bpe_copy_zone_attribution.py`
- `data/bpe_copy_zone_attribution_chck82/bpe_copy_zone_attribution.{json,md}`
- `data/bpe_copy_zone_attribution_chck100/bpe_copy_zone_attribution.{json,md}`

## Text-geometry result: compact views re-expose tail content

Input pair file: `data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`, SHA `d2a3110c110e216180b632ce7a181acdf348b804ea19f172e16b5afeb0d8c9fc`, 12,155 compact pairs. The matched repeat view is the first N source words, where N equals the compact rewrite word count.

Across all compact pairs:

- Mean source length: 21.54 words; mean compact view length: 13.30 words.
- Compact view covers 66.99% of source content-word positions, versus 59.87% for the first-N repeat prefix; mean compact-repeat source-content coverage advantage +7.12 points.
- Compact view recovers tail source content at 70.26% mean coverage; 97.66% of pairs have at least one source-tail content word recovered by the compact view.
- Compact views are denser in content words than first-N repeats: 65.27% content words versus 49.43%, a +15.84 point content-density advantage.
- Source-position decile coverage shows the structural difference: in deciles 6/7/8/9, compact covers 63.60/65.56/70.62/76.18% of content positions, whereas first-N repeat covers 31.69/7.46/0.11/0.00%.

The reciprocal multiview mechanism and scaffold/179 selected subset used in the reciprocal/topology scaffolds is almost identical: 6,071 pairs; compact coverage 66.83%, repeat-prefix coverage 59.68%, compact tail-content coverage 69.99%, and 97.58% of pairs with tail recovery. Therefore the source-wide skeleton property is not an artifact of a different pair subset.

## Tokenizer-level model-lift attribution

I re-annotated the existing reciprocal multiview mechanism and scaffold DeBERTa reciprocal-lift records at the actual BPE/tokenizer level using source token offsets from the checkpoint tokenizer. No model was loaded and no new score was computed.

For `chck_82M` compact rewrite-side targets (`pair_type=compact`, `target_segment=other`, 512 target records):

- Same BPE id appears only in source tail beyond the matched repeat prefix: 160 records, mean lift 6.084254, positive fraction 0.981250.
- Same BPE id appears only in prefix: 222 records, mean lift 5.816492, positive fraction 0.950450.
- Same BPE id appears in both prefix and tail: 33 records, mean lift 2.674222.
- BPE id not in source: 97 records, mean lift only 0.099026, positive fraction 0.443299.

For content-like compact rewrite-side targets at `chck_82M`:

- Tail-only BPE id: 91 records, mean lift 6.330639, positive fraction 0.967033.
- Prefix-only BPE id: 104 records, mean lift 6.731483.
- Not in source: 46 records, mean lift 0.435714.

For `chck_100M`, the pattern persists with slightly lower tail-only lift:

- Compact rewrite-side tail-only same-id: 160 records, mean lift 6.029999.
- Content tail-only same-id: 91 records, mean lift 6.270454.
- Not-in-source BPE id: 97 records, mean lift 0.079276.

This changes the interpretation of reciprocal multiview mechanism and scaffold. The very large copied-token lift is not just trivial prefix repetition. A substantial part of it is lift for tokens whose matching source BPE occurrences lie beyond the first-N repeat prefix. The compact view makes late-source content keys available in the same training row while the matched repeat control does not.

## Scientific interpretation

The best current reading is not `reciprocal semantic paraphrase` in the abstract. The more precise object is:

**source-wide compact skeleton recurrence**: a short, natural, content-dense view reintroduces selected source-wide content keys from across the whole source in the same window; because it is shorter than the source, the saved word budget is reinvested in more distinct official-legal source text. Under MLM, bidirectional masking lets either side use the other side as a reconstruction support. Under causal GPT, the tested one-way packet cannot use future skeleton information for earlier source tokens and the architecture-transfer advantage collapsed outside volatile columns.

This hypothesis preserves and sharpens the strongest old evidence:

- It explains why compactness was load-bearing and spatial-preservation failed: the useful object is a short content-dense source-wide skeleton, not a long spatially faithful paraphrase.
- It explains why pair atomicity/same-window visibility was load-bearing: source and skeleton need to be in one reconstruction field.
- It explains why exact first-N repetition was weaker: it spends many words re-exposing only the source prefix and lower-density function-word material, leaving late source content unseen in the second view.
- It stays compatible with the causal GPT negative: a one-way next-token objective does not give reciprocal access to the source-wide skeleton, and repeat controls can still produce strong second-segment copy lift.

## Caveats and current limits

- This is text geometry and re-annotation of an existing 128-pair model probe, not a new official-compatible training result.
- BPE tail attribution shows source-wide copy/reconstruction opportunity; it does not by itself prove broad capability transfer or endpoint improvement.
- Non-source/non-copy compact lift remains tiny in the reciprocal multiview mechanism and scaffold probe. The source-wide skeleton idea should not be inflated into a general paraphrase-invariance result.
- `chck_82M` and `chck_100M` have similar BPE tail-copy lift, so this measurement does not explain the 82M→100M official relation/state decline.
- A new real-training test must avoid the known failure families: benchmark-shaped targets, edit-state/source-conditioned auxiliaries, coherence margin, alpha/anchor/retention tuning, high-IG masking, graph packet generation, and causal topology packing confounds.

## Concrete discriminating experiment suggested by the evidence

If the pending DeBERTa common-grid results and triangle scores leave the compact-view data mechanism as the best open route, the next scientifically meaningful test is not another reciprocal-topology run. It is a **skeleton-versus-natural-compact dissection** under a bidirectional MLM coordinate:

1. Build an extractive source-wide skeleton arm: replace each generated compact rewrite with a deterministic same-length source-derived skeleton that selects content words from across the source, preserving source order and matching the compact view word count and pair placement. This isolates source-wide coverage without external generated paraphrase style.
2. Compare against the existing compact view and the exact first-N repeat arm under matched filler, pair positions, legal words, tokenizer coordinate, model geometry, seed, and checkpoint/evaluation grid.
3. Interpret results at low cost first (e.g. 20M/40M cheap7 family profile) before any mature exposure. Continue only if source-wide skeleton beats prefix repeat across broad families without being just BLiMP/COMPS/GlobalPIQA redistribution or distribution damage.
4. If extractive skeleton matches compact, the general principle becomes algorithmic source-wide skeletal recoding rather than generated paraphrase. If compact beats extractive skeleton broadly, the natural compressed sentence / semantic recoding component is load-bearing. If both fail versus prefix repeat, the skeleton reading is not trainable in the current coordinate.

Legal/tokenizer note: a route meant for final compliance should not use a tokenizer trained on text outside the candidate 10M pool. For a cheap mechanism screen, keeping the spatial repair route status tokenizer may isolate data-content effects in the known DeBERTa coordinate, but a final legal successor would need a tokenizer trained only on the candidate pool or a truly common legal tokenizer across all arms. This must be decided explicitly before spending H100 time.
