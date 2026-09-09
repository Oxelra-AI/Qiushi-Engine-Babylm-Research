# compact route synthesis Research Synthesis: Compact-View Route Status and Experimental Design

## Current Evidence State

### Three parallel streams converging

1. **Forced full generation (complete)**: 26,567 pairs via Qwen3.5-9B forced compression.
   - Auto-low-risk: 13,370 pairs saving 114,462 words
   - BUT: Forced prompt has ~41% faithful precision from independent_review calibration
   - Negation preservation only 0.4584, modality 0.3245, causal 0.4149
   - This is a damaged candidate pool, not admitted training data

2. **Selective full generation (in progress at the time)**: Same 26,567 pairs with KEEP_CURRENT-enabled prompt.
   - Pilot (512 pairs): 219 auto-low-risk saving 1,432 words (compression 0.88)
   - Expected full-scale: ~11,400 auto-low-risk saving ~75,000 words
   - Higher precision because the prompt allows no change when shortening is unsafe
   - Still needs semantic review before admission

3. **Inherited rewrite errors (new finding)**: Heuristic scan of all 37,594 pairs.
   - 5,647 flagged (15%) but most are false positives from legitimate rephrasing
   - ~5-6 real number errors out of 22 number_changed flags (~25% precision)
   - Confirmed: rw_009884 (negation reversal), rw_001604 (1000 days → 1 year)
   - Estimated real error rate: 0.5-1% (200-400 genuinely contradictory rewrites)
   - DISTINCT intervention from compaction: correspondence REPAIR

### Reference tail evaluation (in progress at the time)

- Fast screen: equal_valid_mean 44.576 vs coherent86 44.564 (+0.012)
- Preliminary official zero-shot: BLiMP 68.48, Supplement 63.56, EWoK 49.93
  vs coherent86 official: BLiMP 68.51, Supplement 63.64, EWoK 50.02
- All slightly below → fast-screen advantage unlikely to translate to official improvement
- Entity: 26.94 vs 27.78 (worse)
- Still need remaining columns (Entity, COMPS, GlobalPIQA, Reading, SuperGLUE, AoA)

## Experimental Design for Learner Comparison

### Three distinct interventions (must be kept separate)

1. **Faithful shortening**: Remove redundant expression while preserving all propositions.
   - Source: auto-low-risk selective generations verified by independent review
   - Control: same source IDs with original inherited rewrites
   - Tests: Does removing redundant words hurt or help per-word efficiency?

2. **Correspondence repair**: Replace contradictory inherited rewrites with source-faithful text.
   - Source: pairs where inherited rewrite demonstrably contradicts source
   - Ideally length-matched to isolate the repair from shortening
   - Control: same source IDs with original (wrong) inherited rewrites
   - Tests: Does fixing false teaching signal improve learning?

3. **Partial-view allocation**: Intentionally reduce second view to fewer source-supported facts.
   - Different from faithful shortening: deliberately omits some facts
   - The source still contains the omitted facts
   - Control: same source IDs with full inherited rewrites
   - Tests: Is selective emphasis better than comprehensive second coverage?

### Scale concern

At ~75,000 saved words from selective generation in a 14M-word tail:
- Changed rows: ~1.6% of tail rows
- Changed words: ~0.5% of total
- This is inherently a small-effect experiment

Possible mitigations:
- Concentrate on metrics most affected by Qwen-pair content (Entity, correspondence probes)
- Use multiple seeds and statistical comparison
- Focus on per-pair measurements rather than aggregate scores
- Compare the compact-reinvest arm (75K new source words) against reference

### Key questions for the comparison

1. Does compacted + reinvested experience produce better Entity/correspondence/reading scores than uncompacted experience?
2. If the effect is positive, does it come from the shortening (removing redundancy) or the reinvestment (new source)?
3. Can we detect correspondence repair effects separately from shortening effects?

## Bridge Cheap7 Suite (complete)

| Arm | Update | equal_valid_mean | Entity |
|-----|--------|-----------------|--------|
| coherent86 | parent | 44.564 | 27.78 |
| reference_tail | 354 | 44.576 | 26.94 |
| ordinary_wwm | 354 | 44.272 | 26.40 |
| answer_alloc | 150 | 44.215 | 25.71 |
| answer_alloc | 200 | 44.199 | 25.61 |
| answer_alloc | 354 | 44.111 | 25.33 |

The bridge policies all degraded relative to both coherent86 and the reference tail.
Answer allocation progressively worsened Entity with more updates.
The relation route needs optimizer/objective redesign, not more exposure.

## Next Steps (ordered by value)

1. **Wait for official eval completion** → determines whether reference tail is a practical finding
2. **Wait for selective full generation** → determines scale of faithful admitted set
3. **Build materialized tails from selective admitted set** → materializer ready (earlier analysis)
4. **Train compact comparison arms** → trainer ready (compact route synthesis)
5. **If selective yield is too small**: consider combining selective + correspondence repair
6. **If reference tail official eval is close**: investigate which specific items differ

## Files

| File | Purpose |
|------|---------|
| `scripts/inherited_rewrite_error_scanner.py` | Scans inherited rewrites for contradictions vs source |
| `data/inherited_rewrite_errors/error_scan_summary.json` | Scan results: 5,647 flagged, ~25% precision for numbers |
| `data/inherited_rewrite_errors/flagged_pairs.jsonl` | All flagged pairs with issue details |
| `scripts/selective_full_generation_pipeline.py` | Full pipeline: prepare → generate → enrich → screen |
| `scripts/compact_comparison_trainer.py` | Train compact comparison arms using bridge trainer |
| `data/full_generation_postprocess/` | Forced generation results (candidate pool only) |
| `data/bridge_cheap7_suite/bridge_cheap7_suite_summary.json` | Complete bridge broad-screen results |
