# bridge atlas geometry result: Coherent88 closure and bridge geometry validation

## Coherent88 alpha sweep — closed

From bridge route recovered from sourcecopy error evaluations:
- **α=0.5**: cheap7 44.126 (-0.056 vs coherent86 α=0.75), stable deltas vs chck84 negative
  (cheap6_no_GP -0.078, EWoK+Entity -0.260)
- **α=0.75**: cheap7 44.192 (+0.011 vs coherent86 α=0.75), but gain is GlobalPIQA-only
  (cheap6_no_GP -0.082, EWoK+Entity -0.315, GlobalPIQA +0.970)

**Decision**: coherent88 does not improve the stable-family endpoint. The chck_84M
backbone provides no meaningfully different coherent-replay signal than chck_82M.
Coherent88 endpoint work is closed; coherent86 α=0.75 remains the strongest
secured local function (projected Overall 42.121).

## Bridge geometry validation — decisive positive result

### Method correction
bridge atlas geometry result identified a critical methodology mismatch: the initial content-only greedy
alignment measured a different quantity than the extractive surface and readout repair atlas. The atlas uses monotone
alignment over ALL normalized words with gap1/skip between consecutive aligned positions
in view order. The corrected atlas-compatible measurement resolves the discrepancy.

### Key result: transformation-like bridge ≈ natural compact on structural geometry

Using atlas-compatible monotone alignment on the 103 retained mechanism evidence synthesis bridge candidates:

| group | mean gap1 | mean skip | pooled gap1 | pooled skip | absent | compress |
|---|---:|---:|---:|---:|---:|---:|
| **bridge transform** (n=55) | **0.764** | **0.236** | **0.795** | **0.205** | 0.039 | 0.682 |
| **matched natcomp** (n=55) | **0.762** | **0.238** | **0.764** | **0.236** | 0.168 | 0.616 |
| bridge extract (n=48) | 0.918 | 0.082 | 0.936 | 0.064 | 0.003 | 0.828 |
| atlas compact (3005 pairs) | — | — | 0.778 | 0.223 | 0.173 | — |
| atlas ext_balanced | — | — | 0.528 | 0.472 | 0.000 | — |
| atlas ext_wide | — | — | 0.647 | 0.354 | 0.000 | — |

**Gap1/skip delta between bridge-transform and matched natural compact: 0.002.**
This is geometric near-identity on the main structural-contiguity dimension.

### Proximity to atlas benchmarks (pooled gap1)
- Bridge transform → compact: distance **0.017** (closest)
- Bridge transform → ext_wide: distance 0.149
- Bridge transform → ext_balanced: distance **0.267** (furthest)

### Scientific interpretation
1. Transformation-like bridge candidates occupy compact-like structural geometry,
   clearly distinct from the failed extractive arms
2. The ONLY major geometric difference vs natural compact is source-absent content
   (0.039 vs 0.168) — novel vocabulary
3. A matched training comparison would cleanly isolate whether novel vocabulary
   (source-absent MLM prediction targets) is necessary for the compact data-
   efficiency mechanism, while controlling for structural geometry
4. Extraction-like bridge candidates are NOT useful for this test (gap1 0.936,
   essentially contiguous source substrings)

### Scale estimate
22% of sampled pairs (55/250) yielded transformation-like outputs → ~2,674 from
full 12,155 pool, enough for ~89% of the 3,005 compact block. Improved prompts
using Tier-A examples as demonstrations could increase this yield.

## Entity-Event-State Binding
The complementary route tests compositional entity-event-state binding. The DeBERTa baseline
EEBF_composite=0.510 (chance) confirms the entity-binding failure. The plan is
frozen-DeBERTa + memory module → joint training → cross-architecture. This is
complementary to our data-geometry decomposition — this finding could explain
WHY compact views help Entity/EWoK specifically (they provide more binding
practice opportunities through re-expressed entity-event-state propositions).
