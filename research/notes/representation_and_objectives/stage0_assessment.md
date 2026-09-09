# stage0 assessment: Stage 0 assessment — rewrite-marginal factorization corpora

## Status: READY FOR STAGE 1 SHORT SCREEN

The v2 marginal corpora pass the Stage 0 audit with good matching on all critical variables.
Content-fraction confound is fully resolved; residual BPE/word gap is modest and inherent.

## What was built

Three candidate arms from the 12,155 compact pairs:

- **Arm C (compact rewrite)**: The original compact rewrite text. Abstractive, fluent, 16.6% source-absent words, BPE/word 1.575.
- **Arm S (semantic extractive)**: Order-preserving source words maximizing content overlap with the compact rewrite, with gap-filling for fluency. Content-matched to rewrite (83% overlap). BPE/word 1.521.
- **Arm G (content-balanced generic extractive)**: Order-preserving source words at evenly-spaced positions, NOT guided by rewrite content, but exactly matching S's content/function word counts per row. Compact overlap 0.584. BPE/word 1.476.

All arms: 161,708 total words across 12,155 rows, perfect per-row word count matching.

## Matching quality

### Critical variables (must match for clean S-G comparison)

| Variable | S | G | S-G diff | Assessment |
|----------|---|---|----------|------------|
| Words per row | 13.30 | 13.30 | 0.000 | ✅ Perfect |
| Content frac | 0.617 | 0.617 | 0.000 | ✅ Perfect (per-row) |
| Content/func count match | — | — | 100% exact | ✅ All 12,155 pairs |
| BPE/word | 1.521 | 1.476 | +0.045 | ⚠️ Residual (3.1%) |
| Total BPE | 244,918 | 237,644 | +7,274 | ⚠️ 3.1% |
| Position spread | 0.925 | 0.977 | −0.052 | ✅ Acceptable |
| Position mean | 0.464 | 0.441 | +0.023 | ✅ Acceptable |
| Span count | 4.0 | 5.9 | −1.9 | ⚠️ G more fragmented |

### Semantic separation (must differ for useful S-G comparison)

| Variable | S | G | Interpretation |
|----------|---|---|---------------|
| Compact overlap | 0.830 | 0.584 | ✅ S captures 83% of rewrite content vs G's 58% |
| S-G position overlap | 65.9% mean, Jaccard 0.505 | — | ✅ ~34% of words differ per arm |
| Identical pairs | 0.5% | — | ✅ Nearly all pairs have different selections |
| Symmetric diff | 9.0 positions mean | — | ✅ Substantial word-level variation |

### C-S separation (for reformulation residual)

| Variable | C | S | C-S diff | Notes |
|----------|---|---|----------|-------|
| Source-absent mass | 16.6% | 0.0% | +16.6% | Defines the abstractive/extractive split |
| BPE/word | 1.575 | 1.521 | +0.054 | Compact rewrites slightly more BPE-heavy |
| Content frac | 0.677 | 0.617 | +0.060 | Compact rewrites are more content-dense |
| Span count | N/A (fluent) | 4.0 | — | C is fluent; S is fragmentary |

## Assessment

### S-G is clean enough for semantic selection

The only remaining S-G confound is BPE/word (+0.045). This residual arises because rewrite-relevant content words (S) tend to be morphologically rarer or more specific than generic content words (G) within the same content/function classes. This is inherent in the semantic selection variable — it cannot be removed without destroying what we're testing.

**At 3.1% total BPE difference and 0% content-fraction difference, the S-G comparison is the cleanest available test of semantic selection quality in source-word extractive compression.**

Under WWM 0.15, the BPE difference translates to roughly 1,091 extra masked BPE targets per epoch for S (0.045 × 161,708 × 0.15 ≈ 1,091 out of ~36k total). This is a modest gradient-signal asymmetry that should be noted in interpretation but is unlikely to dominate downstream competence differences.

### C-S tests reformulation as a package

C and S differ fundamentally: C is fluent abstractive text with 16.6% new words; S is fragmentary extractive text with 0% new words. Any C > S effect cannot separate:
- Fluency/coherence of the training sequence
- Abstractive vocabulary (new function words, reformulated content)
- Slightly higher BPE burden per word

The protocol correctly predicts this: C-S measures "fluent reformulated compression as a package."

### S-G coherence asymmetry

G has mean 5.9 spans vs S's 4.0, meaning G is more fragmented. This arises because content-balanced even spacing creates more isolated word selections. Both are clearly less coherent than C. If S outperforms G, part of the effect could be that S's word clusters form more locally coherent phrases.

## Go/no-go for Stage 1

**GO.** The v2 corpora cleanly separate semantic selection (S vs G) from generic coverage at matched content/function ratio. The C-S-G comparison can identify:
1. Whether compact views win by semantic content selection (S > G with matched coverage)
2. Whether reformulation adds value beyond selection (C > S)
3. Whether generic coverage alone suffices (G ≈ S ≈ C pattern)

## Training stream construction needed

To run Stage 1, the arms need to be embedded in 10M-word training streams using the same geometry as the historical triangle:
- Each stream: source+side pair rows + shared filler rows = exactly 10,000,000 words/epoch
- Same filler as the triangle (from `compact_view_reinvest` 10M pool minus pair rows)
- Same legal16k tokenizer
- 10 epochs → 100M words; Stage 1 uses first 20M–40M words only

Since all three arms have identical pair word counts (423,511), the same filler fills all streams to exactly 10M words. This gives a matched training geometry.

## Files

- Corpora: `data/marginal_corpora_v2/marginal_corpora_csg_v2.jsonl`
- SHA: `d206742ab59154f2f41787a4c005cb19fd33b420b69c755b65cddcd3d7ca63f6`
- Audit: `data/marginal_corpora_v2/marginal_corpora_v2_audit.json`
- Script: `scripts/build_marginal_corpora_v2.py`
- v1 (superseded): `data/marginal_corpora/` (naive G without content balancing)
