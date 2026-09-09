# subdose ladder state: Sub-1x dose ladder construction and state

## What was built

A sub-1x dose ladder in MAX rowholdout geometry, testing the ONSET of the
admission effect below ρ=0.042. All arms share the same 65,313-row 10M pool
structure and the already-trained MAX-geometry DeBERTa clean as ρ=0 reference.

### Row classification (critical finding)

- Pure 1x rows: 3,004 (423,405 words) — the materializer packed 1x pairs first
- Mixed rows: 1 (154 words, 106 1x + 48 increment) — negligible
- Pure increment rows: 4,918

Row-level selection is therefore essentially pair-level clean. No mixing needed.

### Dose ladder

| Dose | Active Rows | 1x Pairs | Inc Pairs | PW | ρ | Docs | Content Types |
|------|-------------|----------|-----------|------|---------|------|---------------|
| clean | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| quarter_1x | 758 | 2,861 | 0 | 105,962 | 0.01060 | 2,861 | 11,202 |
| half_1x | 1,513 | 5,856 | 0 | 211,853 | 0.02119 | 4,247 | 16,264 |
| full_1x | 3,005 | 12,155 | 1 | 423,559 | 0.04236 | 4,529 | 22,734 |
| dose1p82 | (existing) | — | — | 771,199 | 0.07712 | 5,089 | 36,016 |
| MAX | (existing) | — | — | 1,118,587 | 0.11187 | 5,261 | 43,540 |

### Coverage saturation (strongest file-only finding)

Document coverage grows sublinearly and saturates early:
- 0 → 2,861 docs in the first 106k words (quarter)
- +1,386 docs in the next 106k words (half)
- +282 docs in the next 212k words (full 1x)
- +560 docs in the next 348k words (dose1p82)
- +172 docs in the next 348k words (MAX)

By quarter_1x, 63% of all unique docs are already covered.
By full_1x, 86% of all unique docs are covered.

Content types grow more steadily but still diminish:
- 10,570 types per 100k words at quarter
- 7,672 at half
- 5,364 at full 1x
- 4,672 at dose1p82
- 3,891 at MAX

The marginal content-type curve (from earlier analysis) showed 6,354 → 2,620 → 2,166 per
100k words at 1x/1.82x/MAX. The sub-dose data extends this: ~10,570 at quarter_1x.

## Files

- Pools: `data/subdose_ladder_maxgeom_pools/`
  - `subdose_quarter_1x_view_{10M,100M}.jsonl`
  - `subdose_half_1x_view_{10M,100M}.jsonl`
  - `subdose_full_1x_view_{10M,100M}.jsonl`
- Coverage CSV: `subdose_cumulative_coverage.csv` (3,005 rows)
- Metadata: `subdose_ladder_metadata.json`
- Summary: `subdose_ladder_summary.md`
- Training script: `scripts/train_subdose_deberta.py`

## What remains

1. Dry-run preflight for all three training arms (in progress)
2. Launch three DeBERTa view-arm trainings (quarter, half, full_1x) on H100s
   - Each ~45 min; two can run in parallel
3. Score all trained arms against MAX-geometry clean (stable families)
4. Read common-window V-C at each sub-dose point
5. Compare the score onset/knee to the coverage knee
6. If they coincide: the principle is coverage-limited admission saturation
7. If they don't: domain composition or some other property carries the effect

## Pending Evaluations

- Breadth and RoBERTa evaluation remainder.
- First MAX-geometry DeBERTa clean-control scoring.
