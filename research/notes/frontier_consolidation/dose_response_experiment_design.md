# dose response experiment design: Dose-Response Experiment Design

## Scientific Question

Does the fraction of the 10M word budget restructured into source+compact-view
packets produce a dose-dependent competence change? Two possible outcomes, both
scientifically strong:

1. **Compact-minus-repeat grows with dose** → structured semantic compression IS
   the mechanism; the dose-response curve with its optimum and decline quantifies
   the principle with boundaries.

2. **Treatment-vs-clean grows while compact-minus-repeat stays flat** → redundancy
   reduction (freeing budget for novel source diversity) carries the gain; the
   re-expression component is incidental.

## Dose Arms

| Arm | Dose | Changed block | Status |
|-----|------|---------------|--------|
| 0x clean | 0.0 | 0 words | Trained: `complianttok_cleanqwen_seed43022_80M` (80M checkpoints) |
| 1x view | 1.0 | 423,520 words (4.24%) | Trained: `complianttok_reinvest_seed43022_r2` (dense 1M checkpoints) |
| 1x repeat | 1.0 | 423,520 words (4.24%) | Trained: `full_p2c_c2p_abs_repeat_deberta100M_seed43022` (10M checkpoints) |
| MAX view | ~2.8x | ~1.2M words (~12%) | **NEW: build from combined pool** |
| MAX repeat | ~2.8x | ~1.2M words (~12%) | **NEW: build from combined pool** |

## Primary Comparison Points

- **80M**: All arms compared here (clean reference exists at 80M)
- **100M**: 1x view, 1x repeat, MAX view, MAX repeat (no clean reference past 80M)

## Stable Selected Families

BLiMP, Supplement, EWoK, Entity, COMPS, Reading (skip GlobalPIQA, SuperGLUE, AoA)

## Key Metrics

- Treatment-vs-clean at each dose: `dose_N_view - dose_0_clean`
- Compact-vs-repeat at each dose: `dose_N_view - dose_N_repeat`
- Dose-response slope and interior optimum detection
- All measured against seed spread from earlier analysis ladder (seed43022 vs seed43122)

## Recipe (identical across all new arms)

- Architecture: full DeBERTa-v2, p2c+c2p, 8×480, 34,467,424 params
- Tokenizer: `compliant16k_reinvest10M` (spatial repair route status)
- WWM p=0.15, AdamW lr=0.001, warmup 0.06, weight decay 0.01
- Batch 256, seq 256, 100M exposure, 10M checkpoint cadence
- Seed triplet: 43/43022/43023 (same as existing 1x arms)

## Construction Pipeline

1. ✅ Expansion prompts: 18,571 from WAR FineWeb pool
2. 🔄 Generation: GPU0 (9,285), GPU1 (9,286) via Qwen3.5-9B
3. ⏳ Analyze & merge: `analyze_expansion_and_merge.py`
4. ⏳ Materialize pools: `dose_response_materializer.py --dose MAX`
5. ⏳ Train MAX view + MAX repeat on both GPUs simultaneously
6. ⏳ Evaluate all arms at 80M/100M on stable selected families

## Expected Pair Inventory (post-generation)

- Existing accepted: 18,682 pairs, 645,233 pair words
- Expansion: ~16,000 new accepted, ~550,000 pair words (estimated at 87% acceptance)
- Combined: ~34,682 pairs, ~1,195,000 pair words
- Max dose: ~1,195,000 / 423,520 ≈ **2.82x**

## Decision Rule

If dose-response shows monotone positive treatment-vs-clean trend AND growing
compact-minus-repeat above seed spread:
→ Structured semantic compression is quantifiably beneficial with measurable optimum

If treatment-vs-clean grows but compact-minus-repeat stays flat:
→ Source diversity from compression savings, not re-expression quality, drives gain

If treatment-vs-clean saturates or declines relative to noise:
→ The 1x effect size is already at or near the mechanism's capacity in this coordinate
