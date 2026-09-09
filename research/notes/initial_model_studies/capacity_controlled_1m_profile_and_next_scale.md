# capacity controlled 1m profile and next scale — Capacity-controlled 1M Profile: Sparse Signal Does Not Survive Dense Scale


## Purpose

three mechanism 1m profile comparison showed a narrow sparse advantage on BLiMP Supplement, but sparse4x256 and dense6x256 were not matched in parameter count, layer count, or output-head tying. This step added and profiled closer dense controls before any exposure scale-up.

## New dense controls

Two additional dense GPT2LMHeadModel controls were trained for exactly 1,000,000 official whitespace-word exposure with the same corpus revision, tokenizer, sequence length, batch size, word counting, checkpoint naming, and official profile suite.

| run | intended role | actual params | layers | width | heads | LM head | status |
|---|---|---:|---:|---:|---:|---|---|
| `babylm_compare_dense4x256_untied_1M` | accidental V0 rerun; name is misleading | 7,419,392 | 4 | 256 | 4 | tied | trained + verified |
| `babylm_compare_dense5x288_1M` | intermediate dense scale | 9,788,256 | 5 | 288 | 8 | tied | trained + verified + profiled |
| `babylm_compare_dense6x384_1M` | closer capacity control for sparse4x256 | 17,037,312 | 6 | 384 | 6 | tied | trained + verified + profiled |

The `dense4x256_untied` run name is incorrect: `dense_causal` uses GPT2LMHeadModel, whose output head is tied. Treat it as another 4-layer V0-like dense repeat, not an untied dense control.

## Combined 1M official-compatible profile

JSON evidence:

- previous three mechanisms: `experiments/archive/initial_model_studies/data/profile_all3_1m.json`
- new dense controls: `experiments/archive/initial_model_studies/data/profile_densecontrols_1m.json`
- combined table: `experiments/archive/initial_model_studies/data/profile_capacity_controlled_1m.json`

| alias | params | BLiMP | Supplement | EWoK | Entity | COMPS | Reading eye | Reading SPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense6x256 | 8,998,912 | 55.23 | 47.60 | 50.55 | 17.45 | 50.07 | **9.67** | 2.54 |
| sparse4x256 | 17,924,624 | **55.33** | 50.00 | 47.09 | 16.20 | 49.83 | 8.79 | 2.52 |
| morphside4x256 | 11,761,408 | 53.41 | 46.40 | 51.55 | **18.25** | 49.90 | 8.94 | 2.56 |
| dense5x288 | 9,788,256 | 54.13 | 46.00 | 50.45 | 15.62 | **50.09** | 9.34 | **2.65** |
| dense6x384 | 17,037,312 | 54.13 | **51.20** | **51.82** | 15.60 | 49.77 | 8.50 | 2.35 |

## Scientific reading

1. Sparse routing is no longer the leading route on current 1M evidence. Its three mechanism 1m profile comparison Supplement advantage over dense6x256 (50.00 vs 47.60) disappears against the closer dense6x384 capacity control, which scores **51.20** on Supplement and **51.82** on EWoK. Sparse remains slightly better on BLiMP than dense6x384 (55.33 vs 54.13), but that is not enough to justify sparse scale-up given weaker EWoK, entity, COMPS, and reading.

2. Dense scale is a stronger immediate route than sparse routing at 1M. The dense6x384 capacity control gives the best combined NLP proxy among these cheap profiles: Supplement and EWoK both lead, while BLiMP remains competitive. Reading and entity remain weak, so dense scale alone does not solve the human-like/developmental side.

3. Morph-side has a small entity/EWoK signal but loses BLiMP/Supplement. It deserves later redesign if representation becomes the bottleneck, but the present fixed hashed side channel is not yet a route to Overall SOTA.

4. Entity tracking remains severely weak for every candidate (15.6–18.25), and Reading is mostly flat. This suggests the current 1M regime mostly discriminates early NLP columns; it does not yet reveal a strong mechanism for human-like/AoA improvements.

5. Because the official 2026 Overall is a mean over nine columns, the near-term scale-up should not chase a single task. Dense6x384 is the best candidate to test whether more exposure lifts multiple NLP columns while preserving Reading; sparse should not be scaled before it has a cleaner advantage.

## Next execution direction

Patch the comparison trainer so 10M/100M exploratory runs save multiple official-style checkpoints (`chck_1M`, `chck_2M`, ..., `chck_10M`, then later `chck_20M`... as needed). The current one-threshold `chck_1M` behavior is insufficient for AoA and checkpoint-trajectory evaluation.

Then run a **10M exposure curve for dense6x384** first, not sparse, with `checkpoint_words=1000000`, and profile at least the final `chck_10M` plus ideally selected intermediate checkpoints. If dense6x384 improves broadly, use it as the baseline for any later objective/data representation intervention. If it fails to improve entity/reading despite stronger NLP, return to mechanism design focused specifically on state/trajectory or data-order/AoA rather than sparse routing.
