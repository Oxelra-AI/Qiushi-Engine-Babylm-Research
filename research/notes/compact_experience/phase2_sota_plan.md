# phase2 sota plan — Phase 2 SOTA path: research state and execution plan

## Current evidence summary

### Dose-response (mixture eval repaired)
- **mix_25pct is best aggregate** on inherited 8×480 backbone:
  - equal7_fast: 43.451 (+0.595 vs baseline)
  - equal7_fullEntity: 43.716 (+0.784 vs baseline)
  - BLiMP 68.12 (+0.77), Supplement 65.6 (+0.40), EWoK 51.09 (+1.45)
  - GlobalPIQA_mean 38.12 (+2.0), Reading 8.405 (+1.075)
  - Entity_full 22.7 (+0.92), COMPS 51.98 (-1.13)
- Higher aligned fractions (50%, 75%, 100%) redistribute toward Entity at the cost of other columns
- The improvement is broadly distributed, not Entity-dominated

### Leader configuration (Overall 41.8)
| Parameter | Value |
|---|---|
| Architecture | DeBERTa-v2 12×384, 12 heads, intermediate 1280 |
| Params | 34,677,952 |
| Vocab | 40,000 SentencePiece BPE |
| Optimizer | LAMB, LR 0.007, cosine schedule |
| Seq curriculum | 64→128→256 (constant tokens/step) |
| Masking | WWM epochs 1-7, token epochs 8-10 |
| Data | 9,999,969 words FineWeb simplification pairs |
| share_att_key | true |
| position_biased_input | false |

Leader's column scores: BLiMP 67.2, Supplement 56.01, EWoK 56.07, Entity 28.45, COMPS 53.57, GlobalPIQA 39.67, (Super)GLUE 69.79, Reading 5.42, AoA 0.0

### Our advantages over the leader
1. **Entity**: mix_25pct already achieves Entity_full 22.7 with 25% aligned data (fast screen)
2. **BLiMP**: 68.12 vs leader's 67.2
3. **Supplement**: 65.6 vs leader's 56.01 (large gap)
4. **Reading**: 8.405 vs leader's 5.42

### Where the leader excels (columns we need to improve)
1. **EWoK**: 56.07 vs our 51.09 (gap -4.98)
2. **(Super)GLUE**: 69.79 (we haven't evaluated this)
3. **GlobalPIQA**: 39.67 vs our 38.12 (gap -1.55)
4. **Entity**: 28.45 vs our 22.7 (gap -5.75 on full Entity)

### Key insight
- Our mix_25pct already beats the leader on 4 of 9 columns (BLiMP, Supplement, Reading, and likely AoA)
- The leader's architecture (12×384 + LAMB + 40k) may provide EWoK + SuperGLUE gains
- Combining our data advantage with the leader's architecture should be the SOTA path

## phase2 sota plan execution plan

### Phase A: Fixed-init replication (RUNNING)
- **Purpose**: Confirm mix_25pct > official is not an initialization artifact
- **Arms**: official-only vs mix_25pct, both with extra_init_seed=43022, train_rng_seed=43023
- **Config**: Same 8×480 DeBERTa-v2, baseline16k, AdamW, WWM baseline recipe
- **Expected duration**: ~97 min per arm (parallel on 2 GPUs)

### Phase B: Phase 2 SOTA attempt (READY TO LAUNCH)
- **Purpose**: Combine our data advantage with the leader's proven architecture
- **Arms**: 
  - `phase2_official`: Official-only 10M, 40k tokenizer, 12×384, LAMB
  - `phase2_mix25`: mix_25pct data, same everything else
- **Key files**:
  - Trainer: `scripts/phase2_sota_trainer.py`
  - Launcher: `scripts/launch_phase2_sota.sh`
  - Tokenizer: `data/tokenizer/hf_tokenizer_40k` (trained on mix_25pct pool)
  - Official data: `data/fixedinit_replication/official_100M.jsonl`
  - Mix data: `data/mixture/training_files/mix_25pct_100M.jsonl`
- **Config**: 12 layers, 384 hidden, 12 heads, intermediate 1280, LAMB LR 0.007,
  seq curriculum 64→128→256, WWM→token at 70%, extra_init_seed=43200
- **Expected duration**: ~90-120 min per arm (parallel on 2 GPUs)

### Phase C: Full evaluation and SOTA determination
- After Phase B, evaluate both Phase 2 arms on all fast columns + full Entity
- If phase2_mix25 > phase2_official AND > leader on fast screen,
  run full official evaluation including SuperGLUE and AoA

## Risk mitigation
1. **Tokenizer issue**: The 40k SentencePiece tokenizer adds BOS/EOS with `encode()` but 
   the trainer uses `add_special_tokens=False` — verified to work correctly
2. **Initialization**: Using explicit `extra_init_seed=43200` for reproducibility
3. **LAMB stability**: If LAMB fails or diverges, fallback is AdamW at lower LR
4. **Word-start detection**: SentencePiece uses `▁` prefix for word starts — works with WWM

## Why this path can achieve SOTA
- Leader achieves 41.8 with similar architecture on pure FineWeb simplification data
- Our mix_25pct already outperforms the leader on BLiMP (+0.92), Supplement (+9.59!), Reading (+2.99)
- If the 40k+12×384+LAMB upgrade gives even half the EWoK/GlobalPIQA/GLUE improvement 
  that the leader gets, combined with our column-specific advantages, Overall should exceed 41.8
- The critical question is whether SuperGLUE + AoA (not yet evaluated) help or hurt
