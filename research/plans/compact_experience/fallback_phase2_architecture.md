# execution state updated fallback route: leader-matched architecture

## Discovery: Phase 2 infrastructure already exists

phase2 sota plan already built a complete trainer matching the leader's architecture:
- File: `scripts/phase2_sota_trainer.py` (659 lines, tested)
- Architecture: DeBERTa-v2 **12×384**, 12 heads, intermediate 1280
- Optimizer: **LAMB** with LR 0.007, cosine schedule
- Tokenizer: 40k SentencePiece BPE
- Seq curriculum: 64→128→256
- Masking: WWM → token at epoch 8-10
- Includes complete LAMB implementation

The trainer was never run on clean-Qwen data because:
1. It was built during the mix25 era (mixture experiment design)
2. The mix25 route was declared contaminated (earlier analysis)
3. The clean-Qwen route went through on the inherited 8×480 backbone
4. Subsequent work focused on masking/tail/objective experiments on 8×480

## Evidence: Phase 2 was already run (fixedinit and phase2 dynamics-24)

The 12×384 + LAMB + 40k architecture was actually trained and fast-screened:
- **Phase2 official-only**: equal7 = 41.91 (WORSE than 8×480 baseline 42.86!)
  - EWoK 45.91 (vs 8×480 49.64), GlobalPIQA 34.23 (vs 36.12), Reading 5.63 (vs 7.33)
  - BLiMP improved (68.61 vs 67.35), Supplement stable
- **Phase2 mix25** (contaminated): equal7 = 42.50
  - EWoK 51.73, Entity 25.06 (data effect visible)
  
**Critical insight**: The architecture alone does NOT explain the leader's 41.8.
The leader's EWoK (56.07 vs Phase2-official 45.91), Entity (28.45 vs 22.83),
and GlobalPIQA (39.67 vs 34.23) advantages come from their FineWeb simplification
pairs, not from 12×384/LAMB/40k.

## Why clean-Qwen on 12×384 is still worth testing

Our clean-Qwen pairs produce strong EWoK/Entity effects on 8×480 (EWoK 50.19,
Entity 25.76). On 12×384, the contaminated mix25 achieved EWoK 51.73, Entity 25.06.
The clean-Qwen pairs (which are semantically aligned like FineWeb simplification)
on 12×384 might produce the same data×architecture synergy the leader exploits.

The hypothesis is NOT "architecture alone helps" — it's "clean-Qwen pairs + larger
vocabulary + deeper model may synergistically boost the EWoK/Entity/GlobalPIQA columns
that the leader excels at."

## Execution plan (if MNTP auxiliary fails)

### earlier analysis: Train clean 40k tokenizer (~1 min)
- Train SentencePiece BPE on the clean-Qwen 10M pool
  (`data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl` — first 10M words)
- This replaces the old mix25-trained tokenizer with a verifiably clean one

### babylm2026 surface refresh: Materialize corpus with new tokenizer
- The corpus text stays the same (clean-Qwen 10M pool, 100M exposure)
- Just re-tokenize with the new 40k tokenizer
- Examples keep the same order and words

### hf top repo surface: Train 12×384 + LAMB on clean-Qwen data (~90-120 min on H100)
```bash
python scripts/phase2_sota_trainer.py \
  --example_jsonl data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl \
  --example_jsonl_meta data/qwen_clean_aligned/clean_materialization_metadata.json \
  --output_dir training/runs/phase2_clean_qwen_40k_12x384 \
  --tokenizer_path data/tokenizer_40k_clean/hf_tokenizer_40k \
  --tokenizer_label sp40k_clean_qwen \
  --model_type deberta_v2 \
  --n_layer 12 --hidden_size 384 --n_head 12 --intermediate_size 1280 \
  --share_att_key true --position_biased_input false \
  --optimizer lamb --learning_rate 0.007 \
  --max_seq_length 256 --batch_size 256 \
  --seq_len_schedule "0.0:64,0.3:128,0.6:256" \
  --mask_mode wwm --mask_switch_mode token --mask_switch_fraction 0.7 \
  --mask_prob 0.15 \
  --max_word_exposure 100000000 \
  --warmup_fraction 0.06 --weight_decay 0.01 \
  --seed 43 --extra_init_seed 43200 --train_rng_seed 43201 \
  --checkpoint_words 1000000 --log_every 50
```

### earlier analysis: Official control (same arch/tokenizer/optimizer, official-only data)
- Same as above but with official-only 10M corpus
- Required for causal attribution of the Qwen-pair advantage

### earlier analysis: Full nine-column evaluation
- Compare to clean-Qwen 8×480 (41.34) and leader (41.8)
- If > 41.8 → SOTA candidate → second seed → submission

## Can MNTP auxiliary combine with Phase 2?
If MNTP is positive on 8×480 AND Phase 2 is positive independently:
- Test MNTP + 12×384 + LAMB + 40k + clean-Qwen
- Potentially additive contributions

## Key differences from leader
Even with matching architecture, our approach differs:
- **Data source**: Qwen-3.5-9B generated rewrites vs FineWeb simplification
- **Data mix**: 16.568% aligned pairs + 83.432% official vs 100% FineWeb
- **Our proven advantages**: BLiMP (+0.9), Supplement (+9.6), Reading (+3.0) over leader
- **Leader's advantages**: EWoK (+6.0), Entity (+5.8), GlobalPIQA (+1.6)

The architecture match should close the EWoK/GlobalPIQA gap while preserving our column strengths.

## execution state column decomposition analysis

### Where the leader wins vs Phase2-official (same architecture):
- EWoK: -10.16 (leader 56.07 vs Phase2 45.91)
- Entity: -5.62 (leader 28.45 vs 22.83)
- GlobalPIQA: -5.45 (leader 39.67 vs 34.23)
Total leader advantage in sum: +13.41 in these columns

### Where Phase2-official wins vs leader:
- Supplement: +7.99 (Phase2 64.0 vs leader 56.01)
- BLiMP: +1.41
- Reading: +0.205
Total Phase2 advantage in sum: +9.60 in these columns

### Implication
The leader's data (FineWeb simplification) provides ~+13 sum-points in EWoK/Entity/GlobalPIQA.
These are exactly the columns where our clean-Qwen pairs also help (on 8×480):
- Clean-Qwen EWoK 50.19 vs official control ~48
- Clean-Qwen Entity 25.76 vs official ~21
The architecture change alone hurts GlobalPIQA and Reading.

### For Overall arithmetic
To cross 41.8, we need sum > 376.2 (vs our current 372.1, gap 4.1).
Most achievable: boost EWoK (+4 → +0.44 Overall) and Entity (+3 → +0.33 Overall)
without losing Supplement/Reading/GlobalPIQA.

### Risk of pure architecture pivot
Phase2 architecture HURTS GlobalPIQA_parallel (-6.79 vs 8×480 baseline) and Reading (-1.71).
A naive architecture change may not improve Overall even with our good data.
The safe route: find interventions that improve EWoK/Entity/GlobalPIQA on the CURRENT backbone.
