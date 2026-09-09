# curriculum experiment design — AdamW + sequence-length curriculum experiment

## Design

**Factor tested**: Sequence-length curriculum (64→128→256) with word-boundary chunking  
**Control**: Legal40k 8×480 AdamW fixed-256 baseline (cheap7 43.1079, Overall 41.1406)  
**Architecture**: 8×480/8h/FFN1920 DeBERTa-v2 (45.8M params, our best compliant)  
**Data**: compact-view reinvest 100M (same as baseline)  
**Tokenizer**: legal40k byte-BPE (same as baseline)  
**Optimizer**: AdamW lr=0.001, warmup=0.05, wd=0.01, betas=(0.9,0.98) (same as baseline)  
**Masking**: Fixed WWM mask_prob=0.15 (same as baseline)  
**Seed**: init=43022, data=43, same as baseline  
**Precision**: fp32 (same as baseline)  

### Curriculum schedule
- Phase 0: seq_len=64 until 20M cumulative words
- Phase 1: seq_len=128 until 50M cumulative words  
- Phase 2: seq_len=256 until 100M cumulative words

### What changes
1. **More optimizer updates**: ~4935 steps vs 2529 baseline (1.95×)
2. **Word-boundary chunking**: each row splits into multiple seq_len chunks at word boundaries, no mid-wordpiece splits
3. **Different LR schedule**: warmup and cosine decay over ~4935 steps instead of 2529
4. **Shorter initial contexts**: at seq_len=64, model sees 4× more separate contexts for the same word budget

### What stays the same
- Model config: exact match (pos_att_type=['p2c','c2p'], max_relative_positions=256, pad_token_id=3)
- Total word exposure: 100M
- Data order: rows processed in same order
- Optimizer: same AdamW with same hyperparameters
- Masking: same fixed WWM at 0.15
- Batch size: same 256 effective (4×64 accumulation)

## Scientific basis
1. The leader uses 64→256 curriculum (our biggest untested factor from that recipe)
2. Shorter initial sequences give 4× more gradient updates in phase 0
3. May improve early knowledge consolidation before transitioning to longer contexts
4. This is the first clean, matched curriculum test in this study

## Evaluation plan
- Compare cheap7 against baseline 43.1079
- If cheap7 > 43.1079: curriculum helps, investigate which columns improved
- If cheap7 > 43.77 (leader): proceed to full official evaluation (SuperGLUE + AoA)  
- If cheap7 ≤ 43.1079: curriculum alone is insufficient, need to combine with other factors

## Score context (from lamb config audit)
- Baseline 8×480 legal40k: cheap7 43.1079, Overall 41.1406
- Baseline 12×384 depth: cheap7 43.0030, Overall 41.0276
- Leader: cheap7 43.77, Overall 41.80
- To cross 41.80 with our SuperGLUE: need cheap7 ~43.96-44.00

## Active runs
- GPU0: 8×480 AdamW + curriculum → `training/runs/curriculum_adamw_8x480_legal40k_seed43022`
- GPU1: 8×480 contaminated LAMB + curriculum (wrong config, joint-intervention only)

## Contaminated LAMB run (s131_t18_tool2) interpretation
The lamb curriculum route decision 8×480 LAMB run has config defects (pos_att_type=[], max_relative_positions=-1) and is a joint LAMB×curriculum×config-defect intervention. Its result should only be compared to the lamb config audit score context (baseline cheap7 43.1079) and interpreted as contaminated.
