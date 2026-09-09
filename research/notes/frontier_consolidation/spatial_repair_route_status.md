# spatial repair route status — Spatial-Repair Route Status

## Pending Experiments
- **Training in progress**: 2-pass (19,999,968 word) training of spatial-repair corpus on seed43022
- Same recipe as original (DeBERTa-v2 8×480, baseline16k, seq256, WWM 0.15, AdamW 0.001, batch 256)
- Will produce checkpoints at 1M through 19M

## Corpus modification
- 710 core compact pairs substituted with near-length rewrites where spatial preps were lost
- Core spatial retention: 55.0% → 90.8% 
- 312 lowest-value (no_domain) added pairs dropped to maintain budget
- All science_physical (294) and causal_relational (300) added pairs retained
- Total pool: 9,999,984 words (vs original 10,000,000)
- 5,996 unique pair sources (vs original 6,529 with 12,155 pairs)

## Fair comparison plan
- Compare spatial_repair chck_19M vs original chck_19M (same 19M exposure)
- Key metrics: EWoK (overall + domain-level), BLiMP, Supplement, Entity, COMPS, GlobalPIQA, Reading
- Decision: if spatial repair improves EWoK physical/spatial domains without losing Supplement/Entity/BLiMP, escalate to full 100M run

## Decision tree after 19M screen
1. **Clear EWoK improvement + broad surface stable or improved**: 
   → Escalate to full 100M run → compare against frozen 42.0331 endpoint
   
2. **Marginal or zero EWoK change, broad surface similar**:
   → Spatial preservation alone is insufficient at this compression level
   → Consider: (a) larger substitution (replace ALL compact with near-length for spatial sources, not just lossy ones), (b) different training modification
   
3. **EWoK improved but other columns regress significantly**:
   → The near-length substitution changes the compression-diversity tradeoff too much
   → Need a different approach (e.g., spatial-aware generation, or targeted masking)
   
4. **Broad regression**:
   → Route closed, the spatial handle is too small in practice

## Alternative routes not yet tested
- GlobalPIQA improvement (weakest column at 35.62, but risky based on COMPACT_EXPERIENCE evidence)
- Different learning rate schedule during late training
- Targeted data augmentation for practical reasoning
- Source composition rebalancing within the fixed 10M budget
