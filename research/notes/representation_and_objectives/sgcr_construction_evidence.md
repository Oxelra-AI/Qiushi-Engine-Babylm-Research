# corrected entity gate stats — SGCR Construction Evidence & Depth Evaluation Readiness

## Depth Training: COMPLETE
- Task `s74_t5_tool1` completed at 100M words
- Run: `training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/`
- 100 checkpoints (chck_1M through chck_100M) under `hf_model/`
- Final loss: 2.525, 2,529 steps, 9,578s
- 12×384/FFN1280, legal40k, fixed WWM 0.15, seed43022
- Parameters: 38,421,952

## Depth Evaluation: In Progress
The chck_100M endpoint was found, and the legal40k 12x384 depth evaluation began at 12:46:09 UTC. The planned complete readout includes full evaluation, EWoK, AoA and pristine collation.

## SGCR Module: VERIFIED
- `scripts/sgcr_module.py` — 15,653 bytes
- **Additive residual formulation**: W_eff = W_std + (1-rho) * comp_correction
- Cold init preservation: max_diff=0.0 (exact)
- Zero-sharing limit: exact recovery
- After simulated training: 17.9× more deviation for rare vs common tokens
- Gradient flow: word_emb (full), comp_emb (scaled by 1-rho), proj (scaled by 1-rho)

## SGCR Trainer: VERIFIED (dry-run)
- `scripts/sgcr_trainer.py` — 19,248 bytes
- Base params: 38,421,952, SGCR new: 1,073,536, total: 39,495,488
- Zero-sharing verified in dry run
- 2,529 effective steps (same schedule as depth training)
- Handles --intermediate_size 1280 for depth architecture
- Saves sgcr_components.pt at each checkpoint alongside HF model

## Corrected Entity Discriminating-Span Support
- Items: 6,780 (matches pristine collation exactly, with official `nothing` skip)
- disc_over_shared_lt50: **5.23** (strengthened from uncorrected 4.82)
- EWoK: 0.98 (no enrichment — support not the bottleneck)
- GlobalPIQA: 1.87 (enriched in solutions)
- COMPS: 2.92 (enriched in discriminating prefixes)
- Entity: 5.23 (enriched in options, corrected)

## Gate Statistics (K=50)
- Weighted mean rho: 0.936 (93.6% of training mass barely affected)
- Low-rho types (rho<0.5): 24,854 types, 4.1% of training mass
- 94.9% of low-rho tokens have all legal16k components ≥50 support
- Component influence concentrated in count 0-49 band: 17,854 types, 4.1% mass

## Decision Tree After Depth Vector Arrives

### If depth Overall ≥ 41.8:
→ Protect depth endpoint, launch seed43122 reproduction
→ SGCR becomes a possible enhancement route for second-wave improvement

### If depth improves but < 41.8, with GlobalPIQA/COMPS/Entity remaining gap:
→ **SGCR is the indicated next route**
→ Launch legal40k SGCR K50 d64 on 12×384/FFN1280 from scratch
→ Uniform-gate control for attribution if time allows

### If depth improves but < 41.8, with EWoK as the main remaining gap:
→ SGCR less indicated (no EWoK discriminating-span enrichment)
→ Consider experience-utilization U256 (CHILDES/dialogue tails)

### If depth is flat or worse:
→ Re-evaluate entire legal40k coordinate
→ Consider SGCR on 8×480 as a different structural route

## Files
- Module: `scripts/sgcr_module.py`
- Trainer: `scripts/sgcr_trainer.py`
- Arch spec: `notes/sgcr_architecture_specification.md`
- Entity/gate stats: `data/corrected_entity_gate_stats/`
- Dry-run output: `data/sgcr_dryrun/` (no training data, just manifest verification)
