# execution state execution state

## Training status
- Script: `scripts/mlm_mntp_auxiliary_trainer.py`
- Run dir: `training/runs/mlm_mntp_aux015_seed43022`
- Expected completion: ~84 min from launch (~76 min remaining at last check)

## Last observed state (earlier analysis / 2515, ~8M/100M words)
- MLM loss: 4.46 (exactly matches clean-Qwen trajectory)
- Aux loss: 6.41 (model learning shifted prediction)
- Lambda: 0.040 (adapted from 0.167 as aux/mlm ratio grew to ~3.8)
- Effective aux ratio: 0.150 (on target)
- Per-step: ~2.0s

## Key verification points
- MLM loss earlier analysis = 9.811304092407227 = exact match to clean-Qwen
- Confirms byte-identical initialization, data, masking, and first forward pass
- Only difference: gradient includes 0.15× auxiliary contribution

## Prepared for post-training
1. Evaluation script: `scripts/launch_full_eval.sh`
   - Uses custom_endpoint_full_eval.py (proven infrastructure)
   - Full nine columns: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, (Super)GLUE, Reading, AoA
2. Design notes: `notes/mlm_mntp_auxiliary_construction_design.md`
3. Post-training plan: `plans/post_training_plan.md`
4. Fallback plan: `plans/fallback_phase2_architecture.md`
   - Documents Phase 2 evidence: architecture alone doesn't help (equal7 dropped 41.91 vs 42.86)
   - Clean-Qwen on 12×384 is the correct fallback combination

## Decision tree
- If Overall > 41.34 (positive delta vs clean-Qwen):
  - Check column coherence
  - If near 41.8: second seed + submission preparation
- If Overall ≤ 41.34:
  - Close same-stack auxiliary route
  - Pivot to 12×384 + LAMB + 40k + clean-Qwen (data×architecture interaction)
