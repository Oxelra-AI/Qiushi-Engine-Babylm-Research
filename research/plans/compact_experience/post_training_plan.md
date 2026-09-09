# execution state post-training plan

## When training completes (~85 min from launch)

### Immediate actions
1. Read `training/runs/mlm_mntp_aux015_seed43022/scientific_metrics.json`
   - Verify `status == "MLM_MNTP_AUXILIARY_TRAINING_COMPLETE"`
   - Check `mlm_loss_last` — should be near clean-Qwen's 2.507 (if much higher, aux may have hurt)
   - Check `aux_loss_last` — expect ~10-12 (model learning to predict shifted)
   - Check `lambda_last` — expect ~0.04-0.06 (from target_ratio=0.15 / ratio~3)
   - Check `ema_aux_norm_final / ema_mlm_norm_final` — the actual norm ratio at end

2. Run full nine-column evaluation:
   ```
   bash scripts/launch_full_eval.sh
   ```
   Uses custom_endpoint_full_eval.py with:
   - --eval_target mlm_mntp_aux015_100M
   - --run_dir training/runs/mlm_mntp_aux015_seed43022
   - --endpoint chck_100M
   - --out_root data/mlm_mntp_full_eval

### Evaluation interpretation
Compare to clean-Qwen Overall 41.34429066479573:
- **If Overall > 41.34 with coherent column profile** → positive signal
  - Check which columns improved vs clean-Qwen
  - If close to 41.8 → immediate second-seed replication + AoA attention
  - If >41.8 → potential SOTA, prepare submission pipeline
- **If Overall ≤ 41.34** → close same-stack auxiliary route
  - Check if specific columns improved (may inform architecture changes)
  - Pivot to representation/architecture/tokenization

### What "coherent column profile" means
- No catastrophic loss in any single column (>-3.0 vs clean-Qwen)
- Net positive across NLP columns (equal7 > 43.11)
- AoA remains 0.0 (same checkpoint ladder structure)
- SuperGLUE maintains or improves (>70.0)

## Fallback route priorities (if auxiliary fails)

### Priority 1: Tokenizer expansion (16k → 40k)
The leader uses 40k SentencePiece BPE. Our 16k is very small.
Impact: more words per sequence → richer context → potentially better all columns.
Requirement: train new tokenizer on BabyLM data, rebuild corpus, full 100M training.
Time: ~2-3 hours total. High-reward potential.

### Priority 2: LAMB optimizer (replace AdamW)
Leader uses LAMB at 0.007. LAMB is designed for large-batch.
Can combine with tokenizer change. Moderate-reward.

### Priority 3: Architecture changes (GEGLU, layer weighting)
From GPT-BERT: GEGLU activation, attention gating, layer mixing.
Can be applied to current recipe. Lower-risk, moderate reward.

### Priority 4: Lower auxiliary ratios or PCGrad
If aux@0.15 shows partial improvements but net-negative, try:
- target_ratio=0.05 (very light directional signal)
- PCGrad-style projection on conflicting early layers only
Only if column analysis shows promise in specific dimensions.
