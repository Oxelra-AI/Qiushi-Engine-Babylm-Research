# causal attention verification post-training plan

## Current state
- The comparison had started training on both H100s; completion was pending:
  - GPU 0: causal_fraction=0.50, clean-Qwen corpus → mixed_causal50_qwen_seed43022
  - GPU 1: causal_fraction=0.15, clean-Qwen corpus → mixed_causal15_qwen_seed43022
- Expected duration: ~2-3 hours for 100M word exposure at batch 256.
- Clean-Qwen pure-MLM reference: equal7 43.1129, Overall 41.3443.

## When training completes

1. **Verify completion**: Check both `scientific_metrics.json` for status, word exposure,
   loss trajectories, and realized causal batch fractions.

2. **Run no-AoA evaluation**: Execute `scripts/eval_mixed_objective_noaoa.sh`
   which evaluates chck_80M through chck_100M on seven columns per arm.

3. **Compare against reference**: Use `scripts/summarize_mixed_objective_noaoa.py`
   to compute equal7 and deltas vs clean-Qwen.

4. **Decision fork**:
   - If BOTH arms < clean-Qwen equal7: The mixed objective does not help on this
     backbone/data. Consider reducing causal fraction (0.05) or try a different
     objective mixing strategy (e.g., prefix-LM, ELECTRA-style RTD).
   - If either arm > clean-Qwen by ≥0.3 equal7 with coherent profile:
     Proceed to full 9-column evaluation (SuperGLUE + AoA). This is the critical
     test: does from-scratch mixed training preserve AoA ≥ 0?
   - If either arm approaches or exceeds 41.8 Overall: immediate second-seed
     replication (seed43122), then full submission packaging.

5. **AoA expectation**: Since this is from-scratch training (not 80M→continuation),
   the model should have a natural acquisition trajectory. The tail-restart family's
   AoA collapse was likely caused by the restart disrupting the acquisition curve.
   From-scratch training preserves the curve.

## Key metrics to report
- Per-arm best checkpoint and equal7
- Per-column comparison: which tasks benefit from causal signal?
- MLM loss last vs causal loss last (convergence)
- Realized causal batch/target fractions
