# causal intervention: Context-Credit Replay — Training and Evaluation Results

## Training completed

Both arms trained from chck_82M on the legal 82M→86M tail (3,992,800 words, 101 updates).

| Metric | Standard | Carrier_Residual |
|--------|----------|-----------------|
| Final main loss | 2.498 | 4.266 (weighted) |
| Final neutral KL | 0.003 | 0.007 |
| Private RMS mean | ~0.023 | ~0.032 |
| Private RMS L7/L8 | 0.023/0.020 | 0.043/0.052 |
| Carrier easy frac | — | 0.379 |

## Evaluation results (fast eval, 6/7 columns — Reading failed due to missing --data_path)

| Column | Standard | Carrier_Residual | Delta (CR-Std) | Reference (coherent86 α0.75) |
|--------|----------|-----------------|----------------|------------------------------|
| BLiMP | 69.01 | 68.93 | -0.08 | 68.51 |
| Supplement | 64.80 | 64.80 | 0.00 | 63.64 |
| EWoK | 49.45 | **50.82** | **+1.37** | 50.02 |
| Entity | 27.93 | **28.29** | **+0.36** | 28.32 |
| COMPS | 52.19 | 52.29 | +0.10 | 52.05 |
| GlobalPIQA | **38.08** | 37.59 | -0.49 | 38.57 |
| Reading | — | — | — | 8.17 |
| equal6 (no Read) | 50.24 | **50.45** | **+0.21** | — |

## Interpretation

1. **Carrier_residual wins on EWoK (+1.37) and Entity (+0.36)** — these are precisely
   the relational/entity-tracking columns where the carrier-error credit should help most.
   The carrier struggles with tokens requiring broad contextual understanding; the private
   adapter's directed credit on these tokens improves relational and entity competence.

2. **Standard wins on GlobalPIQA (+0.49)** — physical intuition may benefit from
   reinforcing patterns the carrier already partially knows.

3. **BLiMP and Supplement are essentially tied** — syntactic competence is similar
   across both credit modes. Both improve over the reference, likely due to the
   different RNG path from gradient accumulation.

4. **Overall equal6 is slightly better for carrier_residual (+0.21)**. Without Reading,
   this is provisional.

5. **The per-layer private adapter RMS pattern is informative**: carrier_residual shows
   much higher RMS in the deepest layers (L7: 0.043, L8: 0.052 vs 0.023/0.020). This
   suggests the carrier struggles most with tokens requiring deep contextual processing,
   and the private adapter compensates more in those layers.

## Next steps

1. **Fix Reading evaluation** — the reading benchmark requires `--data_path`; need to
   add the correct path to complete the 7-column comparison.
2. **Test different alpha values** — carrier_residual's higher neutral KL (0.007 vs 0.003)
   suggests alpha=0.75 may not be optimal; try alpha=0.5, 0.6, 0.65 to find the sweet spot.
3. **Run full official evaluation** if the Reading-corrected equal7 looks promising.
4. **Compare against chck_82M directly** to verify the private adapter adds value.

## Reading evaluation fix

The reading benchmark needs `--data_path evaluation_data/full_eval/reading` added to
the command. The eval script needs a small fix.

## Files

- Trainer: `scripts/context_credit_trainer.py`
- Evaluator: `scripts/fast_eval.py`
- Standard run: `training/runs/standard_seed43023/`
- Carrier_residual run: `training/runs/carrier_residual_seed43023/`
- Eval results: `data/eval/step025_{standard,carrier_residual}_eval.json`
- Design note: `notes/context_credit_design.md`
- This result: `notes/context_credit_result.md`
