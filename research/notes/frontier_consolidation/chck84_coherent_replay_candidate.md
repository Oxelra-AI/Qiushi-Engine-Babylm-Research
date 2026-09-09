# bridge dose response interpretation — concrete score-improvement candidate: chck_84M coherent replay

## Observation

The coherent86 approach (frozen chck_82M + 4M coherent private-path replay → α=0.75) produced the strongest secured endpoint at projected Overall 42.1210 [coherence margin signal isolation]. But it anchored on chck_82M (cheap7 43.96), not chck_84M (cheap7 44.12, projected Overall 42.02). 

chck_84M is +0.165 cheap7 above 82M on the same trajectory. If the coherent replay mechanism adds a similar ~+0.15-0.22 cheap7 at the optimized alpha, chck_84M-anchored replay could reach cheap7 ~44.27-44.34, projecting to Overall ~42.10-42.20.

## Why this may work

1. Both chck_82M and chck_84M share the same scale1.75 adapter architecture (35,463,008 params)
2. The coherent replay trainer (frozen82_fastpath_replay_trainer.py) freezes all base params and trains only a private adapter (995,584 params)
3. The 84M anchor has better EWoK, Entity, Supplement, and GlobalPIQA than 82M
4. Total exposure: 84M + 4M = 88M words, within 100M budget
5. No architecture or objective change needed

## Why it may NOT work

1. The 84M peak is redistributive, not broad — it improved some families while slightly losing BLiMP
2. The coherent replay gain from 82M was +0.147 cheap7, but the channel decomposition showed 84.1% came from GlobalPIQA [earlier analysis]
3. Cross-seed robustness showed 84M's signed transition doesn't recur (seed43122 route decision)
4. Naive 80/82/84 averaging already failed (seed43122 tail completion rationale)

## Minimum implementation

1. Load chck_84M as frozen anchor (replace chck_82M in the replay trainer)
2. Run same 4M word coherent replay with same private adapter config
3. Test alpha=0.5/0.75/1.0 on cheap7
4. If cheap7 exceeds 44.18 (current α=0.75 from 82M), run selected eval

## Cost

One GPU, ~15 minutes for training + ~30 minutes for selected eval = ~45 minutes total.

## Decision gate

If chck84-coherent88 α=0.75 cheap7 > 44.18 AND the gain is not GlobalPIQA-only, pursue full evaluation. Otherwise close.

## Status

Not launched. Both GPUs running bridge dose-response. Can be run after bridge completes if mechanism results don't motivate something higher-value.
