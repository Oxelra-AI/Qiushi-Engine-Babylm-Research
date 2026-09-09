# Closed branches and no-training probes

## Closed by completed evidence

- Simple developmental first-pass ordering is closed as a 100M training route. The frozen complete evaluations in `data/devcurr_eval/devcurr_eval_summary.json` show both seeds below their same-seed clean-Qwen controls and both with AoA raw/leaderboard score 0.0. This means the legal source/statistics first-pass ordering did not reshape the measured acquisition trajectory and did not improve the nine-column result.
- Cap-length/row-geometry optimization is closed as a 100M expansion route. `data/official_geometry_noaoa_interpretation.json` shows official160_b256 `chck_95M` has only +0.0093 equal7 over clean-Qwen while losing Supplement, Entity, and Reading; cap120geom is below clean-Qwen. The cap120 BLiMP/Reading gains are a profile tradeoff, not broad progress.

## Pair-agreement probe judgment

`data/pair_agreement_probe/pair_agreement_signal.json` uses only clean qwen compliance and validity training pairs and submitted-model representations. It does not justify a new pair-agreement objective run: clean-Qwen improves mean positive-minus-shuffled margin over official by only +0.00186 and is lower than the shuffled-trained control by -0.00596. The high absolute positive/rewrite separability is mostly a property of lexical-semantic relatedness already present in the pair texts, not evidence for an unexploited credit-assignment mechanism.

## SWA probe status

`data/tail_swa_models/tail_swa_metadata.json` created three single averaged model artifacts from clean-Qwen checkpoints: `swa_90_95_100M`, `swa_85_90_95_100M`, and `swa_75_90_95_100M`. They are being evaluated as individual models on BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading only. If none clearly beats clean-Qwen equal7 43.112857 with a profile that can plausibly survive SuperGLUE/AoA measurement, do not run full nine-column eval or new 100M training from SWA.

## Next mechanism requirement

The next expensive training wave should not chase AoA arithmetic leverage, pair-exposure confirmation, or cap geometry. It should be a tightly matched low-cost test of a mechanism that can plausibly recover clean-Qwen's weak EWoK/COMPS/GlobalPIQA region while retaining its Supplement, Entity, SuperGLUE, and Reading strengths. The most defensible direction is corpus-side local coherence/contextual-diversity shaping within the official/Qwen training corpus, with explicit shuffled and original-repeat controls and no use of official AoA/CDI items, curves, predictions, scores, or downstream eval outputs in construction.
