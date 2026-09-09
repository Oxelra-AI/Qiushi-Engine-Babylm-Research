# u256 endpoint mechanism — U256 endpoint consumption state

No final score from U256 is available yet. The official-compatible evaluator is writing to `data/u256_100M_full_eval_hardened/` with target `U256_100M_seed43022`. Do not start another expensive training route until this endpoint evidence is complete or fails in a way requiring a minimal measurement repair.

## File-Only Verification

1. **Endpoint legality/readiness/HF usability**
   - Script: `scripts/u256_endpoint_readiness_smoke.py`
   - Outputs: `data/u256_endpoint_readiness_smoke/u256_endpoint_readiness_smoke.{json,md}`
   - Result: ok=true. The U256 100M endpoint has arm U256, exactly 100,000,000 charged words, 2,530 steps, first loss 9.826857208144903, final loss 2.516624725910071, parameter_count 34,467,424, raw_tokens_per_epoch 14,664,519, 100 checkpoints, complete 19-checkpoint AoA ladder, legal pool SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`, stream SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`, endpoint tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`, and endpoint tokenizer vocab equals the spatial repair route status legal tokenizer. CPU `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` succeeds with an isolated writable HF cache, finite logits, and probe loss 13.028825759887695.

2. **Post-eval analyzer path/schema repair**
   - Patched `scripts/u256_post_eval_analyzer.py` so `DEFAULT_TARGET = 'U256_100M_seed43022'`, matching the actual submitted evaluation command and part filenames.
   - Patched validation to accept and check the evaluator's `endpoint_ready.aoa_ladder_present` and `endpoint_ready.aoa_ladder_missing` fields. AST check passed.

3. **Endpoint mechanism/readout**
   - Script: `scripts/u256_endpoint_mechanism.py`
   - Outputs: `data/u256_endpoint_mechanism/u256_endpoint_mechanism.json`, note `notes/u256_endpoint_mechanism.md`
   - Result: U256 is not a small local perturbation of spatial repair route status; it is a large same-architecture trajectory redirection caused by changing the stream object to make row tails visible. U256 vs spatial repair route status tensor cosine/rel-L2: 20M 0.797325/0.637042; 80M 0.645656/0.843406; 90M 0.645322/0.843782; 100M 0.645323/0.843789. Late update cosine versus spatial repair route status is very low: 20M→80M 0.183527, 80M→90M 0.019275, 90M→100M 0.017136; update norms stay matched (~1.0). Core matrix spectra remain stable (U256 stable-rank mean 33.296 at 80M and 33.315 at 100M vs spatial repair route status 32.919/32.963), so this is not a Muon/LAMB-like spectral collapse/broadening.

4. **Prefix consistency of 20M screen and full 100M run**
   - The standalone U256 20M `chck_20M/model.safetensors` and the full U256 100M run's `chck_20M/model.safetensors` have identical SHA256 `1f6fce560592f6ecbb33ff75ccac841fc77e8ca137e61ecda9267a60aba33386`. Therefore the +0.9179 cheap7 20M screen was a true prefix of the exact endpoint trajectory now under full evaluation, not a separate stochastic run.

## Validation of the pending U256 full evaluation

After U256 full evaluation completes with exit 0 and `data/u256_100M_full_eval_hardened/summary/u256_100M_full_eval_hardened_summary.json` exists, run:

```bash
python -B experiments/archive/frontier_consolidation/scripts/u256_post_eval_analyzer.py
```

It should use the corrected target `U256_100M_seed43022`, validate nine-column arithmetic, check the endpoint and AoA ladder fields, patch the staged payload if needed, and run item-flip comparison against spatial repair route status.

If the evaluator fails in only one column from a measurement-path issue, repair only that missing component and merge from completed split outputs, following the earlier analysis scale1.75 repaired-merge pattern. Do not rerun completed columns or start a new training route before consuming U256.

Scientific interpretation target: U256 must be judged against spatial repair route status Overall 41.257770896404615 and scale1.75 100M Overall 41.57074653643003, and against the 41.8 target. The route matters scientifically only if the mature endpoint keeps the 20M BLiMP/Supplement/COMPS/GlobalPIQA gains without losing EWoK material/social/quantitative/spatial families or SuperGLUE. If it misses, the combination of scale1.75 and U256 evidence should go to a high-level route judgment before more GPU training.

## Added after training-log comparison

Training-log comparison outputs: `data/u256_training_log_comparison/u256_training_log_comparison.json`, note `notes/u256_training_log_comparison.md`, prefix JSON `data/u256_training_log_comparison/u256_20m_prefix_consistency.json`.

Key measurements: U256 and spatial repair route status both end at exactly 100,000,000 counted words, but U256 uses 2,530 updates vs spatial repair route status 2,529. U256 final MLM loss is lower by -0.035937 (2.516624725910071 vs 2.5525617599487305). Total masked token pieces are 22,009,842 for U256 vs 21,449,599 for spatial repair route status (+560,243, about +2.61%), matching the active-token visibility repair scale. Across common step indices LR differs only by at most 3.83e-07, while batch word counts and masked-token counts differ at almost every step because chunking changes the training examples. This reinforces that the pending score should be read as a changed credit-flow/visible-token trajectory under the same counted-word stream, not as a different corpus, tokenizer, model size, or objective.
