# b256 fixedseq pair eval raw — earlier analysis-regime fixed WWM vs WWM→token pair

Final pair summary: `experiments/archive/compact_experience/data/b256_pair_eval/b256_pair_100M_eval_summary.json`
Trajectory summary: `experiments/archive/compact_experience/data/b256_pair_eval/b256_pair_trajectory_eval_summary.json`
Compact CSV: `experiments/archive/compact_experience/data/b256_pair_eval/b256_pair_compact_table.csv`
Machine-readable mechanism summary: `experiments/archive/compact_experience/data/b256_pair_eval/b256_pair_mechanism_summary.json`

## What was recovered from the timed-out background task

Wave 2 was incomplete, but wave 1 completed the matched paired comparison. Both wave-1 runs have `scientific_metrics.json`, `training_log.jsonl`, ten 10M-spaced checkpoints, and valid `hf_model/chck_100M` directories. They use the earlier analysis-like regime: official corpus, DeBERTa-v2 8×480, baseline16k tokenizer, fixed seq256, batch256, seed43, 100M word exposure, and WWM 0.15 until the WWM→token arm switches at 70M.

The pair is internally clean because the arms are identical through 70M except for insignificant evaluation noise; it is not a replacement for the INITIAL_MODEL_STUDIES 40.7028 coordinate because it was trained with the COMPACT_EXPERIENCE curriculum trainer rather than the original earlier analysis trainer and the fixed-WWM Supplement score differs strongly from the old direct recheck.

## Fast task trajectory

Weighted fast proxy = (3/28) × (BLiMP + Supplement + EWoK + Entity + COMPS + GlobalPIQA mean) + (1/8) × Reading. It is a research statistic, not official Overall.

| target | BLiMP | Supplement | EWoK | Entity | COMPS | GPIQA mean | Reading | weighted fast proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_fixed__chck_70M | 66.380 | 61.200 | 49.360 | 23.680 | 52.330 | 39.065 | 7.980 | 32.285 |
| wwm_to_token__chck_70M | 66.370 | 61.200 | 49.360 | 23.680 | 52.330 | 39.065 | 7.980 | 32.284 |
| wwm_to_token_minus_wwm_fixed__chck_70M | -0.010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.001 |
| wwm_fixed__chck_80M | 66.640 | 58.400 | 51.270 | 23.510 | 52.610 | 38.590 | 8.160 | 32.201 |
| wwm_to_token__chck_80M | 67.260 | 61.600 | 50.000 | 24.320 | 52.340 | 38.120 | 7.930 | 32.453 |
| wwm_to_token_minus_wwm_fixed__chck_80M | 0.620 | 3.200 | -1.270 | 0.810 | -0.270 | -0.470 | -0.230 | 0.252 |
| wwm_fixed__chck_100M | 67.360 | 57.200 | 50.910 | 24.950 | 52.720 | 39.580 | 8.230 | 32.392 |
| wwm_to_token__chck_100M | 67.630 | 61.200 | 50.450 | 23.780 | 52.180 | 39.580 | 7.885 | 32.573 |
| wwm_to_token_minus_wwm_fixed__chck_100M | 0.270 | 4.000 | -0.460 | -1.170 | -0.540 | 0.000 | -0.345 | 0.182 |

## Mechanistic reading

At 70M, before the switch, fixed WWM and WWM→token match almost exactly: proxy delta -0.001 and no meaningful column movement. The causal movement begins after token-level masking starts.

At 80M, WWM→token is higher on the fast proxy by 0.252: BLiMP +0.62 and Supplement +3.20, but EWoK -1.27, COMPS -0.27, GlobalPIQA mean -0.47, and Reading -0.23. Entity is temporarily +0.81.
At 100M, WWM→token remains higher by 0.182: BLiMP +0.27 and Supplement +4.00 carry the gain, while EWoK -0.46, Entity -1.17, COMPS -0.54, and Reading -0.345 move down; GlobalPIQA mean is unchanged.

This differs from the earlier seq-length-scheduled b64 run, where WWM→token at 100M had proxy delta -0.239, Supplement -5.20, Entity -1.90, Reading -0.46, and GlobalPIQA mean +3.41. The sign of the Supplement and GlobalPIQA effects is therefore not stable across these two training regimes. The stable part is that the late token switch changes task allocation after 70M and tends to reduce Reading and often Entity/relational continuity, even when it improves BLiMP or Supplement.

The route implication is sharper than either early interpretation. Unconditional WWM→token should not be discarded solely because of the seq-scheduled screen, but it also should not become the main SOTA route from this fast pair alone. The useful signal is late token-level pressure interacting with the training regime; the unresolved problem is how to keep the lexical/sentence gains while preserving entity tracking, reading dynamics, and full-column transfer.

## Alignment with INITIAL_MODEL_STUDIES earlier analysis coordinate

INITIAL_MODEL_STUDIES earlier analysis direct recheck for the original earlier analysis fixed-WWM `chck_100M` reported BLiMP 67.34, Supplement 65.2, EWoK 49.64, Entity 21.24, COMPS 53.11, Reading 7.33. The COMPACT_EXPERIENCE fixed-WWM rerun in this pair reports BLiMP 67.36, Supplement 57.2, EWoK 50.91, Entity 24.95, COMPS 52.72, Reading 8.23. The pair reproduces much of the earlier analysis regime but not the exact old score surface, especially Supplement; conclusions should use within-pair deltas and then be checked by full official evaluation before any scale decision.

## Next scientific work

1. Run a full official-compatible evaluation for the two b256/fixed-seq `chck_100M` checkpoints, including (Super)GLUE, full GlobalPIQA, Reading, and AoA where the current evaluator supports it, so the +0.182 fast proxy is tested on the actual Overall surface.
2. In parallel, inspect why the COMPACT_EXPERIENCE fixed-WWM Supplement score is far below the INITIAL_MODEL_STUDIES earlier analysis direct recheck despite similar BLiMP and recipe; compare data order, checkpoint schedule, trainer implementation, and evaluation output paths rather than treating this rerun as identical to earlier analysis.
3. If full evaluation preserves the same split, build the next mechanism as conditional late granularity rather than a hard global switch: retain WWM on entity/long/clause-rich examples or a fixed fraction of late batches, while adding token-level pressure only where it improves lexical/sentence judgments. Do not mix this with paired-rewrite data until the granularity mechanism is isolated.
