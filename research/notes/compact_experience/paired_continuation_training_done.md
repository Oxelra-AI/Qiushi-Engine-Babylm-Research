# paired continuation training done — Paired continuation training complete

## Summary

Both arms completed successfully from INITIAL_MODEL_STUDIES earlier analysis chck_70M:

| Arm | mask_mode | loss_first | loss_last | duration | checkpoints |
|-----|-----------|-----------|-----------|----------|-------------|
| WWM (control) | wwm | 2.7667 | 2.5579 | 824s | chck_80M, chck_90M, chck_100M |
| Token (treatment) | token | 2.9010 | 2.3374 | 754s | chck_80M, chck_90M, chck_100M |

## Key observations from training

- Token arm has notably lower final loss (2.34 vs 2.56). This is expected: token-level
  masking creates more independent prediction targets per sequence, giving more gradient
  signal. But training loss does NOT predict downstream transfer (established in cs 4m residualized eval).
- Both arms trained at ~1.01-1.12 sec/step on a single H100 with batch 256.
- LR started at 0.000227 (earlier analysis of 2442 total) and decayed to 0 by earlier analysis.
- Both arms used exactly the same post-70M data order from INITIAL_MODEL_STUDIES manifest.

## Shared conditions verified

- Pool: 62,500 examples, 10,000,000 words (matches INITIAL_MODEL_STUDIES exactly)
- Post-70M: 187,500 examples = 30,000,000 words
- Optimizer: fresh AdamW (lr=1e-3, wd=0.01, betas=(0.9, 0.98))
- Schedule: cosine(warmup=122, total=2442) advanced to earlier analysis
- Model: chck_70M (34,467,424 params, DeBERTa-v2 8×480)
- Masking RNG seed: 43

## Artifacts

- WWM: `experiments/archive/compact_experience/training/runs/continuation_wwm_seed43`
- Token: `experiments/archive/compact_experience/training/runs/continuation_token_seed43`
- Each has: `scientific_metrics.json`, `training_log.jsonl`, `hf_model/chck_{80,90,100}M`

## Next: Evaluation

Run `eval_paired_continuation.py` to evaluate both chck_100M endpoints + INITIAL_MODEL_STUDIES
reference on BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading.
The differential (token - wwm) isolates the granularity effect.
