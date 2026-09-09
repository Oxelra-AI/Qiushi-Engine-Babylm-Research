# relation wwm 20m launch — relation-WWM 20M falsification run launch

## Smoke validation

Evidence:

- `data/relation_wwm_smoke_validation.json`
- `notes/relation_wwm_smoke_validation.md`
- trainer: `training/scripts/babylm_masked_train_relation_wwm.py`

Both 20k smokes completed with ai_lab-accepted artifacts and loadable checkpoints.

Key telemetry:

- both arms selected exactly 3000 / 20000 whole-word groups = 0.15 group density.
- `relation_wwm` selected relation groups at fraction 0.166 vs candidate relation fraction 0.088; shuffled selected relation fraction stayed near candidate.
- `relation_wwm` selected entity groups at fraction 0.464 vs candidate entity fraction 0.277; shuffled selected entity fraction was much closer to candidate.
- This confirms the mechanism and shuffled control are active while preserving mask density.

## Launched 20M paired experiment

Scientific goal: test whether entity/relation-enriched WWM on the official corpus moves Entity and GlobalPIQA without harming Reading, BLiMP, or Supplement, before any 100M run.

Shared configuration:

- official BabyLM 2026 Strict-Small corpus only
- baseline16k tokenizer
- DeBERTa-v2 8×480, 8 heads, FFN 1920, p2c/c2p relative attention
- WWM-style MLM with exact-K group selection
- 20,000,000 word exposure, official 10M pool, two passes at most
- batch size 256, fixed 256-token training length, `max_position_embeddings=512`
- `lr_total_steps=2442`, matching the prefix of the protected 100M DeBERTa schedule rather than compressing a full cosine schedule into 20M
- checkpoint words: 10M, yielding `chck_10M` and final `chck_20M`
- seeds 42 / 456 / 789 and same data order controls

Runs:

1. `babylm_relation_wwm_deberta_20M`
   - `--mask_mode relation_wwm`

2. `babylm_relation_wwm_shuffled_deberta_20M`
   - `--mask_mode relation_wwm_shuffled`

Interpretation boundary: these 20M runs are falsification/screening experiments, not SOTA candidates. They should be compared to each other and to the existing protected baseline16k DeBERTa trajectory at `chck_10M`/`chck_20M`. A full 100M run is justified only if relation_wwm beats shuffled and ordinary baseline on target signals without Reading/grammar loss.

Target evaluations after completion:

- Entity official macro and split details
- GlobalPIQA parallel/nonparallel with paired wrong→correct table
- Reading eye/self-paced
- BLiMP, Supplement, COMPS
- optional fast EWoK as non-official internal signal
