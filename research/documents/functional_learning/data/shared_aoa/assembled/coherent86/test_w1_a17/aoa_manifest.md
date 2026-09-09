# aoa shared measurement and densemask control AoA measurement: coherent86
Created: 2026-09-07T21:03:03.151791+00:00

Status: `ASSEMBLED_AOA_MEASURED`

## Trajectory

- Early-stop convention: 17 ancestral checkpoints chck_1M..chck_80M for ~86M/~89M endpoints, plus exact final endpoint; full 19-step max-budget ladder is not borrowed.
- Shared ancestry steps: chck_1M, chck_2M, chck_3M, chck_4M, chck_5M, chck_6M, chck_7M, chck_8M, chck_9M, chck_10M, chck_20M, chck_30M, chck_40M, chck_50M, chck_60M, chck_70M, chck_80M
- Endpoint word count: `86005295`
- Assembled surprisal: `experiments/archive/functional_learning/data/shared_aoa/assembled/coherent86/test_w1_a17/AoA_word/surprisal.json`

## Extraction evidence

- Results: `234`
- Finite: `234`; nonfinite: `0`
- Missing steps: `[]`
- Steps with wrong counts: `{}`

## AoA scoring

- Measured: `True`
- Raw correlation: `0.0`
- Leaderboard score: `0.0`
- Legitimate measured zero: `True`
- Valid words: `0`

The zero is produced by completed extraction plus official curve scoring; it is not a missing-checkpoint placeholder.
