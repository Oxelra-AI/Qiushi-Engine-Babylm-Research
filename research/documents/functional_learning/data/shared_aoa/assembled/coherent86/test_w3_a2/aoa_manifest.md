# aoa shared measurement and densemask control AoA measurement: coherent86
Created: 2026-09-07T21:44:34.661915+00:00

Status: `ASSEMBLED_AOA_MEASURED`

## Trajectory

- Early-stop convention: 17 ancestral checkpoints chck_1M..chck_80M for ~86M/~89M endpoints, plus exact final endpoint; full 19-step max-budget ladder is not borrowed.
- Shared ancestry steps: chck_1M, chck_2M
- Endpoint word count: `86005295`
- Assembled surprisal: `experiments/archive/functional_learning/data/shared_aoa/assembled/coherent86/test_w3_a2/AoA_word/surprisal.json`

## Extraction evidence

- Results: `99`
- Finite: `99`; nonfinite: `0`
- Missing steps: `[]`
- Steps with wrong counts: `{}`

## AoA scoring

- Estimator: `current_official_main_6f825c2_AoAEvaluator`
- Estimator provenance: `experiments/archive/functional_learning/data/aoa_estimator_provenance/aoa_estimator_platform_provenance.json`
- Measured: `True`
- Raw correlation: `0.0`
- Leaderboard score: `0.0`
- Legitimate measured zero: `True`
- Valid words: `0`

The zero is produced by completed extraction plus official curve scoring; it is not a missing-checkpoint placeholder.
