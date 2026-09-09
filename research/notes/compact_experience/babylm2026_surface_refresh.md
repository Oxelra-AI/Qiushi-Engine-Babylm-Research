# babylm2026 surface refresh — BabyLM 2026 Strict-Small surface refresh

Fetched at UTC `2026-08-24T02:39:18.641042+00:00` from `https://babylm-community-babylm-leaderboard-2026.hf.space/config`; public Space `https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026`.
Local eval source used for rule snippets: `experiments/archive/initial_model_studies/repos/babylm-eval` at commit `6f825c291e2c4c78ad33b1935fd64d45f52642dc` with remote `https://github.com/babylm-org/babylm-eval.git`.

## Live Strict-Small target surface

Strict-Small rows parsed: `122`. Current top visible row: `wwm_curriculum_simplification_40k` with Overall `41.8`, NLP `52.97`, Human-like `2.71`.

| rank | model | Overall | NLP | Human-like | BLiMP | Supp | EWoK | Entity | COMPS | GlobalPIQA | (Super)GLUE | Reading | AoA |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | wwm_curriculum_simplification_40k | 41.8 | 52.97 | 2.71 | 67.2 | 56.01 | 56.07 | 28.45 | 53.57 | 39.67 | 69.79 | 5.42 | 0.0 |
| 2 | RecGPT-10M | 41.53 | 52.4 | 3.46 | 73.11 | 61.73 | 52.62 | 16.59 | 55.43 | 40.68 | 66.64 | 6.92 | 0.0 |
| 3 | Wordpiece-24-4 | 41.31 | 52.85 | 0.92 | 70.2 | 66.52 | 52.1 | 21.82 | 54.04 | 36.08 | 69.18 | 1.84 | 0.0 |
| 4 | Bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s0 | 40.94 | 51.81 | 2.92 | 67.85 | 65.21 | 53.0 | 21.47 | 52.24 | 37.59 | 65.29 | 5.84 | 0.0 |
| 5 | bb26_claim_compagg_v22_v15_relay_synp050_synkeep010_s0 | 40.93 | 51.79 | 2.92 | 67.85 | 65.21 | 53.0 | 19.9 | 52.24 | 38.65 | 65.66 | 5.84 | 0.0 |
| 6 | BabySteps_MurphysLaw-10M-mixed | 40.86 | 53.59 | -3.72 | 71.6 | 63.93 | 51.94 | 27.95 | 53.13 | 36.15 | 70.44 | 7.67 | -15.1 |
| 7 | instanton-hybrid | 40.78 | 51.53 | 3.18 | 72.13 | 60.86 | 50.15 | 19.87 | 52.96 | 37.14 | 67.58 | 6.35 | 0.0 |
| 8 | bb26_claim_agg_synonly_s0 | 40.67 | 51.46 | 2.91 | 67.13 | 62.3 | 52.28 | 19.17 | 52.22 | 40.11 | 66.99 | 5.82 | 0.0 |
| 9 | deberta-base-75k-sam_ext-s1 | 40.62 | 48.74 | 12.19 | 67.95 | 52.99 | 51.25 | 19.66 | 51.77 | 32.64 | 64.95 | 1.49 | 22.9 |
| 10 | bb26_claim_compagg_clean_union_s1 | 40.59 | 51.3 | 3.09 | 68.35 | 64.03 | 51.98 | 21.26 | 52.37 | 35.12 | 65.98 | 6.18 | 0.0 |
| 11 | leviosa-baseline-medium | 40.55 | 51.39 | 2.6 | 74.02 | 58.41 | 52.8 | 19.4 | 52.34 | 38.56 | 64.18 | 5.2 | 0.0 |
| 12 | bb26_claim_compagg_w025_synp_w025_s0 | 40.52 | 51.14 | 3.33 | 68.2 | 61.64 | 50.74 | 19.28 | 52.15 | 39.65 | 66.32 | 6.66 | 0.0 |
| 13 | morpheus-10M-v5 | 40.47 | 51.64 | 1.35 | 72.53 | 58.29 | 53.2 | 20.38 | 54.67 | 38.65 | 63.78 | 2.71 | 0.0 |
| 14 | instanton-hybrid-dialogue | 40.43 | 50.96 | 3.57 | 70.77 | 61.06 | 52.9 | 21.93 | 53.15 | 29.25 | 67.64 | 7.14 | 0.0 |
| 15 | ujjwal-very-bored | 40.4 | 51.91 | 0.13 | 69.85 | 64.84 | 51.33 | 21.79 | 51.8 | 34.69 | 69.03 | 0.27 | 0.0 |

## Score arithmetic recovered from local official code

For non-multilingual tracks the displayed Overall is the mean of nine entries: BLiMP, BLiMP Supplement, EWoK, Entity Tracking, COMPS, (Super)GLUE, GlobalPIQA, Reading, and AoA. The NLP average is the first seven entries. The Human-like average is Reading and AoA. GlobalPIQA is the mean of parallel and nonparallel subtasks; Reading is the mean of self-paced and eye-tracking components. Missing submitted tasks score as zero in the leaderboard loader, so a local coordinate must keep all nine entries explicit.

## Evaluation and checkpoint-compatibility implications

- The target to beat is not the INITIAL_MODEL_STUDIES internal 40.7028 coordinate; it is the live Strict-Small Overall around the parsed leader above, with current visible leader `wwm_curriculum_simplification_40k` at 41.8.
- AoA must be generated in the current accepted format; older outputs without the current surprisal key are zeroed by the loader.
- Entity Tracking must use the currently accepted filtered result marker; older entity outputs can be zeroed by current submission processing.
- Any new run must preserve exact checkpoint naming and all intermediate checkpoint outputs needed by the official scripts; fast-only local slices are useful for research but do not replace the full official-compatible coordinate.

## Files written

- `experiments/archive/compact_experience/data/babylm2026_surface/leaderboard_config.json`
- `experiments/archive/compact_experience/data/babylm2026_surface/leaderboard_parsed.json`
- `experiments/archive/compact_experience/data/babylm2026_surface/strict_small_top20.json`
- `experiments/archive/compact_experience/data/babylm2026_surface/score_source_snippets.json`
