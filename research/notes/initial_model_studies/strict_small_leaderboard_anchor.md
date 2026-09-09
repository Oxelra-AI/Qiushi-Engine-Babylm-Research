# strict small leaderboard anchor — Strict-Small leaderboard anchor

The parsed leaderboard snapshot contains separate `strict` and `strict-small` sections. The high `strict` rows around Overall 45+ are **not** Strict-Small. The current parsed `strict-small` top row remains:

| rank | model | Overall | NLP | Human-like | BLiMP | Supp | EWoK | Entity | COMPS | GlobalPIQA | SuperGLUE | Reading | AoA |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `wwm_curriculum_simplification_40k` | 41.80 | 52.97 | 2.71 | 67.20 | 56.01 | 56.07 | 28.45 | 53.57 | 39.67 | 69.79 | 5.42 | 0.00 |
| 2 | `RecGPT-10M` | 41.53 | 52.40 | 3.46 | 73.11 | 61.73 | 52.62 | 16.59 | 55.43 | 40.68 | 66.64 | 6.92 | 0.00 |
| 3 | `Wordpiece-24-4` | 41.31 | 52.85 | 0.92 | 70.20 | 66.52 | 52.10 | 21.82 | 54.04 | 36.08 | 69.18 | 1.84 | 0.00 |

Protected baseline16k DeBERTa-v2 9/9 coordinate: Overall 40.5269, NLP 51.0424, Human-like 3.7228. Gap to Strict-Small top row `wwm_curriculum_simplification_40k`: Overall -1.2731, NLP -1.9276, Human-like +1.0128; column gaps are BLiMP -0.44, Supplement +3.87, EWoK -3.88, Entity -5.83, COMPS -1.38, GlobalPIQA -4.035, SuperGLUE -1.7682, Reading +2.20, AoA -0.1745.

Interpretation: the target to beat remains the Strict-Small row with Overall 41.80 in this parsed snapshot, but the route should be chosen by the multi-column gap profile, not by recovering only Entity. A useful mechanism must extend beyond same-form entity tightening into Entity + EWoK + GlobalPIQA while preserving Supplement/Reading advantages.

Source: `data/leaderboard_probe/leaderboard_config_parsed.json` lines 630-930 and `data/debertav2_b256_true_9of9_coordinate.json`.
