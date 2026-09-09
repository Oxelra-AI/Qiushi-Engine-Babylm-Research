# debertav2 b256 full ewok word tokenize score — first complete local 9/9 coordinate for baseline16k DeBERTa-v2

Evidence JSON: `experiments/archive/initial_model_studies/data/debertav2_b256_true_9of9_coordinate.json`

## 9 columns

| column | score |
|---|---:|
| BLiMP | 66.7600 |
| BLiMP Supplement | 59.8800 |
| EWoK | 52.1900 |
| Entity Tracking | 22.6200 |
| COMPS | 52.1900 |
| (Super)GLUE | 68.0218 |
| GlobalPIQA | 35.6350 |
| Reading | 7.6200 |
| AoA | -0.1745 |

## Aggregates

- NLP Average: **51.0424**
- Human-like Average: **3.7228**
- Overall Average: **40.5269**

## Leaderboard snapshot comparison

- Insertion rank among parsed strict-small rows with non-null Overall: **12** / 16
- Visible reference `wwm_curriculum_simplification_40k` Overall: 41.8000; our gap: -1.2731

## Measurement notes

- EWoK uses local gated parquet with official vocab and `nltk.word_tokenize`; 4374 raw items -> 3809 retained items -> 7618 swapped lines.
- AoA uses direct local checkpoint loading (`hf_model/chck_*M`) with official AoA target words, surprisal extraction and curve-fitness scoring; repaired AoA is negative (-0.1745), not the old flat 0.0 artifact.
- Re-fetch leaderboard before any external claim; this coordinate is a research measurement, not a submission result.

## Main gaps vs `wwm_curriculum_simplification_40k` reference

- BLiMP: -0.4400
- BLiMP Supplement: +3.8700
- EWoK: -3.8800
- Entity Tracking: -5.8300
- COMPS: -1.3800
- (Super)GLUE: -1.7682
- GlobalPIQA: -4.0350
- Reading: +2.2000
- AoA: -0.1745
- Overall Average: -1.2731
