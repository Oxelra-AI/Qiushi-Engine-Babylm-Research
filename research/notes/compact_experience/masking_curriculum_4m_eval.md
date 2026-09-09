# masking curriculum 4m eval — Masking-curriculum 4M screen: valid scores, but effect within seed noise

## What was fixed
masking curriculum training done's evaluator returned all-null because it ran `python -m evaluation_pipeline...`
from the eval-repo root (no module) and passed the top-level `evaluation_data` dir,
triggering `KeyError: 'UID'`. masking curriculum 4m eval (`scripts/eval_curriculum_4m_fast.py`)
reuses the known-good cs 4m residualized eval pattern: cwd = `.../babylm-eval/strict`, exact
`evaluation_data/fast_eval/*` (and `full_eval/comps`) paths, official report parsers,
isolated HF caches per work_tag. A BLiMP smoke on wwm_fixed parsed 53.49. All four
arms then evaluated with returncode 0 on every column across both H100s.

## Valid fast-screen scores (chck_4M, single seed 43)
| arm | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | proxy | loss_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_fixed | 53.49 | 49.20 | 50.73 | 18.08 | 50.14 | 35.24 | 6.85 | 28.378 | 6.644 |
| wwm_to_token | 54.14 | 52.80 | 47.27 | 16.91 | 49.96 | 35.73 | 6.86 | 28.372 | 6.718 |
| amlm_hard | 53.87 | 49.20 | 51.36 | 17.96 | 49.98 | 34.73 | 6.94 | 28.413 | 6.508 |
| amlm_hard_switch | 54.08 | 51.20 | 50.55 | 17.37 | 49.97 | 34.74 | 6.94 | 28.501 | 6.660 |

Weighted fast proxy = (3/28)*sum(6 NLP cols) + (1/8)*Reading. Not official Overall
(no SuperGLUE, no AoA, fast subsets).

## Contrasts (proxy delta vs wwm_fixed)
- wwm_to_token: **−0.006** (Supp +3.6, BLiMP +0.65, but EWoK −3.46, Entity −1.17)
- amlm_hard: **+0.034** (EWoK +0.63, BLiMP +0.38, GPIQA −0.52)
- amlm_hard_switch: **+0.122** (Supp +2.0, BLiMP +0.59, Entity −0.71, GPIQA −0.50)
- amlm_hard_switch − amlm_hard: +0.088 (Supp +2.0)
- amlm_hard − wwm_to_token: +0.040 (EWoK +4.09, Entity +1.05 recovered vs pure switch)

## Decisive caveat — effect is within seed noise, and regime is wrong
1. **Magnitude**: proxy deltas −0.006..+0.122 are the same scale as cs 4m residualized eval's
   random-reference spread (r_b − r_a proxy = −0.060). A single seed cannot separate
   a +0.12 proxy from seed variance.
2. **Regime mismatch**: the live leader trains **10 epochs over ~10M words (100M
   exposure)** with WWM for epochs 1–7 then token masking for epochs 8–10. These
   4M runs are ~0.4 of one pass; the "70% switch" here happens after 2.8M words of a
   single partial epoch, which is NOT the leader's 7-of-10-epoch curriculum. The
   masking-curriculum hypothesis has effectively not been tested in its intended
   regime.

## Internal dynamics do not yet explain transfer (dynamics_traces.jsonl)
- AMLM operated as designed: effective mask rate 0.49→0.31 as mask_prob 0.26→0.15
  over 39 updates; switch arms moved wwm→token at chck_3M (earlier analysis ≈ 0.7).
- BUT the frequency-band diagnostic is uninformative at 4M: mid/low-band masked
  accuracy = 0.0 for all arms/checkpoints; high-band acc only 0.03–0.07; loss bands
  ~identical across arms (low ~11.9, mid ~10.1, high ~5.7). At 34M params / 4M words
  the model predicts too little for these bands to discriminate mechanisms.

## Scientific conclusion for routing
The masking-curriculum idea is NOT falsified, but the 4M single-seed screen is
underpowered and off-regime. Two things must happen before any scaling claim:
(a) run the actual leader-style regime — full 10M pool, 10 epochs, WWM epochs 1–7 →
token epochs 8–10, with AMLM difficulty decay as a factor — so the curriculum is
tested where it is designed to act; and (b) replicate the most promising arm(s)
across ≥3 seeds so a proxy delta can be distinguished from seed noise, and enrich the
dynamics probe so band accuracy is non-degenerate (larger probe set, or use loss/rank
rather than top-1 accuracy).

## Artifacts
- Scores: `data/curriculum_4m_eval/curriculum_4m_eval_summary.json`
- Per-arm: `data/curriculum_4m_eval/per_arm/*.json`
- Dynamics: `training/runs/step009_*_4m_seed43/dynamics_traces.jsonl`
