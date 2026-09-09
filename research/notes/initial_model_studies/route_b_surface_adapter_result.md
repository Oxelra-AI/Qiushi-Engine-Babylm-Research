# route b surface adapter result — Route B surface/morphology adapter 10M result

## Evidence

- Trainer: `scripts/route_b_adapter_screen.py`
- Verification: `data/route_b_adapter_verification.json`
- Training runs:
  - baseline: `training/runs/route_b_baseline`
  - surface: `training/runs/route_b_surface`
  - token-ID control: `training/runs/route_b_token_id`
- Evaluation script: `scripts/eval_route_b_threearm.py` plus resume script `scripts/eval_route_b_tokenid_resume.py`
- Scores: `data/route_b_threearm_scores.json`

## Experimental contract

Three matched 10M arms used protected DeBERTa-v2 8×480 WWM, official corpus only, baseline16k tokenizer, seed 42, batch 256, exact 10M word exposure, same masking and corpus order. The surface adapter and token-ID adapter have equal trainable parameters; fused HF export was verified numerically before evaluation.

## Training dynamics

| arm | final loss | gate |
|---|---:|---:|
| baseline | 6.580070 | — |
| surface | 6.580061 | 0.000922 |
| token-ID | 6.580199 | 0.000705 |

The adapters stayed extremely small and did not change training loss meaningfully.

## Official-compatible 10M scores

| column | baseline | surface | token-ID |
|---|---:|---:|---:|
| BLiMP | 54.14 | 54.12 | 54.16 |
| Supplement | 50.91 | 51.08 | 51.04 |
| Entity | 16.76 | 16.84 | 16.82 |
| COMPS | 49.87 | 50.34 | 49.98 |
| EWoK | 49.24 | 48.64 | 51.31 |
| GlobalPIQA parallel | 21.36 | 22.33 | 20.39 |
| GlobalPIQA nonparallel | 51.00 | 54.00 | 51.00 |
| GlobalPIQA mean | 36.18 | 38.165 | 35.695 |
| Reading mean | 7.09 | 7.09 | 7.09 |

## Decisive deltas

Surface minus baseline:

- Entity +0.08
- COMPS +0.47
- EWoK -0.60
- GlobalPIQA mean +1.985
- Reading 0.00
- BLiMP -0.02
- Supplement +0.17

Surface minus token-ID:

- Entity +0.02
- COMPS +0.36
- EWoK -2.67
- GlobalPIQA mean +2.47
- Reading 0.00
- BLiMP -0.04
- Supplement +0.04

## Scientific interpretation

The surface-sharing adapter does **not** support the morphology/surface hypothesis in this implementation.

The predefined target-column requirement was: surface should outperform both baseline and parameter-matched token-ID control on Entity/EWoK/COMPS while preserving protected strengths. It fails this test:

1. Entity movement is negligible (+0.08 vs baseline, +0.02 vs token-ID).
2. EWoK is worse than baseline and far worse than token-ID (-2.67 surface minus token-ID).
3. The main gain is GlobalPIQA, especially nonparallel, a pattern already known from exported base threeway official interpretation to be insufficient and often unrelated to the missing mechanism.
4. Adapter gates are tiny and losses are almost identical, so the representation change did not become a strong learned component.
5. Token-ID control beating surface on EWoK argues against surface/morphology sharing as the driver.

Route B should not be scaled to 100M in its current form.

## Consequence for current research posture

This closes the active surface-adapter screen as negative or at least not scale-worthy. It should be included in the systematic earlier analysis–route b surface adapter result compression as another instance of the repeated pattern: local representation/objective changes can move GlobalPIQA or small columns, but have not produced robust Entity/EWoK improvement.

The proposed follow-up is complete 9/9 evaluation of the strongest accumulated 100M candidates and synthesis of the surface-adapter evidence into supported findings, closed routes, recurring trade-offs and unresolved scientific questions.
