# adapter 20M closure — Adapter 20M evaluation: decisive negative, route assessment needed

## Results summary

Both zero-output bottleneck adapter arms trained for 20M words (standalone cosine
schedule, 506 steps) with gradient checkpointing on the exact spatial repair route status legal
compact-view-reinvest substrate.

| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial repair route status 20M | 59.69 | 55.45 | 50.73 | 18.65 | 50.26 | 34.20 | 8.67 | 39.66 | — |
| Muon wd-matched 20M | 61.15 | 58.10 | 50.70 | 20.90 | 50.45 | 38.61 | 7.26 | 41.02 | +1.36 |
| **Adapter b=64** | 53.99 | 50.12 | 49.74 | 17.20 | 49.83 | 32.77 | 7.12 | **37.25** | **−2.41** |
| **Adapter b=128** | 53.81 | 50.13 | 51.08 | 17.07 | 50.34 | 33.75 | 7.17 | **37.62** | **−2.04** |

## Mechanistic observations

1. **First loss matches**: both arms reproduce 9.837543 exactly (function-preserving init verified)
2. **Final loss is high**: adapter64 6.898, adapter128 6.877 — similar to LAMB failures (~6.90)
3. **Regression is broad**: BLiMP −5.7/−5.9, Supplement −5.3/−5.3, Reading −1.5/−1.5
4. **All seven columns negative for adapter64**, six of seven for adapter128 (EWoK +0.35)
5. **Adapter recruitment measurement failed** because of an execution failure; no recruitment result was obtained

## Comparison with other 20M standalone screens

| Intervention | cheap7 | Δ vs spatial repair route status 20M |
|---|---:|---:|
| Muon matched-decay | 41.02 | +1.36 |
| RTD/GDES | 39.77 | +0.11 |
| spatial repair route status (reference) | 39.66 | — |
| **Adapter b=128** | 37.62 | −2.04 |
| **Adapter b=64** | 37.25 | −2.41 |
| LAMB lr=0.005 (14M) | 37.30 | −2.36 |
| LAMB lr=0.007 | 37.17 | −2.50 |

## Decision criteria assessment

- ✓ First loss matches spatial repair route status (9.837543)
- ? Adapter recruitment (measurement failed; training loss suggests some activity)
- ✗ Cheap7 ≥ spatial repair route status 20M: 37.25/37.62 << 39.66
- ✗ No dominant damage: BLiMP −5.7/−5.9, Supplement −5.3
- ? Trunk displacement: could not measure

**Criteria 3 and 4 clearly fail. The adapter route at 20M standalone is closed.**

## Open confounds

1. **Gradient checkpointing**: used here but in no previous 20M screen. Unlikely to
   cause a 2-point drop (should be numerically identical), but untested.
2. **Standalone 20M schedule**: the adapter starts at zero output and needs time to
   recruit. The standalone schedule (warmup 30 steps, cosine to 0 over 506 steps)
   gives much less effective learning than spatial repair route status's first 20M (warmup 152 steps, LR
   still ~0.001 at earlier analysis). Muon overcame this via more efficient per-step updates.

## Files

- Training runs: `training/runs/adapter64_seed43022_20M/`,
  `training/runs/adapter128_seed43022_20M/`
- Evaluation: `data/adapter_20M_eval/`
- Architecture: `scripts/adapter_modeling.py`
- Trainer: `scripts/adapter_trainer.py`
- Evaluator: `scripts/eval_adapter_20m.py`
- Mechanical verification: `data/adapter_mechanics/`
- Experiment design: `notes/adapter_experiment_design.md`
