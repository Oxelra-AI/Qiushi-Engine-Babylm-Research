# memory vs dense untied 1m profile — State-memory vs close dense-untied control at 1M

Purpose: test the compact causal prefix-memory candidate against a same-core, untied dense GPT2 control before attributing any score change to state persistence.

Evidence JSON: `experiments/archive/initial_model_studies/data/profile_memory_control_1m.json`

Runs:

| alias | run_id | params | architecture | exposure |
|---|---|---:|---|---:|
| dense_untied4x256 | `babylm_compare_dense_untied4x256_1M_control` | 11,613,696 | 4L/256/4H GPT2, untied LM head | 1M words |
| memory4x256_m64 | `babylm_compare_memory4x256_m64_1M` | 12,339,200 | same core + causal prefix memory (M=64), untied LM head | 1M words |

Official fast/local profile at `chck_1M`:

| benchmark | dense_untied4x256 | memory4x256_m64 | memory - dense |
|---|---:|---:|---:|
| BLiMP fast | 53.82 | 55.17 | +1.35 |
| BLiMP Supplement fast | 48.80 | 48.00 | -0.80 |
| EWoK fast | 46.27 | 51.91 | +5.64 |
| Entity Tracking fast | 16.12 | 17.86 | +1.74 |
| COMPS | 49.99 | 50.05 | +0.06 |
| Reading eye-tracking | 9.07 | 8.95 | -0.12 |
| Reading self-paced | 2.40 | 2.32 | -0.08 |

## Scientific reading

This is the first controlled signal that the compact memory route may be doing something more useful than added untied capacity: relative to a same-core untied dense model, memory improves BLiMP, EWoK, and Entity, with a large EWoK gain and a smaller Entity gain. The result is not yet a solution: Supplement and Reading are slightly lower, Entity remains extremely weak in absolute terms, and this is a single 1M checkpoint. It should not be interpreted as SOTA evidence.

The proper next test is a 10M exposure trajectory for both memory and the close dense-untied control. If the memory advantage on EWoK/Entity survives or grows without degrading BLiMP/Supplement/Reading, the route deserves deeper mechanism refinement and eventual full-evaluation comparison. If the advantage vanishes or Reading worsens, the memory design needs reconstruction, likely with better state update/slot structure or curriculum/objective changes.
