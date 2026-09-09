# fastpath closure and ordinary84 endpoint fast-path closure and ordinary84 endpoint update

## Fast-path result

The fixed four-way readout decides: **STOP_FASTPATH_INSTANTIATION**.

| arm | cheap7 | net items vs anchor80 | prediction-change benefit |
|---|---:|---:|---:|
| anchor80 | 43.813571 | 0 | 0 |
| coherent80 | 43.670000 | +63 | +0.009828 |
| shuffled80 | 42.784286 | -127 | -0.007644 |
| ordinary84 | 44.122143 | +306 | +0.023470 |

R3 discovery wins vs both controls: 0/6. Coherent80's unique gain minus unique loss against both controls is +3.

This closes the private fast-path instantiation as a mechanism: the positive 82→86 endpoint did not recur as retained-plus-new competence at 80→84, and the same-trajectory ordinary continuation is stronger than the private coherent arm.

## Practical endpoint ranking after fastpath closure and ordinary84 endpoint

| endpoint | cheap7 | SuperGLUE | AoA used | projected / measured Overall | reading |
|---|---:|---:|---:|---:|---|
| coherent86 | 44.106429 | 69.777968–69.842796 | 0.0 | 42.058108–42.065311 | strongest practical candidate; mechanism unresolved |
| ordinary84 | 44.122143 | 69.287536 | 0.0 projected | 42.015837 | simple same-trajectory checkpoint; below coherent86, above chck82 if AoA≈0 |
| chck82 protected | 43.959450 | 69.766181 | 0.0 | 41.942481 | packaged fallback |
| ordinary85 | 44.089286 | unmeasured | — | — | cheap7 below ordinary84 |
| ordinary86 | 43.770714 | unmeasured | — | — | cheap7 below ordinary84 and coherent86 |

## ordinary84 projection

- tie_chck82_aoa0_sg: `68.62733050647392`
- tie_coherent86_min_aoa0_sg: `69.66796826428703`
- tie_coherent86_max_aoa0_sg: `69.73279617564907`
- ordinary84 measured SuperGLUE: `69.28753642850036`
- ordinary84 projected Overall with AoA=0: `42.01583738094448`

## Item-level reading

ordinary84 vs chck82: -58 net discrete items despite +0.1888 discrete payload-mean delta.
ordinary84 vs coherent86: +57 net discrete items and only +0.0208 discrete payload-mean delta.
This is trajectory endpoint selection / redistribution, not a new data-efficient learning principle or a repair of the context-conditioned binding deficit.

## Artifacts

- fixed readout: `experiments/archive/representation_and_objectives/data/fastpath80_fourway_readout/task_balanced_readout.json`
- ordinary84 synthesis: `experiments/archive/representation_and_objectives/data/ordinary84_candidate_synthesis/ordinary84_candidate_synthesis.json`
- ordinary84 SuperGLUE payload: `experiments/archive/representation_and_objectives/data/ordinary84_superglue_eval/per_target/ordinary84_superglue.json`
- ordinary85 cheap7 payload: `experiments/archive/representation_and_objectives/data/ordinary85_cheap7_eval/per_target/ordinary85_cheap7.json`
- JSON: `experiments/archive/representation_and_objectives/data/final_synthesis/final_synthesis.json`
