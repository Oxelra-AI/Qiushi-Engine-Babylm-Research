# representational grounding boundary temporal bottleneck diagnostic

## Hypothesis similarity (before vs after)
- Before vs After same-direction cosine: **0.8543** (±0.1043)
- Before vs Before reversed-direction cosine: **0.9799** (±0.0124)
- Verdict: **before_after_moderately_similar**

The before/after temporal cue in the hypothesis may be distinguishable — switching entity direction changes the representation less than switching temporal reference.

## Context temporal cues (earlier vs later)
- Init vs Later same-direction cosine: **0.9915** (±0.0030)
- Init vs Init reversed-direction cosine: **0.9852** (±0.0074)
- Temporal separation (dir_rev - init_later): **-0.0063**
- Verdict: **temporal_cues_collapsed**

## Full context + hypothesis temporal selection
- Stable world before/after cos: **0.9976**
- Changed world before/after cos: **0.9977**
- Gap (stable - changed): **-0.0001**
- Verdict: **before_after_representationally_identical_in_context**

If the gap is near zero, the pretrained model does not distinguish before/after queries even when the context contains contradictory temporal states.
