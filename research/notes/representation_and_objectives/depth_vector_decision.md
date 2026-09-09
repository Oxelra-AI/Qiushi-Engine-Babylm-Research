# depth vector decision depth vector decision

Authoritative 12x384 legal40k depth seed43022 Overall: 41.027595831353.
Matched legal40k 8x480 seed43022 Overall: 41.140577774478.
Delta depth minus 8x480: -0.112981943125.
Delta depth minus visible 41.80 leader: -0.772404168647.

## Depth scores
- BLiMP: 67.475099616887 (delta vs 8x480 -0.474265456145; delta vs leader +0.275099616887)
- Supplement: 60.139549673292 (delta vs 8x480 -0.826020107435; delta vs leader +4.129549673292)
- EWoK: 50.547190046911 (delta vs 8x480 -0.926794347891; delta vs leader -5.522809953089)
- Entity: 27.171427053823 (delta vs 8x480 -0.032247198802; delta vs leader -1.278572946177)
- COMPS: 52.702343210167 (delta vs 8x480 +1.027680086409; delta vs leader -0.867656789833)
- SuperGLUE: 68.227508256701 (delta vs 8x480 -0.282728998989; delta vs leader -1.562491743299)
- GlobalPIQA: 35.635922330097 (delta vs 8x480 +0.970873786408; delta vs leader -4.034077669903)
- Reading: 7.349322294299 (delta vs 8x480 -0.473335251683; delta vs leader +1.929322294299)
- AoA: 0.000000000000 (delta vs 8x480 +0.000000000000; delta vs leader +0.000000000000)

Interpretation: depth alone does not improve the complete acceptance coordinate. It modestly lifts GlobalPIQA/Entity/COMPS relative to the 8x480 seed43022 baseline but gives back Supplement, EWoK, SuperGLUE, and Reading, leaving Overall below both the matched baseline and the public frontier. This supports stopping depth-alone reproduction and using the repaired exact-prefix SGCR experiment as the next single matched expensive test.

JSON: experiments/archive/representation_and_objectives/data/depth_vector_decision/depth_vector_decision.json
