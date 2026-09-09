# S1, cadence and CLBH: evidence and route comparison

## S1 evidence completion changes no route priority

S1 12×384 at 100M is now understood well enough for route judgment even though official-compatible AoA cannot be computed from existing artifacts.

Evidence files:

- S1 available coordinate: `data/s1_100m_available_coordinate.json`
- S1 full EWoK: `data/s1_s2_100m_ewok_scores.json`
- S1 SuperGLUE: `data/s1_s2_100m_superglue_results.json`
- S1 AoA attempt: `scripts/s1_aoa_direct_local_ckpts.py`
- S1 checkpoint artifact fact: `training/runs/babylm_leadershape_s1_100M_aligned_micro128/scientific_metrics.json`

S1 has these eight available columns:

| Column | protected 8×480 | S1 12×384 | S1 - protected |
|---|---:|---:|---:|
| BLiMP | 66.76 | 66.84 | +0.08 |
| Supplement | 59.88 | 60.31 | +0.43 |
| EWoK | 52.19 | 52.02 | -0.17 |
| Entity | 22.62 | 20.24 | -2.38 |
| COMPS | 52.19 | 52.26 | +0.07 |
| SuperGLUE | 68.02 | 65.12 | -2.90 |
| GlobalPIQA | 35.64 | 37.61 | +1.97 |
| Reading | 7.62 | 7.25 | -0.37 |

Across these eight columns, S1 is lower than protected 8×480 by 3.27 summed points, or -0.408 average. Since S1 lacks chck_1M through chck_9M, official strict-small AoA cannot be scored from current files. If one inserted AoA=0, S1 Overall would be about 40.18, still below the protected 40.527. To equal protected Overall from its eight-column sum, S1 would need AoA about +3.09, far outside the observed AoA scale in these experiments and not a defensible substitution for the missing trajectory.

Therefore S1 is not an internal route advance. Its useful scientific message is narrower: deeper/narrower leader shape raises GlobalPIQA but lowers Entity, EWoK, SuperGLUE, and Reading relative to protected 8×480. It does not solve the relation-knowledge cluster.

## Recent closed mechanisms share one failure pattern

Cadence: the matched all-arm experiment showed every cadence factor damages EWoK at 1M. Length-only gives Entity +0.64 but EWoK -3.00; mask-decay gives Entity +1.11 but EWoK -1.63; combined gives Supplement +4.00 but EWoK -3.54 and Entity -0.44. This is optimization emphasis, not the missing reusable relation-knowledge mechanism.

CLBH: the repaired binding-switch test uses semantic token IDs such as `Ġhat` and `Ġball`, verifies both properties in the same context, and changes only the queried entity. The trained pointer copies context tokens but does not switch with the entity-property binding: copy discrimination stays at 0.5, mean copy margin is only +0.0053, and many pairs favor the same property regardless of query entity. CLBH is an output-copy bias, not relation binding.

S1: leader shape is not enough. It improves GlobalPIQA but worsens Entity and SuperGLUE, and does not improve EWoK.

These three findings rule out three tempting escapes: schedule tuning, output copying, and a slightly different already-trained coordinate.

## Data-route evidence should not be erased, but simple forms are closed

The data side remains scientifically important because the public leader profile still points to simplification-pair data as the least explained component. However, the simple forms already tested should not be repeated:

- Pair-adjacent vs pair-shuffled had an early modest signal under a pure WikiAuto-pair condition, but the absolute model was weak and later official mixtures/cross-view variants did not preserve a stable route.
- FineWeb relation-explicit selection had a real 1M EWoK bump (+3.63) but no Entity movement and no stable 3M trajectory.
- FineWeb paired-restatement under repaired no-truncation/equal-row conditions did not beat the stronger controls.

So the data route is not closed in all possible forms, but generic relation density, naive adjacency, and simple paired-restatement are not enough.

## What next work must target

The unresolved target is joint movement of Entity, EWoK, and GlobalPIQA without giving away the protected strengths in Supplement and Reading. A next direction is worth constructing only if its first small test can show relation-specific behavior rather than a generic copy, topic, grammar, or data-quality effect.

Two families remain plausible enough for serious comparison:

1. A relation-representation architecture inside the encoder, not at the output head. It should bind entity identity and property/value roles before candidate scoring. Examples include soft latent binding slots with role-sensitive writes and reads, or relation-factorized attention where entity and property streams interact before the MLM head. The repaired binding-switch test from clbh binding repair result should be reused, but the mechanism must alter the hidden representation or candidate margins in an entity-conditioned way, not merely add a copied token bias.

2. A rebuilt data-mechanism route that learns from why earlier pair data failed. It should not use broad relation filters or naive adjacency. A stronger version would isolate high-retention, entity/number/predicate-preserving paired views; compare local paired, dispersed paired, and shuffled controls with the same text multiset; and measure whether same-content redundancy produces entity-conditioned changes on held-out binding probes before any 10M/100M run.

The proposed follow-up compares these two families against the accumulated closures and choose one concrete small experiment that can change the route. The test must include a parameter- or content-matched control and at least two seeds before any route is scaled.
