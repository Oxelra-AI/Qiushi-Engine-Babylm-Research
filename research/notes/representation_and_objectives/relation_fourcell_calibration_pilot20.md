# relation filtered pair pools — no-update four-cell calibration on active relation pairs

Created: 2026-08-31T04:07:56Z  Runtime: 30.1 s

## What was scored
For each pair, the scorer masked the saved target token positions and computed `M = s(Ca,Ta)+s(Cb,Tb)-s(Ca,Tb)-s(Cb,Ta)` with frozen checkpoint weights. Two controls used same-stratum target and context permutations. No training or optimizer update was performed.

Selected pairs: **102**; by family: `{'causal_connector': 15, 'comparative': 17, 'negation': 17, 'physical_change': 17, 'spatial': 19, 'temporal': 17}`
Donor assignment: `{'pairs': 120, 'assigned': 102, 'failed': 18, 'level_counts': {'exact': 362, 'loose': 553}}`
Context verification: `{'input_pairs': 102, 'good_pairs': 102, 'bad_contexts': 0, 'bad_examples': []}`

## Arm means for four-cell margin
| margin | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 11.823458108119667 | 11.572386627377687 | 11.672216265644952 | compact > interleaved > rowblock | 0.25107148074197916 |
| tperm_m | 1.075330408180461 | 0.9856527704818576 | 1.2184709982544768 | interleaved > compact > rowblock | 0.23281822777261918 |
| cperm_m | 1.3634654099450392 | 1.3439213113457549 | 1.3966804675611795 | interleaved > compact > rowblock | 0.052759156215424596 |

## Family means for true four-cell margin
| family | n | compact | rowblock | interleaved | true rank | true range | target-perm range | context-perm range |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| causal_connector | 15 | 15.17656257947286 | 14.575358239809672 | 14.772005247076352 | compact > interleaved > rowblock | 0.6012043396631874 | 1.1033122857411701 | 0.2602498213450115 |
| comparative | 17 | 10.805089656482725 | 10.307114436893778 | 10.3886440990164 | compact > interleaved > rowblock | 0.4979752195889464 | 0.7239271928282345 | 0.22073743623845718 |
| negation | 17 | 10.691865717663484 | 10.53349033085739 | 10.696921043536243 | interleaved > compact > rowblock | 0.16343071267885279 | 0.3386662707609289 | 0.44577939370099234 |
| physical_change | 17 | 10.917116802843179 | 10.723623349605237 | 10.873456336557865 | compact > interleaved > rowblock | 0.19349345323794154 | 0.3584884335012997 | 0.06892552095301019 |
| spatial | 19 | 11.999556863876549 | 11.552798614690179 | 11.704975231403584 | compact > interleaved > rowblock | 0.4467582491863702 | 0.41496827727869934 | 0.4990043389169793 |
| temporal | 17 | 11.724322406684651 | 12.097530042423921 | 11.95812269694665 | rowblock > interleaved > compact | 0.37320763573926996 | 0.4738157777225269 | 0.45556988435633006 |

## Relation-oriented paired deltas
### rowblock_minus_compact
- true_m: mean=-0.25107148074198005 stderr=0.17427146240295988 success_gt0=0.45098039215686275 n=102
- tperm_m: mean=-0.08967763769860361 stderr=0.1914879513537909 success_gt0=0.47058823529411764 n=102
- cperm_m: mean=-0.019544098599284302 stderr=0.17253961237537374 success_gt0=0.5294117647058824 n=102

### interleaved_minus_compact
- true_m: mean=-0.1512418424747154 stderr=0.1782659383008162 success_gt0=0.4411764705882353 n=102
- tperm_m: mean=0.1431405900740156 stderr=0.1922973783482615 success_gt0=0.49019607843137253 n=102
- cperm_m: mean=0.033215057616140325 stderr=0.1903659170225056 success_gt0=0.47058823529411764 n=102

### rowblock_minus_interleaved
- true_m: mean=-0.09982963826726465 stderr=0.17298684070337397 success_gt0=0.4803921568627451 n=102
- tperm_m: mean=-0.2328182277726192 stderr=0.16529286077961558 success_gt0=0.46078431372549017 n=102
- cperm_m: mean=-0.05275915621542463 stderr=0.17302645973323333 success_gt0=0.45098039215686275 n=102

## Files
- JSON summary: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration_pilot20/fourcell_calibration_summary.json`
- Pair scores: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration_pilot20/fourcell_pair_scores.jsonl`
- CSV: `experiments/archive/representation_and_objectives/data/relation_fourcell_calibration_pilot20/fourcell_pair_scores_summary.csv`
