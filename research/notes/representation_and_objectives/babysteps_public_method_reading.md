# babysteps public method reading BabySteps public method reading

Source read: `data/external/svsatheesh-BabySteps-MurphysLaw-10M-mixed-Hugging-Face.md`.

## What it establishes
`BabySteps_MurphysLaw-10M-mixed` is a strict-small public model at Overall 40.86 with a component pattern relevant to us: high Supplement 63.93 and Reading 7.67, high BLiMP 71.6, Entity 27.95, COMPS 53.13, but only EWoK 51.94, GlobalPIQA 36.15, and AoA -15.1. It uses **only the official BabyLM 2026 strict-small corpus**, not FineWeb or generated data. The method is a GPT-BERT hybrid masked/causal architecture (39.3M params, 12 layers, hidden 384, 16k BPE), AdaMuon optimizer, tuned LR, and tail-weight averaging over the final 20%.

## What it does NOT establish
It is not evidence that FineWeb or source-by-rewrite will improve our model. Its model card explicitly says its team found their data-centric interventions did not beat a faithful GPT-BERT reconstruction with Muon-family optimization and LR sweep. Thus this source should not be used to justify a FineWeb route directly.

## Why it still matters
It provides existence evidence that high Supplement/Reading are not intrinsically incompatible with better BLiMP/Entity/COMPS under strict-small constraints. This matches the babysteps public method reading public tradeoff analysis: knowledge-cluster/Supplement and knowledge-cluster/Reading correlations in top30 are near zero. The implication is not to copy BabySteps wholesale, but to protect our COMPACT_EXPERIENCE clean-Qwen strengths while testing a narrow factual-breadth intervention.

## Relation to current route
- BabySteps suggests an optimizer/architecture route (GPT-BERT + AdaMuon + tail averaging) may be worth later construction if data routes stall, especially because it has high BLiMP/COMPS/Entity and high Supplement/Reading.
- But BabySteps does not solve the precise leader gap: it has lower Overall than ours mainly because AoA is negative and EWoK/GlobalPIQA remain below the go76dof leader.
- For the immediate next H100 decision, the pending semantic-view no-AoA result and the FineWeb source-pool quality/scale should dominate.
