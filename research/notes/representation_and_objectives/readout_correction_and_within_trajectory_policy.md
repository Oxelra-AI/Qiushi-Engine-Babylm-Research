# fastpath closure and ordinary84 endpoint readout correction and within-trajectory policy

## Correction to earlier analysis dry run

The earlier analysis `task_balanced_readout.py` dry run on the already-known 82→86 payloads was invalid as an item-transition readout. It produced zero common items because it did not force the imported fw globalpiqa relevant substrate loader to resolve prediction/data paths from the correct recorded path base and it read `gain/loss` from top-level comparison records instead of the fw globalpiqa relevant substrate `flip_counts` structure. The apparent R3 pass in `data/readout_dryrun_82to86/` should not be used.

A corrected readout was written at:

- `experiments/archive/representation_and_objectives/scripts/task_balanced_readout_75e45868.py`

It sets `pairwise_item_flip_analysis.ROOT` to the correct recorded path base, reconstructs item rows directly for all four arms, computes retention/discovery against the frozen anchor, and adds a prediction-change-zone signature. The change zone is selected without labels by asking where a candidate's top prediction differs from the frozen anchor; only after selecting the zone are candidate/anchor correctness compared.

## 82→86 validation of the corrected readout

Corrected dry-run output:

- JSON: `experiments/archive/representation_and_objectives/data/readout_dryrun_82to86/task_balanced_readout_d3717bb0.json`
- Markdown: `research/documents/representation_and_objectives/data/readout_dryrun_82to86/task_balanced_readout_ef38cf07.md`

Key result on the known 82→86 evidence:

- coherent86 cheap7: 44.10642857142857
- ordinary86 cheap7: 43.770714285714284
- spanbreak/private control cheap7: 43.121428571428574
- chck82 anchor cheap7: 43.95944987645173
- coherent net discrete items vs chck82: -115
- ordinary86 net discrete items vs chck82: +118
- spanbreak net discrete items vs chck82: -491
- coherent prediction-change benefit: -0.01763533200429382
- ordinary86 prediction-change benefit: +0.010804871348777584
- spanbreak prediction-change benefit: -0.02608788055895011

The corrected readout therefore reproduces the fastpath private localization scientific reading: coherent86 is a strong practical endpoint, but the private fast-path instantiation did not demonstrate broad retained-plus-new competence; its cheap7/overall advantage is macro/column redistribution rather than a verified general stability-plasticity learning principle.

## Policy for the 80→84 four-way test

The earlier interpretation requires correction: chck_80M is not a genuinely independent anchor. It is the same seed43022 scale-1.75 trajectory only two million words before the protected chck_82M point. Therefore:

- a positive 80→84 result would establish within-trajectory checkpoint robustness of the fast-path idea, not trajectory-general stability-plasticity;
- a negative or mixed result closes this fast-path instantiation;
- if coherent80 beats only by cheap7/macro redistribution, or if shuffled80 shares the effect, stop the instantiation and return to representation-level mechanism search;
- only if coherent80 shows task-balanced retained-plus-new competence beyond both ordinary84 and shuffled80, with positive prediction-change-zone benefit above both controls, should the line be preserved as a strong lead and then tested on a truly independent seed or trajectory.
