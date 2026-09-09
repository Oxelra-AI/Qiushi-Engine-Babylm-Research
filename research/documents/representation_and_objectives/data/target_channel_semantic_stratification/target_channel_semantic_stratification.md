# target channel semantic structure target-channel semantic stratification

JSON: `experiments/archive/representation_and_objectives/data/target_channel_semantic_stratification/target_channel_semantic_stratification.json`

Positive values mean that removing source-absent labels harms the evaluated compact-side events more than removing the matched copied whole-word labels.

## Focus: chck_20M source_absent_content

### wholeword copied control plan
- eval_set `train_fixed`
  - all: n=4096, pieces=6738, delta=0.080267, boot[0.058738, 0.100172], frac_gt0=1.0
  - relational_or_event_state: n=533, pieces=582, delta=0.302805, boot[0.217176, 0.391957], frac_gt0=1.0
  - not_relational_or_event_state: n=3563, pieces=6156, delta=0.059228, boot[0.03493, 0.083782], frac_gt0=1.0
  - ordinary_nonrel_nonentity: n=3238, pieces=5248, delta=0.094694, boot[0.070575, 0.119631], frac_gt0=1.0
  - capitalized_or_number: n=282, pieces=851, delta=-0.124103, boot[-0.185789, -0.068747], frac_gt0=0.0

### source disjoint target channel result
- eval_set `doc_disjoint_all_accepted`
  - all: n=1012, pieces=1619, delta=0.035586, boot[-0.007037, 0.077787], frac_gt0=0.953
  - relational_or_event_state: n=97, pieces=103, delta=0.240766, boot[0.034848, 0.458488], frac_gt0=0.982
  - not_relational_or_event_state: n=915, pieces=1516, delta=0.021646, boot[-0.019648, 0.064242], frac_gt0=0.838
  - ordinary_nonrel_nonentity: n=837, pieces=1344, delta=0.024428, boot[-0.023968, 0.070145], frac_gt0=0.841
  - capitalized_or_number: n=60, pieces=151, delta=0.026873, boot[-0.141642, 0.196358], frac_gt0=0.633
- eval_set `doc_disjoint_quality`
  - all: n=71, pieces=110, delta=0.114091, boot[-0.053063, 0.325425], frac_gt0=0.894
  - relational_or_event_state: n=5, pieces=5, delta=0.242129, boot[-0.202096, 1.058281], frac_gt0=0.684
  - not_relational_or_event_state: n=66, pieces=105, delta=0.107994, boot[-0.06307, 0.336235], frac_gt0=0.872
  - ordinary_nonrel_nonentity: n=59, pieces=88, delta=0.099507, boot[-0.08365, 0.318509], frac_gt0=0.831
  - capitalized_or_number: n=7, pieces=17, delta=0.151926, boot[-0.075597, 0.408707], frac_gt0=0.902
- eval_set `source_disjoint_quality`
  - all: n=974, pieces=1578, delta=0.079564, boot[0.036636, 0.125664], frac_gt0=1.0
  - relational_or_event_state: n=159, pieces=175, delta=0.436983, boot[0.277822, 0.601146], frac_gt0=1.0
  - not_relational_or_event_state: n=815, pieces=1403, delta=0.034982, boot[-0.01165, 0.083511], frac_gt0=0.93
  - ordinary_nonrel_nonentity: n=732, pieces=1149, delta=0.065697, boot[0.012331, 0.118916], frac_gt0=0.993
  - capitalized_or_number: n=79, pieces=251, delta=-0.102235, boot[-0.201961, 0.01271], frac_gt0=0.04

## Interpretation for the pending packed endpoint

The removed population is not source absence in isolation: packed targetselect static load and readout patch and this stratification should be used to read it as naturally coupled novel-plus-relational/event abstractive compact content unless a later matched separation holds semantics fixed.
