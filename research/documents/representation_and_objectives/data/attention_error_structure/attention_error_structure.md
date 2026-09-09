# clean init and attention execution plan raw-name attention error structure

## Train/eval overview

| run | train_state | train_cmp | graph_same | unchanged | hh_closure | mixed_acc |
|---|---:|---:|---:|---:|---:|---:|
| shared_trunk|bs+1|seed29300 | 1.000 | 1.000 | 0.688 | 0.812 | 0.547 | 0.539 |
| shared_trunk|bs-1|seed29300 | 1.000 | 0.948 | 0.531 | 0.812 | 0.422 | 0.492 |
| tied|bs+1|seed29300 | 1.000 | 1.000 | 0.844 | 0.812 | 0.750 | 0.777 |
| tied|bs-1|seed29300 | 1.000 | 0.979 | 0.312 | 0.812 | 0.578 | 0.305 |
| untied|bs+1|seed29300 | 1.000 | 0.500 | 0.469 | 0.812 | 0.500 | 0.500 |

## Main interpretation

Held-name evaluation is the hard part of the raw-name pilot. If train fit is high while changed, unchanged, and held-held eval are far from 1.0, the failure is not only gauge transport; it includes non-variable candidate/name generalization under the query-attention architecture.

## Per-run coarse error slices

### shared_trunk|bs+1|seed29300
- state by suite: {'cross_template_state_readout': {'n': 256, 'acc': 0.8125, 'margin': 3.597813735017553}, 'name_permutation_base': {'n': 128, 'acc': 0.875, 'margin': 3.457385841757059}, 'name_permutation_swapped': {'n': 128, 'acc': 0.75, 'margin': 4.065826158504933}, 'paired_state_conservation': {'n': 512, 'acc': 0.796875, 'margin': 3.6249565557809547}}
- state by relation_family: {'direct_anchor': {'n': 512, 'acc': 0.78125, 'margin': 3.2510272396029904}, 'graph_transfer': {'n': 512, 'acc': 0.828125, 'margin': 4.053639183752239}}
- state by initial_pattern: {'opposite': {'n': 512, 'acc': 0.8046875, 'margin': 3.6763362257624976}, 'same': {'n': 512, 'acc': 0.8046875, 'margin': 3.6283301975927316}}
- comparison by suite: {'heldheld_unseen_edge_closure': {'n': 128, 'acc': 0.546875, 'signed_margin': 1.7895503948066238}, 'mixed_held_seen_orientation': {'n': 512, 'acc': 0.5390625, 'signed_margin': 1.8559329045624144}}

### shared_trunk|bs-1|seed29300
- state by suite: {'cross_template_state_readout': {'n': 256, 'acc': 0.65625, 'margin': 1.7128376113250852}, 'name_permutation_base': {'n': 128, 'acc': 0.65625, 'margin': 3.0066368263214827}, 'name_permutation_swapped': {'n': 128, 'acc': 0.78125, 'margin': 3.4216944025829434}, 'paired_state_conservation': {'n': 512, 'acc': 0.6875, 'margin': 2.668443107511848}}
- state by relation_family: {'direct_anchor': {'n': 512, 'acc': 0.6875, 'margin': 2.829591202782467}, 'graph_transfer': {'n': 512, 'acc': 0.6875, 'margin': 2.3023535176180303}}
- state by initial_pattern: {'opposite': {'n': 512, 'acc': 0.6875, 'margin': 2.5899753742851317}, 'same': {'n': 512, 'acc': 0.6875, 'margin': 2.5419693461153656}}
- comparison by suite: {'heldheld_unseen_edge_closure': {'n': 128, 'acc': 0.421875, 'signed_margin': 0.3002891646498407}, 'mixed_held_seen_orientation': {'n': 512, 'acc': 0.4921875, 'signed_margin': 0.2807204157449372}}

### tied|bs+1|seed29300
- state by suite: {'cross_template_state_readout': {'n': 256, 'acc': 0.8125, 'margin': 3.9598324873950332}, 'name_permutation_base': {'n': 128, 'acc': 0.84375, 'margin': 5.753778513055295}, 'name_permutation_swapped': {'n': 128, 'acc': 0.84375, 'margin': 4.560166232287884}, 'paired_state_conservation': {'n': 512, 'acc': 0.8203125, 'margin': 3.9364604051224887}}
- state by relation_family: {'direct_anchor': {'n': 512, 'acc': 0.8046875, 'margin': 3.6623385006096214}, 'graph_transfer': {'n': 512, 'acc': 0.84375, 'margin': 4.832524334546179}}
- state by initial_pattern: {'opposite': {'n': 512, 'acc': 0.82421875, 'margin': 4.271434431662783}, 'same': {'n': 512, 'acc': 0.82421875, 'margin': 4.223428403493017}}
- comparison by suite: {'heldheld_unseen_edge_closure': {'n': 128, 'acc': 0.75, 'signed_margin': 2.402217744038052}, 'mixed_held_seen_orientation': {'n': 512, 'acc': 0.77734375, 'signed_margin': 3.676011604332886}}

### tied|bs-1|seed29300
- state by suite: {'cross_template_state_readout': {'n': 256, 'acc': 0.671875, 'margin': 1.3582188147120178}, 'name_permutation_base': {'n': 128, 'acc': 0.65625, 'margin': 1.6553184036165476}, 'name_permutation_swapped': {'n': 128, 'acc': 0.8125, 'margin': 3.5409856201149523}, 'paired_state_conservation': {'n': 512, 'acc': 0.6796875, 'margin': 2.474152655340731}}
- state by relation_family: {'direct_anchor': {'n': 512, 'acc': 0.8046875, 'margin': 4.461121580912732}, 'graph_transfer': {'n': 512, 'acc': 0.578125, 'margin': -0.008783512283116579}}
- state by initial_pattern: {'opposite': {'n': 512, 'acc': 0.69140625, 'margin': 2.2501720483996905}, 'same': {'n': 512, 'acc': 0.69140625, 'margin': 2.2021660202299245}}
- comparison by suite: {'heldheld_unseen_edge_closure': {'n': 128, 'acc': 0.578125, 'signed_margin': 1.416531661238515}, 'mixed_held_seen_orientation': {'n': 512, 'acc': 0.3046875, 'signed_margin': -3.2082199451621083}}

### untied|bs+1|seed29300
- state by suite: {'cross_template_state_readout': {'n': 256, 'acc': 0.640625, 'margin': 1.7659028372727334}, 'name_permutation_base': {'n': 128, 'acc': 0.8125, 'margin': 2.9982799403369427}, 'name_permutation_swapped': {'n': 128, 'acc': 0.59375, 'margin': 2.440551565028727}, 'paired_state_conservation': {'n': 512, 'acc': 0.7265625, 'margin': 2.2906739274039865}}
- state by relation_family: {'direct_anchor': {'n': 512, 'acc': 0.75, 'margin': 2.955274839885533}, 'graph_transfer': {'n': 512, 'acc': 0.6484375, 'margin': 1.5780583824962378}}
- state by initial_pattern: {'opposite': {'n': 512, 'acc': 0.69921875, 'margin': 2.2906696252757683}, 'same': {'n': 512, 'acc': 0.69921875, 'margin': 2.2426635971060023}}
- comparison by suite: {'heldheld_unseen_edge_closure': {'n': 128, 'acc': 0.5, 'signed_margin': 1.5255885961819573}, 'mixed_held_seen_orientation': {'n': 512, 'acc': 0.5, 'signed_margin': 1.5248702556938027}}

