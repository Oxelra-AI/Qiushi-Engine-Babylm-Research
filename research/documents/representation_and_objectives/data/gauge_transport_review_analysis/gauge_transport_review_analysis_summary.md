# shared gauge review and next causal controls gauge-transport review analysis

CPU-only review over saved shared factorization result synthesis predictions. No model was loaded or trained.

## Run integrity

| run | condition | arm | seed | train_state | train_cmp | parse_errors | eval_state_groups | eval_state_bad_len | eval_state_bad_true | eval_state_bad_arm | ties_true |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tied_aligned_state_bridge_seed28801 | tied | aligned_state_bridge | 28801 | 1.0 | 1.0 | 0 | 768 | 0 | 0 | 0 | 0 |
| untied_aligned_state_bridge_seed28801 | untied | aligned_state_bridge | 28801 | 1.0 | 1.0 | 0 | 768 | 0 | 0 | 0 | 0 |
| tied_inverted_state_bridge_seed28801 | tied | inverted_state_bridge | 28801 | 1.0 | 1.0 | 0 | 768 | 0 | 0 | 0 | 0 |
| untied_inverted_state_bridge_seed28801 | untied | inverted_state_bridge | 28801 | 1.0 | 1.0 | 0 | 768 | 0 | 0 | 0 | 0 |
| tied_heldheld_only_seed28801 | tied | heldheld_only | 28801 | 1.0 | 1.0 | 0 | 768 | 0 | 0 | 0 | 0 |
| tied_heldheld_only_seed28802 | tied | heldheld_only | 28802 | 1.0 | 0.875 | 0 | 768 | 0 | 0 | 0 | 0 |
| untied_heldheld_only_seed28801 | untied | heldheld_only | 28801 | 1.0 | 1.0 | 0 | 768 | 0 | 0 | 0 | 0 |
| untied_heldheld_only_seed28802 | untied | heldheld_only | 28802 | 1.0 | 1.0 | 0 | 768 | 0 | 0 | 0 | 0 |

## Aligned/inverted state-margin mirroring, train-only seed 28801

### tied
- target mode `true`, paired_state_conservation graph/same changed: n=64; aligned_correct=n=64, mean=1.000, range=[1.000,1.000]; inverted_correct=n=64, mean=0.000, range=[0.000,0.000]; opposite_sign=n=64, mean=1.000, range=[1.000,1.000]; both_positive=n=64, mean=0.000, range=[0.000,0.000]; aligned_margin=n=64, mean=16.708, range=[16.204,17.888]; inverted_margin=n=64, mean=-14.754, range=[-17.150,-12.512]
- target mode `arm`, paired_state_conservation graph/same changed: n=64; aligned_correct=n=64, mean=1.000, range=[1.000,1.000]; inverted_correct=n=64, mean=1.000, range=[1.000,1.000]; opposite_sign=n=64, mean=0.000, range=[0.000,0.000]; both_positive=n=64, mean=1.000, range=[1.000,1.000]; aligned_margin=n=64, mean=16.708, range=[16.204,17.888]; inverted_margin=n=64, mean=14.754, range=[12.512,17.150]
### untied
- target mode `true`, paired_state_conservation graph/same changed: n=64; aligned_correct=n=64, mean=0.250, range=[0.000,1.000]; inverted_correct=n=64, mean=0.500, range=[0.000,1.000]; opposite_sign=n=64, mean=0.750, range=[0.000,1.000]; both_positive=n=64, mean=0.000, range=[0.000,0.000]; aligned_margin=n=64, mean=-2.281, range=[-7.116,5.492]; inverted_margin=n=64, mean=-0.895, range=[-7.124,3.188]
- target mode `arm`, paired_state_conservation graph/same changed: n=64; aligned_correct=n=64, mean=0.250, range=[0.000,1.000]; inverted_correct=n=64, mean=0.500, range=[0.000,1.000]; opposite_sign=n=64, mean=0.250, range=[0.000,1.000]; both_positive=n=64, mean=0.250, range=[0.000,1.000]; aligned_margin=n=64, mean=-2.281, range=[-7.116,5.492]; inverted_margin=n=64, mean=0.895, range=[-3.188,7.124]

## Aligned/inverted mixed-comparison mirroring, train-only seed 28801

### tied
- target mode `true`, mixed held-seen: n=512; aligned_correct=n=512, mean=1.000, range=[1.000,1.000]; inverted_correct=n=512, mean=0.000, range=[0.000,0.000]; opposite_sign=n=512, mean=1.000, range=[1.000,1.000]; both_positive=n=512, mean=0.000, range=[0.000,0.000]; aligned_margin=n=512, mean=13.809, range=[13.802,13.816]; inverted_margin=n=512, mean=-13.600, range=[-13.816,-12.508]
- target mode `arm`, mixed held-seen: n=512; aligned_correct=n=512, mean=1.000, range=[1.000,1.000]; inverted_correct=n=512, mean=1.000, range=[1.000,1.000]; opposite_sign=n=512, mean=0.000, range=[0.000,0.000]; both_positive=n=512, mean=1.000, range=[1.000,1.000]; aligned_margin=n=512, mean=13.809, range=[13.802,13.816]; inverted_margin=n=512, mean=13.600, range=[12.508,13.816]
### untied
- target mode `true`, mixed held-seen: n=512; aligned_correct=n=512, mean=0.500, range=[0.000,1.000]; inverted_correct=n=512, mean=0.500, range=[0.000,1.000]; opposite_sign=n=512, mean=0.000, range=[0.000,0.000]; both_positive=n=512, mean=0.500, range=[0.000,1.000]; aligned_margin=n=512, mean=4.698, range=[-3.703,13.170]; inverted_margin=n=512, mean=4.698, range=[-3.703,13.170]
- target mode `arm`, mixed held-seen: n=512; aligned_correct=n=512, mean=0.500, range=[0.000,1.000]; inverted_correct=n=512, mean=0.500, range=[0.000,1.000]; opposite_sign=n=512, mean=1.000, range=[1.000,1.000]; both_positive=n=512, mean=0.000, range=[0.000,0.000]; aligned_margin=n=512, mean=4.698, range=[-3.703,13.170]; inverted_margin=n=512, mean=-4.698, range=[-13.170,3.703]

## Heldheld-only anchor control

| condition | seed | train_state | train_cmp | direct_same | graph_same | pair_both_graph_same | unchanged | mixed_acc | mixed_margin | hh_closure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied | 28801 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | -11.493 | 1.000 |
| tied | 28802 | 1.000 | 0.875 | 0.000 | 0.250 | 0.250 | 1.000 | 0.125 | -4.897 | 0.625 |
| untied | 28801 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 1.000 | 0.000 | -8.974 | 1.000 |
| untied | 28802 | 1.000 | 1.000 | 0.500 | 0.250 | 0.250 | 1.000 | 0.500 | -2.468 | 1.000 |

## Reviewer interpretation

The saved rows support a real aligned/inverted arm sign reversal in the tied model. Relative to the true target, tied aligned graph-transfer changed-state margins are positive while tied inverted margins are negative; relative to each arm's installed target, both are positive. The mixed held-seen comparison margins show the same pattern. This reduces the chance that the shared factorization result synthesis result is only a merge-table artifact. The heldheld-only runs do not produce a correct absolute state coordinate without bridge state anchors, even when comparison closure is high, so comparison graph evidence alone is not enough to orient state transfer.

The result still has the limitations identified in shared factorization result synthesis: candidate discovery/correspondence, binary ownership, same-owner comparison, and changed/unchanged routing are supplied by the harness, and unchanged facts use a separate static scorer. The next causal test should manipulate the sign or role permutation between the shared relation coordinate and state readout while keeping local fit possible; this will make the transported gauge directly visible rather than merely inferred from aligned/inverted data arms.

Full JSON: `experiments/archive/representation_and_objectives/data/gauge_transport_review_analysis/gauge_transport_review_analysis.json`
