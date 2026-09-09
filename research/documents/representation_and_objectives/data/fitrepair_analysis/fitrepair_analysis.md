# independent orientation cross and context interface fit-repaired independent-orientation analysis

This note analyzes the one-seed fit-repaired run in which all five arms reached training accuracy 1.0. It should be treated as a fit-verified pilot awaiting the 3-seed replication, not as final evidence.

## Context wording calibration

Mean-pooled pretrained DeBERTa cosine between train contexts and alternative contexts:

- event_train_vs_dirpara: mean 0.847, std 0.021, min 0.791, n 120
- event_train_vs_held: mean 0.821, std 0.040, min 0.744, n 120
- event_train_vs_nonpara: mean 0.640, std 0.032, min 0.578, n 120
- state_train_vs_dirpara: mean 0.749, std 0.025, min 0.690, n 120
- state_train_vs_held: mean 0.687, std 0.090, min 0.543, n 120
- state_train_vs_nonpara: mean 0.742, std 0.023, min 0.678, n 120

The nonparaphrase state context unexpectedly has high cosine (about 0.742), close to the directional state paraphrase (about 0.749), even though it does not encode higher/lower direction. Mean-pooled cosine therefore measures topical/lexical similarity more than relation-direction equivalence. Nonparaphrase performance, not cosine alone, is the important context-side calibration.

## Fit-repaired arm training

- exposure: train acc 1.000
- Etrue_Rtrue: train acc 1.000
- Eflip_Rtrue: train acc 1.000
- Etrue_Rflip: train acc 1.000
- Eflip_Rflip: train acc 1.000

## ATP compound contrastive metrics

### atp_trainTrain_trainHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.537 | 0.283 | 0.221 |
| Etrue_Rtrue | 1.000 | 1.000 | 1.000 |
| Eflip_Rtrue | 0.004 | 1.000 | 0.992 |
| Etrue_Rflip | 1.000 | 0.196 | 0.988 |
| Eflip_Rflip | 0.004 | 0.113 | 0.967 |

Event-flip effect on event/focal/secondary: 0.996/0.000/0.008.
Rank-flip effect on event/focal/secondary: 0.000/0.804/0.012.

### atp_trainTrain_heldHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.446 | 0.279 | 0.300 |
| Etrue_Rtrue | 0.775 | 0.850 | 1.000 |
| Eflip_Rtrue | 0.442 | 0.804 | 0.992 |
| Etrue_Rflip | 0.996 | 0.533 | 0.958 |
| Eflip_Rflip | 0.017 | 0.279 | 0.954 |

Event-flip effect on event/focal/secondary: 0.333/0.046/0.008.
Rank-flip effect on event/focal/secondary: -0.221/0.317/0.042.

### atp_heldHeld_trainHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.858 | 0.208 | 0.129 |
| Etrue_Rtrue | 0.988 | 0.008 | 0.017 |
| Eflip_Rtrue | 0.787 | 0.104 | 0.008 |
| Etrue_Rflip | 0.967 | 0.000 | 0.008 |
| Eflip_Rflip | 0.858 | 0.121 | 0.442 |

Event-flip effect on event/focal/secondary: 0.200/-0.096/0.008.
Rank-flip effect on event/focal/secondary: 0.021/0.008/0.008.

### atp_heldHeld_heldHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.608 | 0.333 | 0.163 |
| Etrue_Rtrue | 0.667 | 0.175 | 0.058 |
| Eflip_Rtrue | 0.467 | 0.163 | 0.021 |
| Etrue_Rflip | 0.562 | 0.071 | 0.033 |
| Eflip_Rflip | 0.733 | 0.142 | 0.217 |

Event-flip effect on event/focal/secondary: 0.200/0.012/0.038.
Rank-flip effect on event/focal/secondary: 0.104/0.104/0.025.

### atp_dirDir_trainHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.963 | 0.117 | 0.100 |
| Etrue_Rtrue | 0.929 | 0.025 | 0.008 |
| Eflip_Rtrue | 0.996 | 0.104 | 0.121 |
| Etrue_Rflip | 0.938 | 0.087 | 0.029 |
| Eflip_Rflip | 0.996 | 0.242 | 0.192 |

Event-flip effect on event/focal/secondary: -0.067/-0.079/-0.113.
Rank-flip effect on event/focal/secondary: -0.008/-0.062/-0.021.

### atp_dirDir_heldHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.662 | 0.196 | 0.142 |
| Etrue_Rtrue | 0.604 | 0.233 | 0.004 |
| Eflip_Rtrue | 0.525 | 0.167 | 0.121 |
| Etrue_Rflip | 0.654 | 0.221 | 0.008 |
| Eflip_Rflip | 0.963 | 0.400 | 0.179 |

Event-flip effect on event/focal/secondary: 0.079/0.067/-0.117.
Rank-flip effect on event/focal/secondary: -0.050/0.013/-0.004.

### atp_trainEvent_dirState_trainHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.554 | 0.083 | 0.113 |
| Etrue_Rtrue | 1.000 | 0.021 | 0.013 |
| Eflip_Rtrue | 0.025 | 0.100 | 0.108 |
| Etrue_Rflip | 0.983 | 0.183 | 0.025 |
| Eflip_Rflip | 0.017 | 0.254 | 0.250 |

Event-flip effect on event/focal/secondary: 0.975/-0.079/-0.096.
Rank-flip effect on event/focal/secondary: 0.017/-0.162/-0.013.

### atp_dirEvent_trainState_trainHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.992 | 0.333 | 0.312 |
| Etrue_Rtrue | 0.967 | 1.000 | 1.000 |
| Eflip_Rtrue | 1.000 | 1.000 | 1.000 |
| Etrue_Rflip | 0.992 | 0.392 | 0.892 |
| Eflip_Rflip | 0.983 | 0.321 | 0.762 |

Event-flip effect on event/focal/secondary: -0.033/0.000/0.000.
Rank-flip effect on event/focal/secondary: -0.025/0.608/0.108.

### atp_trainEvent_nonState_trainHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.608 | 0.504 | 0.479 |
| Etrue_Rtrue | 1.000 | 0.529 | 0.508 |
| Eflip_Rtrue | 0.000 | 0.529 | 0.512 |
| Etrue_Rflip | 1.000 | 0.446 | 0.517 |
| Eflip_Rflip | 0.000 | 0.458 | 0.512 |

Event-flip effect on event/focal/secondary: 1.000/0.000/-0.004.
Rank-flip effect on event/focal/secondary: 0.000/0.083/-0.008.

### atp_nonEvent_trainState_trainHyp

| arm | event | focal | secondary |
|---|---:|---:|---:|
| exposure | 0.504 | 0.375 | 0.192 |
| Etrue_Rtrue | 0.521 | 0.996 | 1.000 |
| Eflip_Rtrue | 0.504 | 0.992 | 1.000 |
| Etrue_Rflip | 0.487 | 0.421 | 0.925 |
| Eflip_Rflip | 0.521 | 0.362 | 0.725 |

Event-flip effect on event/focal/secondary: 0.017/0.004/0.000.
Rank-flip effect on event/focal/secondary: 0.033/0.575/0.075.

## Pair-domain event transfer

| eval | exposure | Etrue_Rtrue | Eflip_Rtrue | Etrue_Rflip | Eflip_Rflip |
|---|---:|---:|---:|---:|---:|
| pair_trainCtx_trainHyp | 0.575 | 1.000 | 0.000 | 1.000 | 0.000 |
| pair_trainCtx_heldHyp | 0.633 | 1.000 | 0.000 | 1.000 | 0.000 |
| pair_heldCtx_trainHyp | 0.883 | 0.992 | 0.767 | 1.000 | 0.908 |
| pair_heldCtx_heldHyp | 0.825 | 1.000 | 0.825 | 1.000 | 0.908 |
| pair_dirparaCtx_trainHyp | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 |
| pair_dirparaCtx_heldHyp | 0.992 | 0.983 | 1.000 | 1.000 | 0.992 |
| pair_nonparaCtx_trainHyp | 0.525 | 0.458 | 0.508 | 0.508 | 0.508 |
| pair_nonparaCtx_heldHyp | 0.500 | 0.517 | 0.500 | 0.508 | 0.500 |

## Scientific reading

1. Familiar trainTrain/trainHyp contexts show clear selective control: Etrue_Rtrue gives 1.000/1.000/1.000; Eflip_Rtrue gives event 0.004 with focal 1.000 and secondary 0.992; Etrue_Rflip gives event 1.000, focal 0.196, secondary 0.988. This is not a head-wide label convention. It demonstrates that sparse event orientation can be inverted without inverting ranking facts, and sparse focal-rank orientation can be inverted while the secondary ranking fact remains mostly conserved.

2. The secondary ranking fact is not simply dragged by the focal-rank flip on familiar contexts: Etrue_Rflip preserves secondary at 0.988 and Eflip_Rflip preserves secondary at 0.967 while focal ranking drops. This is the first controlled evidence of query/position-specific conservation under an intentionally wrong focal orientation.

3. Context side remains the hard boundary. Held/held and dirDir state contexts collapse for focal and secondary state despite full train fit; event often remains strong. This means contrastive wording cross and grounding's failure was not merely head calibration. It also means the law should be stated as compositional use of oriented evidence only when the directional context interface is grounded by the learner, not as a cosine or raw-frequency threshold.

4. Nonparaphrase controls behave as relation-removal controls: when state context is replaced by mention-only text, focal and secondary approach chance even if event is perfectly controlled; when event context is mention-only but state context remains train wording, event is chance while state readout is strong. This is better evidence of relation-specific context use than mean-pooled cosine.

5. Pair-domain event transfer needs caution: in Eflip_Rtrue/Eflip_Rflip, train-context pair event rows invert on familiar context but held/dir/non contexts can revert or collapse depending on context wording. Event orientation is therefore not universally domain/wording invariant; it is strongest on the sparse-training context family and should be evaluated with context controls.

The pending 3-seed run should test whether these selective effects survive seed variation. If it replicates, the next construction problem is to strengthen context-side grounding for ranking direction without training the final held wording directly, e.g. by adding directional paraphrase bridges or contrastive context objectives that preserve independent event/rank labels and secondary-state truth.
