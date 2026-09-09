# corrected orientation route and next experiments corrected readout of complete orientation probe results orientation evidence

This file re-reads the equivariant symmetry repair and macro context/279 symmetry-identification work from saved outputs only. No model loading, training, official evaluation, or upload occurred.

## Central readout
- Mixed held-seen orientation stays at chance: aligned 0.4922, inverted 0.4992, aligned-minus-inverted -0.0070.
- True-labeled changed-state readout improves in state-trained arms: aligned 0.7945, inverted 0.7375, neutral 0.5656.
- The direct orientation prediction from the formal slot-orbit solver is therefore absent in the fine-tuned DeBERTa readout; the state gain is real but not polarity-controlled on mixed relations.

## Arm structure and exposure
| arm | rows | relation rows | state rows | row presentations at 50 epochs |
|---|---:|---:|---:|---:|
| exposure_only | 0 | 0 | 0 | 0 |
| heldheld_only | 192 | 192 | 0 | 9600 |
| aligned_state_bridge | 320 | 192 | 128 | 16000 |
| inverted_state_bridge | 320 | 192 | 128 | 16000 |
| neutral_decoupled | 704 | 192 | 512 | 35200 |
| mixed_event_bridge | 256 | 256 | 0 | 12800 |

The neutral arm has more than twice the supervised rows of the aligned/inverted arms, so its magnitude cannot be read as a matched state-format reference without a new matched run.

## What aggregate train accuracy can and cannot show
| arm | aggregate train acc | state-query rows | possible state-query train accuracy range |
|---|---:|---:|---:|
| aligned_state_bridge | 0.9094 | 128 | [0.7735, 1.0000] |
| inverted_state_bridge | 0.9144 | 128 | [0.7860, 1.0000] |
| neutral_decoupled | 0.9954 | 512 | [0.9937, 1.0000] |

The inverted arm's aggregate train score does not by itself prove that a coherent inverted event-state semantics was learned on the bridge rows; it only constrains the possible bridge-row fit range.

## Aligned-minus-inverted seed-paired differences
| readout | mean diff | seed values |
|---|---:|---|
| mixed_acc | -0.0070 | -0.008, -0.012, -0.010, -0.002, -0.004 |
| state_changed | 0.0570 | 0.188, 0.012, -0.086, 0.051, 0.121 |
| state_unchanged | -0.0844 | -0.105, -0.004, -0.090, -0.078, -0.144 |
| state_pair_both | -0.0328 | 0.047, 0.016, -0.164, -0.023, -0.039 |
| xtempl_changed | 0.0422 | 0.172, -0.008, -0.102, 0.039, 0.109 |
| nperm_changed | 0.0360 | 0.148, 0.000, -0.094, 0.070, 0.055 |

The changed-state aligned advantage is small relative to seed variation and reverses in one seed; several unchanged and joint readouts favor the inverted arm. This is not a clean bridge-polarity interaction.

## Changed plus unchanged joint structure
| arm | changed | unchanged | both | both - minimum overlap | both - independent reference |
|---|---:|---:|---:|---:|---:|
| exposure_only | 0.5016 | 0.5000 | 0.2547 | 0.2531 | 0.0039 |
| heldheld_only | 0.4977 | 0.5101 | 0.2547 | 0.2469 | 0.0008 |
| aligned_state_bridge | 0.7945 | 0.7578 | 0.5594 | 0.0071 | -0.0427 |
| inverted_state_bridge | 0.7375 | 0.8422 | 0.5922 | 0.0125 | -0.0289 |
| neutral_decoupled | 0.5656 | 0.6539 | 0.3656 | 0.1461 | -0.0042 |
| mixed_event_bridge | 0.5172 | 0.4734 | 0.2359 | 0.2359 | -0.0089 |

Aligned and inverted state arms raise both-correct above the no-state baseline, but the observed joint score is close to the overlap forced by marginal changed/unchanged accuracies and below the independence reference. This supports state-task transfer, not a protected conserved state record.

## Corrected interpretation
The durable result is: in this DeBERTa checkpoint and equivariant symmetry repair and macro context surface, state-formatted supervision gives out-of-distribution state-query competence, including new names/templates, while relation-comparison orientation remains at chance and bridge polarity does not install opposite global orientations. The data do not yet identify whether the state gain is driven by pretrained semantic priors, generic state-format calibration, surface-readable grammatical roles, or optimization/credit assignment. Distinguishing those alternatives requires matched neutral/aligned/inverted runs, trainable random-init controls matched on local fit, pretraining-checkpoint comparisons, and contradictory-evidence dose curves with explicit local-vs-new support readouts.

## Files
- JSON readout: `experiments/archive/representation_and_objectives/data/orientation_evidence_readout/orientation_evidence_readout.json`
