# earlier analysis conservation-transition readout

This readout uses only existing prediction/per-item files. It interprets Entity movement through conservation: does an arm move event pairs into `both_correct` (changed entity updated and unaffected entity retained), or does it exchange one side of the pair for the other?

## Binding paired-state flow: late 80/90/100M means

Percentages below are fractions of paired affected/unaffected event rows, averaged over the three late checkpoints. `from` is the second arm in the contrast (e.g. B for V-B), `to` is the first arm.

| contrast | Δ both-correct pp | Δ affected-only pp | Δ unaffected-only pp | Δ neither pp | not-both→both pp | both→not-both pp | unaffected-only→affected-only pp | both→affected-only pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VminusR | +0.694 | +20.660 | -20.486 | -0.868 | +2.257 | +1.562 | +26.389 | +1.215 |
| VminusB | -1.389 | +8.333 | -2.778 | -4.167 | +2.257 | +3.646 | +20.312 | +2.257 |
| BminusR | +2.083 | +12.326 | -17.708 | +3.299 | +3.819 | +1.736 | +18.924 | +0.521 |

## Binding per-checkpoint V-B state flows

| checkpoint | from both | to both | Δ both pp | from affected-only | to affected-only | from unaffected-only | to unaffected-only | from neither | to neither | flow matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| chck_80M | +2.604 | +2.083 | -0.521 | +32.292 | +43.229 | +58.854 | +54.688 | +6.250 | +0.000 | `{"affected_only->affected_only": 37, "affected_only->both_correct": 2, "affected_only->unaffected_only": 23, "both_correct->affected_only": 4, "both_correct->unaffected_only": 1, "neither->affected_only": 3, "neither->unaffected_only": 9, "unaffected_only->affected_only": 39, "unaffected_only->both_correct": 2, "unaffected_only->unaffected_only": 72}` |
| chck_90M | +4.688 | +2.083 | -2.604 | +34.896 | +41.667 | +57.292 | +56.250 | +3.125 | +0.000 | `{"affected_only->affected_only": 36, "affected_only->both_correct": 3, "affected_only->unaffected_only": 28, "both_correct->affected_only": 5, "both_correct->both_correct": 1, "both_correct->unaffected_only": 3, "neither->affected_only": 1, "neither->unaffected_only": 5, "unaffected_only->affected_only": 38, "unaffected_only->unaffected_only": 72}` |
| chck_100M | +4.167 | +3.125 | -1.042 | +34.375 | +41.667 | +58.333 | +55.208 | +3.125 | +0.000 | `{"affected_only->affected_only": 36, "affected_only->both_correct": 4, "affected_only->unaffected_only": 26, "both_correct->affected_only": 4, "both_correct->unaffected_only": 4, "neither->unaffected_only": 6, "unaffected_only->affected_only": 40, "unaffected_only->both_correct": 2, "unaffected_only->unaffected_only": 70}` |

## Official Entity gain/loss and option response

Option0 is the correct completion in the official Entity format. The table separates net accuracy from gain/loss churn and from response-index movement.

| contrast | group | Δ accuracy pp | wrong→correct pp | correct→wrong pp | both-correct pp | both-wrong pp | Δ option0 pp | Δ option1 pp | Δ option2 pp | Δ option3 pp | Δ option4 pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VminusB | all_18_subtasks | +2.679 | +9.312 | +6.632 | +18.486 | +65.570 | +2.679 | -0.919 | -0.733 | -0.452 | -0.575 |
| VminusB | zero_ops | +3.374 | +11.464 | +8.090 | +27.082 | +53.364 | +3.374 | +0.303 | -1.146 | -0.433 | -2.098 |
| VminusB | nonzero_ops | +2.475 | +8.679 | +6.203 | +15.957 | +69.161 | +2.475 | -1.279 | -0.611 | -0.458 | -0.127 |
| VminusB | numops_0 | +3.374 | +11.464 | +8.090 | +27.082 | +53.364 | +3.374 | +0.303 | -1.146 | -0.433 | -2.098 |
| VminusB | numops_1 | +0.733 | +6.463 | +5.730 | +8.817 | +78.990 | +0.733 | +1.361 | -3.192 | +1.518 | -0.419 |
| VminusB | numops_2 | +1.397 | +7.697 | +6.300 | +14.790 | +71.213 | +1.397 | -0.493 | +0.767 | -1.452 | -0.219 |
| VminusB | numops_3 | +1.989 | +8.737 | +6.747 | +17.715 | +66.801 | +1.989 | -1.398 | +1.317 | -1.559 | -0.349 |
| VminusB | numops_4 | +5.305 | +11.007 | +5.702 | +21.560 | +61.730 | +5.305 | -3.858 | -0.596 | -0.879 | +0.028 |
| VminusB | numops_5 | +4.905 | +12.312 | +7.407 | +21.221 | +59.059 | +4.905 | -4.705 | -3.003 | +1.201 | +1.602 |
| BminusR | all_18_subtasks | -0.206 | +9.651 | +9.857 | +15.467 | +65.025 | -0.206 | +0.265 | -5.339 | +2.748 | +2.532 |
| BminusR | zero_ops | -13.065 | +6.230 | +19.295 | +28.942 | +45.533 | -13.065 | +4.824 | +2.336 | +2.379 | +3.526 |
| BminusR | nonzero_ops | +3.576 | +10.657 | +7.082 | +11.503 | +70.758 | +3.576 | -1.075 | -7.597 | +2.857 | +2.240 |
| BminusR | numops_0 | -13.065 | +6.230 | +19.295 | +28.942 | +45.533 | -13.065 | +4.824 | +2.336 | +2.379 | +3.526 |
| BminusR | numops_1 | +1.204 | +7.274 | +6.070 | +7.274 | +79.383 | +1.204 | -5.050 | -3.375 | +3.035 | +4.186 |
| BminusR | numops_2 | +2.465 | +10.244 | +7.779 | +10.846 | +71.131 | +2.465 | -1.835 | -7.943 | +3.917 | +3.396 |
| BminusR | numops_3 | +6.075 | +12.581 | +6.505 | +11.882 | +69.032 | +6.075 | -0.027 | -8.575 | +2.204 | +0.323 |
| BminusR | numops_4 | +4.085 | +12.057 | +7.972 | +15.206 | +64.766 | +4.085 | +1.277 | -9.986 | +2.638 | +1.986 |
| BminusR | numops_5 | +5.606 | +13.013 | +7.407 | +15.616 | +63.964 | +5.606 | +4.705 | -10.410 | +1.502 | -1.401 |
| VminusR | all_18_subtasks | +2.473 | +10.841 | +8.368 | +16.957 | +63.835 | +2.473 | -0.654 | -6.072 | +2.296 | +1.957 |
| VminusR | zero_ops | -9.691 | +7.311 | +17.002 | +31.235 | +44.452 | -9.691 | +5.127 | +1.190 | +1.947 | +1.428 |
| VminusR | nonzero_ops | +6.051 | +11.879 | +5.828 | +12.757 | +69.536 | +6.051 | -2.354 | -8.208 | +2.399 | +2.112 |
| VminusR | numops_0 | -9.691 | +7.311 | +17.002 | +31.235 | +44.452 | -9.691 | +5.127 | +1.190 | +1.947 | +1.428 |
| VminusR | numops_1 | +1.936 | +7.849 | +5.913 | +7.431 | +78.807 | +1.936 | -3.689 | -6.567 | +4.553 | +3.768 |
| VminusR | numops_2 | +3.862 | +10.189 | +6.327 | +12.298 | +71.186 | +3.862 | -2.328 | -7.176 | +2.465 | +3.177 |
| VminusR | numops_3 | +8.065 | +13.011 | +4.946 | +13.441 | +68.602 | +8.065 | -1.425 | -7.258 | +0.645 | -0.027 |
| VminusR | numops_4 | +9.390 | +15.603 | +6.213 | +16.965 | +61.220 | +9.390 | -2.582 | -10.582 | +1.759 | +2.014 |
| VminusR | numops_5 | +10.511 | +16.116 | +5.606 | +17.417 | +60.861 | +10.511 | +0.000 | -13.413 | +2.703 | +0.200 |

## Scientific interpretation

- Binding V-B does **not** show the conservation pattern needed for source-conditioned state-record formation. Late mean Δ both-correct is -1.389 pp, while not-both→both is +2.257 pp and both→not-both is +3.646 pp. The paired flow is therefore churn around a near-chance boundary rather than robust movement into the state where both changed and stable entity readouts are correct.
- The erosive part is visible directly: V-B contains unaffected-only→affected-only flow +20.312 pp and both→affected-only flow +2.257 pp. This is the conservation failure: view can improve the changed query while damaging the stable query in the same event.
- Official Entity still has a real V-B surface effect: late zero-op Δ accuracy +3.374 pp and nonzero-op +2.475 pp. But the official item transition table shows this as ordinary gain/loss movement and option0/correct-response shifts, not as paired retention of unaffected state.
- B-R remains a changed-state/operation prior relative to repeat: official zero-op -13.065 pp versus nonzero +3.576 pp. V-B is arithmetically different from B-R on official Entity, but the paired binding table says that difference should not be promoted to source-correspondence without an aligned-versus-permuted arm.
- Existing broad score files have not supplied a completed broad ex-Entity V-B/B-C/V-C triangle beyond the already noted 80M point. Because the conservation pattern is absent in the cheap binding evidence and broad V-B is still incomplete/weak, a new H100 permuted-companion run is not supported by the current evidence. The compact macro branch should be closed unless later delivers already-running breadth/clean scores that materially change broad V-B or paired conservation.

## Current broad-score file status

Score rows present: 219; missing score rows: 381.
Contrast rows present: 165; missing contrast rows: 715.
Triangle rows present: 3; broad triangle rows: 0.
- deberta_basin1_VminusB exEntity5 late80_100: not yet complete
- deberta_basin1_BminusR exEntity5 late80_100: not yet complete
- deberta_basin1_VminusR exEntity5 late80_100: n=1 mean=+0.3330 range=[+0.3330, +0.3330] checkpoints=chck_90M
- deberta_basin1_VminusB cheap6_no_GlobalPIQA late80_100: not yet complete
- deberta_basin1_BminusR cheap6_no_GlobalPIQA late80_100: not yet complete
- roberta_VminusC_maxgeom exEntity5 late80_100: not yet complete
- roberta_VminusC_maxgeom cheap6_no_GlobalPIQA late80_100: not yet complete
- deberta_basin1_VminusC_maxgeom exEntity5 late80_100: not yet complete
- deberta_basin2_VminusC_maxgeom exEntity5 late80_100: not yet complete
- chck_80M Entity: V-R=+4.5000, V-B=+3.1000, B-R=+1.4000, residual=+0.0000
- chck_90M Entity: V-R=+3.6600, V-B=+3.0800, B-R=+0.5800, residual=+0.0000
- chck_100M Entity: V-R=+3.8900, V-B=+2.8000, B-R=+1.0900, residual=+0.0000

## Files

- summary_md: `research/documents/representation_and_objectives/data/conservation_transition_readout/conservation_transition_summary.md`
- summary_json: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/conservation_transition_summary.json`
- binding_pair_states_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/binding_pair_states.csv`
- binding_state_counts_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/binding_state_counts.csv`
- binding_flow_rows_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/binding_flow_rows.csv`
- binding_flow_detail_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/binding_flow_detail_rows.csv`
- binding_late_flow_summary_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/binding_late_flow_summary.csv`
- entity_prediction_paths_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/entity_prediction_paths.csv`
- entity_item_records_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/entity_item_records.csv`
- entity_model_group_rows_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/entity_model_group_rows.csv`
- entity_transition_rows_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/entity_transition_rows.csv`
- entity_transition_detail_rows_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/entity_transition_detail_rows.csv`
- entity_late_transition_summary_csv: `experiments/archive/representation_and_objectives/data/conservation_transition_readout/entity_late_transition_summary.csv`
