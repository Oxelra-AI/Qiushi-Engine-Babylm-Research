# clean d component ablation clean L1 d-component intervention

## Purpose

causal intervention/026 showed that donor d-component patches can redirect answers. This experiment edits the natural clean forward pass: center or zero the layer-1 d projection across attribute slots, or rotate the d pattern so the original query slot's scalar component moves to a decoy slot. This tests whether ordinary behavior requires the component and whether direct-full familiar recovery uses the same interface.

## Mean results across seeds

### prep

| bank | clean | center | center drop | zero | zero drop | rotate correct | rotate target | rotate margin | dir acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1.000 | 0.283 | -0.717 | 0.264 | -0.736 | 0.026 | 0.923 | +7.433 | 0.982 |
| held_query | 0.875 | 0.266 | -0.609 | 0.250 | -0.625 | 0.116 | 0.676 | +4.905 | 0.831 |
| train_in_held_ctx | 0.902 | 0.283 | -0.620 | 0.277 | -0.625 | 0.079 | 0.767 | +5.908 | 0.842 |

### direct_full

| bank | clean | center | center drop | zero | zero drop | rotate correct | rotate target | rotate margin | dir acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.872 | 0.348 | -0.525 | 0.345 | -0.527 | 0.267 | 0.478 | +3.618 | 0.510 |
| held_query | 0.402 | 0.251 | -0.151 | 0.311 | -0.091 | 0.257 | 0.314 | +1.328 | 0.342 |
| train_in_held_ctx | 0.814 | 0.349 | -0.465 | 0.316 | -0.497 | 0.273 | 0.414 | +3.181 | 0.484 |

### static_1over17

| bank | clean | center | center drop | zero | zero drop | rotate correct | rotate target | rotate margin | dir acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1.000 | 0.367 | -0.633 | 0.404 | -0.596 | 0.042 | 0.835 | +10.427 | 0.913 |
| held_query | 0.755 | 0.312 | -0.443 | 0.336 | -0.419 | 0.188 | 0.482 | +3.628 | 0.581 |
| train_in_held_ctx | 0.966 | 0.337 | -0.629 | 0.389 | -0.577 | 0.070 | 0.792 | +9.317 | 0.871 |

### interleaved_ans_full

| bank | clean | center | center drop | zero | zero drop | rotate correct | rotate target | rotate margin | dir acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1.000 | 0.352 | -0.648 | 0.449 | -0.551 | 0.001 | 0.992 | +11.031 | 1.000 |
| held_query | 0.833 | 0.329 | -0.504 | 0.387 | -0.447 | 0.138 | 0.639 | +5.164 | 0.736 |
| train_in_held_ctx | 0.970 | 0.324 | -0.646 | 0.413 | -0.557 | 0.031 | 0.944 | +10.129 | 0.970 |

## Interpretation

A large center/zero drop means the natural slot-specific d pattern is needed for the answer. A high rotated-target rate means the answer follows the moved d marker rather than the original query. Comparing `train`, `held_query`, and `train_in_held_ctx` shows whether failure is query-token-specific or due to held-containing contexts. If direct-full at epoch 500 has high train drop/rotation but weak held drop/rotation, familiar recovery is using the interface while held transfer is not.

## Files

- Data: `experiments/archive/functional_learning/data/clean_d_component_ablation/results.json`
- Script: `experiments/archive/functional_learning/scripts/clean_d_component_ablation.py`
