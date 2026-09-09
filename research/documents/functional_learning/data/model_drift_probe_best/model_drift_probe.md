# clean d component ablation model drift and fixed-probe LM analysis

Parent: `models/frontier`

## std_87M

Path: `experiments/archive/functional_learning/training/runs/coherent86_continue_standard_seed43023/hf_model/chck_total_87005295w`

### Adapter displacement

All private-adapter delta norm `4.81734`, relative `0.074511`, cos(delta,parent) `-0.00955074897702909`.

### Fixed WWM probes

| section | CE current on | CE parent on | ΔCE current-parent | KL parent→current | KL carrier-off→current | targets |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_training_tail_prefix | 2.3385 | 2.3399 | -0.0014 | 0.001396 | 0.002422 | 12789 |
| new_continuation_tail_prefix | 2.4226 | 2.4275 | -0.0049 | 0.001446 | 0.002443 | 12875 |
| late_stream_prefix | 2.4224 | 2.4236 | -0.0012 | 0.001408 | 0.002428 | 12692 |

## cr_87M

Path: `experiments/archive/functional_learning/training/runs/coherent86_continue_carrier_residual_seed43023/hf_model/chck_total_87005295w`

### Adapter displacement

All private-adapter delta norm `5.22177`, relative `0.0807665`, cos(delta,parent) `0.0041349848346239415`.

### Fixed WWM probes

| section | CE current on | CE parent on | ΔCE current-parent | KL parent→current | KL carrier-off→current | targets |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_training_tail_prefix | 2.3433 | 2.3399 | +0.0034 | 0.005422 | 0.006868 | 12789 |
| new_continuation_tail_prefix | 2.4276 | 2.4275 | +0.0001 | 0.005444 | 0.006969 | 12875 |
| late_stream_prefix | 2.4279 | 2.4236 | +0.0043 | 0.005401 | 0.006878 | 12692 |

## std_final

Path: `experiments/archive/functional_learning/training/runs/coherent86_continue_standard_seed43023/hf_model/final`

### Adapter displacement

All private-adapter delta norm `8.62318`, relative `0.133377`, cos(delta,parent) `-0.03707340039825945`.

### Fixed WWM probes

| section | CE current on | CE parent on | ΔCE current-parent | KL parent→current | KL carrier-off→current | targets |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_training_tail_prefix | 2.3330 | 2.3399 | -0.0068 | 0.001555 | 0.002443 | 12789 |
| new_continuation_tail_prefix | 2.4185 | 2.4275 | -0.0090 | 0.001584 | 0.002432 | 12875 |
| late_stream_prefix | 2.4160 | 2.4236 | -0.0075 | 0.001578 | 0.002470 | 12692 |

## cr_final

Path: `experiments/archive/functional_learning/training/runs/coherent86_continue_carrier_residual_seed43023/hf_model/final`

### Adapter displacement

All private-adapter delta norm `9.54324`, relative `0.147608`, cos(delta,parent) `-0.01142226439653322`.

### Fixed WWM probes

| section | CE current on | CE parent on | ΔCE current-parent | KL parent→current | KL carrier-off→current | targets |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_training_tail_prefix | 2.3470 | 2.3399 | +0.0071 | 0.007703 | 0.009081 | 12789 |
| new_continuation_tail_prefix | 2.4336 | 2.4275 | +0.0061 | 0.007712 | 0.009128 | 12875 |
| late_stream_prefix | 2.4316 | 2.4236 | +0.0081 | 0.007697 | 0.009123 | 12692 |
