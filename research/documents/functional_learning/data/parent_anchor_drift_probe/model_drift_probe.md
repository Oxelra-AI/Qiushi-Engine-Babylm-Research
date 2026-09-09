# clean d component ablation model drift and fixed-probe LM analysis

Parent: `models/frontier`

## pa_87M

Path: `experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023/hf_model/chck_total_87005295w`

### Adapter displacement

All private-adapter delta norm `4.79359`, relative `0.0741437`, cos(delta,parent) `-0.0022082763393947514`.

### Fixed WWM probes

| section | CE current on | CE parent on | ΔCE current-parent | KL parent→current | KL carrier-off→current | targets |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_training_tail_prefix | 2.3383 | 2.3399 | -0.0015 | 0.001385 | 0.002994 | 12789 |
| new_continuation_tail_prefix | 2.4225 | 2.4275 | -0.0050 | 0.001433 | 0.003011 | 12875 |
| late_stream_prefix | 2.4224 | 2.4236 | -0.0012 | 0.001392 | 0.002999 | 12692 |

## pa_final

Path: `experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023/hf_model/final`

### Adapter displacement

All private-adapter delta norm `8.52589`, relative `0.131872`, cos(delta,parent) `-0.03153909685999029`.

### Fixed WWM probes

| section | CE current on | CE parent on | ΔCE current-parent | KL parent→current | KL carrier-off→current | targets |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_training_tail_prefix | 2.3327 | 2.3399 | -0.0072 | 0.001512 | 0.003200 | 12789 |
| new_continuation_tail_prefix | 2.4183 | 2.4275 | -0.0091 | 0.001536 | 0.003180 | 12875 |
| late_stream_prefix | 2.4159 | 2.4236 | -0.0077 | 0.001535 | 0.003239 | 12692 |

## no_kl_87M

Path: `experiments/archive/functional_learning/training/runs/coherent86_continue_no_kl_seed43023/hf_model/chck_total_87005295w`

### Adapter displacement

All private-adapter delta norm `4.87879`, relative `0.0754615`, cos(delta,parent) `0.0011630738424591842`.

### Fixed WWM probes

| section | CE current on | CE parent on | ΔCE current-parent | KL parent→current | KL carrier-off→current | targets |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_training_tail_prefix | 2.3387 | 2.3399 | -0.0011 | 0.002029 | 0.003630 | 12789 |
| new_continuation_tail_prefix | 2.4224 | 2.4275 | -0.0051 | 0.002102 | 0.003673 | 12875 |
| late_stream_prefix | 2.4224 | 2.4236 | -0.0012 | 0.002034 | 0.003654 | 12692 |

## no_kl_final

Path: `experiments/archive/functional_learning/training/runs/coherent86_continue_no_kl_seed43023/hf_model/final`

### Adapter displacement

All private-adapter delta norm `8.84605`, relative `0.136824`, cos(delta,parent) `-0.02349736847784858`.

### Fixed WWM probes

| section | CE current on | CE parent on | ΔCE current-parent | KL parent→current | KL carrier-off→current | targets |
|---|---:|---:|---:|---:|---:|---:|
| coherent86_training_tail_prefix | 2.3330 | 2.3399 | -0.0069 | 0.002847 | 0.004622 | 12789 |
| new_continuation_tail_prefix | 2.4181 | 2.4275 | -0.0094 | 0.002892 | 0.004622 | 12875 |
| late_stream_prefix | 2.4160 | 2.4236 | -0.0075 | 0.002904 | 0.004736 | 12692 |
