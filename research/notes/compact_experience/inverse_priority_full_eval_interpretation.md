# all mask endpoint interpretation inverse-priority full-evaluation interpretation

Visible leader: 41.800; clean seed43022: 41.344291; clean seed43122: 40.650052.

Official Overall includes SuperGLUE and AoA; no-AoA equal7/equal6 are separate internal measurements.

## true 100M: `true_100M_below_visible_leader`

- target: `mask_inverse_priority_100M`; endpoint: `chck_100M`; endpoint_frozen: `False`
- Overall: 39.714154598526164; Δvisible leader: -2.085845401473833; SuperGLUE: 69.83717271817582; AoA raw: -0.18344781331440352; AoA lb: -18.344781331440352; submit-ready: True
- no-AoA columns: BLiMP 67.39, Supplement 63.86, EWoK 50.88, Entity 28.55, COMPS 51.64, GlobalPIQA 35.635, Reading 7.98

## frozen 95M: `endpoint_frozen_below_visible_leader`

- target: `mask_inverse_priority_95M`; endpoint: `chck_95M`; endpoint_frozen: `True`
- Overall: 39.961218980736284; Δvisible leader: -1.8387810192637133; SuperGLUE: 69.78503419725679; AoA raw: -0.1719406337063024; AoA lb: -17.19406337063024; submit-ready: False
- no-AoA columns: BLiMP 67.33, Supplement 63.74, EWoK 51.05, Entity 28.66, COMPS 51.71, GlobalPIQA 36.635, Reading 7.9350000000000005

Next action: `inspect_column_collapse_before_spending_gpu_on_replication`.
