# stabilization precheck while grids pending late-weight geometry probe

CPU/file-only parameter geometry for same-trajectory late-iterate stabilization. No BabyLM task evaluation, model upload, or leaderboard submission was performed.

## Main readout

- Reference late path has a directly measured turn from 82->84 to 84->86; total cosine=0.078230, adapter cosine=0.061092, backbone+heads cosine=0.078720.
- reference_scale1p75_seed43022: mean 2M update L2 total=1.837740 with mean successive cosine=0.101629; adapter mean successive cosine=0.088860.
- scale1p25_seed43022: mean 2M update L2 total=1.843914 with mean successive cosine=0.101662; adapter mean successive cosine=0.089062.
- scale1p75_seed43122: mean 2M update L2 total=1.840498 with mean successive cosine=0.100553; adapter mean successive cosine=0.086355.
- The built 80/82/84 average moves from 84M in a direction whose cosine with 84->86 is -0.057732 and with 84->100 is -0.038974; negative values mean the low-pass point backs away from later-training drift.
- These measurements can make a later weight-average screen interpretable, but they do not replace selected BabyLM scoring and should not be used to choose a public endpoint.

## Trajectory summary

| trajectory | mean 2M update L2 | mean update cosine | min update cosine | adapter mean cosine | turn 82→84 vs 84→86 total cosine |
|---|---:|---:|---:|---:|---:|
| reference_scale1p75_seed43022 | 1.837740 | 0.101629 | 0.061445 | 0.088860 | 0.078230 |
| scale1p25_seed43022 | 1.843914 | 0.101662 | 0.063125 | 0.089062 | 0.080160 |
| scale1p75_seed43122 | 1.840498 | 0.100553 | 0.062847 | 0.086355 | 0.079424 |

## Built average candidate geometry

Candidate: `experiments/archive/frontier_consolidation/data/late_weight_average_scaffold/candidates/reference_scale1p75_seed43022__center_80_82_84_uniform/model.safetensors`

Distances from the average to selected reference endpoints (total L2):

- `chck_80M`: total L2 `2.510192`, adapter L2 `0.414593`, backbone+heads L2 `2.475718`
- `chck_82M`: total L2 `1.408431`, adapter L2 `0.236399`, backbone+heads L2 `1.388450`
- `chck_84M`: total L2 `2.254685`, adapter L2 `0.371743`, backbone+heads L2 `2.223828`
- `chck_86M`: total L2 `3.242651`, adapter L2 `0.534061`, backbone+heads L2 `3.198369`
- `chck_100M`: total L2 `4.180735`, adapter L2 `0.684169`, backbone+heads L2 `4.124373`

Projection cosines for avg - 84M:

- `avg_minus_84_vs_80_to_84`: total `-0.951816`, adapter `-0.950022`, backbone+heads `-0.951866`
- `avg_minus_84_vs_82_to_84`: total `-0.866288`, adapter `-0.863548`, backbone+heads `-0.866366`
- `avg_minus_84_vs_84_to_86`: total `-0.057732`, adapter `-0.039779`, backbone+heads `-0.058242`
- `avg_minus_84_vs_84_to_100`: total `-0.038974`, adapter `-0.017077`, backbone+heads `-0.039588`

## Reference score rows used only for orientation

- `chck_80M` cheap7 `43.812143`
- `chck_82M` cheap7 `43.958571`
- `chck_84M` cheap7 `44.123571`
- `chck_86M` cheap7 `43.770714`
- `chck_100M` cheap7 `43.542143`

JSON: `experiments/archive/frontier_consolidation/data/late_weight_geometry_probe/late_weight_geometry_probe.json`
