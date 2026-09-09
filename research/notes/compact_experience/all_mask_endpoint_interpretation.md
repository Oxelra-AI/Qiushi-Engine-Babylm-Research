# all mask endpoint interpretation all mask endpoint full-eval interpretation

Visible leader: 41.800; clean-Qwen reference: 41.344291.

Official Overall includes AoA in leaderboard units. no-AoA equal7 is shown only to interpret the source of changes.

| target | endpoint | ready | Overall | Overall if AoA=0 | AoA lb | AoA raw | SuperGLUE | no-AoA equal7 | needed AoA lb | Δleader |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mask_evidence_visible_100M | chck_100M | True | 40.083133 | 41.712749 | -14.667 | -0.1467 | 70.560 | 43.5507 | +0.785 | -1.7169 |
| mask_evidence_visible_90M | chck_90M | False | 40.034893 | 41.752228 | -15.456 | -0.1546 | 70.195 | 43.6536 | +0.430 | -1.7651 |
| mask_inverse_priority_95M | chck_95M | False | 39.961219 | 41.871670 | -17.194 | -0.1719 | 69.785 | 43.8657 | -0.645 | -1.8388 |
| mask_inverse_priority_100M | chck_100M | True | 39.714155 | 41.752464 | -18.345 | -0.1834 | 69.837 | 43.7050 | +0.428 | -2.0858 |
| mask_uniform_control_100M | chck_100M | True | 39.669536 | 41.369830 | -15.303 | -0.1530 | 69.848 | 43.2114 | +3.872 | -2.1305 |
| mask_uniform_control_90M | chck_90M | False | 39.582438 | 41.549416 | -17.703 | -0.1770 | 69.905 | 43.4343 | +2.255 | -2.2176 |

## Interpretation

- Best available endpoint: `mask_evidence_visible_100M` Overall 40.083133.
- Best submit-ready endpoint in this family: `mask_evidence_visible_100M` Overall 40.083133.
- Best true-100M endpoint: `mask_evidence_visible_100M` Overall 40.083133.
- If no endpoint has non-negative AoA with strong no-AoA columns, the masking-tail family is not a current SOTA route; use it as a mechanism clue for later AoA-safe continuation design.
