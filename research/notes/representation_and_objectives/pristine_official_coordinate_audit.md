# official coordinate and seed43122 pristine official coordinate audit

Purpose: build a fresh upstream BabyLM evaluation coordinate, hash the evaluation data, and decide whether the EWoK size mismatch comes from old local files or from a code/data release inconsistency.

- Audit JSON: `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/pristine_official_coordinate_audit.json`
- Fresh code directory: `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval`
- Expected code commit: `6f825c291e2c4c78ad33b1935fd64d45f52642dc`; observed head: `6f825c291e2c4c78ad33b1935fd64d45f52642dc`
- Strict eval HF dataset revision: `8d52da9424a9ff30b9e8266c4f751aba9c504233`
- EWoK HF dataset revision: `34d912a608066c92e2990a0328ffc3bd9a716042`
- Generated EWoK counts equal local counts: False
- Collator EWOK_SIZES equal source raw domain counts×2: False
- Collator EWOK_SIZES equal generated ewok_filtered counts: False

## EWoK domain counts

| domain | collator constant | pristine ewok_filtered | local ewok_filtered | raw source×2 | filtered source×2 |
|---|---:|---:|---:|---:|---:|
| agent-properties | 2210 | None | 1846 | None | None |
| material-dynamics | 770 | None | 770 | None | None |
| material-properties | 170 | None | 130 | None | None |
| physical-dynamics | 120 | None | 100 | None | None |
| physical-interactions | 556 | None | 436 | None | None |
| physical-relations | 818 | None | 818 | None | None |
| quantitative-properties | 314 | None | 276 | None | None |
| social-interactions | 294 | None | 294 | None | None |
| social-properties | 328 | None | 308 | None | None |
| social-relations | 1548 | None | 1548 | None | None |
| spatial-relations | 490 | None | 140 | None | None |

## Pristine collation probe with old min_context=20 AoA

- Return code: `0`
- AoA surprisal null: `True`
- EWoK null: `True`
- Collator stdout excerpt:

```
The sub-data agent-properties from ewok has 1846 datapoints, when it should have 2210 datapoints!
The sub-data material-properties from ewok has 130 datapoints, when it should have 170 datapoints!
The sub-data physical-dynamics from ewok has 100 datapoints, when it should have 120 datapoints!
The sub-data physical-interactions from ewok has 436 datapoints, when it should have 556 datapoints!
The sub-data quantitative-properties from ewok has 276 datapoints, when it should have 314 datapoints!
The sub-data social-properties from ewok has 308 datapoints, when it should have 328 datapoints!
The sub-data spatial-relations from ewok has 140 datapoints, when it should have 490 datapoints!
Warning: EWoK data has incorrect size, setting to None
There are 6560 predictions for checkpoint chck_1M in AoA data, when there should be 8005
Warning: AoA word data has incorrect size, setting to None

```

## Interpretation

If the pristine official script regenerates the same ewok_filtered counts/files as the local vendored data while EWOK_SIZES matches the raw source-domain counts rather than the filtered files, then the mismatch is an upstream code/data coordinate inconsistency, not an local corruption. The endpoint must still be reproduced through one accepted official coordinate before using the scalar against the public leader.

No official code or data were patched. This audit does not replace the pending official-min_context=0 AoA rerun; it defines the coordinate that the corrected endpoint must pass through end to end.
