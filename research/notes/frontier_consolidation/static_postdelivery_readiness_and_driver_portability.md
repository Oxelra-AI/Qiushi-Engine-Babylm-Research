# static postdelivery readiness and driver portability — post-delivery evaluation readiness and path-portability repair

## Analysis Scope

This static analysis adds no further newline or subset comparisons. Active training outputs were not inspected, no model evaluation was performed, and no endpoint score is inferred.

Pending training comparisons:

- spatial repair route status same-pool compliant-tokenizer reinvest retrain, target `complianttok_reinvest_seed43022`.
- byte-alphabet compliant-tokenizer reinvest retrain, target `bytealphatok_reinvest_seed43022`.

The useful execution work was to verify that the already prepared post-delivery path can be run immediately once the runtime delivers a terminal retrain result.







No peer update was received.

## Static post-delivery readiness check

Script created:

- `experiments/archive/frontier_consolidation/scripts/static_postdelivery_readiness_check.py`

Final outputs:

- `experiments/archive/frontier_consolidation/data/postdelivery_static_readiness/static_postdelivery_readiness.json`
- `research/documents/frontier_consolidation/data/postdelivery_static_readiness/static_postdelivery_readiness.md`

Final status:

- `STATIC_POSTDELIVERY_READY`

The check only reads static scripts, tokenizer artifacts, and dry-run post-delivery manifests. It does not inspect active run directories.

Static results:

- Script syntax: 5/5 passing.
  - `compliant_postdelivery_driver.py`
  - `inspect_compliant_retrain.py`
  - `evaluate_compliant_endpoint.py`
  - `project_compliant_eval_continuation.py`
  - `supplement_prediction_slices.py`
- Tokenizer artifacts:
  - `complianttok_reinvest_seed43022`: vocab 16,384, tokenizer SHA256 `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9` matched.
  - `bytealphatok_reinvest_seed43022`: vocab 16,384, tokenizer SHA256 `b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf` matched.
- Dry-run post-delivery manifests:
  - `complianttok_reinvest_seed43022`: all expected stages present and isolated output paths preserved.
  - `bytealphatok_reinvest_seed43022`: all expected stages present and isolated output paths preserved.
- The two pristine-collate summary paths are distinct.
- No dry-run target writes to `deliverables/`.

## Driver portability repair

Patched:

- `experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py`

Reason:

- A static check found the dry-run manifest used absolute filesystem paths for the pristine-collate `--model-root`, because the driver called `.resolve()` before building that command.
- This was not a scoring change, but root-relative command paths are safer and clearer for portable execution and subsequent comparisons.

Change:

- Added `path_arg(...)`, which emits a path relative to the recorded root when possible.
- Applied it to inspector, evaluation, Supplement slice, continuation, and pristine-collate command arguments.
- Removed `.resolve()` from the collator-facing paths.

Regenerated dry-run manifests:

- `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/complianttok_reinvest_seed43022_postdelivery_driver.json`
- `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/bytealphatok_reinvest_seed43022_postdelivery_driver.json`

The regenerated manifests now use root-relative paths for:

- run directory
- model root
- evaluation root
- collation root
- AoA output root
- posthoc Supplement slice output root

## Endpoint policy preserved

- `bytealphatok_reinvest_seed43022` keeps resource priority because it is the standard ByteLevel-alphabet compliant tokenizer endpoint and removes avoidable `<unk>` coverage on official strings.
- `complianttok_reinvest_seed43022` remains a legal spatial repair route status same-pool tokenizer endpoint. Evaluate it if it delivers and doing so does not delay the byte-alphabet priority path.
- Endpoint selection must come from isolated pristine official nine-column collation, not tokenizer speculation.
- Do not change tokenizer design, do not mine more evaluation subsets before results, and do not start new 100M training until actual compliant endpoint results support a new route.

## Exact next action when a terminal result arrives

For target `bytealphatok_reinvest_seed43022`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py \
  --arm reinvest \
  --target bytealphatok_reinvest_seed43022 \
  --run-dir experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022 \
  --expected-tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet \
  --expected-tokenizer-sha256 b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf \
  --gpu 1
```

For target `complianttok_reinvest_seed43022`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py \
  --arm reinvest \
  --target complianttok_reinvest_seed43022 \
  --run-dir experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2 \
  --expected-tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer \
  --expected-tokenizer-sha256 91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9 \
  --gpu 0
```

The driver will inspect the delivered endpoint, run cheaper columns first, run the existing Supplement slice only as posthoc interpretation, apply the hard upper-bound continuation rule, then run SuperGLUE/AoA and pristine collation when warranted.
