# dual mechanism 20m overlap — scale1.75 100M endpoint readiness

This is a file-only training-artifact check. It does not run or replace the official-compatible full evaluator.

- Run: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder`
- Valid artifact checks: `True`; errors: `[]`
- Word exposure: 100000000; steps: 2529; loss first/last: 9.837543487548828 → 2.5441808700561523.
- Checkpoint ladder: 100 directories, expected `chck_1M` ... `chck_100M`; missing=0, extra=0.
- `chck_100M` record: target 100000000, actual 100000000.
- Config: architecture ['AdapterDebertaV2ForMaskedLM'], AutoModelForMaskedLM `adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM`, adapter scale 1.75, bottleneck 128, enabled True.
- 100M model SHA256: `7349475846ef2a193df850cc23876b2a86f99e82b9f051816064e4561b0f6f52`; size 141878192 bytes.

## Deterministic prefix reproduction

- 20M: new/old model hashes equal = `True`; SHA `b37e243b5fc9c9459efe0b6b4f2abce40f4597d82c9621949e0d458e85e7b031`.
- 50M: new/old model hashes equal = `True`; SHA `995c167832d83a721e1cccb6c3d4f348d11443283845ed5d72ba39bb8e802034`.
- 80M: new/old model hashes equal = `True`; SHA `c37f6665df84109a266428e606e1bf665db70d9a5e7842816b5717d36922edda`.

Scientific consequence: if `valid` is true, the 100M endpoint is a deterministic continuation of the previously interpreted 20M/50M/80M scale1.75 trajectory. The endpoint score itself is still unknown here and must come from the full nine-column official-compatible evaluator.

JSON: `experiments/archive/frontier_consolidation/data/scale1p75_endpoint_readiness/scale1p75_endpoint_readiness.json`
