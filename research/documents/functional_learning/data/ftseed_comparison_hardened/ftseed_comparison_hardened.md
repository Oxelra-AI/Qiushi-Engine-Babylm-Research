# evidence hardening and pending ftseed44 hardened matched ftseed SuperGLUE comparison

Created: `2026-09-08T05:14:19Z`

This artifact compares only SuperGLUE primary-metric means. Optional Overall values from seeded wrappers are explicitly ignored.

Seed42 complete: `3/3`; seed44 valid complete: `0/3`.

## SuperGLUE primary means

| model | seed42 | seed44 | seed44 valid | Δ(44-42) |
|---|---:|---:|---:|---:|
| coherent86 | 68.945712 | - | False | - |
| ms_acquisition | 68.887842 | - | False | - |
| clean64 | 69.047711 | - | False | - |

## Key SuperGLUE comparisons

| comparison | seed42 | seed44 | sign relation |
|---|---:|---:|---|
| clean64_minus_coherent86 | +0.101999 | - | pending |
| clean64_minus_ms_preservation_specific | +0.159869 | - | pending |
| ms_minus_coherent86_acquisition_only | -0.057870 | - | pending |

## Task-level values

| task | metric | coherent86_s42 | coherent86_s44 | ms_s42 | ms_s44 | clean64_s42 | clean64_s44 |
|---|---|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | 67.278287 | - | 67.951070 | - | 67.951070 | - |
| multirc | accuracy | 68.275578 | - | 67.904290 | - | 68.151815 | - |
| rte | accuracy | 64.028777 | - | 63.309353 | - | 63.309353 | - |
| wsc | accuracy | 63.461538 | - | 63.461538 | - | 63.461538 | - |
| mrpc | f1 | 87.586207 | - | 87.889273 | - | 88.275862 | - |
| qqp | f1 | 71.557648 | - | 71.430417 | - | 71.589391 | - |
| mnli | accuracy | 60.431948 | - | 60.268949 | - | 60.594947 | - |

## Seed44 validity

- `coherent86`: missing
- `ms_acquisition`: missing
- `clean64`: missing

## Scientific reading

The matched downstream-seed comparison is not yet complete. Seed42 values are reproduced exactly from the completed clean replication and ms direct pending profile, and any available seed44 summaries are validated for SuperGLUE only. Rerun this script after coherent86, exact `(M,S)`, and clean64 seed44 summaries are all present and valid.
