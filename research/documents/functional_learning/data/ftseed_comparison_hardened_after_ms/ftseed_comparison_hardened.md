# evidence hardening and pending ftseed44 hardened matched ftseed SuperGLUE comparison

Created: `2026-09-08T06:43:40Z`

This artifact compares only SuperGLUE primary-metric means. Optional Overall values from seeded wrappers are explicitly ignored.

Seed42 complete: `3/3`; seed44 valid complete: `3/3`.

## SuperGLUE primary means

| model | seed42 | seed44 | seed44 valid | Δ(44-42) |
|---|---:|---:|---:|---:|
| coherent86 | 68.945712 | 69.317987 | True | +0.372275 |
| ms_acquisition | 68.887842 | 69.346053 | True | +0.458211 |
| clean64 | 69.047711 | 69.626595 | True | +0.578884 |

## Key SuperGLUE comparisons

| comparison | seed42 | seed44 | sign relation |
|---|---:|---:|---|
| clean64_minus_coherent86 | +0.101999 | +0.308608 | same sign |
| clean64_minus_ms_preservation_specific | +0.159869 | +0.280542 | same sign |
| ms_minus_coherent86_acquisition_only | -0.057870 | +0.028066 | opposite sign |

## Task-level values

| task | metric | coherent86_s42 | coherent86_s44 | ms_s42 | ms_s44 | clean64_s42 | clean64_s44 |
|---|---|---:|---:|---:|---:|---:|---:|
| boolq | accuracy | 67.278287 | 67.889908 | 67.951070 | 67.706422 | 67.951070 | 67.706422 |
| multirc | accuracy | 68.275578 | 68.028053 | 67.904290 | 67.161716 | 68.151815 | 67.904290 |
| rte | accuracy | 64.028777 | 66.187050 | 63.309353 | 66.906475 | 63.309353 | 69.064748 |
| wsc | accuracy | 63.461538 | 61.538462 | 63.461538 | 61.538462 | 63.461538 | 61.538462 |
| mrpc | f1 | 87.586207 | 88.732394 | 87.889273 | 88.501742 | 88.275862 | 87.889273 |
| qqp | f1 | 71.557648 | 72.051347 | 71.430417 | 72.319859 | 71.589391 | 72.280525 |
| mnli | accuracy | 60.431948 | 60.798696 | 60.268949 | 61.287694 | 60.594947 | 61.002445 |

## Seed44 validity

- `coherent86`: valid=True path=`experiments/archive/functional_learning/data/matched_ftseed44_superglue/coherent86_alpha075_ftseed44/faithful_superglue_seeded_summary.json` errors=[]
- `ms_acquisition`: valid=True path=`experiments/archive/functional_learning/data/ms_ftseed44_superglue/densemask_sparselabel_seed62064_ftseed44/faithful_superglue_seeded_summary.json` errors=[]
- `clean64`: valid=True path=`experiments/archive/relation_learning/data/clean_pres62064_ftseed44_superglue/clean_pres62064_u0080_ftseed44/faithful_superglue_seeded_summary.json` errors=[]

## Scientific reading

All three seed44 SuperGLUE summaries are present and valid. Interpret the seed44 clean64-minus-exact-`(M,S)` difference as a matched downstream-finetuning-seed check of the SuperGLUE portion of the preservation-specific increment, not as a new Overall coordinate. The zero/Reading/AoA components remain those of the fixed official evaluation; only the downstream fine-tuning randomness in SuperGLUE has been perturbed.
