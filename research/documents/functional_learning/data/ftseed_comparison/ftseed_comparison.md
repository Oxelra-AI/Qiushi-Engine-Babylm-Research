# scientific account three levels matched finetuning-seed SuperGLUE comparison

Created: `2026-09-08T05:00:57Z`

Complete: seed42=3/3, seed44=0/3

## SuperGLUE primary-metric means

| model | seed42 | seed44 | Δ(44-42) |
|---|---|---|---|
| coherent86 | 68.94571192183594 | missing | - |
| ms_acquisition | 68.88784158902877 | missing | - |
| clean64 | 69.047710989885 | missing | - |

## Key comparisons

| comparison | seed42 | seed44 | sign consistent |
|---|---|---|---|
| clean64_minus_coherent86 | +0.1020 | - | - |
| clean64_minus_ms_preservation_specific | +0.1599 | - | - |
| ms_minus_coherent86_acquisition_only | -0.0579 | - | - |

## Task-level details

| task | metric | coherent86_s42 | coherent86_s44 | ms_s42 | ms_s44 | clean64_s42 | clean64_s44 |
|---|---|---|---|---|---|---|---|
| boolq | accuracy | 67.2783 | - | 67.9511 | - | 67.9511 | - |
| mnli | accuracy | 60.4319 | - | 60.2689 | - | 60.5949 | - |
| mrpc | f1 | 87.5862 | - | 87.8893 | - | 88.2759 | - |
| multirc | accuracy | 68.2756 | - | 67.9043 | - | 68.1518 | - |
| qqp | f1 | 71.5576 | - | 71.4304 | - | 71.5894 | - |
| rte | accuracy | 64.0288 | - | 63.3094 | - | 63.3094 | - |
| wsc | accuracy | 63.4615 | - | 63.4615 | - | 63.4615 | - |

**Note:** Only 0/3 seed44 results are available. Missing: coherent86, ms_acquisition, clean64. Rerun this script after all jobs complete.
