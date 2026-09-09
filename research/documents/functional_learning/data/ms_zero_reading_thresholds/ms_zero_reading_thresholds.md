# clean replication and ms direct pending thresholds for pending exact `(M,S)` zero-shot/Reading

The exact acquisition-only `(M,S)` endpoint has completed repaired-AutoModel SuperGLUE and measured AoA. Its official zero-shot/Reading run is still pending. This file gives threshold arithmetic only; it does not infer the pending scores.

## Known `(M,S)` components

- SuperGLUE: 68.887842
- AoA: 0.000000

## Seven-component zero/Reading sums already observed

- coherent86: sum 309.270000, mean-over-7 44.181429
- dense_seed62064: sum 310.810000, mean-over-7 44.401429
- dense_seed62065: sum 310.710000, mean-over-7 44.387143
- clean_pres_lambda1_eval_seed62064: sum 311.170000, mean-over-7 44.452857
- clean_pres_lambda1_eval_seed62065_zero_reading_only: sum 311.065000, mean-over-7 44.437857

## Required zero/Reading sums for exact `(M,S)`

| target to equal | target Overall | required zero/Reading sum | required mean over 7 | above coherent86 zero sum | above dense64 zero sum | above clean64 zero sum |
|---|---:|---:|---:|---:|---:|---:|
| coherent86 | 42.023968 | 309.327870 | 44.189696 | 0.057870 | -1.482130 | -1.842130 |
| dense_seed62064 | 42.149091 | 310.453978 | 44.350568 | 1.183978 | -0.356022 | -0.716022 |
| dense_seed62065 | 42.168423 | 310.627963 | 44.375423 | 1.357963 | -0.182037 | -0.542037 |
| clean_pres_lambda1_eval_seed62064 | 42.246412 | 311.329869 | 44.475696 | 2.059869 | 0.519869 | 0.159869 |

## Transparent scenarios using completed official zero/Reading blocks

| scenario | Overall with `(M,S)` SG/AoA | delta vs coherent86 | delta vs clean64 | delta vs dense64 | delta vs dense65 |
|---|---:|---:|---:|---:|---:|
| ms_zero_reading_equals_coherent86 | 42.017538 | -0.006430 | -0.228874 | -0.131553 | -0.150885 |
| ms_zero_reading_equals_dense_seed62064 | 42.188649 | 0.164681 | -0.057763 | 0.039558 | 0.020226 |
| ms_zero_reading_equals_dense_seed62065 | 42.177538 | 0.153570 | -0.068874 | 0.028447 | 0.009115 |
| ms_zero_reading_equals_clean_pres_lambda1_eval_seed62064 | 42.228649 | 0.204681 | -0.017763 | 0.079558 | 0.060226 |
| ms_zero_reading_equals_clean_seed62065_zero_reading | 42.216982 | 0.193014 | -0.029430 | 0.067891 | 0.048560 |

## Scientific reading

Exact `(M,S)` SuperGLUE is -0.057870 below coherent86 and -0.159869 below clean seed62064. To equal coherent86 Overall, `(M,S)` needs its zero/Reading sum to be 0.057870 above coherent86's seven-component sum. To equal clean seed62064 Overall, it would need 0.159869 above clean64's already observed zero/Reading sum. Therefore if `(M,S)` zero/Reading matches clean64 exactly, clean still keeps an Overall advantage of 0.017763, equal to the known SuperGLUE difference divided by nine. The direct comparison was unresolved at the time of this note because the matched zero/Reading evaluation was not yet complete and validated.
