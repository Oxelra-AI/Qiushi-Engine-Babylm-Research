# earlier analysis frozen-bank rollback source-response control

All source-response rows in this file were scored on one newly materialized frozen bank. earlier analysis's Qwen comparison loaded parent/clean/dense source summaries from earlier analysis while regenerating rollback tasks, so its striking Qwen rollback sign is not used as evidence here.

## Ordinary-text KL match

- Clean KL(parent||model): `0.013327214029512199` over `270` calibration positions.
- Dense-mask alpha=1 KL: `0.028201220503593225`.
- Selected alpha: `0.775` with KL `0.013623876969210693` and relative error `0.02225993662603109`.
- Calibration bracket: lower alpha `0.7625` KL `0.01302356283658115`, upper alpha `0.775` KL `0.013623876969210693`.

## Frozen-bank source summaries

- `coherent86`: Qwen Δspecific Tfit `0.0`, Δrank `0.0`; common Δswing Tfit `0.0`, Δrank `0.0`, both `20`/25.
- `densemask_sparselabel_seed62064`: Qwen Δspecific Tfit `0.1523705547081772`, Δrank `51.799618055555555`; common Δswing Tfit `0.5443574689115799`, Δrank `117.79809523809524`, both `21`/25.
- `clean_pres_lambda1_eval_full80`: Qwen Δspecific Tfit `0.10681787415472273`, Δrank `33.59809027777778`; common Δswing Tfit `0.3458754409494855`, Δrank `74.87380952380953`, both `21`/25.
- `rollback_alpha_0p0`: Qwen Δspecific Tfit `0.0`, Δrank `0.0`; common Δswing Tfit `0.0`, Δrank `0.0`, both `20`/25.
- `rollback_alpha_0p7625`: Qwen Δspecific Tfit `0.0876489159079372`, Δrank `29.524791666666665`; common Δswing Tfit `0.4187026541005997`, Δrank `86.19047619047622`, both `21`/25.
- `rollback_alpha_0p775`: Qwen Δspecific Tfit `0.0904747925089517`, Δrank `30.355833333333333`; common Δswing Tfit `0.42576338259946744`, Δrank `87.90714285714289`, both `21`/25.
- `rollback_alpha_1p0`: Qwen Δspecific Tfit `0.152370608449629`, Δrank `51.799618055555555`; common Δswing Tfit `0.544358221760818`, Δrank `117.79809523809524`, both `21`/25.

## Endpoint interpolation checks

- `rollback_alpha_0_vs_parent`: qwen max |ΔNLL_T1| `0.0`, common max |ΔNLL_T1| `0.0`, pass `True`.
- `rollback_alpha_1_vs_densemask`: qwen max |ΔNLL_T1| `9.5367431640625e-06`, common max |ΔNLL_T1| `1.049041748046875e-05`, pass `False`.

## Interpretation

- source_bank_repair: Parent, acquisition-only dense-mask, clean preservation, and rollback alphas are scored on one frozen bank, so source-specific deltas now share task_id, target word, masked input, and wrong-source assignment.
- alpha_matching: Alpha is selected only by ordinary non-Qwen KL(parent||model). Behavioral source-response values are reported for the selected alpha and the calibration bracket to avoid treating a 6% KL mismatch as exact.
- endpoint_validation: Rollback alpha=0 must reproduce coherent86 and alpha=1 must reproduce dense-mask on this bank; failure would invalidate interpolation scoring rather than support a mechanism.
- mechanism_reading: If the selected/bracket rollback resembles clean preservation on source response as well as evidence-availability costs, drift attenuation explains most of the clean benefit. If clean remains materially different on the same bank, trajectory or training-time constraints remain plausible but must be separated from official score evidence.
