# earlier analysis frozen-bank rollback source-response control

All source-response rows in this file were scored on one newly materialized frozen bank. earlier analysis's Qwen comparison loaded parent/clean/dense source summaries from earlier analysis while regenerating rollback tasks, so its striking Qwen rollback sign is not used as evidence here.

## Ordinary-text KL match

- Clean KL(parent||model): `0.007061894240905531` over `33` calibration positions.
- Dense-mask alpha=1 KL: `0.015065510273747019`.
- Selected alpha: `0.75` with KL `0.007379046892200484` and relative error `0.04491042211562281`.
- Calibration bracket: lower alpha `0.5` KL `0.0029121054550077215`, upper alpha `0.75` KL `0.007379046892200484`.

## Frozen-bank source summaries

- `coherent86`: Qwen Δspecific Tfit `0.0`, Δrank `0.0`; common Δswing Tfit `0.0`, Δrank `0.0`, both `20`/25.
- `densemask_sparselabel_seed62064`: Qwen Δspecific Tfit `0.10219120979309082`, Δrank `212.71875`; common Δswing Tfit `0.5443574689115799`, Δrank `117.79809523809524`, both `21`/25.
- `clean_pres_lambda1_eval_full80`: Qwen Δspecific Tfit `0.06465880572795868`, Δrank `150.5`; common Δswing Tfit `0.3458754409494855`, Δrank `74.87380952380953`, both `21`/25.
- `rollback_alpha_0p0`: Qwen Δspecific Tfit `0.0`, Δrank `0.0`; common Δswing Tfit `0.0`, Δrank `0.0`, both `20`/25.
- `rollback_alpha_0p5`: Qwen Δspecific Tfit `0.019182369112968445`, Δrank `74.59375`; common Δswing Tfit `0.26612446291106084`, Δrank `51.93809523809526`, both `21`/25.
- `rollback_alpha_0p75`: Qwen Δspecific Tfit `0.047783441841602325`, Δrank `137.59375`; common Δswing Tfit `0.41160754104455316`, Δrank `84.77714285714288`, both `21`/25.
- `rollback_alpha_1p0`: Qwen Δspecific Tfit `0.10219018906354904`, Δrank `212.71875`; common Δswing Tfit `0.544358221760818`, Δrank `117.79809523809524`, both `21`/25.

## Endpoint interpolation checks

- `rollback_alpha_0_vs_parent`: qwen max |ΔNLL_T1| `0.0`, common max |ΔNLL_T1| `0.0`, pass `True`.
- `rollback_alpha_1_vs_densemask`: qwen max |ΔNLL_T1| `5.7220458984375e-06`, common max |ΔNLL_T1| `1.049041748046875e-05`, pass `False`.

## Interpretation

- source_bank_repair: Parent, acquisition-only dense-mask, clean preservation, and rollback alphas are scored on one frozen bank, so source-specific deltas now share task_id, target word, masked input, and wrong-source assignment.
- alpha_matching: Alpha is selected only by ordinary non-Qwen KL(parent||model). Behavioral source-response values are reported for the selected alpha and the calibration bracket to avoid treating a 6% KL mismatch as exact.
- endpoint_validation: Rollback alpha=0 must reproduce coherent86 and alpha=1 must reproduce dense-mask on this bank; failure would invalidate interpolation scoring rather than support a mechanism.
- mechanism_reading: If the selected/bracket rollback resembles clean preservation on source response as well as evidence-availability costs, drift attenuation explains most of the clean benefit. If clean remains materially different on the same bank, trajectory or training-time constraints remain plausible but must be separated from official score evidence.
