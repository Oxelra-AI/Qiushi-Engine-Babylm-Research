# earlier analysis dense-corruption endpoint validation

Created: `2026-09-08T08:00:01Z`

All validation measurements pass: `True`.

## Endpoint

- run directory: `experiments/archive/functional_learning/data/densecorruption_preservation_lambda1_full80`
- endpoint: `experiments/archive/functional_learning/data/densecorruption_preservation_lambda1_full80/checkpoints/update_0080`
- endpoint model SHA256: `36851b571be1babbe933e29e019a63903c9cff98a98dd70cbcdf85bee95a0c49`
- parent model SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`

## Run and support measurements

- completed updates: `80`
- schedule indices: `[101, 102, 103, 104, 105] ... [176, 177, 178, 179, 180]`
- acquisition words: `3162742`
- counted preservation words: `517332`
- conservative endpoint exposure: `89685369`
- preservation targets: `148017`
- final preservation KL: `0.00803593651754727`

## Parameter movement

- changed tensors: `48`
- changed private tensors: `48`
- changed non-private tensors: `0`
- max private delta: `0.0025386190973222256`
- RMS private delta: `0.0006197825258640293`

## Measurements

| measurement | pass |
|---|---:|
| identity.method_matches_densecorr_policy | `True` |
| identity.completed_updates_80 | `True` |
| identity.train_seed_62064 | `True` |
| identity.private_optimizer_48_tensors | `True` |
| identity.private_optimizer_995584_params | `True` |
| identity.student_private_scale_075 | `True` |
| identity.teacher_private_scale_075 | `True` |
| identity.student_teacher_intended_class | `True` |
| identity.lambda_pres_1 | `True` |
| identity.focus_lambda_015 | `True` |
| identity.kl_direction_teacher_to_student | `True` |
| identity.pres_student_eval_mode | `True` |
| identity.prefix_words_3162742 | `True` |
| identity.pres_words_517332 | `True` |
| identity.endpoint_exposure_89685369 | `True` |
| identity.focus_targets_28590 | `True` |
| identity.ordinary_targets_592858 | `True` |
| identity.preservation_targets_148017 | `True` |
| identity.support_counts_expected | `True` |
| identity.checkpoint_0080_exists | `True` |
| update_log.n_update_logs_80 | `True` |
| update_log.updates_1_to_80 | `True` |
| update_log.schedule_idx_101_to_180 | `True` |
| update_log.all_lr_match_schedule | `True` |
| update_log.sum_words_match_summary | `True` |
| update_log.sum_pres_words_match_summary | `True` |
| update_log.sum_focus_targets_match_summary | `True` |
| update_log.sum_ordinary_targets_match_summary | `True` |
| update_log.sum_pres_targets_match_summary | `True` |
| update_log.support_counts_match_summary | `True` |
| update_log.support_counts_match_expected | `True` |
| update_log.final_update_is_80 | `True` |
| update_log.final_lr_match_schedule | `True` |
| update_log.final_kl_matches_recorded | `True` |
| rng.three_rng_records_present | `True` |
| rng.updates_1_2_3_recorded | `True` |
| rng.all_restored_after_preservation | `True` |
| rng.all_forked_preservation_forward_rng | `True` |
| rng.after_restore_equals_after_acquisition_cuda_digest | `True` |
| rng.after_restore_equals_before_acquisition_cpu_digest | `True` |
| tensor_delta.parent_and_endpoint_same_keyset | `True` |
| tensor_delta.exactly_48_changed_private_tensors | `True` |
| tensor_delta.no_non_private_tensor_movement | `True` |
| tensor_delta.private_only_changed | `True` |

## Scientific use

This endpoint is validated as a private-adapter-only dense-corruption preservation policy with matched schedule and recorded support totals. Use it for bounded acquisition-retention readouts; it is not promoted as a release candidate by this validation alone.
