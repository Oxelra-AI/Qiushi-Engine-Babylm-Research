# Core Research Programs

These are relocated original programs, not newly written approximations. They retain distinct historical configurations and controls. This release preparation used static checks only; it did not retrain models.

| Scientific operation | Original implementation |
| --- | --- |
| First-generation backbone | [adapter_scaled_trainer.py](archive/frontier_consolidation/scripts/adapter_scaled_trainer.py) |
| Frozen-parent replay | [frozen82_fastpath_replay_trainer.py](archive/frontier_consolidation/scripts/frozen82_fastpath_replay_trainer.py) |
| Final acquisition and preservation | [clean_preservation_train.py](archive/functional_learning/scripts/clean_preservation_train.py) |
| Dense masking with sparse supervision | [densemask_sparselabel_train.py](archive/functional_learning/scripts/densemask_sparselabel_train.py) |
| Ordinary and focused continuation | [real_stream_train_weighted.py](archive/functional_learning/scripts/real_stream_train_weighted.py) |
| Word-paced bridge | [corrected_bridge_trainer.py](archive/functional_learning/scripts/corrected_bridge_trainer.py) |
| Dense supervision (focus probability 1, cap 128) | [real_stream_train_weighted.py](archive/functional_learning/scripts/real_stream_train_weighted.py) |
| Functional reach trajectory | [causal_interface_trajectory.py](archive/functional_learning/scripts/causal_interface_trajectory.py) |
| Held-symbol directions | [held_fitted_direction_test.py](archive/functional_learning/scripts/held_fitted_direction_test.py) |
| Causal signal intervention | [causal_intervention.py](archive/functional_learning/scripts/causal_intervention.py) |
| Split-window construction | [materialize_split_inwindow_controls.py](archive/relation_learning/scripts/materialize_split_inwindow_controls.py) |
| Split-window training | [train_split_inwindow_control.py](archive/relation_learning/scripts/train_split_inwindow_control.py) |
| Three-seed source-use readout | [original_threeseed_neutral_anchor.py](archive/relation_learning/scripts/original_threeseed_neutral_anchor.py) |
| Target-form controls | [component_and_targetclass_checks.py](archive/relation_learning/scripts/component_and_targetclass_checks.py) |
| Clean paired-text construction | [clean_materialize_qwen_pairs.py](archive/compact_experience/scripts/clean_materialize_qwen_pairs.py) |
| Compact density overlay | [materialize_density_on_cleanqwen_base_rowholdout.py](archive/frontier_consolidation/scripts/materialize_density_on_cleanqwen_base_rowholdout.py) |
| Continuation-tail construction | [revision_047b_reference_tail_builder.py](archive/functional_learning/scripts/revision_047b_reference_tail_builder.py) |
| Unchanged pair annotation | [annotate_unchanged_qwen_tail.py](archive/functional_learning/scripts/annotate_unchanged_qwen_tail.py) |
| Original column evaluation | [evaluate_compliant_endpoint.py](archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py) |
| Local AoA trajectory evaluation | [aoa_local_ckpts_minctx.py](archive/frontier_consolidation/scripts/aoa_local_ckpts_minctx.py) |

Read the [training and data guide](../reproducibility/TRAINING.md) before selecting an endpoint. Full source/configuration/result discovery is available through `python3 tools/inspect_materials.py list`.
