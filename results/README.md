# Result Data

This directory is the sole numerical source for report tables and figures. The Chinese and English reports must not maintain separate copies of the same CSV files.

| File | Contents |
| --- | --- |
| [training_strategy_comparison.csv](training_strategy_comparison.csv) | Nine local endpoints, all nine component scores, continuation seeds and cumulative word counts |
| [leaderboard_comparison.csv](leaderboard_comparison.csv) | Two released model generations and eight external submissions; publicly displayed scores without a rank column |
| [public_model_revisions.csv](public_model_revisions.csv) | Public locations and pinned revisions for the two released model generations |
| [compact_budget.csv](compact_budget.csv) | Compact restatements and budget reallocation |
| [late_consolidation_controls.csv](late_consolidation_controls.csv) | Late continuation and incremental-learning controls |
| [private_scale_sweep.csv](private_scale_sweep.csv) | Scale selection for the dedicated incremental branch, not a privacy experiment |
| [relation_context_use.csv](relation_context_use.csv) | Changes in source advantage for compact-restatement targets whose token IDs are absent from the source, with standard deviations across three training seeds |
| [relation_window_controls.csv](relation_window_controls.csv) | Same-window and split-window controls for each relation, with two training seeds and explicit checkpoint aggregation |
| [natural_restatement_transfer.csv](natural_restatement_transfer.csv) | Natural-restatement transfer by target class, averaged within source pairs and across three training seeds |
| [finetuning_seed_comparison.csv](finetuning_seed_comparison.csv) | Existing (Super)GLUE results for three model endpoints under fine-tuning seeds 42 and 44, including all seven task scores |
| [blimp_scoring_comparison.csv](blimp_scoring_comparison.csv) | Reconciliation of option-index scoring and saved-answer text scoring for the two released models |
| [interface_reach.csv](interface_reach.csv) | Familiar/unseen query performance and internal functional interventions |
| [source_disjoint_target_loss.csv](source_disjoint_target_loss.csv) | Target losses with source-pair and document separation |
| [target_deletion_100m.csv](target_deletion_100m.csv) | Seven-component results under target deletion, not full Overall |
| [state_preservation_readouts.csv](state_preservation_readouts.csv) | Acquisition and preservation measurements under different input conditions |
| [preservation_gradient_diagnostic.csv](preservation_gradient_diagnostic.csv) | Preservation gradients and support-set diagnostics on common targets |
| [research_catalog.tsv](research_catalog.tsv) | Research topics, operations, findings, status and report references |

Public scores retain the report's snapshot from 8 September 2026 at 20:53 Beijing time. Nine-component Overall is the arithmetic mean of the nine top-level metrics. Other subset means, losses in nats and intervention selection rates are not Overall.

The BLiMP local reports use the selected option index; the released prediction files store the answer text. Seven evaluation items have identical good and bad answer strings. Text matching counts three additional answers as correct for the first model and five for the second, reproducing the displayed 68.52/68.27 rather than the local 68.51/68.26. The released predictions match the saved predictions from the corresponding local runs. The scoring comparison records this distinction without overwriting either result vector. No model inference was rerun for this reconciliation.

These files preserve the values from the checked report. This file reorganization did not retrain or reevaluate models, and the directory is not a live leaderboard.
