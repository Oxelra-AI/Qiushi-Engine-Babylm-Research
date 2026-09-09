# Corrected-tokenizer endpoint interpretation

Current public Strict-Small leader: `wwm_curriculum_simplification_40k` Overall `41.8` fetched `2026-08-30T00:43:20.858782+00:00`.

Training artifacts complete: `True`. Corrected seeds present: `['43022', '43122']`.

## Scientific reading

- Neither corrected-tokenizer seed clears the current visible Strict-Small leader 41.8; the old inherited-tokenizer 42.033 result remains mechanism evidence, not a compliant endpoint.
- Corrected two-seed mean Overall 40.863975; seed spread 0.320038.
- legal tokenizer SHA 4a95a2a2ead2 has zero missing ByteLevel alphabet entries, zero <unk> on the 10M pool, and zero <unk> across 389343 official scored-text strings; separate same-pool tokenizer construction issue does not reclassify running evaluations.
- Old inherited-tokenizer EWoK atlas is a full-coordinate mechanism baseline: 7618 rows, treatment effect seed43022 +1.706pp, seed43122 -0.683pp, interaction -2.389pp, negative-interaction rows 1766, and only 9.29% of those have all four margins within ±1. Strongest old negative domains: material-dynamics -17.53pp, physical-dynamics -10.83pp, spatial-relations -8.37pp, physical-interactions -5.94pp, social-relations -2.13pp.
- Training-text concept exposure does not explain the old EWoK instability: negative-interaction rows have both EWoK concepts in the full 10M pool 84.82% of the time and in the compact changed block 60.87%; exposure correlations with old accuracy interaction are near zero (min-full r=-0.0171, sum-full r=-0.0168).
- On the same 553-row focused EWoK subset, item-level old-vs-token-length features have max |Pearson r| 0.041 with corrected 100M margins, so the local corrected relation behavior is not a crude length-fragmentation artifact.
- Seed43022 corrected-minus-inherited largest column movements: Supplement -3.995, EWoK -3.699, SuperGLUE -1.862, Entity -1.550.
- Seed43022 EWoK falls by -3.699 while tokenizer lengthens EWoK text by ratio 1.0322; inspect EWoK domains and old-atlas row classes before attributing loss to the compact-view data mechanism alone.
- Seed43122 corrected-minus-inherited largest column movements: Supplement -6.185, GlobalPIQA +3.471, EWoK -2.466, Entity +2.302.
- Seed43122 EWoK falls by -2.466 while tokenizer lengthens EWoK text by ratio 1.0322; inspect EWoK domains and old-atlas row classes before attributing loss to the compact-view data mechanism alone.

## Next action

`rebuild_legal_representation_data_or_learning_dynamics_from_score_movement`

## Files used

- corrected_two_seed_comparison: `experiments/archive/representation_and_objectives/data/corrected_tokenizer_two_seed_comparison/corrected_tokenizer_two_seed_comparison.json`
- training_completion: `experiments/archive/representation_and_objectives/data/strictsmalltok_training_completion/strictsmalltok_training_completion_summary.json`
- a01_tokenizer_surface: `experiments/archive/representation_and_objectives/data/tokenizer_surface_contingency/tokenizer_surface_contingency.json`
- relation_phase_dynamics: `experiments/archive/representation_and_objectives/data/relation_phase_dynamics_summary/relation_phase_dynamics_summary.json`
- old_ewok_atlas: `experiments/archive/representation_and_objectives/data/full_old_ewok_atlas_synthesis/full_old_ewok_atlas_synthesis.json`
- training_exposure_link: `experiments/archive/representation_and_objectives/data/ewok_old_atlas_training_exposure_link/ewok_old_atlas_training_exposure_link.json`
- a01_byte_coverage_record: `experiments/archive/representation_and_objectives/data/tokenizer_byte_coverage_audit/tokenizer_byte_coverage_audit.json`
- focus_tokenization_margin_link: `experiments/archive/representation_and_objectives/data/ewok_focus_tokenization_margin_link/ewok_focus_tokenization_margin_link.json`
- leaderboard: `experiments/archive/representation_and_objectives/data/babylm2026_live_surface/leaderboard_parsed.json`
- note: `research/notes/representation_and_objectives/corrected_endpoint_interpretation.md`
