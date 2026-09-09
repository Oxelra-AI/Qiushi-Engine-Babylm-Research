# Measurement Chronology and AoA Corrections

Scientific status: retrospective correction record; interim interpretations are identified explicitly.

An AoA value of 0.0 must not automatically be called a placeholder. The standalone 19-checkpoint measurement had raw correlation about -0.035 with p=0.60 and received a genuine clipped evaluator score of 0.0. A measured clipped zero is different from an unmeasured component filled with zero.

The initial correction distinguished a 19-checkpoint, 152,095-row trajectory with 225 fitted words from an 18-point shared trajectory with 144,090 rows and 504 vocabulary entries. The later clarification is more precise: the repaired main comparison used the shared 18-point trajectory, while the standalone first-generation measurement and its prediction package used 19 checkpoints. Both produced 0.0. Fitted-word counts, vocabulary entries, row counts, and trajectory lengths must not be interchanged.

Historical Overall 41.344 preceded the exposure audit. It was not a score obtained after the overlap had been removed. The later 33,431-hit finding withdrew the corresponding unseen-text generalization interpretation while retaining the trained-text-fit measurement. Its exact scope was diagnostic holdout/probe overlap, not a blanket finding about all official evaluation datasets.

The source-distance values 0.5026571895641201 for acquisition, 0.3525180022714145 for ordinary-state preservation, and 0.009356491809946708 for dense-state preservation are dense-state KL measurements from the common-support probe. They must not be substituted for ordinary-mask preservation-surface KL.

A figure comparing the preserved endpoint with matched ordinary continuation does not measure a difference from the first-generation model. Similarly, suppressing within-view completion evidence is not proof that only original-source evidence can explain a prediction. These distinctions correct the interpretation without changing any measurement.

Evidence: [common-support result](../../../experiments/archive/functional_learning/data/common_support_endpoint_probe_with_densecorr/common_support_endpoint_probe.json), [final AoA clarification](aoa_standalone_and_shared_trajectory_identity.md), [precise overlap correction](../functional_learning/overlap_scope_and_model_identity_corrections.md).
