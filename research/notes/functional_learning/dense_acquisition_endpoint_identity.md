# Dense Acquisition Endpoint Identity

The endpoint `dense64_scale0p75` in the vocabulary/source-response comparison is
the dense-input, dense-target **(M,M)** model. A historical plotting label called
it (M,S); the [plotting source](../../../experiments/archive/functional_learning/scripts/regen_cdi_and_dense_common.py) now uses
the correct label. This is a label correction, not a change to the measurements.

The [probe configuration](../../../experiments/archive/relation_learning/data/shrinkage_source_cdi_probe/plan.json)
selects the repaired dense endpoint. Its
[loading repair record](../../../experiments/archive/functional_learning/data/automodel_repair/repair_validation.json) identifies
the original dense-focus checkpoint. The
[training configuration](../../../experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/train_config.json)
uses `focus_prob=1.0` with a cap of 128 content groups. In the
[original training implementation](../../../experiments/archive/functional_learning/scripts/real_stream_train_weighted.py), the
same selected positions are both masked and supervised. The
[realized summary](../../../experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/train_summary.json)
records 132,283 selected groups out of 132,283 candidate groups.

The later [dense-mask, sparse-label implementation](../../../experiments/archive/functional_learning/scripts/densemask_sparselabel_train.py)
selects the two supports separately. That (M,S) branch is a different checkpoint;
its label in the separate common-support panel remains unchanged.

No model, result table or existing figure was regenerated for this correction.
When reusing a historical rendered figure, check its endpoint label against this
record and the plotting source.
