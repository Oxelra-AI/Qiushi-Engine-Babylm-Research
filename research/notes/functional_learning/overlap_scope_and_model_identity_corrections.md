# Overlap Scope and Model-Identity Corrections

Scientific status: retrospective corrections to the interpretation of existing measurements.

## Exposure Audit Scope

The audit covered 37,594 pairs, or 75,188 texts. It found 24,071 matches to a 6,992-row reference set, 9,359 to a 2,647-row reference set, and one to a Wikipedia simplification probe: 33,431 blocking hits in total. This was overlap with diagnostic reference material, not a demonstration that every official benchmark had leaked.

The 6,992-row set was not genuinely held out from both training arms. The recorded NLL contrasts were -0.0166/-0.0215 on hit rows versus -0.0104/-0.0084 on clean rows. These values must retain their separate subset identities. The affected results can support familiar-text fitting claims but not the original unseen-text generalization claim.

Exposure findings occurred after the historical evaluation. They must not be narrated as a cleaning operation performed before that endpoint was trained or scored. Recombining fixed corpora can reintroduce reference text, so the relevant unit of audit is the constructed stream, not only the initial corpus list.

## Model Identity Through Serialization

A prediction interface must preserve the trained private-adapter computation. The saved configuration's automatic model mapping must continue to select the adapter-enabled implementation after save and reload. The recorded full load, forward, save, reload, and equality checks gave maximum logit difference 0.0 and encoder hidden-state difference 0.0 for both representative packages; gradients still reached all 48 private tensors.

This is reproducibility evidence about the evaluated function, not a new benchmark run. A stock loader that omits the trained branch defines another model and another score coordinate.

Evidence: [original save/reload verification](../../../experiments/archive/functional_learning/data/package_save_reload/package_save_reload_verify.json), [corrected policy coordinate](endpoint_policy_interpretation_corrections.md), [measurement chronology](../relation_learning/measurement_chronology_and_aoa_corrections.md).
