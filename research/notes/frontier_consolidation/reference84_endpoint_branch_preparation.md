# partial deberta grid and endpoint branch reference `chck_84M` endpoint-branch preparation

## Scientific context

The protected reference trajectory was previously selected at `chck_82M` because the available dense panel included 77--83M and 100M. The newly running common 70--100M/2M selected-grid scoring has revealed in a bounded diagnostic tail that `chck_84M` may have higher cheap7 than `chck_82M`:

- diagnostic `chck_82M`: cheap7 43.958571; scores BLiMP 68.48, Supplement 62.94, EWoK 50.06, Entity 28.31, COMPS 52.19, GlobalPIQA 37.58, Reading 8.15.
- diagnostic `chck_84M`: cheap7 44.123571; scores BLiMP 68.25, Supplement 63.48, EWoK 50.07, Entity 28.58, COMPS 52.21, GlobalPIQA 38.12, Reading 8.155.

This partial result requires confirmation from the completed reference common grid and integrity check. If confirmed, the cheap7 improvement is not purely GlobalPIQA/Reading-carried: excluding GlobalPIQA, cheap6 improves by about +0.1025, with Supplement and Entity gains offsetting the BLiMP decline.

## Endpoint branch, not mechanism branch

A confirmed `chck_84M` improvement would be a separate endpoint opportunity on the already legal scale1.75 seed43022 trajectory. It would not explain why compact-view reinvestment works, and it must not become the organizer of the transferable-learning mechanism route. The source conditioned ordering interaction synthesis source-use probe and lead route assessment after source use probe route assessment still govern mechanism work.

## Provenance already audited

Script: `scripts/checkpoint_identity_audit.py`

Output: `data/checkpoint_identity_audit/chck_84M_identity_audit.{json,md}`

Audited `chck_84M` identity:

- run: `training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder`
- actual exposure: 84,028,405 words = 8.4028405 epochs of the 10M legal pool
- model SHA256: `2217917c687faf4de26ef6f381be3048d0bd66b2025c06382683c24d78e8d8c9`
- static files match `chck_82M` except model weights: true
- trusted-code CPU load: true; class `AdapterDebertaV2ForMaskedLM`; parameter count 35,463,008

## Prepared missing-column tool

Script: `scripts/superglue_for_selected_mlm_endpoint.py`

Purpose: after the selected cheap7 grid confirms a candidate endpoint, run only the missing official-compatible SuperGLUE column through `evaluate_compliant_endpoint.py`, then combine measured SuperGLUE with cheap7 and AoA=0 arithmetic.

Tool check: CPU preflight on existing `chck_100M` selected row succeeded and computed thresholds. For `chck_100M` cheap7=43.542143, SuperGLUE would need 72.6873 to beat `chck_82M`, so it should not be evaluated as an endpoint branch.

For the diagnostic `chck_84M` cheap7=44.123571, the approximate SuperGLUE thresholds are:

- beat protected `chck_82M` Overall 41.942481: SuperGLUE >= 68.6202 with AoA=0
- reach Overall 42.0 with AoA=0: SuperGLUE >= 69.1350
- match coherent86 alpha0.75 projected Overall 42.121025 with AoA=0: SuperGLUE >= 70.2242

Thus, if terminal cheap7 remains near 44.12, a single SuperGLUE evaluation is the minimum reliable expensive work to decide whether `chck_84M` is a real endpoint improvement. AoA/checkpoint-ladder work can remain deferred unless the SuperGLUE projection makes the endpoint competitive enough to justify complete official-material preparation.

## Salvage tooling for near-timeout grid tasks

Script: `scripts/reconstruct_selected_mlm_trajectory.py`

Purpose: if the selected-grid bg task terminates before writing `selected_trajectory.json`, reconstruct completed cheap7 rows from `OUT_DIR/eval/per_target/<label>_<endpoint>.json`. It performs no model evaluation. Compile/tool-check succeeded on the already complete cached `chck_100M` payload.

## Next use

After the protected-reference common grid completes, run `scripts/selected_mlm_integrity_check.py`, reconstruct if necessary, and interpret with `scripts/interpret_deberta_common_grid.py`. If `chck_84M` remains the best reference endpoint and the family profile is not driven only by volatile columns, `superglue_for_selected_mlm_endpoint.py` provides the next endpoint comparison for `chck_84M`.
