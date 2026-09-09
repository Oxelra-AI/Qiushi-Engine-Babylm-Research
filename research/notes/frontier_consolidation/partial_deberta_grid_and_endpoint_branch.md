# partial deberta grid and endpoint branch partial DeBERTa common-grid result and endpoint branch

## What is authoritative now

The original causal transfer result synthesis selected-grid bg tasks timed out at the managed 10,800s cap. They are terminal as timed-out tasks, but they left substantial official-compatible per-target payloads. `scripts/reconstruct_selected_mlm_trajectory.py` reconstructed completed rows from those payloads without rerunning model evaluation.

Recovered reference scale1.75 seed43022 common-grid rows:

- valid 13/16 endpoints; missing 96M and 98M; 94M incomplete; 100M cache-seeded from prior exact-hash payload.
- best recovered endpoint: `chck_84M`, cheap7 `44.12357142857143`.
- `chck_82M` cheap7 `43.95857142857143`; `chck_84M` is +0.1650 cheap7 above it.
- `chck_84M` scores: BLiMP 68.25, Supplement 63.48, EWoK 50.07, Entity 28.58, COMPS 52.21, GlobalPIQA 38.12, Reading 8.155.
- compared with 82M, the improvement is not pure GlobalPIQA/Reading: BLiMP -0.23, Supplement +0.54, EWoK +0.01, Entity +0.27, COMPS +0.02, GlobalPIQA +0.54, Reading +0.005. cheap6(no GlobalPIQA) rises by about +0.1025.
- recovered trajectory falls sharply after 84M: 86M cheap7 43.7707, 88M 43.7107, 90M 43.2721, 92M 43.3671, 100M 43.5421.

Recovered scale1.25 seed43022 rows:

- valid 11/16 endpoints; 92M incomplete; 94/96/98/100M missing.
- best recovered endpoint so far: `chck_86M`, cheap7 `43.53785714285714`.
- the partial comparison against reference through common valid rows shows lower residual energy is not a broad improvement: mean delta cheap7 -0.485844, cheap6(no GlobalPIQA) -0.857273, cheap5(no GP/Reading) -0.995818, relation_state -0.48. The only positive cheap7 row so far is 90M (+0.045714) and is 98% volatile-positive share.

Partial interpreter output: `data/partial_deberta_common_grid_interpretation/deberta_common_grid_interpretation.{json,md}`.

## What remains unsettled

- The reference grid is not fully complete: 94M lacks COMPS/GlobalPIQA/Reading, 96M and 98M are missing. Existing trend makes it unlikely but not impossible that a late row competes with 84M.
- The scale1.25 late end (92M Reading and 94--100M) is missing. The current evidence already weakens the hypothesis that lower adapter scale broadly improves or broadens the late peak, but complete rows are needed to characterize whether it has a later volatile optimum or merely underfits broad families.
- SuperGLUE is missing for the new reference `chck_84M` endpoint. Cheap7 alone cannot determine Overall.

## Targeted GPU work launched in partial deberta grid and endpoint branch

After the timeout, two new bg tasks were submitted:

1. SuperGLUE-only evaluation for reference `chck_84M` on GPU0.
   - Purpose: decide whether the recovered cheap7 peak is a real Overall endpoint improvement over protected/submitted `chck_82M`, and whether it approaches coherent86 alpha0.75.
   - Minimum scope: SuperGLUE only. No AoA and no leaderboard submission.
   - Tool: `scripts/superglue_for_selected_mlm_endpoint.py`.
   - Inputs: selected cheap row from `data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json` and checkpoint `training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_84M`.

2. resume missing DeBERTa common-grid cheap endpoints on GPU1.
   - Purpose: complete the 70--100M/2M trajectory evidence after salvaging the timed-out task outputs.
   - Minimum scope: only missing or incomplete cheap-task columns/checkpoints; completed payloads are skipped.
   - Tool: `scripts/resume_mlm_common_grid.py`.

## Checkpoint provenance already secured

`scripts/checkpoint_identity_audit.py` audited reference `chck_84M`:

- actual exposure 84,028,405 words = 8.4028405 epochs of the legal 10M pool
- model SHA256 `2217917c687faf4de26ef6f381be3048d0bd66b2025c06382683c24d78e8d8c9`
- static files match `chck_82M` except model weights
- trusted-code CPU load succeeds as `AdapterDebertaV2ForMaskedLM`, 35,463,008 params

## Interpretation boundary

A confirmed `chck_84M` Overall improvement would be an endpoint opportunity on the already legal scale1.75 trajectory. It does not explain compact-view reinvestment and should remain separate from the transferable-mechanism route. The mechanism picture after source conditioned ordering interaction synthesis--184 remains: ordered source-conditioned retrieval is a general bidirectional MLM property rather than compact-specific, and the cleanest possible future training contrast is still compact ordered vs compact scrambled only after directional-fork results and completed grids are integrated.
