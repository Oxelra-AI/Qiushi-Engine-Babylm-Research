# chck84 carrier and scale125 grid synthesis synthesis: chck84 carrier closed, reference/scale1.25 common grids complete

## What changed

The `chck_84M` endpoint branch is now reproducibility-ready through the shortest safe route, and no further endpoint-carrier work is needed for this format.

Native validated carrier:

- Carrier: `experiments/archive/frontier_consolidation/data/chck84_native_validated_full_carrier/all_full_preds_truthful_chck84_mlm.json`
- Carrier SHA256: `b55e1997c3fbc0255527f35c8225f788fd1dae6a1427b13b142a13ed9e64189d`
- Manifest: `experiments/archive/frontier_consolidation/data/chck84_native_validated_full_carrier/chck84_truthful_full_carrier_manifest.json`
- Native saved-Space prediction validation: `is_valid_predictions=True`, message `Upload successful.`
- Contents: exact `chck_84M` cheap-task prediction blocks, exact `chck_84M` SuperGLUE prediction blocks, scalar `aoa={"aoa":0.0}`, and no `fast_eval_results` copied from any other endpoint.
- Score arithmetic comes from existing official-compatible local evaluation records rather than the direct local Space scorer, because the local scorer path expects a text-task dataset schema not present in the cached snapshot used in earlier analysis.
- Overall(AoA0): `42.0189129742181`; delta vs protected `chck_82M`: `+0.0764318068321117`.
- Model identity carried by the public HF artifact remains `leslie721007/babylm-strict-small-scale1p75-chck84`, revision `040284de9ac49eee6dc1b30cf65aea97ec17d86e`, model SHA `2217917c687faf4de26ef6f381be3048d0bd66b2025c06382683c24d78e8d8c9`, trusted params `35,463,008`.

No leaderboard submission and no model upload occurred.

## Completed common-grid evidence

The remaining selected cheap-task rows were completed for:

- Reference scale1.75 seed43022 trajectory: `experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json`
- Scale1.25 seed43022 dense trajectory: `experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M/selected_trajectory.json`

Integrity check output:

- `research/documents/frontier_consolidation/data/deberta_common_grid_integrity_reference_scale1p25/selected_mlm_integrity_check.md`
- Overall OK: `True`
- Both trajectories have all 16 endpoints from `chck_70M` to `chck_100M`, all payload cheap7 values match trajectory rows.

Interpretation output:

- `research/documents/frontier_consolidation/data/deberta_common_grid_interpretation_reference_scale1p25/deberta_common_grid_interpretation.md`
- JSON: `experiments/archive/frontier_consolidation/data/deberta_common_grid_interpretation_reference_scale1p25/deberta_common_grid_interpretation.json`

## Reference trajectory: 84M is the completed selected-grid peak

Reference scale1.75 seed43022 selected cheap7 values:

- `chck_82M`: `43.95857142857143`
- `chck_84M`: `44.12357142857143`
- `chck_86M`: `43.770714285714284`
- `chck_100M`: `43.542142857142856`

Thus the completed 70M--100M grid confirms the earlier partial result: `chck_84M` is the best selected cheap-task checkpoint on this trajectory, `+0.16412155211970259` cheap7 over protected/submitted `chck_82M`, and `-0.5814285714285745` by 100M. The near-best contiguous band within 0.2 cheap7 is only `chck_82M` and `chck_84M`, span 2M. This remains a narrow late competence-allocation phase, not a monotone training-law solution.

Column peaks are dispersed across the late window: BLiMP peaks at 94M, Supplement at 76M, EWoK at 74M, Entity at 88M, COMPS at 78M, GlobalPIQA at 84M, Reading at 78M. The 84M aggregate peak is therefore an alignment of several shallow movements, not a single global improvement signal.

## Scale1.25 contrast: lower residual energy is broadly worse

Scale1.25 seed43022 is the seed/mask-matched lower adapter-energy contrast. It is a real lower-energy trajectory: mean effective adapter-up/stock ratio `0.081414`, about `0.780` of the scale1.75 reference `0.104429`.

But selected cheap-task behavior is worse:

- Best scale1.25 endpoint: `chck_86M`, cheap7 `43.53785714285714`
- Mean delta vs reference over 16 endpoints:
  - cheap7 `-0.4109821428571454`
  - cheap6 without GlobalPIQA `-0.8152083333333335`
  - cheap5 without GlobalPIQA/Reading `-0.9491250000000013`
  - EWoK/Entity mean `-0.4253125000000004`
  - syntax/COMPS mean `-1.298333333333333`
  - volatile mean `+0.9343749999999984`
- Only one endpoint has positive cheap7 delta vs reference (`chck_90M`, `+0.04571428571428271`), and that row is almost entirely GlobalPIQA-carried: positive gain share volatile `0.9798792756539246`; cheap6 and cheap5 remain negative.

Conclusion: reducing residual-adapter effective amplitude to 1.25 does not preserve broad competence and does not solve the late phase. It delays the best selected endpoint by 2M but does not broaden the contiguous near-best band and does not reduce the reference late falloff in a scientifically useful way.

## Same-trajectory average probe status

A one-endpoint selected cheap-task evaluation of the already-built 80/82/84 uniform average was launched after the scale1.25 grid completed. It tests whether weight averaging remains a candidate stabilization method if the seed43122 grid reveals stochastic peak movement; its result is not established here.

The task completed with no valid score because the run-dir view lacked `scientific_metrics.json`:

- Failed output: `experiments/archive/frontier_consolidation/data/avg80_82_84_selected_eval_if_authorized/selected_trajectory.json`
- Error: `FileNotFoundError: Training metrics missing; inspect run before evaluation: experiments/archive/frontier_consolidation/data/average_candidate_selected_eval_plan/run_view/scientific_metrics.json`

This is a wrapper metadata defect, not a model result. The average candidate itself remains mechanically valid from chck84 endpoint carrier validation and functionally noncollapsed on the stabilization precheck while grids pending CPU masked-token probe, but it has no official-compatible selected score. Do not infer positive or negative BabyLM competence from this failed task.

A later repair, if still useful after seed43122 evidence, should create a pseudo-run metadata file that explicitly records the average's source checkpoints and zero additional training exposure, then rerun in a fresh output directory or remove the failed per-target payload before reuse. Do not spend further time on this before the cross-seed result unless a Lead decision makes stabilization the active question.

## Scientific state after chck84 carrier and scale125 grid synthesis

- Public protected fallback: `chck_82M`, displayed 41.94, challenge-counted.
- Cleaner ordinary-training endpoint branch: `chck_84M`, native-valid prediction carrier and public HF model ready, Overall(AoA0) `42.0189129742181`; no submission is reported here.
- Strongest local endpoint carrier remains coherent86 alpha0.75 at projected Overall(AoA0) `42.1210247099666`, but its mechanism is amplitude-controlled competence redistribution.
- Lower residual energy via scale1.25 is now disfavored by completed common-grid evidence.
- The key unresolved evidence is still scale1.75 seed43122 common-grid scoring. It decides whether the 82--84M late competence phase is reproducible under another initialization plus mask stream or whether the next valuable route should explicitly target stochastic stabilization of broad competence.
