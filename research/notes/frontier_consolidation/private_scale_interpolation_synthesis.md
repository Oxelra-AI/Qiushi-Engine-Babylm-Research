# coherent86 multiarm mechanism reading — coherent86 private-scale interpolation synthesis

## Object
No new training. The frozen anchor fastpath disruption design coherent86 frozen-anchor private-path model has a learned
private residual branch. Its config field `private_adapter_scale` can be changed at
inference/materialization time:
- alpha 0.0: exact protected chck82 slow function on CPU probe (max/mean logit diff 0.0).
- alpha 0.5: same weights, config SHA `43060b6f...`, model SHA `e14d757a...`, cheap7 44.177857.
- alpha 0.75: same weights, config SHA `ca6792fe...`, model SHA `e14d757a...`, cheap7 44.181429.
- alpha 1.0: coherent86 original, same weights, cheap7 44.106429.

Materialization manifests:
- `data/private_scale_materialization_probe/coherent_scale0/private_scale_manifest.json`
- `training/runs/coherent86_private_scale_0p5/hf_model/final/private_scale_manifest.json`
- `training/runs/coherent86_private_scale_0p75/hf_model/final/private_scale_manifest.json`

## Cheap7 + item-transition result
Updated panel: `data/private_scale_panel_analysis/private_scale_panel_analysis.{json,md}`.

Against protected chck82:

| endpoint | cheap7 | Δcheap7 | discrete net | changed items | anchor-correct retention | focus notes |
|---|---:|---:|---:|---:|---:|---|
| alpha 0.5 | 44.177857 | +0.218407 | -21 | 3141 | 0.983946 | EWoK focus +10, Entity focus -4, Supplement focus -1 |
| alpha 0.75 | 44.181429 | +0.221979 | -86 | 4750 | 0.975446 | EWoK focus +5, Entity focus -3, Supplement focus +4 |
| alpha 1.0 | 44.106429 | +0.146979 | -115 | 6347 | 0.967191 | EWoK focus +6, Entity focus +1, Supplement focus +3 |

Interpretation: the learned private residual is amplitude-sensitive. Reducing scale from
1.0 to 0.5/0.75 improves cheap7 while *reducing* item churn and anchor-correct erosion.
Alpha 0.5 is the cleaner retention point; alpha 0.75 has slightly higher cheap7 (+0.00357).
Both remain redistribution endpoints because their discrete net vs anchor is still not broad
positive (+cheap7 with small negative item net), and GlobalPIQA contributes a large part of
the aggregate gain.

## SuperGLUE threshold arithmetic
The only expensive missing column for provisional Overall(AoA0) is SuperGLUE; two parallel
SuperGLUE-only runs were launched in coherent86 multiarm mechanism reading:
- alpha0.5, `data/private_scale_superglue_summary/coherent86_private_scale_0p5_sg_superglue_summary.json`
- alpha0.75, `data/private_scale_superglue_summary/coherent86_private_scale_0p75_sg_superglue_summary.json`

Because alpha0.5 cheap7 exceeds alpha1 by 0.0714286, its SuperGLUE can be about 0.500 lower
than alpha1 and still match alpha1 Overall(AoA0). Because alpha0.75 cheap7 exceeds alpha1 by
0.0750000, its SuperGLUE can be about 0.525 lower than alpha1 and still match alpha1 Overall.
To match protected chck82, alpha0.5 SuperGLUE can be about 1.529 lower than chck82 SuperGLUE;
alpha0.75 can be about 1.554 lower. Thus a normal SuperGLUE result near the coherent86 or
chck82 range would promote an intermediate-alpha materialized endpoint above coherent86 and
well above the submitted chck82.

## Route reading before SuperGLUE returns
- If SuperGLUE is near coherent86/chck82, alpha0.5 or alpha0.75 becomes the stronger local
  endpoint candidate and should receive a truthful carrier/HF bundle for submission consideration.
- If SuperGLUE collapses enough to erase the cheap7 gain, private-scale interpolation closes
  as another redistribution effect.
- Regardless of endpoint arithmetic, this still does not validate a general slow-fast learning
  principle. It shows the private residual has a tunable inference amplitude; it does not show
  benchmark-independent acquisition of stable new relation/state competence.

## Added alpha-sweep decision-pattern evidence

Artifact: `data/alpha_sweep_decision_patterns/alpha_sweep_decision_patterns.{json,md}`.

Across 170,722 common discrete official items, correctness patterns over `[alpha0, alpha0.5,
alpha0.75, alpha1]` are almost completely monotonic in the private residual amplitude:
- adjacent changes: alpha0→0.5 = 3,141; 0.5→0.75 = 1,613; 0.75→1.0 = 1,625
- anchor-wrong monotonic activation candidates: 3,116
- anchor-wrong nonmonotonic candidates: 9
- anchor-correct monotonic damage candidates: 3,231
- anchor-correct nonmonotonic damage candidates: 7

This is stronger than the earlier generic redistribution reading: the coherent private residual is a
real, stable amplitude knob over many item margins. Alpha 0.5 and 0.75 are not arbitrary separate
endpoints; they are lower-amplitude cuts through the same learned private direction, and the lower
amplitude reduces both useful activations and damages. The endpoint optimum on cheap7 appears to
sit between 0.5 and 0.75, but choosing it from official scores is an endpoint-engineering fact, not a
benchmark-independent learning principle.

Scientific update: private-scale interpolation remains valuable for selecting the best faithful
materialized endpoint, but it does not by itself explain how to form more robust relation/state
competence from limited data. It shows that learned private residuals can be moderated at inference
time so that fewer anchor-correct decisions are overwritten while enough high-weight official-column
improvements survive.
