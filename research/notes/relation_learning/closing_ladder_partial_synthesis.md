# closing ladder partial synthesis closing ladder synthesis while final evaluations run

This interim synthesis records measurements available at the closing round and distinguishes them from measurements still pending.

## Active scientific reading

The comparison should foreground clean preservation against the matched ordinary-continuation rung, not only against the coherent86 parent. On seed62064, the repaired-coordinate ladder is:

| rung | Overall | BLiMP | Supplement | EWoK | Entity | COMPS | SuperGLUE | GlobalPIQA | Reading | AoA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coherent86 | 42.0239679913 | 68.51 | 63.64 | 50.02 | 28.32 | 52.05 | 68.9457119 | 38.565 | 8.165 | 0.0 |
| ordinary64 | 42.0926058632 | 68.47 | 63.63 | 49.83 | 28.16 | 52.00 | 68.9884528 | 39.535 | 8.22 | 0.0 |
| exact `(M,S)`64 | 42.2025379543 | 68.12 | 63.08 | 49.95 | 29.39 | 52.15 | 68.8878416 | 40.05 | 8.195 | 0.0 |
| clean64 | 42.2464123322 | 68.26 | 63.28 | 49.82 | 29.40 | 52.16 | 69.0477110 | 40.05 | 8.20 | 0.0 |

Source: `research/documents/functional_learning/data/same_coordinate_with_ordinary_complete/same_coordinate_with_ordinary.md`.

The seed62064 practical contrast clean64 minus ordinary64 is `+0.1538064690` Overall. Its component deltas are: BLiMP `-0.21`, Supplement `-0.35`, EWoK `-0.01`, Entity `+1.24`, COMPS `+0.16`, SuperGLUE `+0.0592582`, GlobalPIQA `+0.515`, Reading `-0.02`, AoA `0`. Divided by the nine-component arithmetic, Entity contributes `+0.1378` Overall and GlobalPIQA contributes `+0.0572`; BLiMP+Supplement together contribute `-0.0622`. The clean-over-ordinary gain is therefore mostly the mechanism-predicted Entity/source-responsive column, with the residual GlobalPIQA movement coming from a very small fixed item set.

## Seed62065 landed values

All landed seed62065 values so far use repaired checkpoints with trusted custom loading and `FrozenSlowPrivateDebertaV2ForMaskedLM`, 36,458,592 total parameters, and 995,584 private parameters where the local payload records identity.

### Ordinary continuation seed62065 (`O62065`)

| component | value | source |
|---|---:|---|
| Supplement | 63.6198528917 | `experiments/archive/relation_learning/data/o62065_Supplement/o62065/with_special/Supplement_component_payload.json` |
| EWoK | 49.8131075509 | `experiments/archive/relation_learning/data/o62065_EWoK/o62065/with_special/EWoK_component_payload.json` |
| Entity | 28.0891913492 | `experiments/archive/relation_learning/data/o62065_Entity/o62065/with_special/Entity_component_payload.json` |
| GlobalPIQA parallel | 31.0679611650 = 32/103 | `experiments/archive/relation_learning/data/o62065_GlobalPIQA_parallel/o62065/with_special/GlobalPIQA_parallel_component_payload.json` |
| GlobalPIQA nonparallel | 48.0 = 48/100 | `experiments/archive/relation_learning/data/o62065_GlobalPIQA_nonparallel/o62065/with_special/GlobalPIQA_nonparallel_component_payload.json` |
| GlobalPIQA mean | 39.5339805825 | `research/documents/relation_learning/data/globalpiqa_ladder_item_anatomy/summary.md` |
| Reading | 8.1965529497 | `experiments/archive/relation_learning/data/o62065_Reading/o62065/with_special/Reading_component_payload.json` |

Interpretation: O62065 reproduces O62064 on landed columns. It stays at parent-like Supplement/EWoK and below-parent Entity, so ordinary continuation by itself does not produce the Entity movement that motivates the principle-guided recipe. O62065 GlobalPIQA has the same count pattern as O62064, 32/103 parallel and 48/100 nonparallel.

### Dense-mask/sparse-label acquisition seed62065 (`(M,S)62065`)

| component | value | source |
|---|---:|---|
| BLiMP | 68.09 | `experiments/archive/functional_learning/data/parallel_eval/ms62065_BLiMP/result_BLiMP.json` |
| Supplement | 63.0903403312 | `experiments/archive/relation_learning/data/ms62065_Supplement/ms62065/with_special/Supplement_component_payload.json` |
| EWoK | 49.8181276606 | `experiments/archive/relation_learning/data/ms62065_EWoK/ms62065/with_special/EWoK_component_payload.json` |
| GlobalPIQA mean | 40.0485436893 = 31/103 and 50/100 | `research/documents/relation_learning/data/globalpiqa_ladder_item_anatomy/summary.md` |

Interpretation: `(M,S)62065` already reproduces the dense-credit cost pattern on BLiMP/Supplement within about 0.03 of seed62064. EWoK is close to the ordinary floor and does not look like a preservation-specific cost. Entity is still pending at this stage; it is one of the main numbers deciding the seed62065 attribution.

## GlobalPIQA item anatomy

The closing ladder partial synthesis item analyzer found no missing prediction files and exact deterministic split counts:

| group | seed64 | seed65 | relation to coherent86 |
|---|---|---|---|
| ordinary | 32/103 parallel, 48/100 nonparallel | 32/103 parallel, 48/100 nonparallel | `+0.970874` GlobalPIQA points, `+0.107875` Overall contribution |
| `(M,S)` and clean | 31/103 parallel, 50/100 nonparallel | 31/103 parallel, 50/100 nonparallel | `+1.485437` GlobalPIQA points, `+0.165049` Overall contribution |

Ordinary seed-shared changed items are exactly two parallel gains: sealed air-filled bag under load and candle wick height. The `(M,S)`/clean seed-shared set has five items: those same two gains, one parallel loss on cookies/housework, and two nonparallel gains on polymer-clay baking time and dry-towel counter protection. Therefore the v4-to-v5 GlobalPIQA contribution of about `+0.165` Overall decomposes into about `+0.108` already obtained by matched ordinary continuation and only about `+0.057` beyond ordinary continuation. This is reproducible on the fixed item pool but it should not carry broad scientific weight.

Source: `research/documents/relation_learning/data/globalpiqa_ladder_item_anatomy/summary.md`.

## Mechanism readout tied to the ladder

Ordinary continuation seed62064 stays near coherent86 on the mechanism surfaces: Qwen source specificity moves only `+0.000075` Tfit and rank `-1.24`; ordinary full-row preservation-surface KL is `0.000551` with ΔCE `-0.002291`. By contrast, exact `(M,S)64` moves Qwen specificity by `+0.146789` Tfit and rank `+66.97`, and clean64 retains `+0.100746` Tfit and rank `+42.70` while reducing preservation-surface drift relative to `(M,S)`. Sources: `research/documents/functional_learning/data/temperature_source_readout_ordinary_full/temperature_source_readout.md` and `research/documents/functional_learning/data/preservation_drift_profile_ordinary/preservation_drift_profile.md`.

The dense-corruption preservation arm gives a useful negative rendering result. Matching the parent on dense-corrupted nonlabel positions collapses source specificity to `+0.003914` Tfit and gives only rank `+1.58`, while clean ordinary-mask preservation keeps much larger source movement. On common dense support, exact `(M,S)64` has CE gain `0.757052`, rank gain `132.848`, and parent KL `0.502657`; clean64 has CE gain `0.682168`, rank gain `128.344`, and KL `0.352518`; dense-corruption preservation has CE gain only `0.158699`, rank gain `34.825`, and KL `0.009356`. This supports the current principle sentence: the amount of private movement governs much of the benchmark trade, but the surface where parent matching is applied governs which distributions are allowed to move. Sources: `research/documents/functional_learning/data/common_support_endpoint_probe_with_densecorr/common_support_endpoint_probe.md` and `research/documents/functional_learning/data/temperature_source_readout_densecorr_full/temperature_source_readout.md`.

Evidence-absent costs remain central. In the fixed evidence-availability bank, dense `(M,S)64` improves correct-source NLL slightly but worsens wrong-source/view-only/source-erased conditions; clean preservation keeps correct-source benefit with smaller wrong/absent evidence damage. In CDI, dense64/65 worsen NLL and rank relative to chck82, while clean64/65 improve relative to chck82 but still do not fully recover coherent86. Dense64 at evaluation private scale 0.60 matches clean64 closely on the landed zero-shot cheap columns and on Qwen specificity, but not on CDI: it remains CDI-damaging while clean64 is CDI-protective. Thus the trained leash’s distinct support is distributional selectivity rather than a broad leaderboard separation from smaller private scale on the cheap surface. Sources: `research/documents/functional_learning/data/evidence_availability_readout_densecorr/evidence_availability_readout.md`, `research/documents/relation_learning/data/cdi_endpoint_rank_candidate_family/cdi_endpoint_rank_candidate_family.md`, and `research/documents/relation_learning/data/shrinkage_source_cdi_probe/shrinkage_source_cdi_probe.md`.

## Current Runtime work and launch status

Tasks delivered and consumed in closing ladder partial synthesis:

- `(M,S)62065` EWoK completed with score `49.8181276606`.
- `O62065` Entity completed with score `28.0891913492`.
- The BoolQ and MultiRC attempts were cancelled before official fine-tuning started; no score should be inferred.

Tasks newly accepted in closing ladder partial synthesis:

The `O62065` SuperGLUE RTE, WSC, BoolQ retry, MultiRC retry and MRPC evaluations were pending at this stage.

No evaluation snapshots or scores were available for those pending measurements. QQP, MNLI and O62065 AoA had not yet been started.

A research loop was also started from `experiments/archive/relation_learning/analysis/Research_Project_Proposal.md` to draft Chinese report skeleton and Stage I/II research fragments under its own workspace. It returned `running` with worker PID `3739353`; no result has been read yet.
