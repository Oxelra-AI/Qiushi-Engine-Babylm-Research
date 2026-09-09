# earlier analysis updated evidence decomposition — what drove the v5 movement

This note supersedes the evidence decomposition and missing controls version where the proposed ordinary-continuation reimplementation was still treated as the relevant control. earlier analysis rejected that nonmatched reimplementation and established the already executed earlier analysis `inherited_wwm` endpoint as the matched ordinary seed62064 continuation. earlier analysis-106 then repaired its loading interface, measured zero-shot/Reading and AoA, added source/evidence probes, and prepared the guarded same-coordinate admission pending repaired SuperGLUE.

## 1. Stage II understanding mapped to Stage III modifications

| Stage II understanding | Evidence before Stage III modification | Stage III implementation | Current status |
|---|---|---|---|
| Effective input matters more than experimenter-described surface form. Learning changes only when the manipulated structure survives into the learner's actual tokens and masks. | Early cue/canonicalization experiments showed that surface diversity disappears when canonicalization removes the manipulated cues. | Dense masking of Qwen second-view content removes local completion cues from the actual MLM input. | Supported by `(S,S)` → `(M,S)`: sparse labels/focus weighting held fixed while dense input masking changed the learned function. |
| Credit allocation under competition shapes which interface forms. Broad prediction pressure can dilute or suppress a target computation. | Query-indexed binding and held-symbol transfer work showed that the same evidence can form different computations under different prediction pressure. | `(M,S)` keeps sparse source-visible labels while dense masking makes earlier/source evidence consequential; retained labels receive much larger per-label focus weight than dense `(M,M)`. | Exact `(M,S)` reaches `42.20253795433653`, supplies most of clean64's aggregate movement, and exceeds both dense `(M,M)` endpoints on the repaired fixed coordinate. |
| Acquisition can damage inherited broad competence; preservation should constrain harmful drift without erasing the acquired movement. | Dense and `(M,S)` readouts showed source-absent, wrong-source, CDI, BLiMP/Supplement/EWoK costs before clean preservation was designed. | Clean preservation adds eval-mode `KL(teacher || student)` from a frozen coherent86 private-scale-0.75 teacher on ordinary full-row WWM Qwen renderings before the same optimizer step. | Clean64 reaches `42.246412332209445`; clean65 replicates the full policy at `42.23173113265801`; rollback shows attenuation explains much of the repair, so teacher-specific mechanism remains a question. |

Teacher identity is now explicit: preservation controls and corrected interpretation loads both student initialization and teacher through `corrected_bridge_trainer.load_model`, i.e. coherent86 private scale 0.75 with 48 private-adapter tensors and 995,584 private parameters. Preservation distills the frozen coherent86 private initialization function, not a slow-only, chck82-only, or private-disabled parent.

## 2. Learned-versus-lost behavior: context-sensitive redistribution, not uniform enhancement

The direct source/evidence readouts remain essential because benchmark deltas alone do not reveal whether the model learned a reusable context computation or merely shifted probability mass.

### Source-response and evidence availability

- Dense and exact `(M,S)` improve correct-source NLL only slightly, while wrong-source, view-only, source-erased, and CDI conditions worsen more strongly. For clean64 relative to coherent86, the recorded evidence-availability deltas include correct-source NLL about `-0.02155`, wrong-source `+0.09856`, view-only `+0.07631`, this-source-erased `+0.12699`, and all-sources-erased `+0.01009`.
- earlier analysis ordinary seed62064 does not show that asymmetric cost pattern. In the evidence-availability probe, ordinary relative to coherent86 is slightly better under all six evidence states: pair-correct `-0.010063`, pair-wrong `-0.013720`, view-only `-0.016015`, full-row original `-0.010564`, this-source-erased `-0.013781`, all-sources-erased `-0.013888`.
- In the temperature/source probe, ordinary is almost parent-like: Qwen fitted source-specificity delta `+0.000075` and rank delta `-1.24`; exact `(M,S)` is much larger (`+0.146789`, `+66.97`), and clean64 retains a reduced version (`+0.100746`, `+42.70`).

The current mechanism is therefore a redistribution toward stronger response to available evidence together with weaker context-independent prediction. This does not imply that no contextual behavior was acquired: Entity and some GlobalPIQA items may benefit when the evaluation format supplies the relevant evidence. It also does not justify saying the model simply learned context use in general.

## 3. Ordinary continuation control: verified endpoint and current evaluation state

The valid seed62064 ordinary-control endpoint is:

`experiments/archive/functional_learning/data/unchanged_focus_weighted_train/inherited_wwm/checkpoints/update_0080`

earlier analysis identity result: `experiments/archive/functional_learning/data/ordinary_control_identity/ordinary_control_identity.json`

Key identity facts:

- Same coherent86 parent and unchanged-Qwen prefix as exact `(M,S)`.
- Same seed62064, 80 updates, schedule offset/total `101/455`, warmup `10`, peak LR `5e-5`, mask probability `0.15`, private scale `0.75`.
- 3,162,742 consumed words, 697,102 ordinary targets, zero focus targets.
- Exactly 48 changed private-adapter tensors; zero non-private movement.
- Endpoint SHA256 `1e5b3ec6eade0559f4eb3b9bed7553e89e88ddac10e4595a421d6f2a5cd30be3`.

Evaluation state after earlier analysis:

| Component | Status | Values |
|---|---|---|
| Zero-shot/Reading | complete official-sized repaired AutoModel | BLiMP `68.47`, Supplement `63.63`, EWoK `49.83`, Entity `28.16`, COMPS `52.00`, GlobalPIQA `39.535`, Reading `8.22` |
| AoA | complete measured batched AoA | `0.0`, with 144,090 finite assembled rows and platform-matching scorer provenance |
| SuperGLUE | running | output root `experiments/archive/functional_learning/data/repaired_ordinary_superglue` |

A guarded partial table is at `experiments/archive/functional_learning/data/same_coordinate_with_ordinary_after_aoa/same_coordinate_with_ordinary.json`. It admits ordinary zero/Reading and AoA but withholds Overall because SuperGLUE is absent. Present-component deltas ordinary minus coherent86 are BLiMP `-0.04`, Supplement `-0.01`, EWoK `-0.19`, Entity `-0.16`, COMPS `-0.05`, GlobalPIQA `+0.97`, Reading `+0.055`, AoA `0.0`, SuperGLUE pending.

Interpretation so far: ordinary additional WWM can move some benchmark surfaces, especially the small GlobalPIQA item set, but it has parent-like source/evidence geometry and does not reproduce the exact `(M,S)`/clean Entity and source-dependence pattern. The complete repaired coordinate is still needed before comparing total score.

## 4. Preservation geometry: matched local measurement before any full arm

earlier analysis measured the proposed dense-corruption preservation states at the exact acquisition-only `(M,S)` checkpoint before launching a full training arm:

`experiments/archive/functional_learning/data/preservation_geometry_matched_diagnostic_full_v2/preservation_geometry_matched_diagnostic.json`

Selected macros: update indices `[0, 20, 40, 60]`, 1,021 rows, 172 Qwen rows, 158,522 words. Target supports:

- clean-like ordinary full-row WWM support: 4,762 target tokens;
- dense-corrupted non-label support: 6,665 target tokens;
- common positions between ordinary-WWM targets and dense-corrupted masked non-label positions: 1,003 tokens.

At the same 1,003 target positions, dense-corrupted inputs differ sharply from ordinary-WWM inputs:

| Common-support comparison | Dense minus ordinary |
|---|---:|
| Teacher entropy | `+1.632988` |
| Teacher top-1 probability | `-0.217614` |
| Teacher target CE | `+1.315477` |
| Student target CE | `+0.580109` |
| KL(teacher||student) | `+0.473810` |
| Private gradient L2 ratio | `11.835×` |

Gradient alignment at common support:

- ordinary-common KL gradient cosine with full acquisition gradient: `+0.2241`;
- dense-common KL gradient cosine with full acquisition gradient: `+0.2834`;
- ordinary-common cosine with focus-only acquisition gradient: `-0.4821`;
- dense-common cosine with focus-only acquisition gradient: `-0.5054`;
- ordinary-common versus dense-common KL gradient cosine: `+0.7288`.

Thus equal lambda does not create equal preservation strength. Dense-corrupted KL sees more uncertain parent states and produces a much larger private-adapter constraint. The full dense-nonlabel branch also changes target support (6,665 tokens versus 4,762 clean-like ordinary targets), so a later full arm must be described as a joint rendering-and-support intervention. A geometry-specific result should be judged by source/evidence behavior and drift at comparable acquired source response, not by selecting the best score.

## 5. Seed and downstream fine-tuning stability

| Factor | Current seed evidence | Current interpretation |
|---|---|---|
| Dense `(M,M)` acquisition | seeds 62064 and 62065 | Reproducible item-level displacement; GlobalPIQA changes shared. |
| Exact `(M,S)` acquisition | seed62064 complete; seed62065 pending | Seed stability for acquisition alone is not yet established. |
| Clean `(M,S)+KL` full policy | seeds 62064 and 62065 complete | Full policy is highly reproducible on the fixed evaluation items. |
| Ordinary continuation | seed62064 identity, zero/Reading, AoA, probes; seed62065 pending | Current seed62064 looks source-parent-like; seed stability pending. |
| Preservation residual | same-seed clean64 minus `(M,S)`64 complete; no `(M,S)`65 yet | Anchoring-associated residual, not an isolated teacher-only causal estimate. |

Matched downstream SuperGLUE ftseed44 is complete and parsed at `experiments/archive/functional_learning/data/ftseed_comparison_hardened_after_ms/ftseed_comparison_hardened.json`: coherent86 `69.31798722319077`, exact `(M,S)` `69.34605281040608`, clean64 `69.62659512687073`. Clean64 exceeds exact `(M,S)` at seed42 and seed44, strengthening the SuperGLUE part of the clean residual. Exact `(M,S)` versus coherent86 changes sign across downstream seeds, so ftseed44 does not change the fixed Overall coordinate and does not replace training-seed counterparts.

## 6. Current strongest evidence chain

1. Repaired fixed coordinate: clean seed62064 is highest among six complete repaired endpoints (`42.246412332209445`), with clean seed62065 reproducing the full policy (`42.23173113265801`).
2. Acquisition dominates the aggregate: exact `(M,S)` seed62064 reaches `42.20253795433653`, supplying most of clean64's score movement.
3. Ordinary seed62064, so far, is not a copy of `(M,S)`: it has parent-like source/evidence geometry and ordinary-function drift while its complete repaired score awaits SuperGLUE.
4. Clean preservation reduces ordinary full-row drift relative to exact `(M,S)`, but rollback/attenuation explains much of the repair; earlier analysis geometry measurement shows the dense-corrupted parent KL would impose a far stronger and more uncertain-state constraint at equal lambda.
5. The transferable scientific principle is not a particular mask recipe. It is the interaction of effective input, credit allocation, and retention under finite training: useful evidence must be made responsible for predictions, while inherited competence needs protection from collateral drift.

## 7. Work still needed before final synthesis and release statements

| Priority | Scientific comparison | What it resolves | Status at the time |
|---|---|---|---|
| 1 | Ordinary seed62064 repaired SuperGLUE and guarded table | Whether ordinary continuation's total repaired coordinate resembles parent, exact `(M,S)`, or clean | Evaluation in progress |
| 2 | Ordinary62065 and `(M,S)62065 artifacts | Training-seed stability of ordinary and acquisition-only rungs | Incomplete; validation required before use |
| 3 | Preservation geometry at comparable acquired response, using the tested preservation-control implementation | Whether dense-corrupted preservation changes acquisition-retention behavior beyond stronger suppression | Proposed, conditional on the ordinary seed62064 comparison |

Any official platform result remained distinct from repaired-coordinate leadership. Historical, repaired and external evaluation coordinates were not interchangeable.
