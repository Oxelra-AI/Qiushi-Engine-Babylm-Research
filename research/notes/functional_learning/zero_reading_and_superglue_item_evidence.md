# zero reading and superglue item evidence item-level evidence for clean preservation while SuperGLUE jobs run

## Purpose

The active Stage III question is whether clean eval-mode parent-function preservation turns the `(M,S)` dense-clue-suppression acquisition policy into a lawful, usable BabyLM Strict-Small improvement rather than a fragile score redistribution. At zero reading and superglue item evidence start, two repaired-AutoModel SuperGLUE evaluations remained in progress:

- exact acquisition-only `(M,S)` seed62064: output root `experiments/archive/functional_learning/data/repaired_densemask_sparselabel_superglue`;
- clean preservation seed62065: output root `experiments/archive/functional_learning/data/repaired_clean_seed62065_superglue`.

This note records independent CPU-side analyses that do not use those unfinished payloads.

## Zero-shot and Reading item stability across clean seeds

Script: `experiments/archive/functional_learning/scripts/zero_reading_item_stability.py`

Outputs:

- summary JSON: `experiments/archive/functional_learning/data/zero_reading_item_stability/zero_reading_item_stability_summary.json`
- markdown: `research/documents/functional_learning/data/zero_reading_item_stability/zero_reading_item_stability.md`
- subtask table: `experiments/archive/functional_learning/data/zero_reading_item_stability/zero_shot_subtask_family_table.csv`
- flip samples: `experiments/archive/functional_learning/data/zero_reading_item_stability/zero_shot_flip_samples.json`

The parser reads actual official prediction files and full_eval gold files for coherent86, dense seed62064, dense seed62065, clean seed62064, and clean seed62065. It reproduces official-style scores from predictions to within ordinary rounding differences, so the item comparisons are grounded in the evaluator outputs.

Main seed62064/seed62065 clean replication evidence:

- BLiMP: 59,875 items; clean seed correctness agreement `0.998146`; shared parent-relative gains `551`, shared losses `694`, gain Jaccard `0.9371`, loss Jaccard `0.9036`; no opposite changed items. The BLiMP loss is therefore a repeated displacement, not seed noise.
- Supplement: 5,218 items; agreement `0.999425`; shared gains `37`, shared losses `36`; essentially identical profile.
- EWoK: 7,618 items; agreement `0.996718`; shared gains `158`, shared losses `160`; repeated but small net loss.
- Entity: 6,780 items; agreement `0.997345`; shared gains `181`, shared losses `136`; gain Jaccard `0.9526`, loss Jaccard `0.9379`; the Entity improvement is stable across clean seeds and concentrated in the same operation/split structure.
- COMPS: 91,028 items; agreement `0.997330`; shared gains `2529`, shared losses `2389`; repeated small positive movement.
- GlobalPIQA_parallel/nonparallel: exact prediction/correctness agreement across clean seeds. The gain is real in the fixed official item set but small in count: parallel has 2 shared gains and 1 shared loss over 103 items; nonparallel has 2 shared gains over 100 items.
- Reading vector: clean64/clean65 surprisal predictions have Pearson `0.99999499`, previous-token surprisal Pearson `0.99999078`, mean absolute prediction difference `0.011876` over 1,726 rows. Reading is effectively replicated at the prediction-vector level.

The stable clean seed profile supports the view that the zero-shot/Reading part of clean preservation is a repeated functional displacement: Entity and GlobalPIQA gains and BLiMP/Supplement/EWoK costs recur on the same items/subtasks. It does not prove complete model-improvement replication because clean seed62065 SuperGLUE remains pending.

## Dense-to-clean zero-shot decomposition

The same zero-shot analysis decomposes seed-matched dense-to-clean changes against coherent86. Clean preservation recovers many dense-induced parent losses in BLiMP and Supplement while keeping most dense gains, but it is not uniformly superior to dense on all zero-shot items:

- BLiMP seed62064: clean recovers `338` dense parent losses, keeps `562` dense gains, drops `214` dense gains, adds `16` new gains, loses `17` parent items dense kept; net clean minus dense item accuracy `+0.2054` pp. Seed62065 decomposition is similar (`+0.2021` pp).
- Supplement: net clean minus dense `+0.1150`/`+0.1342` pp for seed64/seed65, mostly by recovering dense parent losses without introducing new parent-item losses.
- EWoK: clean is worse than dense item-weighted (`-0.1838`/`-0.1050` pp), because it drops more dense gains than it recovers; this matches the official EWoK loss and should remain part of the trade interpretation.
- Entity: clean roughly matches dense item-wise (`+0.0147`/`+0.0590` pp clean-dense), while retaining the source/evidence-dependent gains.
- COMPS: clean is marginally above dense item-wise (`+0.0264`/`+0.0088` pp), a small repeated movement.
- GlobalPIQA: clean and dense have identical correctness on the official items, so preservation does not explain the GlobalPIQA gain relative to dense; it preserves it.

This strengthens the acquisition-retention account: preservation repairs part of dense’s ordinary-language damage, especially in BLiMP/Supplement, while mostly retaining Entity/GlobalPIQA movement. EWoK remains an unresolved cost.

## Completed SuperGLUE task/item profile for clean seed62064

Script: `experiments/archive/functional_learning/scripts/superglue_completed_item_profile.py`

Outputs:

- summary JSON: `experiments/archive/functional_learning/data/superglue_completed_item_profile/superglue_completed_item_profile.json`
- markdown: `research/documents/functional_learning/data/superglue_completed_item_profile/superglue_completed_item_profile.md`
- task deltas CSV: `experiments/archive/functional_learning/data/superglue_completed_item_profile/superglue_task_primary_deltas.csv`
- flip samples: `experiments/archive/functional_learning/data/superglue_completed_item_profile/superglue_clean64_dense64_flip_samples.json`

This parser uses only completed repaired-AutoModel SuperGLUE payloads: coherent86, dense seed62064, dense seed62065, and clean seed62064. It reproduces all payload primary metrics exactly from prediction files and `glue_filtered/*.valid.jsonl`.

Completed SuperGLUE means:

- coherent86: `68.94571192183594`
- dense seed62064: `68.5318200743064`
- dense seed62065: `68.80580432073164`
- clean seed62064: `69.047710989885`
- clean64 minus coherent86: `+0.10199906804906789`
- clean64 minus dense64: `+0.5158909155786091`
- clean64 minus dense65: `+0.24190666915336578`

Task-primary deltas show the clean64 SuperGLUE advantage over dense is not a single-task artifact:

- BoolQ: clean64 equals dense64 and is `+0.6728` over coherent86.
- MultiRC: dense64/dense65 are `-0.7013`/`-0.5363` vs coherent86; clean64 is only `-0.1238`, improving over dense by `+0.5776`/`+0.4125`.
- RTE: clean64 remains at the dense value, `-0.7194` vs coherent86.
- WSC: clean64 equals coherent86 and dense65; it improves over dense64 by `+1.9231`.
- MRPC F1: clean64 is `+0.6897` over coherent86 and both dense seeds.
- QQP F1: clean64 is `+0.0317` over coherent86 and `+0.1968`/`+0.2447` over dense64/dense65.
- MNLI: clean64 is `+0.1630` over coherent86 and `+0.2241`/`+0.2852` over dense64/dense65.

Relative to coherent86, clean seed62064’s SuperGLUE is still a small mixed improvement: BoolQ/MRPC/QQP/MNLI improve, MultiRC/RTE lose, WSC is equal. Relative to dense controls, clean broadly repairs supervised-transfer cost across MultiRC, WSC, MRPC, QQP, and MNLI. This supports the interpretation that parent preservation bounds dense-induced transfer damage, but it does not by itself prove the preservation increment over exact `(M,S)` acquisition-only because `(M,S)` SuperGLUE and zero/Reading are still unfinished.

## Consequence for the next execution work

The strongest next work remains unchanged:

1. The planned validation of completed results checks clean seed62065 SuperGLUE strictly: exit code 0, all seven subtasks returncode 0, primary metric details present, repaired-AutoModel source path, and no substantive stderr. Then rerun `experiments/archive/functional_learning/scripts/same_coordinate_all_candidates.py` into a fresh directory to compute complete seed62065 Overall.
2. The same planned validation checks exact `(M,S)` SuperGLUE with the same rules. The subsequent planned measurement was the preflighted exact `(M,S)` official zero-shot/Reading evaluation into `experiments/archive/functional_learning/data/repaired_densemask_sparselabel_zero_reading` using repaired model `experiments/archive/functional_learning/data/automodel_repair_candidates/repaired_densemask_sparselabel_seed62064_u0080`.
3. After exact `(M,S)` zero/Reading completes, rerun the guarded same-coordinate table so `(M,S)` enters Overall arithmetic. The direct clean64 minus `(M,S)` comparison is the load-bearing method test.
4. After complete official payloads, a proposed extension was to apply the zero reading and superglue item evidence item parsers to include clean seed62065 SuperGLUE and exact `(M,S)` zero/Reading/SuperGLUE, but do not substitute item analysis for the official coordinate.
