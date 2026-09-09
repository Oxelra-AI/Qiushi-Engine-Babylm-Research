# compact repeat causal read plan causal read plan for compact_repeat_core full surface

This note records the scientific purpose and current interpretation of the already-trained `compact_repeat_core` contrast before the remaining SuperGLUE subtasks arrive.

## Why this contrast matters

compact core full loss anatomy proposed a possible explanation for the compact-core full failure: the density cleanqwen overlay medium riskhard overlay displaced 423,520 words of inherited clean-Qwen official-source mass with FineWeb compact-core material, so losses in Supplement, GlobalPIQA, and SuperGLUE might come from source-mixture displacement rather than compact semantic transformation itself.

The interpretation requires a qualification: the same observed losses are also compatible with the documented compact-view semantic risks from density cleanqwen overlay medium riskhard—role loss, modality/hedge changes, relation weakening, causal-direction changes, numeric attachment errors, and coreference repair. Before using the 234k-283k Qwen-internal compacting budget from compact core full loss anatomy, the cheaper already-trained contrast is `compact_repeat_core`: it shares the FineWeb compact-core source substrate and neutral top-up with `compact_view_core`, but replaces compact generated views with exact repetition. Full Supplement and SuperGLUE behavior can therefore separate source/repetition substrate effects from compact semantic transformation effects without a new 100M pretraining run.

## Work launched in compact repeat causal read plan

A direct foreground run of:

`PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/full_eval_density_compact.py --target compact_repeat_core --gpu 0 --columns Supplement SuperGLUE`

completed full Supplement and the BoolQ SuperGLUE subtask before the 1800s foreground timeout. It wrote:

- `experiments/archive/frontier_consolidation/data/density_full_eval/per_target/compact_repeat_core.json`
- full Supplement report/predictions under `experiments/archive/frontier_consolidation/data/density_full_eval/official_outputs/compact_repeat_core/Supplement`
- BoolQ finetuning output under `experiments/archive/frontier_consolidation/data/density_full_eval/superglue_results/compact_repeat_core/boolq`

The remaining SuperGLUE subtasks were resumed using the same evaluator and output root; their completed results remained pending.

## What is already known from full Supplement

CPU parsing script:

- `experiments/archive/frontier_consolidation/scripts/repeat_core_full_surface_contrast.py`
- JSON: `experiments/archive/frontier_consolidation/data/repeat_core_full_surface_contrast/repeat_core_full_surface_contrast.json`
- note: `research/notes/frontier_consolidation/repeat_core_full_surface_contrast.md`

Full Supplement triad:

| model | full Supplement |
|---|---:|
| clean-Qwen | 62.84 |
| compact_repeat_core | 59.96 |
| compact_view_core | 61.85 |

Decomposition:

- `compact_repeat_core - clean_Qwen = -2.88`
- `compact_view_core - compact_repeat_core = +1.89`
- `compact_view_core - clean_Qwen = -0.99`

UID-level pattern:

| UID | clean | repeat | view | repeat-clean | view-repeat | view-clean |
|---|---:|---:|---:|---:|---:|---:|
| qa_congruence_easy | 70.31 | 65.62 | 62.50 | -4.69 | -3.12 | -7.81 |
| qa_congruence_tricky | 49.09 | 44.85 | 45.45 | -4.24 | +0.61 | -3.64 |
| turn_taking | 64.64 | 64.29 | 68.21 | -0.36 | +3.93 | +3.57 |
| subject_aux_inversion | 80.76 | 74.42 | 82.49 | -6.34 | +8.07 | +1.73 |
| hypernym | 49.41 | 50.59 | 50.59 | +1.19 | +0.00 | +1.19 |

Interpretation: on the aggregate Supplement column, the FineWeb repetition substrate loses more than compact semantic views. Compact views recover turn-taking and subject-aux inversion strongly, leave hypernym unchanged, slightly recover tricky QA, and further hurt easy QA. This means the Supplement result cannot be used as a simple argument that compact semantic views are the cause of the Supplement loss. It also cannot justify immediate Qwen-internal compaction, because easy QA role/answer-type loss still worsens under compact views.

The fast Supplement screen was misleadingly favorable to compact views: compact-view minus repeat was +7.60 on the fast set but only +1.89 on the full set.

## Interpretation After SuperGLUE Completion

Rerun:

`PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/repeat_core_full_surface_contrast.py`

Then inspect the updated `repeat_core_full_surface_contrast.md` and JSON.

The important SuperGLUE patterns are:

1. If `compact_repeat_core` is already weak on RTE and MRPC relative to clean-Qwen, and `compact_view_core` is similar or better than repeat, the likely cause of the full SuperGLUE loss is the FineWeb/repetition substrate and displaced official mixture rather than compact semantic transformation. In that case, Qwen-internal density remains scientifically live because it would preserve the inherited official/Qwen substrate while testing redundancy reduction inside a block already known to help.

2. If `compact_repeat_core` preserves RTE/MRPC near clean-Qwen but `compact_view_core` loses them, the compact semantic transformation itself likely damages entailment/paraphrase calibration. In that case, compacting inherited Qwen generated sides may transfer the same weakness into the strongest existing block unless the transformation is repaired to preserve roles, modality, and relational force.

3. If the pattern is mixed by task—e.g. repeat loses RTE but view loses MRPC, or view improves BoolQ/MultiRC while hurting pairwise equivalence—the next design must be narrower: do not train a broad Qwen-internal compacting arm until a lower-cost subset or scoring analysis identifies which relation types are safe to compress.

Current BoolQ partial result: clean-Qwen 68.379, compact_repeat_core 67.339, compact_view_core 67.829. This is a small substrate loss partly recovered by compact views; it is not enough to decide SuperGLUE because the compact core full loss anatomy full loss was dominated by RTE and MRPC.

## Relation to AoA

`compact_repeat_core` has the full 19-checkpoint ladder. The pending AoA ladder evaluation is required for trajectory comparison. The scalar AoA score should not be treated as a mechanism by itself because compact_view_core's -12.687 column came from a small raw-correlation shift crossing the official p-value discontinuity while clean-Qwen was already weakly negative. After the AoA results become available, compare word-level fitted trajectories and common fitted-word support across near-repeat, near-view, compact-repeat-core, compact-view-core, and compact-view-reinvest.

## Current route stance

No new corpus generation or 100M training should start from compact core full loss anatomy's Qwen-internal budget until the remaining compact_repeat_core SuperGLUE subtasks and the pending AoA localization are read. The already measured Supplement triad makes the cause more nuanced: source-mixture displacement is still plausible, compact-view semantic loss is still plausible for easy QA/role details, and the correct next decision depends on full RTE/MRPC behavior plus AoA trajectory evidence.
