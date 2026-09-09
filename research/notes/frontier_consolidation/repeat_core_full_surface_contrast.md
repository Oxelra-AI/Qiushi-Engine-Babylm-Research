# compact repeat causal read plan compact-repeat-core full-surface contrast

Purpose: use the already-trained compact_repeat_core model as the missing contrast between the FineWeb/repetition substrate and compact semantic transformation. The script only parses existing prediction files; the only new GPU work was the separate official-compatible Supplement+SuperGLUE run for compact_repeat_core, whose Supplement and BoolQ outputs are already present and whose remaining SuperGLUE subtasks are pending.

## Full Supplement triad
- Official equal-UID Supplement: clean-Qwen 62.84, compact_repeat_core 59.96, compact_view_core 61.85.
- Decomposition on this column: repeat minus clean -2.88; view minus repeat +1.89; view minus clean -0.99.
- Fast-screen Supplement overstated the compact-view advantage: fast view-repeat +7.60, full view-repeat +1.89.

| UID | clean | repeat | view | repeat-clean | view-repeat | view-clean |
|---|---:|---:|---:|---:|---:|---:|
| qa_congruence_easy | 70.31 | 65.62 | 62.50 | -4.69 | -3.12 | -7.81 |
| qa_congruence_tricky | 49.09 | 44.85 | 45.45 | -4.24 | +0.61 | -3.64 |
| turn_taking | 64.64 | 64.29 | 68.21 | -0.36 | +3.93 | +3.57 |
| subject_aux_inversion | 80.76 | 74.42 | 82.49 | -6.34 | +8.07 | +1.73 |
| hypernym | 49.41 | 50.59 | 50.59 | +1.19 | +0.00 | +1.19 |

Supplement reading: the large full Supplement shortfall is not introduced only by compact semantic views. The repetition substrate is already far below clean-Qwen, while compact views recover part of that loss rather than worsen it on the column aggregate. However compact views still remain below clean-Qwen, and the fast mini-set was too favorable to the view model, so Supplement alone supports neither immediate abandonment of compact views nor immediate Qwen-internal compaction.

## SuperGLUE currently available for compact_repeat_core
- Available compact_repeat_core SuperGLUE tasks: boolq.
- Missing compact_repeat_core SuperGLUE tasks: multirc, rte, wsc, mrpc, qqp, mnli.

| task | clean | repeat | view | repeat-clean | view-repeat | view-clean |
|---|---:|---:|---:|---:|---:|---:|
| boolq | 68.379 | 67.339 | 67.829 | -1.040 | +0.489 | -0.550 |

SuperGLUE reading so far: only finished compact_repeat_core subtasks should be interpreted. The decisive RTE and MRPC comparisons are not present until the managed resume finishes; do not infer them from compact_view_core alone.

## Consequence for the next route
- The new Supplement contrast corrects the compact core full loss anatomy explanation: official-source displacement remains possible, but the observed QA/RTE/MRPC structure is also compatible with role/modality/relation losses in compact views, and Supplement now shows a mixed pattern where the FineWeb repetition substrate loses more than the compact view model.
- Qwen-internal density should remain conditional. If full compact_repeat_core SuperGLUE already has the RTE/MRPC weakness, then the FineWeb/repetition substrate and displaced official mixture are implicated; if repeat preserves RTE/MRPC while view loses them, compact semantic transformation is implicated and internal compaction could transfer the same weakness into the strongest inherited Qwen block.
- The pending AoA ladder still matters because the scalar AoA discontinuity alone is not a mechanism; compare stable word-level trajectories across repeat/view/reinvest after the AoA results become available.

Machine-readable JSON: `experiments/archive/frontier_consolidation/data/repeat_core_full_surface_contrast/repeat_core_full_surface_contrast.json`
