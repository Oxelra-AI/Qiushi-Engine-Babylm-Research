# extractive surface and readout repair — extractive source-only contrast: readout repair and surface geometry

## Current scientific object

The running earlier analysis experiment compares two source-only substitutes for the natural compact-view block in the legal 100M DeBERTa stream:

- `extractive_balanced`: closest to compact on content density and active-token mass, but with higher source coverage, no source-absent content, and telegraphic source-word deletion surface.
- `extractive_wide`: near-complete source coverage and high content density, again with no source-absent content and a deletion surface.

The result will decide whether the DeBERTa compact-reinvestment benefit can be reproduced by selecting original source words alone, or whether generated compact views add value through their bundled natural surface: fluent re-expression, source-absent lexical material, context recomposition around copied/shared content, and a different source-position/active-token profile.

## Repair made before selected scoring

`scripts/extractive_training_integrity_reader.py` had inherited an incorrect expectation of 100 saved checkpoints. earlier analysis deliberately trains with `checkpoint_words=10,000,000`, so a valid 100M extractive run should save ten checkpoint directories (`chck_10M` through `chck_100M`). I changed the expected saved-checkpoint count to 10 and reran the reader into:

`experiments/archive/frontier_consolidation/data/extractive_training_integrity_repaired/extractive_training_integrity.json`

The rerun still reports both arms not ready because `launcher_result.json` and `scientific_metrics.json` are absent while the managed jobs remain unresolved. The fixed reader now should not falsely reject a valid terminal run because of the 10M checkpoint cadence.

## Surface geometry added for interpretation

I wrote and ran:

`experiments/archive/frontier_consolidation/scripts/extractive_surface_discontinuity_atlas.py`

Outputs:

- `experiments/archive/frontier_consolidation/data/extractive_surface_discontinuity_atlas/extractive_surface_discontinuity_atlas.json`
- `research/documents/frontier_consolidation/data/extractive_surface_discontinuity_atlas/extractive_surface_discontinuity_atlas.md`
- `experiments/archive/frontier_consolidation/data/extractive_surface_discontinuity_atlas/per_pair_surface_discontinuity.csv`

Important pooled values across 12,155 compact candidate pairs:

| variant | source-absent content / content | copied adjacent gap=1 | copied adjacent skip | mean source-span fraction |
|---|---:|---:|---:|---:|
| compact | 0.1729 | 0.7775 | 0.2225 | 0.8072 |
| extractive_balanced | 0.0000 | 0.5278 | 0.4722 | 0.9715 |
| extractive_wide | 0.0000 | 0.6465 | 0.3535 | 0.9281 |
| prefix_repeat_local | 0.0000 | 1.0000 | 0.0000 | 0.6246 |

`prefix_repeat_local` is included only as a local structural anchor. correction remains active: the historical selected repeat arm is hash-rotated cyclic repetition, not simple prefix-first-N repetition.

## How this should affect the pending result reading

The extractive arms are not only "source-tail visible" variants. They are source-only deletion views with broader source-span exposure than compact but far more discontinuous copied adjacency. Natural compact views, by contrast, combine source-shared/copied target repetition with generated fluent adjacency and about 17% source-absent content among content words.

Therefore:

- If either extractive arm is near legal compact on stable late selected families, source-word selection/coverage is more sufficient than the current mechanism account expects, despite the deletion surface.
- If both extractive arms lag legal compact, the result supports a bundled natural generated compact advantage. The deficit should not be attributed to missing tail coverage alone, because both extractive variants have higher source-span and tail/content coverage than compact.
- The selected readout should still rely on stable late selected families (`cheap6_no_GlobalPIQA`, `cheap5_no_GlobalPIQA_Reading`, `EWoK+Entity`, `Supplement`, `Entity`, `COMPS`) rather than training loss or GlobalPIQA-only movement.

No selected evaluation, SuperGLUE, AoA, upload, or leaderboard action occurred in extractive surface and readout repair.

## independent_review verifier integration and post-result readout refinement

independent_review verifier integration saved at:

`data/external/independent_review01_verifier1_integration.md`

The independent scientific reading agrees with the current interpretation: the extractive comparison is a bundled realization contrast, not an isolation of one factor. The verifier emphasized that source-only success would mean ordered source-word selection / broad source access / repeated selected-content exposure can reproduce much of the compact benefit without requiring generated paraphrase or source-absent lexical additions for that reproduced portion. If both source-only arms lag compact, the supported conclusion is a bundled natural compact-surface advantage beyond source-tail access; tail starvation is disfavored because both extractive variants have broader source-span and tail/content coverage than compact.

Important cautions to carry forward:

- The historical repeat correction must stay visible: the selected repeat arm is hash-rotated cyclic repetition, while `prefix_repeat_local` in extractive surface and readout repair is only a structural local anchor.
- Checkpoint points are correlated trajectory observations, not independent replicates; use common-checkpoint late means and paired item/subtask movement, not a best-checkpoint story.
- Training loss, local pseudolikelihood, and GlobalPIQA-only movement cannot stand in for stable selected BabyLM competence.
- Values from extractive view pool preflight and training plan, audits, and extractive surface and readout repair use related but not identical metric definitions; compare variants within the same atlas rather than mixing exact decimals across atlases.

I also patched `scripts/extractive_selected_eval_panel.py` after independent_review so source-only readings compare extractive and legal compact over the same checkpoints, preventing the reused legal-compact 70/80/100 rows from being averaged against a different extractive checkpoint set. The plan now records a low-cost first selected readout: after both trainings finish, evaluate only extractive `chck_80M` and `chck_100M` first, because legal compact per-target payloads already exist at both. Continue to `chck_60M/70M/90M` and missing legal references only if the first two endpoints are ambiguous, near compact, or temporally inconsistent enough to change interpretation.

Finally I created `scripts/extractive_selected_movement_driver.py` and AST-checked it. Its `--plan-only` output currently has ready_count 0 because extractive selected predictions do not exist yet. After selected panel summaries exist, it can run the existing earlier analysis item/subtask movement reader for extractive-vs-legal-compact comparisons. This is CPU/file-only interpretation and should be used after selected scoring, not before.
