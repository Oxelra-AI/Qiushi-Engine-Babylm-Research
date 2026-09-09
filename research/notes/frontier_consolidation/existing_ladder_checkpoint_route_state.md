# existing ladder checkpoint route state — existing-ladder checkpoint route state

This note preserves the evidence generated in existing ladder checkpoint route state before any new training or corpus construction. It is research-state memory, not a final report.

## Full seed43122 official coordinate arrived from representation_and_objectives

The missing same-coordinate seed43122 collation is now available:

- `experiments/archive/representation_and_objectives/data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json`
- `research/notes/representation_and_objectives/seed43122_official_failure_localization.md`
- `research/notes/representation_and_objectives/same_coordinate_2x2_decision_memo.md`
- `experiments/archive/representation_and_objectives/data/consistent_official_2x2/consistent_official_2x2.json`

Official-coordinate `compact_view_reinvest` seed43122 scores:

- Overall 41.24823958912208, below visible 41.8 by -0.5517604108779182.
- Columns: BLiMP 65.6644, Supplement 61.9524, EWoK 51.8917, Entity 26.2853, COMPS 51.5392, SuperGLUE 69.9002, GlobalPIQA 35.1359, Reading 8.8650, AoA 0.0.
- Seed43122 minus seed43022: Overall -0.784895; BLiMP -1.208, Supplement -1.323, EWoK -1.645, Entity -1.460, COMPS -0.430, SuperGLUE -1.136, GlobalPIQA -0.485, Reading +0.623, AoA 0.

same-coordinate 2×2 changes the interpretation: clean43022 41.347633, clean43122 40.767775, reinvest43022 42.033135, reinvest43122 41.248240. Treatment effects are positive in both seeds (+0.685502 and +0.480465 Overall). The treatment×seed interaction is only -0.205037 Overall and is concentrated in EWoK (-2.462123), with smaller Entity/COMPS effects; Supplement treatment is almost identical across seeds (+0.006589 interaction). Thus seed43122 is not a failed treatment, but it is not an above-leader endpoint.

## existing ladder checkpoint route state selected-checkpoint broad no-AoA screen

The broad screen used existing checkpoints only, no training/corpus changes/SuperGLUE/AoA. Outputs:

- `experiments/archive/frontier_consolidation/data/broad_noaoa_seed43022/broad_noaoa_late_checkpoints_summary.json`
- `experiments/archive/frontier_consolidation/data/broad_noaoa_seed43122/broad_noaoa_late_checkpoints_summary.json`
- merged: `research/documents/frontier_consolidation/data/broad_noaoa_late_checkpoint_merge/broad_noaoa_late_checkpoint_merge.md`

Key rows (equal7 uses BLiMP, Supplement, EWoK-fast, Entity_full, COMPS, GlobalPIQA mean, Reading):

| target | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r43022_80M | 66.630 | 66.000 | 51.090 | 27.600 | 51.780 | 35.135 | 8.160 | 43.771 |
| r43022_90M | 66.800 | 67.600 | 53.360 | 27.620 | 52.050 | 35.120 | 8.170 | 44.389 |
| r43122_45M | 62.700 | 65.600 | 48.000 | 25.530 | 51.470 | 38.550 | 8.900 | 42.964 |
| r43122_80M | 65.400 | 65.200 | 49.730 | 26.350 | 51.470 | 36.590 | 8.935 | 43.382 |
| r43022_100M fast ref | 66.630 | 66.400 | 53.090 | 27.750 | 51.970 | 35.620 | 8.240 | 44.243 |
| r43122_100M fast ref | 65.750 | 63.200 | 49.360 | 26.290 | 51.540 | 35.135 | 8.865 | 42.877 |

Scientific read:

- 80M as a common stopping rule is not strong enough by itself: seed43122 improves over its 100M broad fast surface (+0.505 equal7_full_entity), but seed43022 80M loses -0.472 equal7_full_entity and -2.0 EWoK-fast versus its 100M reference.
- Raw seed43022 90M is the most plausible single endpoint continuation: it improves fast Supplement (+1.2), EWoK-fast (+0.27), BLiMP (+0.17), COMPS (+0.08), and equal7_full_entity (+0.146) versus seed43022 100M, with the main loss being GlobalPIQA mean (-0.5) and tiny Reading/Entity changes. This makes raw 90M more promising than seed43022 80M or an immediate broad 80M rule.
- Seed43122 45M improves Supplement and GlobalPIQA but loses BLiMP/EWoK/Entity_full enough that it should not be the endpoint. Seed43122 80M is a useful robustness/consolidation point but remains below the raw seed43022 route.

## existing ladder checkpoint route state checkpoint averaging screen

Constructed arithmetic averages of existing checkpoints only:

- `experiments/archive/frontier_consolidation/data/checkpoint_averages/reinvest43022_avg_90_100`
- `experiments/archive/frontier_consolidation/data/checkpoint_averages/reinvest43122_avg_80_100`

Merged broad no-AoA output:

- `research/documents/frontier_consolidation/data/checkpoint_average_noaoa_merge/checkpoint_average_noaoa_merge.md`

Averaged-model scores:

| target | BLiMP | Supp | EWoK-fast | Entity full | COMPS | GPIQA | Reading | equal7 fullEnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| avg43022_90_100 | 66.730 | 66.800 | 53.550 | 27.850 | 52.070 | 34.620 | 8.210 | 44.261 |
| avg43122_80_100 | 65.340 | 64.800 | 50.270 | 26.190 | 51.600 | 36.090 | 8.915 | 43.315 |

Interpretation:

- Averaging improved seed43122's measured broad fast surface over its 100M reference (+0.438 equal7_full_entity, +1.6 Supplement, +0.91 EWoK, +0.955 GlobalPIQA mean), but it is still far below an above-leader absolute endpoint.
- Averaging did not improve seed43022 enough to supersede raw 90M: avg43022_90_100 is +0.019 equal7_full_entity versus 100M but -0.127 versus raw90, and it worsens GlobalPIQA mean by -1.0 versus 100M. Raw90 remains the sharper candidate.

## Current expensive-work decision logic

No new 100M training or corpus generation is justified by existing ladder checkpoint route state. Existing-ladder work is still cheaper and more informative.

Pending evaluation:

- output `experiments/archive/frontier_consolidation/data/official_zeroshot_r43022_90m`, evaluates raw seed43022 90M on full BLiMP, full Supplement, and pristine official 7,618-row EWoK. It asks whether the fast raw90 advantage survives the full official zero-shot coordinate before any SuperGLUE or AoA work.

Decision from that task:

- If raw90 full BLiMP/Supplement/EWoK plus known raw90 Entity_full/COMPS/GlobalPIQA/Reading require a realistic SuperGLUE+AoA and project above the frozen 100M official endpoint when SuperGLUE≈100M and AoA=0, then the next step should run targeted SuperGLUE/AoA only for raw90.
- If full Supplement falls back enough, or EWoK/full BLiMP erase the fast advantage, raw90 should not be escalated; the research should return to relation-stability mechanism work over existing EWoK margins/trajectories and not launch a repaired-corpus 100M run yet.

The frozen reference remains `compact_view_reinvest` seed43022 100M official-coordinate Overall 42.0331347900748.
