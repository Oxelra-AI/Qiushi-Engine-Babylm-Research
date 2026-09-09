# full core route localization and existing ladder plan route localization after compact-core full result

## What the full compact-core result changes

`compact_view_core` is not a near-SOTA endpoint after complete official-compatible evaluation. Its full score is:

- Overall 39.9877
- BLiMP 67.15, Supplement 61.85, EWoK 51.26, Entity 27.85, COMPS 52.18, GlobalPIQA 35.135, SuperGLUE 68.9012, Reading 8.25, AoA -12.6873

The important arithmetic is stronger than “fix AoA.” If AoA were simply restored from -12.6873 to 0 while every other measured column stayed fixed, Overall would become only `(67.15 + 61.85 + 51.26 + 27.85 + 52.18 + 68.9012 + 35.135 + 8.25 + 0)/9 = 41.3974`, still below the visible 41.8 surface. The shortfall is therefore distributed: the compact transform produced a useful EWoK/Entity/COMPS profile, but the full split lowered the fast Supplement impression, GlobalPIQA stayed weak, and SuperGLUE fell below clean-Qwen.

The density mechanism remains scientifically alive, but unchanged compact-core training should stop as a SOTA route. It improved several hard columns relative to clean-Qwen (EWoK +1.07, Entity +2.09, COMPS +0.40, BLiMP +0.31, Reading +0.49) while losing Supplement -0.99, GlobalPIQA -1.485, SuperGLUE -1.407, and AoA -12.687. A new 100M run on the same compact-core substrate would not be justified.

## What the existing trained ladders can still answer

The density eval repair/017 trained models separate factors without additional training:

1. `near_repeat` versus inherited clean-Qwen: FineWeb replacement / row-holdout effects under near-length source repetition.
2. `near_view` versus `near_repeat`: generated near-length view effect with little compression.
3. `compact_repeat_core` versus `near_repeat`: compact-core source-family and neutral top-up effects without generated compact views.
4. `compact_view_core` versus `compact_repeat_core`: generated compact transformation on the shared core.
5. `compact_view_reinvest` versus `compact_view_core`: added compact source exposure through saved words.

The missing AoA values for near repeat, near view, compact repeat core, and reinvest are now the lowest-cost way to localize the acquisition-timing inversion. All four ladders have complete 19-step coverage according to `experiments/archive/frontier_consolidation/data/aoa_localization/aoa_localization_preflight.json`.

AoA localization on the four already-trained ladders was launched using `experiments/archive/frontier_consolidation/scripts/run_aoa_localization_ladders.py`, with outputs under `experiments/archive/frontier_consolidation/data/aoa_localization`. This evaluation performs no training.

## Early AoA mechanism result already available

CPU recomputation from existing `compact_view_core` and inherited clean-Qwen surprisal files is saved in `research/documents/frontier_consolidation/data/aoa_mechanism_analysis/aoa_mechanism_analysis.md` and `.json`.

Key reading:

- clean-Qwen seed43022: r=-0.1109, p=0.1382, leaderboard AoA 0.0; late-minus-early fitted model AoA = -0.1355 log10 words.
- clean-Qwen seed43122: r=-0.1019, p=0.1543, leaderboard AoA 0.0; late-minus-early = -0.0980.
- compact_view_core: r=-0.1269, p=0.0787, leaderboard AoA -12.687; late-minus-early = -0.1506.

This means the large leaderboard penalty is not a large new ordering reversal relative to clean-Qwen. Clean-Qwen already has a mildly negative fitted relation that is not significant enough to count. Compact-view core makes the negative relation only somewhat stronger and more statistically stable. On the 159 common fitted words, compact-minus-clean fitted-AoA deltas have mean -0.0262 log10 words and almost no relation to child AoA (r=0.0251, p=0.754). The official penalty appears to come from strengthening a pre-existing negative tendency and changing fit validity/variance, not from a simple global delay of child-early words.

## Reinvestment should be judged from complete surface, not duplicated locally

`compact_view_reinvest` has the strongest fast surface: equal7 44.2886, seven-column sum 310.02. From this fast surface alone, it needs SuperGLUE+AoA >= 66.18 to reach Overall 41.8 and >=67.98 to reach 42.0. Therefore, with SuperGLUE near compact-core 68.90, reinvest can tolerate only about AoA -2.72 for 41.8; with a full-split zero-shot degradation like compact-core Supplement, the tolerance shrinks sharply.

Full `compact_view_reinvest` evaluation remains pending. Its completed result determines whether reinvestment retains enough benefit to remain an endpoint candidate. If it shares compact-core's large negative AoA or loses its fast zero-shot advantage on full splits, same-substrate density training should stop. If it preserves most full zero-shot gains with AoA near 0 or only modestly negative, it remains a trained density endpoint with a possible path to 41.8.

## Consequence for next research

Before designing another corpus, read the existing-ladder AoA localization and reinvest full result. A redesigned density route is justified only if those results reveal a mechanism that can preserve the EWoK/Entity/COMPS movement while also recovering GlobalPIQA/SuperGLUE and preventing a counted negative AoA. Possible directions should be benchmark-independent and broad-task motivated, such as: retaining compact views only where the original has concrete relations useful for GlobalPIQA/COMPS, mixing compact and near views to avoid over-accelerating abstract/late-style word contexts, or changing where compact packets are injected relative to the inherited clean-Qwen base. None of these should be trained until the existing-ladder localization identifies which factor actually carries the timing penalty.
