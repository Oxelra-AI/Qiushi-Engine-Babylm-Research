# compact reinvest full eval summary reinvest route state

## Main scientific line

The mechanistic anchor remains compact generated views on the shared source core, but the SOTA-facing endpoint is now `compact_view_reinvest`: the same anchor-preserving compression with the recovered word budget used for additional compact source-view packets.

## Fast evidence

- `compact_view_reinvest` fast seven-column mean: 44.2886; full-Entity mean: 44.2429.
- Reinvest minus core: equal7 +0.4393, EWoK +1.54, Entity +0.77, GlobalPIQA_mean +0.485, with BLiMP -0.30 and COMPS -0.21.
- From the compact-view density calculation, reinvest needs SuperGLUE+AoA 66.180 for Overall 41.8.

## Fixed-budget principle

- Core used 10094 source-view pairs; reinvest uses 12155 pairs, adding 2061 compact pairs while staying within the same 423,520-word changed-block budget.
- Core pair words 353945 plus 69575 neutral top-up words; reinvest pair words 423511 plus 9 top-up words.
- Reinvest source words 261803 vs core 218542; rewrite/source ratio remains ~0.618; entity recall 0.996; number recall 1.000; content recall 0.663.

## Active work after compact reinvest full eval summary

- `s25_t24_tool1`: companion analysis-owned full official-compatible evaluation of already-trained `compact_view_reinvest` seed43022; output root `experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval`.
- `s25_t33_tool1`: train `compact_view_reinvest` seed43122 and run fast no-AoA screen; output root `experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast`.
- companion analysis still owns the core full-eval output directory `experiments/archive/frontier_consolidation/data/density_full_eval`; read it only after the runtime/group makes it available.

## Closed work

The older cached-FineWeb seqsafe96 branch and the core-only seed43122 controller were stopped before producing new BabyLM scores. They should not be interpreted as evidence for or against compact-view density.

JSON: `experiments/archive/representation_and_objectives/data/reinvest_route_state/reinvest_route_state.json`
