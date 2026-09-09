# full core route localization and existing ladder plan route refinements after independent_review and corrected reinvest arithmetic

## Core route judgment that remains supported

The compact-density result is scientifically important but not currently a SOTA endpoint. `compact_view_core` improved the hard NLP profile against clean-Qwen in EWoK, Entity, COMPS, BLiMP, and Reading, but its complete official-compatible Overall is 39.9877 because it lost Supplement, GlobalPIQA, SuperGLUE, and received AoA -12.687. If AoA alone is set to 0 while all other measured columns stay unchanged, Overall becomes 41.39735, still below 41.8.

Therefore, the next scientific work should read already-trained ladders rather than start another 100M run on the same substrate. The existing models separate useful factors: near repetition, near generated views, compact repetition, compact generated views, and compact-source reinvestment. Their AoA trajectories plus reinvest full evaluation can reveal whether the density effect is worth a redesigned corpus.

## Corrections and sharpened interpretation

1. Official-like seven-column arithmetic must use `Entity_full` from the fast no-AoA screen, not the smaller fast Entity subset, when projecting toward official Overall.

2. The compact-core non-AoA movement is approximately a redistribution, not a net non-AoA gain against clean-Qwen. Relative to clean-Qwen, the eight non-AoA column deltas sum to about -0.016, matching NLP_average -0.00178. The value of the line is that it moves the task profile toward EWoK/Entity/COMPS while paying elsewhere.

3. The exact official AoA code in `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline/utils.py` returns 0.0 when p>0.1 and otherwise returns the raw Pearson correlation; the full-eval wrapper uses leaderboard score = 100*raw correlation. This explains why clean-Qwen r≈-0.11 with p≈0.14 becomes 0.0, while compact-core r=-0.1269 with p=0.0787 becomes -12.687.

4. The currently supported AoA reading is narrow: compact-core slightly strengthens a weak negative fitted relation already present in clean-Qwen and makes it count under the official p≤0.1 mapping. On 159 common fitted words, compact-minus-clean fitted-AoA shift averages -0.0262 log10 words and has no relation to child AoA (r=+0.0251, p=0.754). The source of the counted penalty may be changed fitted-word support, variance, a few influential words, or category-specific movement; it is not yet a broad child-age-linked global reversal.

5. Reinvestment remains the only already-trained density endpoint with a plausible route, but only if its full-split columns and AoA survive. Corrected official-like reinvest fast seven-column sum using `Entity_full` is 309.700, so it needs SuperGLUE+AoA ≥ 66.500 for Overall 41.8 and ≥ 68.300 for 42.0. If its full split receives the same BLiMP/Supplement/EWoK changes observed for compact-core, the sum falls to 305.880 and it needs SuperGLUE+AoA ≥ 70.320 for 41.8. This makes the independent full reinvest evaluation decisive.

## Useful low-cost analyses before any redesigned corpus

- Rerun `experiments/archive/frontier_consolidation/scripts/analyze_aoa_from_surprisals.py` after AoA localization finishes, including all density ladders. Then run `harmonized_aoa_support.py` on the expanded analysis.
- Interpret localization patterns:
  - near repetition already counted-negative: FineWeb replacement / row-holdout source mix is implicated.
  - near view moves more negative than near repetition: generated same-source views contribute even without compression.
  - compact repetition is negative: compact-core source selection and neutral top-up matter even before generated compact text.
  - compact view much more negative than compact repetition: compact generated form or context concentration matters.
  - reinvest less negative than compact core while retaining full gains: broader compact source exposure may stabilize the route.
  - reinvest strongly negative or full surface collapses: stop same-substrate density training and redesign only from a clear mechanism.
- Build evaluation-independent corruption probes from ordinary corpus sentences for role reversal, event order, causal/temporal substitution, instrument/material replacement, spatial relation reversal, negation, and modality. Compare likelihood margins for clean-Qwen, compact-repeat, and compact-view checkpoints to test whether compact views made representations too gist-like.
- Rank existing source/view pairs by source-internal properties: compression ratio, event-frame preservation, lexical overlap, contradiction risk, unresolved coreference, and discourse force. Relate those strata to source/view likelihood and hidden-state similarity before choosing a new view mixture.

## Consequence

Do not spend H100 time on another unchanged compact-core or reinvest-style corpus until the existing AoA localization and paired reinvest full result are read. A strong next density route would probably be a semantically gated mixed-resolution construction: compact views only for complete stable propositions, near or role-explicit views for physical/event/discourse relations, and original-only retention for malformed or ambiguous sources. But this should remain a hypothesis until the current ladders show which factor actually damages full official performance.
