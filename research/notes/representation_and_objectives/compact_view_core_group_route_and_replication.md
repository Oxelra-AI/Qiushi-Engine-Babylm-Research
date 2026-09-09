# compact view core group route and replication group route update: compact-view density is now the strongest candidate

## What changed

frontier_consolidation seqsafe96 interpretive foundation exposed a stronger group signal than companion analysis's pending cached-FineWeb source-breadth route. The protected contrast is **compact semantic view on the shared FineWeb core vs. exact repetition on the same shared core**, both overlaid on the inherited clean-Qwen base and trained with the same DeBERTa-v2 8x480 / baseline16k / fixed seq256 / fixed WWM / AdamW recipe.

Fast official-compatible no-AoA screen:

| column | compact repeat core | compact view core | view - repeat |
|---|---:|---:|---:|
| BLiMP | 67.09 | 66.93 | -0.16 |
| Supplement | 58.00 | 65.60 | +7.60 |
| EWoK | 48.64 | 51.55 | +2.91 |
| Entity fast | 25.11 | 27.30 | +2.19 |
| Entity full | 25.42 | 27.85 | +2.43 |
| COMPS | 51.09 | 52.18 | +1.09 |
| GlobalPIQA mean | 34.105 | 35.135 | +1.03 |
| Reading | 8.02 | 8.25 | +0.23 |
| equal7 mean | 41.7221 | 43.8493 | +2.1271 |
| equal7 full-Entity | 41.7664 | 43.9279 | +2.1614 |

Evidence: `experiments/archive/frontier_consolidation/data/density_noaoa_eval_compact_core/density_noaoa_eval_summary.json` and `research/notes/frontier_consolidation/compact_core_protected_result_and_full_eval_launch.md`.

Arithmetic projection from companion analysis: if compact_view_core keeps the fast seven-column surface, it needs only **SuperGLUE+AoA >= 69.255** to reach Overall 41.8. With the inherited clean-Qwen SuperGLUE 70.3086 and AoA 0, the projected Overall is **41.9171**. Evidence: `experiments/archive/frontier_consolidation/data/projection_thresholds/compact_core_full_eval_thresholds.json`.

## Interpretation

This does **not** yet prove a SOTA result. The screen omits SuperGLUE and AoA, and it is one seed. But it is the first group result with a plausible path above the visible 41.8 surface under the protected local recipe. It also moves the hard deficit families (EWoK, Entity, COMPS, GlobalPIQA) rather than only improving Supplement/Reading.

The mechanism is better stated as **information-density via moderately compressed, faithful same-source views**: Qwen views compress source wording enough to free training words while preserving local proposition adjacency, making the model see more reusable relational states per fixed word budget than literal source repetition. This is stronger than earlier SimpleWiki same-source views (real but low-ceiling) and stronger than near-length view contrast, because the compact view retained transfer gains while reducing redundancy.

Remaining vulnerabilities:

- Full official-compatible evaluation may reduce the fast surface or produce negative AoA.
- SuperGLUE could fall below the needed threshold even if fast columns remain strong.
- The result could be seed-fragile; one seed cannot support a final SOTA claim.
- Against the visible leader, compact_view_core still trails in EWoK (-4.52), COMPS (-1.39), and GlobalPIQA (-4.535) on the fast screen, so later refinement may still need factual/source breadth or representation changes even if Overall crosses 41.8.

## independent endpoint-robustness plan, followed by paired replication only if justified

companion analysis should now support the strongest group route rather than extend its weaker cached-FineWeb machinery. The immediate companion analysis task is best described as an **independent endpoint-robustness test** of companion analysis's compact_view_core, not a complete paired causal replication. It changes initialization and training RNG for the view arm only. This is the cheapest first check because if the view endpoint collapses, a paired repeat seed is unnecessary. If the view endpoint stays strong and companion analysis's full official-compatible result is near/SOTA-relevant, the next expensive check should be a matched `compact_repeat_core` seed43122 run so the mechanism is tested as `view43122 - repeat43122`.

An independent-seed endpoint test of compact_view_core was prepared:

- Training controller: `experiments/archive/representation_and_objectives/training/scripts/replicate_compact_view_core_seed43122.sh`
- Fast no-AoA evaluator: `experiments/archive/representation_and_objectives/training/scripts/fast_eval_replication.py`
- run directory: `experiments/archive/representation_and_objectives/training/runs/repl_compact_view_core_seed43122`
- output root: `experiments/archive/representation_and_objectives/data/compact_view_core_replication`
- Frozen data: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_core_neutral_100M.jsonl`
- Hash preflight succeeded: expected and actual sha256 `5380686975f6126eb1d9d88dfd91dea7a511d10a8e5a0cc45f2cc394a2853604`.
- Replication seeds: seed 43, extra_init_seed 43122, train_rng_seed 43123.
- Recipe: same COMPACT_EXPERIENCE DeBERTa-v2 8x480 / baseline16k / fixed WWM / AdamW / 100M word exposure.

The controller waits for a genuinely isolated GPU (>=70GB free and <=15% utilization across three samples), then trains the seed43122 replication and runs only the fast no-AoA screen. It exits without training if no GPU frees within the wait budget. This avoids repeating the semantic full eval route update/seqsafe96 interpretive foundation oversubscription failure.

## Decision use

- If companion analysis full official-compatible evaluation exceeds or approaches the visible 41.8 target and companion analysis seed43122 preserves the compact_view_core fast surface near seed43022 (especially a high equal7_full_entity and concurrent EWoK/Entity/COMPS/GlobalPIQA strength), the group should prioritize a **matched compact_repeat_core seed43122** run to convert endpoint robustness into a same-seed treatment-effect replication, then full evaluation/submission engineering if the paired result holds.
- If full evaluation fails because of SuperGLUE/AoA, the fast-screen component pattern remains valuable but the next work should specifically repair the missing full-score component rather than repeat the same corpus unchanged.
- If seed43122 collapses toward the seed43022 repeat baseline, the seed43022 fast screen is not stable enough for a SOTA claim; the group should analyze which component was seed-fragile before spending more H100 time.
- pending cached-FineWeb source-breadth tasks may still finish as useful evidence, but they should not receive additional machinery unless their real results change the route.

## independent_review verification nuance

independent_review verifier result `data/external/independent_review01_verifier1_integration.md` emphasized that the evidence currently supports only the bounded claim: under seed43022 and a tightly matched fast-screen design, compact generated same-source views improved several transfer aggregates relative to literal repetition. It does not yet isolate compression from rewriting/order/truncation/semantic alteration, establish full Overall, demonstrate seed-stable treatment effects, or establish official SOTA. It also notes that the seed43122 view-only run is endpoint robustness; the true paired treatment-effect replication is `view43122 - repeat43122`.
