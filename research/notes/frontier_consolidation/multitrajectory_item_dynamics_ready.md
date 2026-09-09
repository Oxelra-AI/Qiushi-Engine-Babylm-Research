# multitrajectory item dynamics ready: generalized multi-trajectory item-dynamics readout (ready for seed43122)

CPU/file-only analysis without model inference. The seed43122 selected grid remains pending.

## What was built

`experiments/archive/frontier_consolidation/scripts/multitrajectory_item_dynamics.py` generalizes the chck84 item movement synthesis/chck84 late item dynamics synthesis single-trajectory item analysis into a reusable multi-trajectory tool. For any set of completed selected-grid directories it:

- reconstructs official-like item correctness for all classification columns (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA) using the exact chck84 item movement synthesis parsers, plus payload Reading scalar;
- computes per-endpoint cheap7/cheap6-no-GP/cheap5-no-GP-Reading/relation-state/syntax-COMPS/volatile, the aggregate best endpoint, near-best contiguous band, local excess, and column peak vector;
- measures item churn dynamics **around each trajectory's own aggregate best endpoint** (gains/losses prev→best, best→next, best→final; transient vs persistent gains) so peaks at different exposure bins are compared on equal footing;
- late-window (80–100M) per-column correctness signature concentration;
- subtask-unit bootstrap of best-minus-prev / next-minus-best / final-minus-best;
- pairwise item-overlap vs a reference label, both same-endpoint and best-vs-best, giving net item differences, disagreement fraction, and correct-set Jaccard per column;
- a `trajectory_score_alignment` block that records max |item-recomputed minus selected| per metric, so endpoint arithmetic still uses the selected/payload carriers while item analysis is used only for churn/overlap.

## Validation smoke (completed reference + scale1.25 grids)

Output: `experiments/archive/frontier_consolidation/data/multitrajectory_item_dynamics_smoke_reference_scale125_v2`.

- Reference scale1.75 seed43022 aggregate best = `chck_84M`, cheap7 44.123626 (item-recomputed), local excess cheap7 +0.258062, cheap5-no-GP/Reading +0.102375, relation/state +0.051195 — matches chck84 late item dynamics synthesis.
- Scale1.25 seed43022 aggregate best = `chck_86M`, cheap7 43.538267, local excess cheap7 +0.153455 — matches chck84 carrier and scale125 grid synthesis.
- Item recomputation is faithful: max |cheap7 diff vs selected| = 0.002799 (scale1.25) and 0.002456 (reference); max |BLiMP diff| ≈ 0.013. These small differences are payload-rounding/tie effects and do not move the aggregate best endpoint. Endpoint arithmetic must still use the selected/payload carrier scores.

## New item-level cross-trajectory finding (reference vs scale1.25)

Best-vs-best (reference `chck_84M` vs scale1.25 `chck_86M`), correct-set Jaccard per column: BLiMP 0.773, Supplement 0.843, EWoK 0.556, Entity 0.586, COMPS 0.504, GlobalPIQA 0.612. Net (scale1.25 − reference) items: BLiMP −1161 (−1.94%), Supplement +102 (+1.95%), EWoK −59, Entity −24, COMPS +106 (+0.12%), GlobalPIQA +4.

So even at each trajectory's own best endpoint, the lower-adapter-energy scale1.25 run does not reproduce reference competence at item level: it loses broadly on BLiMP, is roughly even on the heavily weighted COMPS pool, and gains only on volatile/small columns. This is item-level confirmation of the chck84 carrier and scale125 grid synthesis aggregate conclusion that scale1.25 is broadly worse and delayed, not a broadened peak.

Column peak vectors also differ substantially between reference and scale1.25 (reference BLiMP 94 / Supplement 76 / EWoK 86 / Entity 88 / COMPS 78 / GlobalPIQA 84; scale1.25 BLiMP 88 / Supplement 96 / EWoK 78 / Entity 86 / COMPS 88 / GlobalPIQA 84), consistent with the lead cross seed decision framework view that the aggregate best endpoint is an alignment of several shallow family movements rather than one global optimum.

## How to use after seed43122 delivers

Command after seed43122 selected-grid completion and a passing `selected_mlm_integrity_check.py` on its output:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/multitrajectory_item_dynamics.py \
  --trajectory scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M \
  --trajectory scale1p75_seed43122_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M \
  --trajectory scale1p25_seed43022_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M \
  --reference-label scale1p75_seed43022_reference \
  --out-dir experiments/archive/frontier_consolidation/data/multitrajectory_item_dynamics_all3 \
  --bootstrap 2000
```

Predeclared reading for seed43122 (init + mask stream change vs reference; same scale 1.75):

- **Structural robust**: seed43122 has a comparable narrow late aggregate peak, a related column peak vector (per-column peaks within a few 2M bins of reference), and best-vs-best item overlap with reference materially higher than the reference-vs-scale1.25 Jaccard — the late competence-allocation phase is a property of the adapter-coordinate learning dynamics.
- **Structural but stochastic**: seed43122 shows a narrow late peak with similar family dispersion but the aggregate peak bin and item set differ substantially (Jaccard near the scale1.25 level) — the phase is real but its exact location/alignment is seed/mask-sensitive, and the next high-value route is stabilizing broad competence across stochastic trajectories rather than tuning the seed43022 peak.
- **Unstructured**: seed43122 has no comparable late broad-family peak or an unrelated column peak vector — seed43022 `chck_84M` is an endpoint asset, not a robust learning law, and future work should target variance/stabilization or a stronger inductive mechanism.

Use `cross_seed_common_grid_analyzer.py` for the score-level peak-vector correlation and `multitrajectory_item_dynamics.py` for the item-level churn/overlap; together they decide the readout above. Endpoint arithmetic and any HF carrier work continue to use selected/payload carrier scores.
