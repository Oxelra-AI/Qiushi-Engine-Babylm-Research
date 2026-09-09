# aligned shuffled full ewok residual slices — late-window evidence and contrastive-margin coordination

## Independent Trajectory Evidence

companion analysis does **not** currently have an exact non-scale1.75 late-window official cheap7 surface for `chck_77M`–`chck_83M` plus `chck_100M` that could serve as independent ground truth for a new selector. Local searches in `experiments/archive/representation_and_objectives/data` and `experiments/archive/frontier_consolidation/data` found:

- exact 77–83M cheap7 sweep exists only for the scale1.75 trajectory:
  - `experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep/summary/scale1p75_checkpoint_sweep_summary.json`
  - values: 77M 43.2821, 78M 43.7021, 79M 43.5786, 80M 43.8121, 81M 43.6493, 82M 43.9600, 83M 43.8079.
- non-scale compact-view evidence exists at coarser or different coordinates, not the exact requested late window:
  - inherited/noncompliant dense 76–84 listings appear in early compact summaries, but those rely on the out-of-budget tokenizer and should not be used as compliant independent ground truth.
  - legal compact-view seed43022 no-AoA/broad checks exist at 80M/90M/100M, e.g. `experiments/archive/frontier_consolidation/data/broad_noaoa_late_checkpoint_merge/broad_noaoa_late_checkpoint_merge.json` and `experiments/archive/frontier_consolidation/data/checkpoint_average_noaoa_merge/checkpoint_average_noaoa_merge.json`.
  - Legal compact/control treatment checks exist at 70M/80M, not 77–83M.
- fixed retention-vector route is already negative as a selector: `research/documents/frontier_consolidation/data/retention_cross_trajectory/retention_cross_trajectory_summary.md` says the unchanged probe picks 100M for mean-like NLL metrics and 77M for forgetting-min across scale1.75, U256, and aoa mincontext discrepancy audit, while the known scale1.75 official peak is 82M.

Consequence: a corpus-derived contrastive-margin signal should not be calibrated on the protected scale1.75 peak alone. If companion analysis needs independent late-window ground truth before trusting it, companion analysis can only offer cheaper coarse anchors already measured (legal compact seed43022 80/90/100 and seed43122 45/80/100, plus averages) unless a new no-training evaluation pass is deliberately launched. That launch would need a narrow purpose: whether the label-free signal ranks a non-scale trajectory's real late official surface, not endpoint chasing.

## Useful numbers from existing non-scale anchors

From `experiments/archive/frontier_consolidation/data/broad_noaoa_late_checkpoint_merge/broad_noaoa_late_checkpoint_merge.json`:

- legal compact seed43022: 80M equal7_full_entity 43.7707, 90M 44.3886, 100M fast 44.2429. The coarse peak among these is 90M, with EWoK 53.36 and Supplement 67.6.
- legal compact seed43122: 45M equal7_full_entity 42.9643, 80M 43.3821, 100M fast 42.8771. The coarse peak among these is 80M, with GlobalPIQA_mean 36.59, EWoK 49.73, Supplement 65.2.
- averages: `avg43022_90_100` equal7_full_entity 44.2614; `avg43122_80_100` 43.3150. These are not new trajectories and do not replace exact checkpoint ground truth.

## contrastive-margin scout: useful but not yet a trained object

`research/documents/frontier_consolidation/data/contrastive_margin_feasibility/contrastive_margin_feasibility_scout.md` scanned 64,739 legal-pool rows without official labels and found ample heuristic frame yields:

- belief/report role binding: 57,683 events / 31,650 unique rows.
- comparative relation: 6,559 events / 5,388 unique rows.
- multi-operation entity state: 28,349 events / 28,349 unique rows.
- polarity relation composition: 35,703 events / 23,991 unique rows.
- spatial directional relation: 172,037 events / 57,601 unique rows.
- temporal/procedure order: 64,037 events / 36,790 unique rows.

The scout's own warnings are central: counts are heuristic frame yields, not validated minimal pairs; transformations/filters/tokenizer-length controls/aggregation must be frozen before scoring checkpoints; natural state/procedure frames may remain noisy.

## Low-cost construction path if full-EWoK turnover closes the coupled route

The next non-training construction should turn the heuristic corpus scan into a fixed small measurement object, not into training immediately:

1. Select 3–4 transformation families that induce a known alternative change without official text: polarity flips, comparative direction swaps, entity-state operation order, and belief/report source-target role changes.
2. For each family, create a rule-generated coherent sentence and one controlled perturbation from the same legal-pool example. Reject examples with unmatched named-entity count, large token-length imbalance, target tokens absent from tokenizer vocabulary boundaries, or surface artifacts.
3. Score existing checkpoints only at first: scale1.75 77–83/100 and one non-scale coarse trajectory if available. The readout should measure coherent-minus-perturbed pseudo-likelihood margin, not target-token reconstruction loss.
4. Compare ranking against already known official late surfaces without using official labels in the transformation creation. If the signal does not distinguish known 82M scale1.75 peak or fails on the coarse legal compact anchors, it should remain a source-analysis object, not a training objective.
5. Only such evidence would justify designing a training loss. A premature objective would risk repeating ACS/PVDM: local likelihood changes without context-conditioned relation transfer.

## Pending evidence

The following full-EWoK readouts remain unresolved; no result is inferred:

- full EWoK four-cell readout for `mlm_only_20M`.
- full EWoK four-cell readout for `coupled_aligned_20M`.
- full EWoK four-cell readout for `coupled_shuffled_20M`.

When they finish, aggregate with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B experiments/archive/representation_and_objectives/scripts/full_ewok_coupled_turnover.py --aggregate
```
