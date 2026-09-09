# lead cross seed decision framework lead synthesis: how to read the pending seed43122 grid

## Active state

The research now has three separate objects that must not be merged.

1. The protected public fallback is scale1.75 `chck_82M`, displayed 41.94 and already challenge-counted.
2. The cleaner ordinary-training branch is scale1.75 `chck_84M`, public HF model `leslie721007/babylm-strict-small-scale1p75-chck84` revision `040284de9ac49eee6dc1b30cf65aea97ec17d86e`, native-valid prediction carrier SHA256 `b55e1997c3fbc0255527f35c8225f788fd1dae6a1427b13b142a13ed9e64189d`, and local official-compatible Overall(AoA0) `42.0189129742181`.
3. The numerically stronger local carrier remains coherent86 alpha0.75 at projected Overall(AoA0) `42.1210247099666`, but its evidence reads as competence redistribution, not the transferable data-efficient learning principle.

The completed reference and scale1.25 grids changed the residual-capacity story. The reference trajectory's best selected checkpoint is `chck_84M` with cheap7 `44.12357142857143`; no 86M--100M reference checkpoint competes. The scale1.25 seed/mask-matched lower-energy trajectory is broadly weaker: mean delta versus reference is `-0.410982` cheap7, `-0.815208` cheap6 without GlobalPIQA, `-0.949125` cheap5 without GlobalPIQA/Reading, and `-0.425313` on EWoK/Entity. Lower adapter energy is therefore not the current route.

The single unresolved result is the already-running scale1.75 seed43122 common-grid scoring. It decides whether the seed43022 late 82--84M phase is a structural property of the adapter-coordinate learning dynamics or a favorable stochastic alignment on one seed/mask stream.

## independent_review-supported reframing

The important readout from seed43122 is not just the aggregate cheap7 peak bin. In the reference trajectory, the component-family peaks are spread across a wide late window: BLiMP at 94M, Supplement at 76M, EWoK at 74M, Entity at 88M, COMPS at 78M, GlobalPIQA at 84M, Reading at 78M. Thus the 84M aggregate peak is the moment when several shallow family movements align; it is not a single global maximum of all abilities.

The primary comparison after seed43122 delivers should be the per-column peak vector and its relation to the reference vector. There are three scientifically different outcomes:

- If the seed43122 column peak vector resembles seed43022 and the aggregate peak also lands near 82--84M, residual-capacity allocation dynamics become the stronger research object. The next low-cost work should localize where family competences live in the existing checkpoints before any new training.
- If the column peak vector resembles seed43022 but the aggregate peak shifts, the competence-family schedule is structural but the aggregate alignment is seed/mask-sensitive. The next work should combine family-wise allocation analysis with stabilization of broad competence.
- If the column peak vector is unrelated and no comparable late broad-family phase appears, the seed43022 `chck_84M` endpoint is still a valid asset but not a robust learning law. The next route should study stochastic competence stabilization or a more reliable inductive mechanism, not continue tuning the original peak.

This framing is consistent with the earlier negative evidence: compact semantic views did not transfer broadly into the tested GPT2 causal coordinate, the source-conditioned ordering interaction was general bidirectional retrieval rather than compact-specific, alpha/interpolation was redistributive, and scale1.25 produced a mostly volatile positive row while losing stable aggregates.

## New lead cross seed decision framework assets

A reusable analyzer was written before seeing seed43122:

- `experiments/archive/frontier_consolidation/scripts/cross_seed_common_grid_analyzer.py`

It compares selected common-grid trajectories by aggregate peak, connected near-best band, local peak excess, per-column peak vector, column-peak spread, peak-vector correlations, derived family aggregates, and volatile contribution. It was smoked on the completed reference and scale1.25 grids:

- `research/documents/frontier_consolidation/data/cross_seed_analyzer_smoke_reference_scale125/cross_seed_common_grid_analysis.md`
- `experiments/archive/frontier_consolidation/data/cross_seed_analyzer_smoke_reference_scale125/cross_seed_common_grid_analysis.json`

The smoke reproduces the known scale1.25 negative interpretation: aggregate peak shifts by +2M but peak-vector Pearson/Spearman are near zero, mean absolute column-peak shift is 8.286M, mean delta cheap7 is `-0.410982`, and no broad positive row appears.

When seed43122 delivers, the command form is:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_integrity_check.py \
  --out-dir experiments/archive/frontier_consolidation/data/deberta_common_grid_integrity_seed43122 \
  --trajectory experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M/selected_trajectory.json

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/interpret_deberta_common_grid.py \
  --trajectory scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json \
  --trajectory scale1p25_seed43022_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M/selected_trajectory.json \
  --trajectory scale1p75_seed43122_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M/selected_trajectory.json \
  --adapter-energy-json experiments/archive/frontier_consolidation/data/adapter_norm_trajectory/adapter_norm_trajectory.json \
  --reference-label scale1p75_seed43022_reference \
  --out-dir experiments/archive/frontier_consolidation/data/deberta_common_grid_interpretation_all3 \
  --strict-complete

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/cross_seed_common_grid_analyzer.py \
  --trajectory scale1p75_seed43022_reference=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json \
  --trajectory scale1p25_seed43022_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p25_seed43022_dense_common2M/selected_trajectory.json \
  --trajectory scale1p75_seed43122_dense=experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43122_dense_common2M/selected_trajectory.json \
  --reference-label scale1p75_seed43022_reference \
  --out-dir experiments/archive/frontier_consolidation/data/cross_seed_common_grid_analysis_all3 \
  --strict-complete
```

## Stabilization route: not active until seed43122 points there

Same-trajectory weight averaging remains only a prepared stabilization probe. chck84 endpoint carrier validation/192 established that the `80/82/84` average is mechanically valid and functionally smooth on an MLM sample; chck84 carrier and scale125 grid synthesis established that its attempted selected score failed before model evaluation because the run view lacked `scientific_metrics.json`. There is no BabyLM score for the average.

lead cross seed decision framework wrote a repair helper, but only dry-ran it:

- `experiments/archive/frontier_consolidation/scripts/prepare_average_run_view_metadata.py`

The dry run confirms it would write explicit pseudo-run metadata with `additional_training_word_exposure=0`, `actual_training_steps=0`, source endpoints `80M/82M/84M`, and the candidate model SHA `d47c15f96e3424fc0946cfa4e747f9a6374006d9a3cf68d58d5031509585ac50`. It was deliberately not executed with `--write` and no average scoring was run.

If seed43122 makes stabilization the active question, the first safe average test is a single selected cheap-task endpoint after writing this pseudo-run metadata and using a fresh output directory or removing the failed per-target state. Its value must be read through cheap7, cheap6 without GlobalPIQA, cheap5 without GlobalPIQA/Reading, and EWoK/Entity. A GlobalPIQA/Reading-carried bump would repeat the scale1.25 and alpha0.75 failure pattern.

## Low-cost science after seed43122

If the late phase reproduces, do not harvest only the endpoint. Use existing checkpoints to localize the allocation mechanism. Useful next measurements include adapter-block/function patching across family-peak checkpoints and 100M, and family-restricted function drift between checkpoints. These can convert the late-phase description into a mechanism before new H100 training.

If the late phase does not reproduce, do not tune seed43022 further. Use the two seed trajectories to measure cross-seed competence covariance and then test stabilization only through official-compatible selected scoring. Cross-seed parameter averaging is not valid without a separate alignment argument; same-trajectory averaging can be tested within each seed if it becomes scientifically motivated.

No new GPU task, model upload, or leaderboard submission was performed in lead cross seed decision framework.
