# Exact-swap interpretation while the strict probability-reallocation run is pending

## What was read

The completed analysis is documented in `experiments/archive/frontier_consolidation/analysis/Research_Report.md`, with supporting artifacts under `analysis/`. Its main structured outputs are:

- `analysis/innovation_metadata_summary.json`
- `analysis/innovation_wwm_smoke.json`
- `analysis/frozen_stream_batch_audit.json`
- `analysis/DESIGN_ANALYSIS.md`
- `analysis/PORTING_GUIDE.md`

The probability-reallocation training/evaluation results were not used in this analysis.

## Exact-swap construction

The proposed exact-swap conditional-innovation WWM variant is:

1. Start from the exact baseline 15% WWM Bernoulli selection.
2. For each changed row, propose at most one strict pair-local source-absent rewrite innovation group, using a stable hash rotation.
3. If the proposed group was already selected, do nothing and log a baseline collision.
4. Otherwise, replace one already-selected ordinary non-pair donor group of identical token length elsewhere in the batch, preferring ordinary rows.
5. Protect every group touching source/rewrite pair spans; source and copyable-pair masks therefore remain exactly unchanged.

The exact-swap metadata is legal: it uses only the frozen compact-view reinvest stream SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`, spatial repair route status tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`, and route portfolio and intervention assets span map SHA `f3f5ab9d425e777a0ac461c3d755bd7a4bf806fc73a5a266fbd284b2df353078`. Its lean trainer payload `innovation_metadata_train.jsonl` contains no normalized strings.

Key exact-swap counts per 10M pool:

- 26,315 pair-local source-absent innovation groups / 36,899 tokens
- 129,472 copyable groups / 195,608 tokens
- 3,004 rows with innovation
- 12,145 both-visible pairs; seven source-only and three invisible pairs excluded
- 52 changed rows are truncated at fixed seq256

CPU stress-test result:

- selected groups: 527,935 baseline and 527,935 treatment
- selected tokens: 764,428 baseline and 764,428 treatment
- per-batch group/token mismatches: zero
- selected innovation groups: 15,637 -> 25,820, gain +10,183 over four replicates
- copyable selections: 78,046 -> 78,046, delta zero
- source selections: 155,690 -> 155,690, delta zero
- donors all came from ordinary rows; no copyable/source/protected-pair donor violations

Projected full-100M effect from the smoke: about +25.5k added innovation selections over baseline, raising innovation group exposure by about 64.6%, while changing only ~0.174% of global selected-group mass.

## Difference from probability reallocation

The probability-reallocation run remained in progress. It is not the exact-swap design; it tests a different, already-checked intervention:

- Probability reallocation targets 17,968 stricter row-unique content-like innovation groups / 28,007 tokens per 10M.
- It uses p_strict=0.5 and reduces copyable rewrite probability to p_copy=0.1024016587 from a token-mass calculation.
- Source/filler/duplicate/function-like groups remain ordinary at 0.15 in expectation, but copyable rewrite labels are intentionally reduced.
- Expected per 10M change: strict token labels +9,802.45 and copyable token labels -9,802.45; strict group exposure multiplier x3.333 and copyable token/group exposure multiplier x0.683.

The durable comparison is `data/innovation_intervention_comparison/innovation_intervention_comparison.{json,md}`.

## Scientific reading

The exact-swap result does **not** make the already-running probability-reallocation experiment obsolete. Probability reallocation directly tests whether copyable rewrite supervision is surplus enough to fund much stronger strict-content innovation pressure. That question is still meaningful because pair identity/copyability was nearly saturated by 80M, while strict source-absent content innovation remained high-loss and source-conditioned in route reopen conditional innovation.

Probability reallocation and exact-swap are not interchangeable. A negative probability-reallocation score should close the p_strict/p_copy probability intervention, especially if it damages BLiMP/Supplement/EWoK or repeats the GlobalPIQA-nonparallel redistribution pattern. It should not automatically close exact-swap, because exact-swap has four material differences: broader pair-local source-absent target pool, one proposal per changed row, no copyable-pair suppression, and exact same-token donor replacement from non-pair ordinary groups.

Conversely, if probability reallocation shows broad cheap-column gains at 70M/80M versus spatial repair route status token-mean reinvest, especially on BLiMP/Supplement/EWoK and not only GlobalPIQA_nonparallel/COMPS, it is the stronger already-running route and should be continued toward 100M/full official evaluation before launching a second innovation variant.

## Interpretation of the pending probability-reallocation results

Read, do not infer:

- `training/runs/strict_content_innovation_wwm_reinvest_seed43022_80M/scientific_metrics.json`
- `data/strict_innovation_70_80M_eval/per_target/strict_content_innovation_seed43022_70M.json`
- `data/strict_innovation_70_80M_eval/per_target/strict_content_innovation_seed43022_80M.json`
- `data/strict_innovation_trajectory/strict_innovation_vs_trajectory.json`

Then decide:

- broad positive 70/80M cheap surface -> continue probability reallocation to 100M/full evaluation;
- innovation-probe improvement but broad score damage or copyable-like damage -> stop probability reallocation, consider exact-swap as a lower-perturbation next construction;
- no innovation-probe improvement and weak columns -> weaken the whole masking-pressure route;
- GlobalPIQA_nonparallel-only or COMPS-only movement -> stop probability reallocation as another redistribution route.
