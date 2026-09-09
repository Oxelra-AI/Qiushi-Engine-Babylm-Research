# coupled control interpretation infrastructure — coupled shuffled control interpretation infrastructure

## Current scientific state
The matched `coupled_sparse20_shuffled_20M` control launched in coupled correspondence control is still the decisive evidence for the dual-view mechanism. Do not change route before it finishes.

Question: endpoint resolved dualview hardsurface showed `coupled_sparse20_aligned_20M` strongly repairs the fixed full ewok interaction synthesis EWoK stable-reversal surface (stable failures 922 -> 430; accuracy 0.26581 -> 0.51258) and moves fixed fw globalpiqa relevant substrate GlobalPIQA hard52 (3.846% -> 9.615%, mean top-minus-correct 1.6229 -> 1.4432 nats), but it collapses broad cheap7 (38.7936 vs 39.7864 mlm_only). The missing control asks whether the repair depends on true source-rewrite correspondence or is only a coupled auxiliary trajectory effect.

## Matched-control status and provenance
Task: `s177_t14_tool1`  
Run dir: `experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022`  
Trainer: `experiments/archive/frontier_consolidation/scripts/dual_view_corrected_trainer.py` with `--mode shuffled`, otherwise matched to `coupled_sparse20_aligned_20M_seed43022`.

Single status check in coupled control interpretation infrastructure confirmed the run is a clean matched control:
- first loss `9.837543487548828`, bit-identical to aligned and mlm-only
- params `35,463,008`, adapter params `995,584`
- aux path active at update 1 (`aux_targets=24`, 4 conditioned + 4 source-free views)
- aux exposure is charged: update 50 has `cumulative_main_words=1,976,087`, `cumulative_aux_words=18,414`, `cumulative_charged_words=1,994,501`
- this mirrors the aligned run, which stopped at `19,995,185` charged words / 501 updates (`stopped_before_cap=true`); mlm_only has no aux and reached 20,000,000 / 506 updates, so mlm_only is a payload baseline, not the exact aux-charged control.

No further polling should occur. Collect only when the runtime delivers terminal state or when the next scientific decision truly depends on it.

## Upgraded readout script
Script upgraded and parsed/runs ready-only:
`experiments/archive/representation_and_objectives/scripts/coupled_shuffled_control_readout.py`

It reuses endpoint resolved dualview hardsurface readers verbatim:
- fixed fw globalpiqa relevant substrate GlobalPIQA hard52, all-option length-normalized paired rank/margin reader
- fixed full ewok interaction synthesis 1,471-row EWoK stable-reversal subset reader

New behavior:
- computes `coupled_aligned_minus_coupled_shuffled`, `coupled_aligned_minus_mlm_only`, and `coupled_shuffled_minus_mlm_only`
- adds `correspondence_classification` to the summary JSON
- decomposes stable-failure repair: total repair = aligned-minus-mlm_only; correspondence-free part = shuffled-minus-mlm_only; true-correspondence share = `(residual - total)/abs(total)` when aligned repairs relative to mlm_only
- interprets EWoK as the primary graded interaction surface and GlobalPIQA through ranks/margins as well as accuracy.

Ready-only output currently resolves `mlm_only_20M` and `coupled_sparse20_aligned_20M`; only `coupled_sparse20_shuffled_20M` is missing until training finishes.

## Natural-resolution interpretation rule
Do not read the result as a binary vote between unequal readouts.

1. **EWoK stable subset (1,471 rows) is the cleaner interaction measure.** Read aligned-minus-shuffled changes in:
   - accuracy
   - stable-failure count and fraction
   - `interaction_sum_wrong` mean/median/p05/p95 shift
   - domain and ContextDiff breakdowns saved by the reader.

   Large aligned-over-shuffled EWoK effect preserves the mechanism even if only a few GlobalPIQA hard52 choices flip.

2. **GlobalPIQA hard52 is small (52 rows).** Read:
   - hard52 rank counts (rank1/rank2/rank3/rank4)
   - hard52 mean/median top-minus-correct margin
   - small wrong-margin buckets
   - whole parallel-set accuracy and margin summary.

   Coherent margin/rank movement counts even if discrete hard52 accuracy moves by only one or two items.

3. **Route verdicts**
   - If aligned strongly beats shuffled on EWoK and GlobalPIQA margins/ranks move coherently: preserve coupled dual-view true-correspondence mechanism and design the smallest broad-preserving coupled variant.
   - If shuffled cuts EWoK stable failures near aligned: the endpoint resolved dualview hardsurface repair is mostly correspondence-free auxiliary perturbation; do not scale the sparse recipe.
   - Broad-score movement alone without alignment-specific hard-row transitions does not support the mechanism.

## Consolidated existing evidence now run and saved
Previously unwritten endpoint resolved dualview hardsurface consolidation script has been syntax-checked and run:
- JSON: `experiments/archive/representation_and_objectives/data/endpoint_and_dualview_consolidation/endpoint_and_dualview_consolidation.json`
- note: `research/notes/representation_and_objectives/endpoint_resolved_dualview_hardsurface.md`

It preserves:
- chck_82M endpoint: from-corpus reproduction PASS, full+fast carrier PASS, full blocks byte-equal to trusted earlier analysis collation
- dual-view broad scores: mlm_only cheap7 39.7864, separated aligned 40.2929, separated shuffled 40.0636, coupled aligned 38.7936
- hard surfaces: coupled aligned EWoK stable failures 430 vs mlm_only 922; separated aligned worsens EWoK stable failures to 988 despite broad preservation.

## Boundary
The reproduced `chck_82M` endpoint (Overall 41.942481, packaged and fast-materialized) remains untouched. The current work is mechanism verification for the requested generalizable learning advance, not endpoint packaging or final expression.
