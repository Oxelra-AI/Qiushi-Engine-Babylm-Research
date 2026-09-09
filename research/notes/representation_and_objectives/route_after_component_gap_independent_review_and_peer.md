# babysteps public method reading route synthesis after component gap, independent_review, peer evidence, and FineWeb source-pool audit

## Current hard fact
No new BabyLM score has been produced yet in representation_and_objectives. The decisive active artifact remains the managed no-AoA evaluator `s8_t14_tool1`, which is writing to `experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval`; the directory remained write-locked during babysteps public method reading, so no result was inferred.

The matched semantic-view training pair is complete:
- treatment `semantic_view_treatment`: exact 100M exposure, loss_last 2.547764;
- control `original_packet_local`: exact 100M exposure, loss_last 2.412606;
- core recipe/seed/tokenizer/shape/checkpoint ladder matched;
- changed block exposure 8,201,870 words each.
Training-loss difference is not BabyLM competence evidence.

## Component-gap result that changes the next route
Strict-small public target remains `go76dof/wwm_curriculum_simplification_40k` at Overall 41.8. The 41.93 `go76dof/spangeo05_simplification` row is strict track, not strict-small.

Against the inherited trusted local `qwen_clean_aligned` 41.3443 coordinate, the 41.8 leader gap is concentrated in:
- EWoK: ours 50.19 vs leader 56.07, contribution -0.653 Overall;
- GlobalPIQA: ours 36.62 vs leader 39.67, contribution -0.339;
- Entity: ours 25.76 vs leader 28.45, contribution -0.299;
- COMPS: ours 51.78 vs leader 53.57, contribution -0.199;
- BLiMP small deficit: -0.040.

Our advantages are large and valuable:
- Supplement: ours 62.84 vs leader 56.01, contribution +0.759;
- Reading: ours 7.76 vs leader 5.42, contribution +0.260;
- SuperGLUE: ours 70.31 vs leader 69.79, contribution +0.058.

The total displayed gap is about 0.455 Overall. To exceed the leader with all other columns fixed requires >4.10 summed column points. A safer one-seed target is >4.5 summed points because small deltas can be seed/eval noise.

## What this means mechanistically
The active SimpleWiki semantic-view contrast is low-ceiling for the actual SOTA gap because it re-expresses the same SimpleWiki propositions and cannot add broad factual/entity/commonsense experience. It can still answer whether generated second views are useful relative to same-source repetition. Its result should inform whether to pay the generation cost, but it should not be treated as the only SOTA route.

The next route must target the deficit columns while preserving the strengths. The clearest hypothesis is not simply "more FineWeb" or "more simplification"; it is:

> A small, high-precision, broader factual source component inserted into the protected clean-Qwen/developmental mixture may raise EWoK/Entity/COMPS/GlobalPIQA enough to beat 41.8 if it does not erase Supplement/Reading. Faithful simplification is a separate possible accelerator and must be isolated from source breadth.

## Corrections from independent_review and source-pool audit
independent_review verified the arithmetic but corrected several over-broad claims:
- The four deficit columns should not be treated as one proven causal faculty. EWoK and COMPS are broadly weak; Entity and GlobalPIQA are leader-relative deficits.
- The available clean rewrite substrate is not yet 1.5-2M high-quality source words. frontier_consolidation high-anchor is ~97,605 source words; high-precision is ~214,771; medium repaired is ~469,887 but lower-quality. representation_and_objectives/deduplicated priority pool across available tiers contains 33,512 unique normalized sentences / 725,008 source words / 5,293 docs, but reaching ~1.38M paired words at rewrite/source=0.9 requires including lower-quality medium/balanced material.
- Inflating a high-precision set by heavy repetition would change a breadth test into a repetition/consolidation test.
- A clean experiment must separate source breadth from generated rewrite utility.

CPU source-pool audit: `experiments/archive/representation_and_objectives/data/fineweb_source_pool_audit/fineweb_source_pool_audit.json`.
Priority sources: `experiments/archive/representation_and_objectives/data/fineweb_source_pool_audit/fineweb_priority_unique_sources.jsonl`.

## Public BabySteps reading
`svsatheesh/BabySteps_MurphysLaw-10M-mixed` is relevant but not evidence for FineWeb. It uses only the official strict-small corpus with a GPT-BERT hybrid objective, AdaMuon, LR tuning, and tail averaging. It preserves high Supplement/Reading and improves BLiMP/Entity/COMPS relative to many baselines, but Overall is 40.86 because AoA is negative and EWoK/GlobalPIQA remain below the go76dof leader. This source says architecture/optimizer may later matter, but it does not replace the immediate data question.

## Decision interpretation for the pending semantic-view comparison
When `semantic_view_packet_local_delta_summary.json` becomes available, read component trajectories, not just final equal7.

Let Ksum = ΔEWoK + ΔEntity + ΔCOMPS + ΔGlobalPIQA. Let protected = Supplement and Reading.

1. If semantic-view gives positive Ksum and preserves Supplement/Reading:
   - Generated second views are empirically useful, but because the source is same-SimpleWiki the next source-breadth experiment still needs separate B and C arms.
   - Run selected full evaluation for the best semantic endpoint and matched packet endpoint only if the no-AoA equal7 movement is large enough to plausibly alter Overall after SuperGLUE/AoA.
   - Then test FineWeb source+rewrite using the three-arm decomposition below.

2. If semantic-view helps only BLiMP/Supplement/Reading but not Ksum:
   - It is not addressing the leader gap. Do not train the semantic hybrid as the main route.
   - Use FineWeb primarily as source-breadth, with rewrite as an optional arm after faithfulness yield is known.

3. If semantic-view is near-zero or negative:
   - Same-source generated views are not supported as a high-value lever.
   - The next low-cost action should be a Qwen3.5-9B faithfulness pilot on high-anchor/high-precision FineWeb sources, not raw FineWeb source-swap training and not another SimpleWiki variant.

4. If semantic-view damages Supplement/Reading:
   - Be cautious about any generated-text insertion. Source-only FineWeb repetition becomes essential; source+rewrite cannot be interpreted as breadth.

## Clean FineWeb experiment design after a generation/faithfulness pilot
Use three arms under the protected 8x480/baseline16k/fixed-WWM/AdamW recipe and the same seed/checkpoint ladder:

A. **Protected slot control**: clean-Qwen/developmental mixture with an official-corpus slot matching the FineWeb intervention's word count, row-length distribution, packing, and filler schedule.

B. **FineWeb source repetition**: the same selected FineWeb source sentences, repeated/packet-local-filled to match the word count and row geometry that C will use. This estimates factual source breadth versus A.

C. **FineWeb source + accepted faithful rewrite**: exactly the same selected FineWeb source sentences as B, plus one accepted faithful simplification per source, with source/rewrite order balanced or randomized. This estimates rewrite utility relative to identical factual coverage.

Important contrasts:
- Breadth effect = B - A.
- Rewrite effect = C - B.
- SOTA candidate = whichever of B or C improves the deficit cluster enough while preserving Supplement/Reading.

Do not train C alone; without B it confounds factual breadth with generated second views. Do not inflate high-precision sources by repetition unless the experiment is explicitly about repetition/consolidation.

## Generation/faithfulness pilot
The first pilot should use frontier_consolidation high-anchor 1024 or 2048 prompts, optionally supplemented by representation_and_objectives's 544 stratified slice for tier-stress testing. Required outputs before any corpus materialization:
- accepted rewrite yield by tier;
- entity/date/number/unit/location retention;
- polarity/modality/argument-role preservation;
- new entity/new number/new fact rates;
- near-copy rate;
- rewrite/source word ratio;
- examples of failures.

If high-anchor/high-precision acceptance is high, generate the full high-precision pool. If acceptance or quality is weak, pivot to source-only FineWeb breadth or construct a better source selector, not to launch a weak generated corpus.

## Next research requirement
The pending component deltas must be inspected before choosing the main route. If they support the three-arm FineWeb comparison, construct its materialization/generation pipeline under the controls above; otherwise reassess the mechanism and route value.
