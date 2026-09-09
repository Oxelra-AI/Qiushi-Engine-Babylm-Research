# dense focus official and mechanism state dense-focus official and mechanism state

## Historical candidate status
The active practical target remains a lawful BabyLM Strict-Small v5 improvement beyond coherent86/v4 projected Overall(AoA0) `42.1210247099666`. Dense unchanged-Qwen focus is currently the leading practical candidate, but no v5 is established. Two comparisons had started but remained incomplete at the time of this note:

- dense seed62064 u0080 official-compatible evaluation under `experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064`.
- fixed-policy dense seed62065 train+eval under `experiments/archive/functional_learning/data/dense_focus_rep_seed62065_{train,eval,pipeline}`.

Partial files did not establish the unfinished evaluations' outcomes.

## True coherent86 reference coordinate
The reference for official comparison is the combined coherent86 alpha0.75 coordinate, not the causal interface trajectory fast-screen parent:

- zero-shot/Reading payload: `experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json`
- SuperGLUE payload: `experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json`
- scores: BLiMP `68.51`, Supplement `63.64`, EWoK `50.02`, Entity `28.32`, COMPS `52.05`, SuperGLUE `69.81922238969935`, GlobalPIQA `38.565`, Reading `8.165`, AoA projected as `0.0`
- cheap7 `44.18142857142857`; projected Overall(AoA0) `42.1210247099666`

A self-comparison using `official_transition_compare.py --include-superglue` wrote `research/documents/functional_learning/data/coherent86_reference_selfcheck/coherent86_reference_self_official_transition.md` and confirmed the comparator reproduces this coordinate with small payload-vs-computed rounding differences only.

## Dense fast-screen profile from earlier analysis
Fast-screen item table: `research/documents/functional_learning/data/dense_fast_transition_compare/dense_vs_coherent86_fast_official_transition.md`.

Against the fast causal interface trajectory parent, dense u0080 has computed deltas: BLiMP `-0.552239`, Supplement `-0.400000`, EWoK `+0.363636`, Entity `+0.658300`, COMPS `+0.106543`, GlobalPIQA `+1.485437`, Reading `+0.055000`, cheap7 mean `+0.245240`. This is a real fast-screen improvement but not a uniform broad uplift; much of the summed positive movement comes from GlobalPIQA and Entity while BLiMP/Supplement fall.

## Partial official full-evaluation status
The official-compatible dense payload had completed BLiMP, Supplement, EWoK, and Entity by the time of this note:

- payload: `experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json`
- score readout: `research/documents/functional_learning/data/dense_official_score_readout/dense_official_score_readout.md`
- partial transition table: `research/documents/functional_learning/data/dense_official_partial_transition/dense_vs_coherent86_official_partial_official_transition.md`

Completed-column deltas vs true coherent86 full eval are BLiMP `-0.45`, Supplement `-0.60`, EWoK `-0.10`, Entity `+1.05`. The known non-AoA delta sum over these four columns is `-0.10`. Therefore COMPS + SuperGLUE + GlobalPIQA + Reading need a total delta above `+0.10` to exceed coherent86 under AoA0 arithmetic. This is arithmetic only; unknown columns are not estimated.

The partial full-eval table changes the interpretation of the fast screen:

- BLiMP remains negative but slightly less negative on full eval than fast (`-0.450` vs `-0.552`), with strong subtask sign agreement.
- Supplement remains negative and is less stable across fast/full samples.
- EWoK reverses from fast positive to full slightly negative.
- Entity strengthens from fast `+0.658` to full `+1.045`, with exact agreement on the regular-operation subtasks present in both sets.

Fast-vs-full comparison file: `research/documents/functional_learning/data/fast_vs_official_partial/fast_vs_official_partial.md`.

## Mechanism interpretation after independent_review
independent_review verifier integration: `data/external/independent_review01_verifier1_integration.md`.

Strongest faithful statement: dense focus produces a source-conditioned common-target gain and a structured Entity shift toward multi-operation cases, consistent with reduced reliance on local completion cues, but it does not yet isolate correct source-to-view correspondence. Dense and sparse both optimize `0.15 * mean(focus_loss) + 0.85 * mean(ordinary_loss)`; dense does not multiply the aggregate focus coefficient. The material change is that dense selects all Qwen content groups as targets, masking many second-view content clues while the source remains visible. This confounds two possible causes: input-side clue suppression and added focused target coverage.

independent_review recommended the most direct low-cost separator: evaluate aligned source vs length/position-matched shuffled source crossed with retained vs suppressed second-view clues. Dense should show a reproducibly larger aligned-over-shuffled advantage specifically under clue suppression if it has strengthened correct source-to-view routing. Equal gains for aligned and shuffled sources would favor generic context, coverage, or mask-density adaptation.

## New alignment x clue probe
Script: `experiments/archive/functional_learning/scripts/qwen_alignment_clue_factorial.py`.

Dry run under `experiments/archive/functional_learning/data/qwen_alignment_clue_factorial_dryrun` verified the planned geometry: 80 sampled segments, 160 base targets, balanced 6-condition grid (`aligned_source`, `shuffled_source`, `no_source`) x (`retained_view_clues`, `suppressed_view_clues`), all shuffled sources target-string absent and length-matched, mean token delta `-0.2`, max abs `10`.

A tiny CPU pilot under `experiments/archive/functional_learning/data/qwen_alignment_clue_factorial_cpu_pilot` scored 20 base targets on parent, sparse u0080, and dense u0080. It is only a functionality smoke/readout, not enough for a mechanism conclusion. Results:

- parent mean aligned-over-shuffled advantage: retained `1.321`, suppressed `1.784`, clue interaction `0.462`.
- sparse u0080: retained `1.362`, suppressed `1.799`, interaction `0.437`; delta interaction vs parent `-0.025`.
- dense u0080: retained `1.270`, suppressed `1.741`, interaction `0.471`; delta interaction vs parent `+0.009`.

The pilot shows the code runs and the parent already has strong aligned-source advantage, especially under clue suppression. It does not yet show a dense-specific increase; a larger GPU run after the pending jobs should compare parent, ordinary WWM, sparse, dense seed62064, and dense seed62065 using the same sampled targets.

## Instance-diversity evidence
The tight-private 480-map pilot reached heldout raw `0.5883`, pairs `39.85%`, quads `26.96%`, with eval-unseen `54.7%`, while a 116-map all-parameter run was much weaker (`0.315/7.04%/0.75%`). This supports instance diversity as a load-bearing variable for relational-operator acquisition, distinct from capacity alone. Isolated-all and half-coherent/half-isolated formation arms were in progress. Comparing the instance-diversity result against dense-focus Entity depth and source-alignment interactions remained a proposed analysis.

## Planned completed-endpoint analysis
The planned official full-transition comparison, conditional on completed seed62064 evaluation, included SuperGLUE:

```bash
python experiments/archive/functional_learning/scripts/official_transition_compare.py \
  --a-label coherent86_official_ref \
  --a-payload experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json experiments/archive/frontier_consolidation/data/private_scale_superglue_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75_sg_retry.json \
  --b-label dense_focus_seed62064_u0080_official \
  --b-payload experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json \
  --out-root experiments/archive/functional_learning/data/dense_official_full_transition \
  --tag dense_vs_coherent86_official_full \
  --include-superglue --max-examples 12
```

The completed-endpoint analysis uses `official_score_readout.py`. The planned replication assessment uses the `eval_summary.json`, train summary, common-target summary, qwen view-surface summary, and fast payloads; replication was the criterion for official-compatible seed62065 evaluation and item/score concordance. Failure to replicate would require item-level differences to separate seed movement from the dense effective-input mechanism.
