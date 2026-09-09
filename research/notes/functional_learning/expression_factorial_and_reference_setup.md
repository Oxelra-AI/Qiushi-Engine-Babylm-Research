# expression factorial and reference setup expression factorial and matched-reference setup

## Why this step matters

The main bridge runs are still pending, so this step did not reinterpret their scientific result. It strengthened two pieces needed to read those results once they arrive:

1. The transfer boundary and accounting expression-transfer pilot was reanalyzed as a small update/query factorial rather than as one pooled non-original robustness score.
2. The matched no-substitution ordinary-tail reference was turned into an executable wait-then-train task, so the corrected bridge can be compared against a same-trainer ordinary continuation rather than only against historical clean d component ablation.

## Factorial structure in the expression-transfer pilot

The transfer boundary and accounting probe has four factor cells on the same literal-value state maps:

- `orig`: original update wording and original query frame.
- `query_only`: query frame changed while update wording stays original (`query_short`, `query_sentence`).
- `update_only`: update wording changed while query frame stays original (`update_record`).
- `both_changed`: both update and query wording changed (`both_record_short`, `both_correction_sentence`).

I wrote `experiments/archive/functional_learning/scripts/expression_factorial_analyzer.py` and ran it on the six-pair pilot records. Output:

- `experiments/archive/functional_learning/data/expression_factorial_pilot6/factorial_analysis.json`
- figure: `experiments/archive/functional_learning/figures/expression_factorial_pilot6.png`

The analysis averages concrete variants within a factor cell for per-pair effects, because `query_only` and `both_changed` each have two phrasings per pair. This is a pilot with only 6 held maps (4 birthplace, 2 death_place), so the purpose is to preserve the mechanism clue and analysis contract, not to settle expression robustness.

## Pilot result: retrieval and update use dissociate

For the answer-only specialist seed40040, pair-level cell means were:

| factor cell | neutral source | distractor retains source | self-update accepts new | complete recipient contrast |
|---|---:|---:|---:|---:|
| original | 1.000 | 0.833 | 1.000 | 0.833 |
| query changed | 0.833 | 0.417 | 0.833 | 0.250 |
| update changed | 1.000 | 0.833 | 0.667 | 0.500 |
| both changed | 0.833 | 0.417 | 0.750 | 0.333 |

The parent stayed at zero complete recipient contrasts in every cell, despite nearly always accepting a self-update; it lacks reliable source selection and remains dominated by the shared replacement in distractor contexts.

The specialist pattern is not simple uniform weakening:

- Query-frame change mainly damages the distractor-retention side. In the pilot, neutral source retrieval remains mostly available under changed query wording (0.833), but retention falls from 0.833 to 0.417 and complete contrast from 0.833 to 0.250. This matches the concrete failures where the correct source still beats the wrong source, but the shared replacement wins in the distractor-update context.
- Update wording change leaves neutral and distractor retention near the original level in this pilot, but damages legitimate revision: self-update falls from 1.000 to 0.667 and complete contrast to 0.500. This corresponds to the opposite failure mode: the model retains or favors the source even when the queried entity was the recipient of the update.
- Changing both factors is not simply additive in this tiny sample. Complete contrast is 0.333, with both replacement-overriding-source failures and source-inertia-on-update failures present.

The per-pair contrast estimates in `factorial_analysis.json` quantify this:

- complete operation: query effect at original update `-0.583`, update effect at original query `-0.333`, extra interaction when both change `+0.417`;
- retain source: query effect `-0.417`, update effect `0.000`;
- self-update new: query effect `-0.167`, update effect `-0.333`;
- neutral source: query effect `-0.167`, update effect `0.000`.

The strongest scientific reading is therefore that the acquired operation contains a usable source-selection computation under many expression changes, but its use is coupled to specific query and update realizations. Strengthening only the source preference is insufficient: it can improve neutral retrieval while still failing complete state revision, and it may even resist a legitimate update. Conversely, making update acceptance stronger without preserving source selection can collapse into the parent-like replacement preference.

## How this shapes the next intervention without changing the pending bridge

This does not justify altering the already running bridge. The bridge tests whether answer-span allocation inside legal ordinary continuation can acquire and preserve the original relation operation under controlled word/update accounting. The expression factorial instead supplies a follow-up hypothesis if the full 30-pair screen confirms the pilot pattern:

> Data-efficient operation learning may require connecting the same acquired state-selection computation to independently varied query and update expressions. Repeating more state maps or simply increasing source preference may not improve the complete operation, because portability can fail through replacement override in distractor contexts or through source inertia when a revision should be accepted.

A future training test should keep the train/test separation at the expression-combination level:

- use training maps for expression-linking rows, not held maps;
- hold out some expression combinations and/or variant families so success is not evaluation familiarity;
- measure four separable quantities: neutral source retrieval, distractor-update retention against both wrong source and replacement, self-update acceptance against both source and wrong source, and complete recipient-only contrast;
- compare query-diversified, update-diversified, and crossed expression-linking schedules under the same row/word budget before adding any such policy to BabyLM continuation.

If a bridge checkpoint acquires the original operation but has weak expression robustness, the expression factorial and reference setup analyzer can be run on bridge expression records to identify whether the bottleneck is query realization, update interpretation, or their interaction. This would guide the next legal overlay design more sharply than blindly adding more relation repeats.

## Matched no-substitution reference setup

The no-substitution reference was prepared but had not yet begun training. It used the corrected bridge trainer on:

- `experiments/archive/functional_learning/data/reference_tail/reference_tail_ordinary_wordpaced.jsonl`

with output:

- `experiments/archive/functional_learning/data/reference_tail_corrected_ordinary`

The command uses the same corrected trainer, `ordinary_wwm` arm, 354 word-paced macro-updates, private-adapter-only training, and seed `49049`. Because all rows in this stream are ordinary tail rows, the relation component has lambda 0 and the result is a no-substitution same-trainer ordinary reference. This task deliberately waits for an idle GPU to avoid contending with the two corrected bridge arms. Its result is needed because historical clean d component ablation is useful but not matched to the repaired earlier analysis training objective and masking implementation.

## Current unresolved work

Pending runtime tasks remain:

The following comparisons were incomplete at the time of this note:
- Corrected bridge ordinary-WWM relation-substituted arm.
- Corrected bridge answer-allocation arm.
- Full 30-pair operation-preserving expression-transfer screen.
- Shortcut-prone natural-row descriptive screen.
- No-substitution same-trainer ordinary-tail reference; training had not yet started.

No BabyLM improvement, general data-efficient learning principle, or new SOTA result was established in this step. The research state is stronger because the expression-transfer measurement now separates the failure modes that matter for a general learning principle, and because the matched ordinary reference is executable without resource contention.
