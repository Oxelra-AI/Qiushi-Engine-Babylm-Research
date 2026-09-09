# roberta selected transfer resolution — RoBERTa selected-transfer resolution

## Why this evaluation was run

The RoBERTa compact-vs-repeat experiment had become the decision-bearing execution bottleneck. The managed compact order result and roberta transfer decision selected-evaluation supervisor had not delivered an integrated trajectory after long elapsed time, and its visible output tree contained only 10M/20M/30M summaries plus incomplete 40M folders. To resolve the compact-vs-repeat question without colliding with the managed output tree, roberta selected transfer resolution ran a separate-output selected cheap-column evaluation on already trained checkpoints. This did not train, upload, run SuperGLUE/AoA, or submit to the leaderboard.

The scientific question was narrow and necessary: does the positive compact-trained local source-absent relational/event pseudolikelihood channel in stock RoBERTa correspond to official-compatible BabyLM selected competence on stable families, or is it another local-learning/downstream split?

## Completed selected endpoints

### chck_100M

Output: `data/roberta_minimal_selected_eval/minimal_selected_eval_summary.json`

Compact scores: BLiMP 56.09, Supplement 51.33, EWoK 51.79, Entity 19.64, COMPS 51.73, GlobalPIQA 35.635, Reading 7.825, cheap7 39.148571.

Repeat scores: BLiMP 55.60, Supplement 53.54, EWoK 50.74, Entity 20.05, COMPS 51.62, GlobalPIQA 34.65, Reading 7.545, cheap7 39.106429.

Compact minus repeat:

- cheap7: +0.042143
- cheap6 without GlobalPIQA: -0.115000
- cheap5 without GlobalPIQA/Reading: -0.194000
- EWoK+Entity: +0.640000, but internally split as EWoK +1.050 and Entity -0.410
- Supplement: -2.210
- Entity: -0.410
- COMPS: +0.110
- BLiMP: +0.490
- GlobalPIQA: +0.985
- Reading: +0.280

The tiny positive cheap7 is not a broad stable-family transfer signal: it is offset by Supplement/Entity damage and relies on EWoK plus volatile columns.

### chck_60M

Output summaries:

- compact: `data/roberta_lateband_selected_eval/compact/chck_60M/roberta_compact_chck_60M_summary.json`
- repeat: `data/roberta_lateband_selected_eval/repeat/chck_60M/roberta_repeat_chck_60M_summary.json`

Compact scores: BLiMP 56.22, Supplement 51.87, EWoK 50.34, Entity 17.67, COMPS 51.43, GlobalPIQA 33.71, Reading 7.045, cheap7 38.326429.

Repeat scores: BLiMP 55.85, Supplement 54.28, EWoK 49.60, Entity 18.46, COMPS 50.99, GlobalPIQA 35.21, Reading 7.620, cheap7 38.858571.

Compact minus repeat:

- cheap7: -0.532143
- cheap6 without GlobalPIQA: -0.370833
- cheap5 without GlobalPIQA/Reading: -0.330000
- EWoK+Entity: -0.050000
- Supplement: -2.410
- Entity: -0.790
- COMPS: +0.440
- BLiMP: +0.370
- EWoK: +0.740
- GlobalPIQA: -1.500
- Reading: -0.575

This confirms the negative stable-family direction at a separated late checkpoint and makes the 100M result unlikely to be only endpoint overtraining.

A broader separate roberta selected transfer resolution late-band attempt for 60M/70M/80M/90M exceeded the 1800s tool limit after finishing 60M and starting 70M. Repeat 70M finished in that separate tree, but compact 70M did not produce a summary there, so that separate 70M pair is not used.

After this note was first written, the long-running managed compact order result and roberta transfer decision evaluator returned a failed terminal state rather than an authoritative integrated file, but its partial output tree contains complete paired summaries through chck_70M. These partial pairs strengthen the selected-trajectory reading:

- 10M: cheap7 +0.154, cheap6-no-GP +0.019, cheap5 -0.024, EWoK+Entity -1.32.
- 20M: cheap7 -0.059, cheap6-no-GP +0.272, cheap5 +0.300, EWoK+Entity +0.24.
- 30M: cheap7 +0.022, cheap6-no-GP +0.026, cheap5 +0.014, EWoK+Entity -0.02.
- 40M: cheap7 -0.118, cheap6-no-GP -0.304, cheap5 -0.372, Supplement -1.46, COMPS -1.08.
- 50M: cheap7 -0.049, cheap6-no-GP +0.193, cheap5 +0.178, but Supplement -1.03 and COMPS -0.32.
- 60M: cheap7 -0.532, cheap6-no-GP -0.371, cheap5 -0.330, EWoK+Entity -0.05.
- 70M: cheap7 -0.493, cheap6-no-GP -0.161, cheap5 -0.278, EWoK+Entity -0.24, Supplement -1.17, Entity -0.99.

Thus the available trajectory is not a late stable transfer signal: any early positives are small and inconsistent, and the mature 60M/70M/100M readouts are selected-stable negative or mixed with key stable damage.

## Item/subtask movement at 100M

Output: `data/roberta100_selected_prediction_movement/selected_prediction_movement.{json,md}` and `per_item_movement.csv`.

The reader covered 170,722 common selected items with zero loader warnings. It is file-only interpretation of already produced prediction payloads.

Important movement:

- Stable-five raw item pool: compact +0.168 item-accuracy points, but this raw item pool does not replace official column/subtask weighting; official stable composites remain negative on cheap6/cheap5 because Supplement and Entity are down.
- BLiMP: +0.49 official with 9,347 flips and only +261 net correct; large redistribution rather than uniform acquisition. Losses concentrate in superlative quantifiers, adjunct islands, only-NPI licensing, Principle A domains, NPI present, and agreement distractors; gains concentrate in other NPI scope/island/agreement templates.
- Supplement: official -2.21 even though raw item net is +9/5,218, showing subtask weighting matters; worst visible subtasks include QA congruence easy and turn-taking.
- Entity: -0.41 official and -11 raw net.
- COMPS: almost flat officially (+0.11) and raw (-13 net over 91,028) despite 26,457 flips.
- EWoK: +1.05 official, but paired with Entity and Supplement losses it does not establish broad transfer.

## Scientific reading

The RoBERTa training pair is mechanically matched and compact has much lower endpoint MLM loss than repeat. The local bridge is also real: compact-trained RoBERTa strongly improves compact-side source-absent relational/event pseudolikelihood. But official-compatible selected evidence now shows that this local semantic channel does not become stable BabyLM competence in the tested stock RoBERTa absolute-position MLM coordinate.

Therefore the RoBERTa compact-vs-repeat route should not receive an independent-seed robustness run. The correct result is a local-learning/downstream split: compact views can create a transferable local denoising signal for novel relational/event content, but that signal is not sufficient to improve the selected stable BabyLM surface in this coordinate.

This agrees with mechanism evidence synthesis's coupled-system synthesis rather than competing with it: the measured result is that source-absent words are useful context but not a special anchor-conditioning channel, and that the leading account shifts to input-side high-density faithful-rewrite distribution plus dense source-shared supervision. The RoBERTa result says the local compact semantic signal alone is not sufficient in a different encoder; the next work should decompose the natural compact data system, not pursue another RoBERTa seed or direct source-absent target pressure.

This bounds only the tested coordinate and seed: stock RoBERTa 8x480, same legal tokenizer, ordinary WWM p=0.15, compact-reinvested 100M stream versus first-N-repeat matched stream. It does not falsify the original DeBERTa compact-reinvestment result, and it does not identify a single architectural cause. It does remove this RoBERTa replication path as the next expensive work.

## Next scientific work

Return to mechanism formation around the natural compact-data factors that remain coupled in the validated DeBERTa result:

- content density and abstraction rather than raw source-absentness;
- source-wide/tail coverage;
- lexical recurrence across source positions;
- faithful compression and surface naturalness;
- reinvested source diversity under the fixed 10M-word budget.

The next H100 work should be a tightly matched data-factor experiment under ordinary WWM, not another RoBERTa seed and not explicit source-absent target weighting. The immediate design requirement is to choose a minimal legal pool contrast that separates one of these factors while preserving the compact-reinvested substrate and pretraining interface enough that selected stable-family movement is interpretable.
