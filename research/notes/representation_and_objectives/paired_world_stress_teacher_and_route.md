# paired world stress teacher and route — Paired-world stress tests, teacher verification, and route update

## Scientific Motivation

paired world pilot result built a source-attested paired-world role-flip pilot from tennis, badminton, and LaLiga records. Its grouped bag-of-words classifier was near chance, but this does not exclude sequence-aware shortcut routes: finite template identity, token order, participant position, and context--hypothesis alignment can solve a templated corpus without yielding a general data-efficient learning principle.

This analysis treated teacher inference only as language-realization verification, and ran stronger shortcut and perturbation tests before any student or BabyLM training.

## Inputs

- paired world pilot result families: `data/paired_world_pilot/families_train.jsonl` and `families_held.jsonl`.
- paired world pilot result summary: `data/paired_world_pilot/pilot_summary.json`.
- Cross-architecture compact-effect analysis: effects are near-random at item level; future paired-world probes should compare compact/repeat DeBERTa/RoBERTa only after a sound substrate exists.

## 1. Sequence-aware shortcut stress

Script: `scripts/sequence_shortcut_stress_and_teacher_verify.py`.

Outputs:
- `data/paired_world_sequence_teacher/sequence_shortcut_stress_summary.json`
- `data/paired_world_sequence_teacher/sequence_shortcut_stress_summary.md`
- original examples: `data/paired_world_sequence_teacher/sequence_examples_original_ablated.jsonl`
- perturbation examples: `data/paired_world_sequence_teacher/sequence_examples_perturbations.jsonl`

The linear sequence baselines did not find a shortcut:

| view | group-family CV | held original | held held-template | lost-to perturbation | retemplate opposite order |
|---|---:|---:|---:|---:|---:|
| raw word 1-2 | 0.500 +/- 0.004 | 0.518 | 0.520 | 0.497 | 0.500 |
| raw char 3-5 | 0.505 +/- 0.007 | 0.507 | 0.515 | 0.520 | 0.507 |
| canonical word 1-2 | 0.504 +/- 0.008 | 0.510 | 0.515 | 0.515 | 0.505 |
| canonical char 3-5 | 0.523 +/- 0.004 | 0.515 | 0.510 | 0.502 | 0.522 |
| symbolic order text | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |

However, a transparent metadata parser that is told whether the template is winner-first or loser-first and reads the first participant solves the covered 67.8% of rows perfectly. This is not a learned shortcut result, but it exposes the actual structure of the task: the present corpus is still a controlled role-parsing substrate whose surface grammar is finite and explicit.

## 2. Teacher realization verification

Purpose: verify that Qwen3.5-9B and Llama3.1-8B-Instruct can read the generated contexts and hypotheses. This is not evidence of student learning.

Prompt set:
- 200 families: 100 train + 100 held; 139 tennis, 42 badminton, 19 football.
- 3,200 prompts per teacher: original score-ablated, original score-visible, name-swapped context, and lost-to hypothesis variants.
- Commands used both H100s in parallel with max 8 generated tokens.
  - Qwen: 3,200 prompts, 143.858 s, output `training/data/paired_world_teacher_verify/teacher_outputs_qwen.jsonl`.
  - Llama: 3,200 prompts, 102.536 s, output `training/data/paired_world_teacher_verify/teacher_outputs_llama.jsonl`.

Analysis outputs:
- `data/paired_world_sequence_teacher/teacher_realization_summary.json`
- `data/paired_world_sequence_teacher/teacher_realization_summary.md`
- `data/paired_world_sequence_teacher/teacher_labeled_rows.jsonl`
- `data/paired_world_sequence_teacher/teacher_cross_model_pairs.jsonl`
- `data/paired_world_sequence_teacher/teacher_failure_postanalysis.json`
- `data/paired_world_sequence_teacher/teacher_failure_postanalysis.md`

Teacher result did **not** pass the intended stability level:

| measure | value |
|---|---:|
| Qwen accuracy | 0.944 |
| Llama accuracy | 0.935 |
| cross-teacher agreement / both-correct | 0.879 |
| original score-ablated both-correct | 0.900 |
| held-template both-correct | 0.734 |
| train-template both-correct | 0.929 |
| lost-to hypothesis both-correct | 0.830 |

Failure postanalysis:
- There were 2,813 both-correct pairs, 209 Qwen-correct/Llama-wrong, and 178 Llama-correct/Qwen-wrong.
- There were **no shared same-wrong cases** in the postanalysis state counts, so the deterministic labels are not obviously corrupted by a common teacher reading.
- Failures concentrate in wording/predicate/template brittleness:
  - template 20, `went down to`: 0.344 both-correct;
  - template 10, `was unable to overcome`: 0.600;
  - template 17, `succumbed to`: 0.716;
  - loser-first templates overall: 0.755;
  - mixed templates: 0.963;
  - winner-first templates: 0.934.
- Score-visible did not act as a clean ceiling (both-correct 0.879), likely because some sports score strings are source-format dependent and because teachers sometimes overread/underread the relational sentence despite scores.

Scientific meaning: current realizations are often semantically readable, but not yet reliable enough as a teacher-stable substrate. Any distillation or BabyLM-scale training from this exact template set is blocked.

## 3. Tiny learned sequence-parser shortcut probe

Script: `training/scripts/tiny_gru_sequence_baseline.py`.

Execution limitations:
- The first attempt was rejected because the script contains `backward` and updates a learned object, so it is a training probe rather than evaluation-only inference.
- A subsequent attempt failed because the declared output directory already existed. Neither failed attempt produced a GRU result.
- Direct CPU execution of the same small probe produced the results under `training/runs/tiny_gru_sequence_baseline_direct`.

Output:
- `training/runs/tiny_gru_sequence_baseline_direct/tiny_gru_sequence_baseline_summary.json`
- `training/runs/tiny_gru_sequence_baseline_direct/tiny_gru_sequence_baseline_summary.md`

Key result: with participant names canonicalized to `ENTITY_A/ENTITY_B`, a tiny GRU learns the finite original-template role parser extremely well:

| training regime | mode | held original | held held-template | name-swap context | lost-to hypothesis | passive hypothesis | retemplate opposite order |
|---|---|---:|---:|---:|---:|---:|---:|
| original only | canonical | 0.965 | 0.931 | 0.935 | 0.045 | 0.035 | 0.990 |
| original only | raw names | 0.547 | 0.559 | 0.472 | 0.487 | 0.480 | 0.500 |
| augmented variants | canonical | 0.955 | 0.912 | 0.960 | 0.955 | 0.955 | 1.000 |
| augmented variants | raw names | 0.502 | 0.505 | 0.500 | 0.500 | 0.502 | 0.500 |

Interpretation:
- The current sports outcome corpus is learnable by a small finite sequence parser once entity identities are abstracted.
- It generalizes to held families and many held templates, so a high student score on the current direct-outcome NLI rows would not by itself show a general data-efficient learning principle.
- It catastrophically fails cross-predicate transfer to `lost to` and passive hypotheses when trained only on `defeated`, showing that predicate-level transfer is not automatic.
- When trained on those variants, the same GRU learns them, which confirms the task is a learnable template grammar rather than a broad semantic update mechanism.

This sequence-parser result changes the route: the present paired-world sports outcome set should be used as a probe and source-construction scaffold, not as training evidence.

## 4. Distinct event-to-state construction pilot

Script: `scripts/event_to_state_final_pilot.py`.

Purpose: construct a stronger source-derived operation: final-match outcome assigns entity state (`champion` vs `runner-up`) rather than only the binary relation `defeated(A,B)`.

Outputs:
- `data/event_to_state_final_pilot/event_to_state_summary.json`
- `data/event_to_state_final_pilot/event_to_state_summary.md`
- `data/event_to_state_final_pilot/state_families_train.jsonl`
- `data/event_to_state_final_pilot/state_families_held.jsonl`

Result:
- tennis final events: 46,972; reversed final pairs: 1,472.
- BWF final events: 2,889; reversed final pairs: 178.
- built 300 paired-world state families: 220 tennis, 80 badminton.
- 0 label-consistency errors.
- 2,400 state NLI rows + 600 invariant rows.
- sequence probes near chance: raw word 0.498, raw char 0.493, canonical word 0.500, canonical char 0.532.

Boundary: the champion/runner-up state is derived from the final-match schema rather than independently present as a separate official title field. This is still sports-derived, but it is a better next substrate because the model must compose event type (`final`), event outcome, and queried entity state. It needs teacher verification and transfer design before training.

## Current scientific judgment

The paired-world route remains scientifically meaningful because it directly attacks the rawtoken bridge screen and route judgment bottleneck: latent occurrence-role assignment from raw text. It also now has a distinct event-to-state extension. But the current direct sports outcome corpus is **not** sufficient evidence for the desired general data-efficient learning principle.

What was strengthened:
- Score-ablated paired worlds can be source-attested at scale.
- Linear bag-of-words and tf-idf sequence baselines are near chance on the original rows.
- Deterministic labels survive adversarial name swaps by construction.
- Event-to-state final-match families exist at useful scale.

What was weakened or blocked:
- The teacher-stable realization check failed the intended >=0.95 stability target, especially on held templates and loser-first expressions.
- A tiny canonicalized GRU solves the original direct-outcome task at 0.965 on held families and 0.931 on held-template rows, so a student score on this task can be a finite template parser.
- Cross-predicate transfer is absent in the original-only GRU: `lost_to` 0.045 and passive 0.035.
- Therefore no TinyMLM, DeBERTa, RoBERTa, compact/repeat comparison, distillation, BabyLM training, official evaluation, upload, or submission should be launched from the current paired world pilot result direct-outcome substrate.

## Next research work

The next strong route is not more training on the current outcome templates. It is to build a substrate whose held evaluation requires transfer across predicates and event-to-state updates:

1. Repair realization language before any teacher-stable use:
   - remove or rewrite ambiguous/brittle templates (`went down to`, `was unable to overcome`, `succumbed to`);
   - avoid score-visible supervision unless score perspective is normalized;
   - retain templates with high teacher stability (`defeated`, `was defeated by`, `was beaten by`, `proved too strong`, `triumph over`, `victorious over`, `prevailed against`) but do not make them the whole task.

2. Build a cross-predicate transfer protocol:
   - train/probe on one predicate family and test on held predicate families (`defeated`, `beat`, `lost_to`, `was_defeated_by`, `champion`, `runner_up`);
   - keep participant identities held out and canonicalized controls explicit;
   - require transfer to event-to-state queries, not just relation paraphrases.

3. Teacher-check the event-to-state final pilot separately, with simple unambiguous contexts and hypotheses.

4. Search for or construct non-sports source-attested event-to-state/temporal families. The finals pilot is a bridge, not enough by itself for a general principle.

5. Only after the substrate passes realization stability and cross-predicate/event-state shortcut tests should a small student or architecture comparison be run. Then compare compact-trained vs repeat DeBERTa and RoBERTa on the same role-assignment items to test whether apparent treatment effects are stochastic/metric-specific.
