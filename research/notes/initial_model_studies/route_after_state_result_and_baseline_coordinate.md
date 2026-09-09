# route after state result and baseline coordinate — After the procedural state result: what must be separated next

## Direct reading of state coherent vs corrupted 1m profile

The state-story experiment was a clean 90/10 comparison:

- 900k official words + 100k hand-coded state stories;
- exact 1,000,000-word exposure;
- same 8x256 BERT-WWM base, baseline16k tokenizer, fixed 256 length;
- 98 updates with batch 83;
- coherent and corrupted arms matched on official rows, word counts, state target values, target mask positions, kept-token totals, and story supervision.

The 1M official-compatible profile is not a useful Entity result:

| mean coherent - corrupted | value |
|---|---:|
| BLiMP | +0.01 |
| Supplement | +0.80 |
| EWoK | +0.23 |
| Entity | 0.00 |
| COMPS | -0.03 |
| Reading eye | 0.00 |
| Reading SPR | 0.00 |

Per seed, Entity was exactly unchanged: 18.31 vs 18.31 for seed42 and 18.28 vs 18.28 for seed43. The coherent/corrupted training summaries are nearly identical, including final losses and selected token budgets. The state target itself occupied only about 2.7k cross-view tokens out of about 197k selected tokens, while the official WWM part dominated the loss.

The result does not support scaling the current simple procedural story design.

## Revised scientific interpretation

The important next question is not “make another synthetic story variant.” We must first separate two possibilities:

1. The model did not learn the procedural state task at all under the current objective/update budget.
2. The model learned the procedural state task but it did not transfer to official Entity Tracking.

These two outcomes imply different routes. If the state task was not learned, the immediate issue is target budget, update geometry, or representation. If it was learned but did not transfer, the issue is mismatch between the synthetic substrate and the official Entity/EWoK capability.

## Probe constructed for the next execution step

Script written:

- `experiments/archive/initial_model_studies/scripts/probe_state_learning.py`

It scores masked final-state words directly using the trained models. For each coherent/corrupted story pair, it compares the model score for:

- the target value supported by the coherent story;
- the alternative value implied by the corrupted story.

It reports, for both the training-template distribution and held-out disjoint-vocabulary probe:

- accuracy in coherent contexts;
- accuracy in corrupted contexts;
- both-context accuracy;
- mean coherent margin;
- mean corrupted margin;
- mean event sensitivity.

Default models include:

- official WWM seed42: `training/runs/babylm_masked_wwm_pos512_1M/hf_model/chck_1M`
- official WWM seed43: `training/runs/babylm_masked_wwm_pos512_seed43_1M/hf_model/chck_1M`
- state coherent/corrupted seed42 and seed43 from state coherent vs corrupted 1m profile.

A meaningful pattern would be:

- **not learned:** state-coherent and state-corrupted models are near official WWM on train-distribution and held-out probes;
- **learned but no transfer:** state-coherent model strongly beats official WWM and corrupted model on the train-distribution probe, possibly also held-out, while official Entity remains unchanged;
- **corrupted learns the wrong mapping:** corrupted model prefers the alternative value in corrupted contexts more than coherent model, revealing that the objective taught the surface target slot rather than state consistency.

The probe is not a new training route; it is needed to interpret state coherent vs corrupted 1m profile.

## Nine-column coordinate is now necessary

Route choices are still being made from fast/local fragments. The real BabyLM Strict-Small target is the 2026 Overall across nine columns:

1. BLiMP
2. BLiMP Supplement
3. EWoK
4. Entity Tracking
5. COMPS
6. (Super)GLUE fine-tuning
7. GlobalPIQA
8. Reading
9. AoA

The protected WWM base has only partial profile coverage. Before selecting another large mechanism route, the research needs a nine-column coordinate for the protected WWM base, at least for seed42 and preferably both protected seeds. The evaluation repository confirms:

- full/fast zero-shot scripts: `strict/scripts/eval_zero_shot.sh`, `eval_zero_shot_fast.sh`;
- AoA script: `strict/scripts/eval_aoa.sh` using `evaluation_data/full_eval/aoa/cdi_childes.json`;
- GlobalPIQA script: `strict/scripts/eval_zero_shot_global_piqa.sh`;
- fine-tuning script: `strict/scripts/eval_finetuning.sh` over `evaluation_data/full_eval/glue_filtered/*`;
- submission requires final full evaluation plus fast checkpoint evaluations.

This baseline coordinate is not final submission work. It is route evidence: it tells where Overall can actually move. For example, if WWM already has high Reading but low SuperGLUE/GlobalPIQA, the next route should not be another Entity-only story mechanism. If AoA is zero or unstable, an entirely different learning trajectory may matter.

## Immediate next work

1. Run `probe_state_learning.py` and inspect `data/state_learning_probe.json`.
2. In parallel or next, build a bounded official-evaluation runner for the protected WWM seed42 model first, covering the missing columns: full zero-shot where feasible, GlobalPIQA, AoA, and (Super)GLUE fine-tuning. Save a `wwm_nine_column_coordinate.json` or equivalent with raw subtask files and aggregation.
3. Only after these two evidence objects exist should the route be reopened. Do not create another similar synthetic story variant before knowing whether the current state task was learned. Do not scale WikiAuto or state stories to 10M/100M from the current evidence.
