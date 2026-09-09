# Corrected Strict-Small-tokenizer experiment summary

## Current active experiment

The decisive experiments are still the earlier analysis corrected-tokenizer retrains:

- seed43022, run dir `experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022`
- seed43122, run dir `experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122`

At the time of this summary, both runs had reached `chck_80M` and about 82.13M cumulative word exposure:

- seed43022: earlier analysis, 82,130,718 words, loss 2.399269
- seed43122: earlier analysis, 82,130,718 words, loss 2.398272

These progress measurements do not establish endpoint competence; completed evaluation is required. The only submission-relevant endpoint is the reinvest run whose tokenizer and pretraining pool are the same 10M corpus. Any clean-Qwen run with this reinvest tokenizer is a fixed-tokenizer scientific control, not a separate valid Strict-Small artifact, because tokenizer-fitting text plus clean pretraining text exceeds 10M.

## Evidence and analysis assets

### 1. MLM loss cannot select the winning seed

Script: `experiments/archive/representation_and_objectives/scripts/loss_trajectory_vs_overall.py`
Output: `experiments/archive/representation_and_objectives/data/loss_trajectory_vs_overall/loss_trajectory_vs_overall.json`

Old inherited-tokenizer reinvest seed43022 beat seed43122 by +0.784895 Overall, but had slightly higher all-step mean loss (+0.003586) and last-50-step mean loss (+0.000543). Loss windows were tiny/sign-flipping. Training loss is only a health signal, not a seed-selection signal.

### 2. Official-compatible focused EWoK margin dynamics

Scripts:

- `experiments/archive/representation_and_objectives/scripts/strictsmalltok_midtrain_ewok_margin_probe.py`
- `experiments/archive/representation_and_objectives/scripts/oldtok_checkpoint_ewok_phase_probe.py`
- `experiments/archive/representation_and_objectives/scripts/summarize_relation_phase_dynamics.py`

Compact summary:

- `experiments/archive/representation_and_objectives/data/relation_phase_dynamics_summary/relation_phase_dynamics_summary.json`
- Note: `research/notes/representation_and_objectives/loss_and_relation_phase_dynamics.md`

Focused 553-row old-instability EWoK subset:

| coordinate/checkpoint | seed43022 | seed43122 | seed43022 advantage |
| --- | ---: | ---: | ---: |
| oldtok 50M | 50.0904 | 43.9421 | +6.1483 |
| oldtok 80M | 60.7595 | 34.3580 | +26.4014 |
| oldtok 100M endpoint | 66.1844 | 30.5606 | +35.6239 |
| strictsmalltok 50M | 40.1447 | 56.6004 | -16.4557 |
| strictsmalltok 70M | 51.5371 | 45.3888 | +6.1483 |
| strictsmalltok 80M | 51.8987 | 46.4738 | +5.4250 |

Old-tokenizer relation polarization formed late: old 80M seed-delta correlates 0.938 with old 100M endpoint seed-delta, but old 50M only 0.427. Corrected-tokenizer 80M seed-delta has near-zero/slightly negative correlation with old endpoint seed-delta (-0.0837). Therefore old seed behavior and old relation phenotype cannot be imported as predictions for the compliant coordinate. Final judgment must come from corrected 100M official evaluation.

### 3. Projection and surface-first evaluation infrastructure

Projection helper:

- `experiments/archive/representation_and_objectives/scripts/projection_after_partial_surface.py`
- output dir `experiments/archive/representation_and_objectives/data/projection_after_partial_surface`

Validated on old seed43022 and seed43122 pristine summaries after patching both score-summary shapes. Arithmetic: after seven non-SuperGLUE/non-AoA columns, required `SuperGLUE+AoA` to hit 41.8 is `9*41.8 - sum(seven known columns)`. With AoA=0, this is the required SuperGLUE value. Old seed43022 needed only 68.94 and cleared; old seed43122 would need 74.87 and did not.

Surface-first controller:

- `experiments/archive/representation_and_objectives/scripts/strictsmalltok_surface_first_eval_controller.py`

Dry-runs passed structurally and currently only complain about not-yet-finished checkpoints. Once a retrain completes, this controller can run cheaper surface columns first:

```bash
python3 -B experiments/archive/representation_and_objectives/scripts/strictsmalltok_surface_first_eval_controller.py --seed 43022 --gpu 0
python3 -B experiments/archive/representation_and_objectives/scripts/strictsmalltok_surface_first_eval_controller.py --seed 43122 --gpu 1
```

It evaluates BLiMP, Supplement, Entity, COMPS, GlobalPIQA, Reading, plus repaired official EWoK, then writes a projection without running SuperGLUE or AoA. Use it when deciding whether to spend SuperGLUE/AoA. The full controller remains:

```bash
python3 -B experiments/archive/representation_and_objectives/scripts/strictsmalltok_posttrain_eval_controller.py --seed 43022 --gpu 0
python3 -B experiments/archive/representation_and_objectives/scripts/strictsmalltok_posttrain_eval_controller.py --seed 43122 --gpu 1
```

Run the full controller for any seed whose cheap surface leaves a plausible path to exceed 41.8, and for any seed selected for an authoritative final record. AoA must be measured with min_context=0 before any final scalar.

## Related Experimental Evidence

Related experimental evidence:

- companion analysis independently agrees old 42.0331 is non-submission evidence because of the out-of-budget inherited tokenizer.
- union audit: reinvest tokenizer+pretraining union is exactly 10M; clean-Qwen with reinvest tokenizer has designed union 10,423,520 words, so clean is only a fixed-tokenizer control.
- old-tokenizer raw90 checkpoint route did not justify extra SuperGLUE/AoA escalation.



## Next action

After both strict-tokenizer seed training runs complete:

1. Collect authoritative task results.
2. Inspect training completion, `scientific_metrics.json`, `training_log.jsonl`, complete checkpoint ladders, `example_order_manifest.json`, and `hf_model/chck_100M/tokenizer.json` SHA (`4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738`).
3. Start with surface-first official-coordinate evaluation if GPU time is tight; otherwise run the full strictsmall tokenizer visibility audit controller for the seed(s).
4. Use the projection helper after cheap surface outputs to decide whether SuperGLUE+AoA can still plausibly restore/confirm a >41.8 endpoint.
5. If a corrected-tokenizer seed clears 41.8, protect it and then run the missing complementary official checks needed for submission. If both fail, reassess the mechanism before any new 100M training route.
6. Collect `s49_t23_tool1` when delivered; it is the full old four-cell EWoK margin atlas and should be used only as mechanism baseline, not as endpoint evidence.
