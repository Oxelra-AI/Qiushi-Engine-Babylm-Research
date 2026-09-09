# fw absolute progress decision discipline — FW compact-view vs whole-sentence source-breadth result

## What was tested

earlier analysis trained two matched 100M-word legal arms from repaired FW mechanism-scale construction:

- `compact_view`: expanded preserved FineWeb companion budget spent on faithful compact restatements.
- `source_breadth`: same companion budget spent on coherent whole independent FineWeb sentences.

Both arms were matched in row sequence, pass order, tokenizer, architecture, seed, optimizer recipe, word exposure, and checkpoint ladder. The only intended difference is the 318,851 FineWeb companion words. Both trainings completed to exactly 100,000,000 words with 100 checkpoints.

Actual saved tokenizer SHA in both arms: `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`, trained on 9,681,149 words, so the arms are legal-tokenizer and matched-tokenizer despite the cosmetic `tokenizer_label=baseline16k` string in trainer metrics.

## Cheap official-compatible results

Reference legal compact-view-reinvest trajectory:

| Checkpoint | existing legal ref cheap7 |
|---|---:|
| 70M | 42.6086 |
| 80M | 42.9486 |
| 100M | 43.0057 |

Results:

| Checkpoint | Arm | cheap7 | Δ vs legal ref | Δ compact-breadth |
|---|---|---:|---:|---:|
| 70M | compact_view | 42.6571 | +0.0485 | +0.6257 |
| 70M | source_breadth | 42.0314 | -0.5772 | — |
| 80M | compact_view | 42.9864 | +0.0378 | +0.0207 |
| 80M | source_breadth | 42.9657 | +0.0171 | — |
| 100M | compact_view | 43.1814 | +0.1757 | +0.5443 |
| 100M | source_breadth | 42.6371 | -0.3686 | — |

Full cheap-column deltas versus the existing legal reference are recorded in `data/fw_absolute_decision/fw_absolute_decision.{json,md}`.

## Interpretation

The within-FW-family result supports compact restatement over independent source breadth at 100M (+0.5443 cheap7) and at 70M (+0.6257), though the two arms are nearly tied at 80M (+0.0207). Source breadth has a severe persistent Entity deficit and does not become the endpoint winner.

The SOTA-progress result is negative: the best arm, compact_view, is only +0.1757 cheap7 over the existing legal endpoint at 100M, with positive movement in only 3/7 cheap columns. The cheap-column pattern is not broad: BLiMP +1.00, Entity +0.96, GlobalPIQA +2.57, but Supplement -2.31, EWoK -0.14, COMPS -0.17, Reading -0.683. If SuperGLUE/AoA were unchanged, this would raise Overall by only about `(0.1757*7)/9 = 0.1367`, projecting roughly 41.3945 rather than 41.8.

Therefore this trained FW family does not provide the SOTA-scale substrate sought by the active goal. The source-repeat attribution arm should not be launched merely to explain compact-vs-breadth because it would spend 100M training on a family that has not shown enough absolute progress over the existing legal trajectory.

## Evidence files

- Evaluation summary: `data/fw_comparison_eval/fw_comparison_eval_summary.{json,md}`
- Absolute-progress decision: `data/fw_absolute_decision/fw_absolute_decision.{json,md}`
- Training verification: `notes/fw_training_complete_verification.md`
- Decision discipline note: `notes/fw_absolute_progress_decision_discipline.md`
- Runs:
  - `training/runs/fw_compact_view_shared16k_seed43022/`
  - `training/runs/fw_source_breadth_shared16k_seed43022/`

## Next research implication

Preserve compact semantic second views as the protected scientific core, but the simple expanded FineWeb mechanism-scale route is not enough under the legal representation coordinate. The next route should not be source-repeat, FW breadth attribution, generic LR/mask sweeps, or final expression. It should reopen around a distinct corpus-general learning/representation mechanism that can produce broad gains in Supplement/EWoK/Reading/SuperGLUE while not sacrificing the compact-view reinvestment core.
