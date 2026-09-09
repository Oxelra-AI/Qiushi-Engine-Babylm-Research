# fast eval results carrier schema — official `fast_eval_results` carrier schema for Strict-Small

Official source read:
- README/guidelines lines 166–173: a Strict/Strict-Small submission uploads the predictions file produced by `collate_preds`; adding `--fast` includes checkpoint evaluation results and is required for full BabyLM Challenge submission.
- Current leaderboard validation source: `experiments/archive/frontier_consolidation/data/live_leaderboard_scoring_probe/source/src/submission/check_validity.py` lines 292–328.
- Task/key constants: `.../source/src/display/utils.py` lines 193–231 and 402–405.

Schema interpretation from validation code:
- Top-level prediction file key: `fast_eval_results`.
- If absent or `None`, the validator returns upload success, but comments state missing fast results are scored as zero. This is accepted upload format but not the full challenge material expected by the README.
- If present, `fast_eval_results` is treated as a dictionary keyed by the fast task names in `FAST_TASKS`:
  - `reading`: subtasks `spr`, `rt`
  - `entity_tracking_filtered`: `regular_0_ops` ... `regular_5_ops`, `ambiref_0_ops` ... `ambiref_5_ops`, `move_contents_0_ops` ... `move_contents_5_ops`
  - `global_piqa_parallel`: `global_piqa_parallel`
  - `global_piqa_nonparallel`: `global_piqa_nonparallel`
  - `ewok`: 11 domain groups
  - `blimp_supplement`: 5 groups
  - `blimp`: 67 BLiMP UIDs
- Each `fast_eval_results[task]` maps to a list of checkpoint-result dictionaries.
- For Strict-Small, the expected checkpoint order is 19 word counts: `1M..9M,10M,20M,...,100M`.
- Missing fast tasks and missing checkpoint entries are accepted, but a present checkpoint dict must contain every subtask key for that task.

Immediate consequence for `chck_82M`:
- The current score-bearing collated file already has complete full predictions and AoA ladder, and the score is reproduced.
- The remaining carrier work is to either rerun official collation with `--fast`, or construct the exact `fast_eval_results` block from existing fast checkpoint outputs if those outputs exist for the 19 checkpoints.
- This should be done as an artifact-preparation/validation task, not as new model research or endpoint changing.
