# stabilization precheck while grids pending average-candidate selected-eval dry plan

No evaluation, upload, or submission was run. This only prepares a run-dir view for a possible later one-endpoint selected cheap-task score.

- Candidate: `experiments/archive/frontier_consolidation/data/late_weight_average_scaffold/candidates/reference_scale1p75_seed43022__center_80_82_84_uniform`
- Endpoint link path: `experiments/archive/frontier_consolidation/data/average_candidate_selected_eval_plan/run_view/hf_model/chck_84M` (symlink)
- Endpoint target: `experiments/archive/frontier_consolidation/data/late_weight_average_scaffold/candidates/reference_scale1p75_seed43022__center_80_82_84_uniform`
- Dry command: `PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/data/average_candidate_selected_eval_plan/run_view --gpu 0 --out-dir experiments/archive/frontier_consolidation/data/avg80_82_84_selected_eval_if_authorized --label reference_scale1p75_seed43022_avg80_82_84_uniform --endpoints chck_84M --dry-run`
- Future command if authorized: `PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/data/average_candidate_selected_eval_plan/run_view --gpu 0 --out-dir experiments/archive/frontier_consolidation/data/avg80_82_84_selected_eval_if_authorized --label reference_scale1p75_seed43022_avg80_82_84_uniform --endpoints chck_84M`
- Scientific use: Run the future command only after delivered seed43122/evidence makes same-trajectory stabilization the right target. The average carries no new training exposure; the pseudo-endpoint is a scoring wrapper label, not a chronological checkpoint.

JSON: `experiments/archive/frontier_consolidation/data/average_candidate_selected_eval_plan/average_candidate_selected_eval_plan.json`
