# dense6x384 10m curve profile fixed — Profiling bug identified; 10M trajectory artifacts are ready

## What happened

The 10M dense6x384 curve (`babylm_compare_dense6x384_10M_curve`) trained and saved 10 clean checkpoints (`chck_1M`–`chck_10M`), each verified as loadable HF model directories. However, two profiling attempts (`profile_babylm_curve_revisions.py`) returned identical scores for `chck_1M`, `chck_5M`, and `chck_10M` — all 56.31 BLiMP, 55.20 Supplement, 51.73 EWoK, 17.88 Entity, 49.73 COMPS, 8.98/2.24 Reading.

The identical scores are **not** evidence that the model did not learn between 1M and 10M. Training loss fell from 9.76 to 4.30, and the model clearly learned more. The identical scores indicate a **profiling method failure**: the evaluator's `--model_path_or_name` argument, even when pointed at a subdirectory, may still resolve to the root model path due to how HF local checkpoints with `revision_name` are loaded. The fix attempted in dense6x384 10m curve profile fixed (passing the actual checkpoint directory as `model_path`) did not change the output because the BabyLM evaluator internally uses `revision_name` to locate output subdirectories but may still load the root model.

## What the research needs next

The correct trajectory profile requires either:
1. Inspecting how `evaluation_pipeline.sentence_zero_shot.run` resolves its model path and `revision_name` arguments, then invoking it correctly for local checkpoint subdirectories.
2. Or using a standalone evaluation loop that loads each checkpoint directory directly with `AutoModelForCausalLM.from_pretrained(checkpoint_path)` and runs the same official data. This is more transparent and avoids the evaluator's revision-name ambiguity.

## Artifacts available for correct profiling

- checkpoint trajectory root: `experiments/archive/initial_model_studies/training/runs/babylm_compare_dense6x384_10M_curve/hf_model`
- verified checkpoints: `chck_1M` through `chck_10M`
- each verified in `experiments/archive/initial_model_studies/training/runs/babylm_compare_dense6x384_10M_curve/checkpoint_trajectory_verify.json`
- each loads cleanly as `GPT2LMHeadModel` with 17,037,312 parameters
- official fast evaluation data is already downloaded and available under `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data`

## Next work

The proposed correction is a standalone trajectory evaluator that loads each checkpoint directory directly, runs the official fast tasks, and saves scores per revision. This is the highest-value action because without it the 10M curve cannot expose whether Entity/Reading remain mechanism-limited or improve with exposure.
