# BabyLM Leaderboard Source Snapshots

These unchanged Python files are source snapshots of the public
[BabyLM 2026 leaderboard](https://huggingface.co/spaces/BabyLM-community/babylm-leaderboard-2026):

| Local snapshot | Upstream source path | Scientific role |
| --- | --- | --- |
| [src__leaderboard__read_evals.py](../../../../../experiments/archive/initial_model_studies/data/leaderboard_probe/src__leaderboard__read_evals.py) | `src/leaderboard/read_evals.py` | Reading and aggregating submitted evaluation results. |
| [src__submission__check_validity.py](../../../../../experiments/archive/initial_model_studies/data/leaderboard_probe/src__submission__check_validity.py) | `src/submission/check_validity.py` | Required result columns and submission validation. |
| [src__submission__eval_submission.py](../../../../../experiments/archive/initial_model_studies/data/leaderboard_probe/src__submission__eval_submission.py) | `src/submission/eval_submission.py` | Evaluation submission and score handling. |

The captured upstream README declares `license: apache-2.0`. The
[Apache License, Version 2.0](../../../../../experiments/archive/initial_model_studies/data/leaderboard_probe/LICENSE) is included for these source files.
Existing source notices are retained. These are source-inspection dependencies
for understanding the reported metrics, not a standalone leaderboard
application. No original source bytes have been changed. The historical
capture does not identify a Git commit, and no commit from a different
evaluation repository is assigned to these snapshots.

[leaderboard_config_parsed.json](../../../../../experiments/archive/initial_model_studies/data/leaderboard_probe/leaderboard_config_parsed.json) is the existing
parsed observation of the public
[leaderboard configuration](https://babylm-community-babylm-leaderboard-2026.hf.space/config).
It preserves the observed table values, missing values and public model
references; it is not a newly fetched leaderboard or a new evaluation. Model
and dataset names are identifiers, not copies of their weights or training
corpora. The application's code license is not a blanket license grant for
those separately referenced materials.
