import json
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import f1_score
from typing import Any

from src.envs import EVAL_DATASETS_PATH
from src.about import Tasks
from src.submission.hidden_multilingual import (
    SERVER_SCORED_TASKS,
    score_hidden_predictions,
)

# Canonical metric per GLUE subtask, taken from about.py so the leaderboard scores
# with the same metric the finetuning pipeline reports/selects on locally
# (e.g. F1 for MRPC/QQP, accuracy elsewhere). Any subtask not listed defaults to accuracy.
GLUE_SUBTASK_METRICS = {
    t.value.task_name: t.value.metric
    for t in Tasks
    if t.value.benchmark == "glue" and t.value.task_name is not None
}


def evaluate_submission(
    path_to_submission: str,
    multilingual: bool = False,
    predictions_path: str | None = None,
) -> dict[str, dict[str, float]]:
    results_dict = json.load(open(path_to_submission, "r"))
    data_path = Path(EVAL_DATASETS_PATH)

    # Multilingual submissions use lm_eval format with pre-computed accuracy scores
    if multilingual:
        processed = _process_multilingual_results(results_dict)
        if predictions_path is not None:
            with open(predictions_path, "r") as f:
                raw_predictions = json.load(f)
            processed.update(
                score_hidden_predictions(
                    raw_predictions.get("hidden", {}),
                    EVAL_DATASETS_PATH,
                )
            )
        return processed

    processed_results_dict = {
        'fast_eval_results' : results_dict.get('fast_eval_results'),
        'aoa_surprisals' : results_dict.get('aoa_surprisals'),
        # Marker recording that entity tracking was scored from the "nothing"-filtered
        # predictions (keyed 'entity_tracking_filtered'). Legacy submissions predate this
        # key, so the leaderboard zeroes their entity tracking score to force re-collation
        # (see read_evals.py, mirroring the aoa_surprisals gate).
        'entity_tracking_filtered' : bool(results_dict.get('entity_tracking_filtered')),
    }

    def _preds(key):
        return results_dict.get(key) or {}

    # BLiMP — missing subtasks produce 0.0 via _calculate_target_results
    processed_results_dict["blimp"] = _calculate_target_results(_preds("blimp"), data_path / "blimp")
    # BLiMP Supplement
    processed_results_dict["blimp_supplement"] = _calculate_target_results(_preds("blimp_supplement"), data_path / "blimp_supplement")
    # EWoK
    processed_results_dict["ewok"] = _calculate_target_results(_preds("ewok"), data_path / "ewok")
    # Entity Tracking — gold is filtered to match the "nothing"-dropped predictions.
    # Predictions are keyed "entity_tracking_filtered" (they arrive already filtered).
    processed_results_dict["entity_tracking"] = _calculate_target_results(_preds("entity_tracking_filtered"), data_path / "entity_tracking", skip_nothing_targets=True)
    # COMPS
    processed_results_dict["comps"] = _calculate_target_results(_preds("comps"), data_path / "comps")
    # Global PIQA (predictions keyed by example_id; gold is a flat example_id -> solution JSON)
    processed_results_dict["global_piqa_parallel"] = _calculate_global_piqa_results(_preds("global_piqa_parallel"), data_path / "global_piqa_parallel.json", "global_piqa_parallel")
    processed_results_dict["global_piqa_nonparallel"] = _calculate_global_piqa_results(_preds("global_piqa_nonparallel"), data_path / "global_piqa_nonparallel.json", "global_piqa_nonparallel")
    # Reading — fixed subtask keys so missing reading shows as 0 not absent
    preds = _preds("reading")
    processed_results_dict["reading"] = _calculate_reading_results(preds, data_path / "reading" / "reading_data.csv") if preds else {"spr": 0.0, "rt": 0.0}
    # GLUE — per-subtask canonical metric (F1 for MRPC/QQP, accuracy elsewhere) from about.py
    processed_results_dict["glue"] = _calculate_target_results(_preds("glue"), data_path / "glue", GLUE_SUBTASK_METRICS)
    # AoA — new format is {"aoa": <float>} (score computed in the pipeline). Old submissions
    # put the raw surprisals ({"metadata", "results"}) under "aoa" instead; those, and any
    # missing/malformed value, map to 0.0 so legacy files parse without error.
    aoa = results_dict.get("aoa")
    if isinstance(aoa, dict) and isinstance(aoa.get("aoa"), (int, float)) and not isinstance(aoa.get("aoa"), bool):
        processed_results_dict["aoa"] = {"aoa": float(aoa["aoa"])}
    else:
        processed_results_dict["aoa"] = {"aoa": 0.0}
    return processed_results_dict


def _score_predictions(preds: list, targets: list, metric: str) -> float:
    """Score a single subtask under the given metric.

    "f1" computes binary F1 (positive class 1), matching the finetuning pipeline's
    `f1_score(labels, predictions)`. Everything else defaults to exact-match accuracy.
    """
    if not targets:
        return 0.0
    if metric == "f1":
        return float(f1_score(targets, preds, zero_division=0))
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / len(targets)


def _calculate_target_results(results_dict: dict[str, dict[str, list[dict[str, str]]]], path_to_data: Path, metric_by_subtask: dict[str, str] | None = None, skip_nothing_targets: bool = False) -> dict[str, float]:
    if not path_to_data.exists() or not any(path_to_data.glob("*.jsonl")):
        raise FileNotFoundError(f"No dataset files found at {path_to_data}. Ensure the evaluation datasets are downloaded before submitting.")
    metric_by_subtask = metric_by_subtask or {}
    processed_results = {}
    for subtask_file in sorted(path_to_data.glob("*.jsonl")):
        subtask = subtask_file.stem
        if subtask not in results_dict or results_dict[subtask] is None:
            processed_results[subtask] = 0.0
            continue
        preds, targets = [], []
        with subtask_file.open("r") as data_file:
            subtask_results = results_dict[subtask]["predictions"]
            gold = [json.loads(line) for line in data_file.readlines()]
            # entity_tracking drops datapoints whose answer is "nothing" at
            # prediction-generation time (see read_files.py), so the gold must be
            # filtered identically to keep predictions and targets aligned in the zip.
            if skip_nothing_targets:
                gold = [data for data in gold if not (isinstance(data["target"], str) and "nothing" in data["target"])]
            for result, data in zip(subtask_results, gold):
                res = result["pred"].strip() if isinstance(result["pred"], str) else result["pred"]
                target = data["target"].strip() if isinstance(data["target"], str) else data["target"]
                preds.append(res)
                targets.append(target)
        processed_results[subtask] = _score_predictions(preds, targets, metric_by_subtask.get(subtask, "acc"))
    return processed_results


def _calculate_global_piqa_results(results_dict: dict[str, dict[str, list[dict[str, str]]]], answers_path: Path, benchmark: str) -> dict[str, float]:
    """Score a Global PIQA task.

    Unlike the other text tasks, Global PIQA predictions are keyed by `example_id`
    (one prediction per example) and the gold answers live in a single flat JSON
    file mapping `example_id -> correct solution string`. We iterate over the gold
    answers so that missing predictions count as incorrect (consistent with the
    "missing -> 0" policy used elsewhere).
    """
    if not answers_path.exists():
        raise FileNotFoundError(f"No answer file found at {answers_path}. Ensure the evaluation datasets are downloaded before submitting.")
    with answers_path.open("r") as f:
        answers = json.load(f)

    correct = 0
    total = 0
    for example_id, gold in answers.items():
        total += 1
        res = results_dict.get(example_id) if results_dict else None
        if not res:
            continue
        pred = res["predictions"][0]["pred"]
        pred = pred.strip() if isinstance(pred, str) else pred
        gold = gold.strip() if isinstance(gold, str) else gold
        if pred == gold:
            correct += 1
    return {benchmark: correct / total if total > 0 else 0.0}


def _calculate_reading_results(results_dict: dict[str, dict[str, list[dict[str, int | float]]]], path_to_data: Path) -> dict[str, float]:
    data = pd.read_csv(path_to_data, dtype={'item': str})
    preds = [item["pred"] for item in results_dict["reading"]["predictions"]]
    prev_preds = [item["prev_pred"] for item in results_dict["reading"]["predictions"]]
    data["pred"] = preds
    data["prev_pred"] = prev_preds
    eye_tracking_vars = ['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound']
    eye_tracking_result = []
    for dv in eye_tracking_vars:
        # Baseline model
        temp = data[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        # Predictive model
        temp = data[[dv, "Subtlex_log10", "length", "context_length", "pred"]].dropna()
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred', data=temp).fit()
        R2_model = float(OLS_model.rsquared)
        eye_tracking_result.append(((R2_model-R2_baseline)/(1-R2_baseline)))
    eye_tracking_result = sum(eye_tracking_result) / len(eye_tracking_result)

    # Baseline model
    temp = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
    OLS_baseline = smf.ols(formula='self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred', data=temp).fit()
    R2_baseline = float(OLS_baseline.rsquared)
    # Predictive model
    temp = data[["self_paced_reading_time", "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred", "pred"]].dropna()
    OLS_model = smf.ols(formula='self_paced_reading_time ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred + pred', data=temp).fit()
    R2_model = float(OLS_model.rsquared)
    self_paced_reading_result = ((R2_model-R2_baseline)/(1-R2_baseline))

    processed_results = {"spr": self_paced_reading_result, "rt": eye_tracking_result}

    return processed_results


def _process_multilingual_results(results_dict: dict) -> dict[str, dict[str, float]]:
    """Pass through pre-computed multilingual accuracy scores.

    Input format (from collate_results.py):
        {task_name: {subtask_or_lang: float_score, ...}, ...}

    Scores are stored as-is — no server-side evaluation is performed.
    When multiple subtask/language scores are present for a task, the leaderboard
    averages them at display time via _get_task_results.
    """
    processed = {}
    for task_key, task_data in results_dict.items():
        # Hidden tasks are always recomputed from the separate raw prediction
        # file. Never trust participant-supplied final scores for these keys.
        if task_key in SERVER_SCORED_TASKS:
            continue
        if isinstance(task_data, dict):
            float_scores = {k: v for k, v in task_data.items() if isinstance(v, (int, float))}
            if float_scores:
                processed[task_key] = float_scores
        elif isinstance(task_data, (int, float)):
            processed[task_key] = {task_key: task_data}
    return processed


if __name__ == "__main__":
    path_to_submission = "eval-queue/babylm/all_full_preds.json"
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_path = Path(f"eval-results/demo-leaderboard/test/results_{current_time}.json")
    output_dict = {"track": "strict-small", "config": {"model_name": "Sub_tester", "hf_repo": "", "model_sha": ""}}
    output_dict["results"] = evaluate_submission(path_to_submission)
    json.dump(output_dict, output_path.open("w"))
