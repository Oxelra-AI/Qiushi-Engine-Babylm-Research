import json
import os
import numpy as np
from collections import defaultdict

import huggingface_hub
from huggingface_hub import ModelCard
from huggingface_hub.hf_api import ModelInfo
from transformers import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.display.utils import TEXT_TASKS, NUM_EXPECTED_EXAMPLES, AOA_SIZE, GLOBAL_PIQA_SIZES
from src.envs import EVAL_DATASETS_PATH
from src.submission.hidden_multilingual import (
    SERVER_SCORED_TASKS,
    validate_hidden_predictions,
)
from src.display.utils import NUM_CHECKPOINTS_10M, NUM_CHECKPOINTS_100M, CHECKPOINT_WORD_COUNTS_10M, \
    CHECKPOINT_WORD_COUNTS_100M, FAST_TASKS


def check_model_card(repo_id: str) -> tuple[bool, str]:
    """Checks if the model card and license exist and have been filled"""
    try:
        card = ModelCard.load(repo_id)
    except huggingface_hub.utils.EntryNotFoundError:
        return False, "Please add a model card to your model to explain how you trained/fine-tuned it."

    # Enforce license metadata
    if card.data.license is None:
        if not ("license_name" in card.data and "license_link" in card.data):
            return False, (
                "License not found. Please add a license to your model card using the `license` metadata or a"
                " `license_name`/`license_link` pair."
            )

    # Enforce card content
    if len(card.text) < 200:
        return False, "Please add a description to your model card, it is too short."

    return True, ""


def is_model_on_hub(model_name: str, revision: str, token: str = None, trust_remote_code=False, test_tokenizer=False) -> tuple[bool, str]:
    """Checks if the model model_name is on the hub, and whether it (and its tokenizer) can be loaded with AutoClasses."""
    try:
        config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=trust_remote_code, token=token)
        if test_tokenizer:
            try:
                _ = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=trust_remote_code, token=token)
            except ValueError as e:
                return (
                    False,
                    f"uses a tokenizer which is not in a transformers release: {e}",
                    None
                )
            except Exception:
                return (False, "'s tokenizer cannot be loaded. Is your tokenizer class in a stable transformers release, and correctly configured?", None)
        return True, None, config

    except ValueError:
        return (
            False,
            "needs to be launched with `trust_remote_code=True`. For safety reason, we do not allow these models to be automatically submitted to the leaderboard.",
            None
        )

    except Exception:
        return False, "was not found on hub!", None


def get_model_size(model_info: ModelInfo, precision: str):
    """Gets the model size from the configuration, or the model name if the configuration does not contain the information."""
    try:
        model_size = round(model_info.safetensors["total"] / 1e9, 3)
    except (AttributeError, TypeError):
        return 0  # Unknown model sizes are indicated as 0, see NUMERIC_INTERVALS in app.py

    size_factor = 8 if (precision == "GPTQ" or "gptq" in model_info.modelId.lower()) else 1
    model_size = size_factor * model_size
    return model_size


def get_model_arch(model_info: ModelInfo):
    """Gets the model architecture from the configuration"""
    return model_info.config.get("architectures", "Unknown")


def already_submitted_models(requested_models_dir: str) -> set[str]:
    """Gather a list of already submitted models to avoid duplicates"""
    depth = 1
    file_names = []
    users_to_submission_dates = defaultdict(list)

    for root, _, files in os.walk(requested_models_dir):
        current_depth = root.count(os.sep) - requested_models_dir.count(os.sep)
        if current_depth == depth:
            for file in files:
                if not file.endswith(".json"):
                    continue
                with open(os.path.join(root, file), "r") as f:
                    info = json.load(f)
                    try:
                        file_names.append(f"{info['model']}_{info['revision']}_{info['track']}")
                    except Exception:
                        continue

                    # Select organisation
                    if info["model"].count("/") == 0 or "submitted_time" not in info:
                        continue
                    organisation, _ = info["model"].split("/")
                    users_to_submission_dates[organisation].append(info["submitted_time"])

    return set(file_names), users_to_submission_dates


def is_valid_predictions(predictions: str, track: str) -> tuple[bool, str]:
    predictions = json.load(open(predictions, "r"))

    # Multilingual track uses a different validation path
    if track == "multilingual":
        return is_valid_multilingual_predictions(predictions)

    # First check full evals
    valid_full, full_message = is_valid_full_predictions(predictions, track)
    if not valid_full:
        return valid_full, full_message

    # Then check fast evals
    valid_fast, fast_message = is_valid_fast_metrics(predictions, track)
    return valid_fast, fast_message


def is_valid_multilingual_predictions_file(path: str) -> tuple[bool, str]:
    """Validate the predictions file produced by `collate_results.py`.

    Required: at least one of `zeroshot`, `finetune`, or `hidden`.
    Optional: `finetune` dict, `fast_eval_results` list (only present with --fast).
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return False, f"Error: predictions file could not be read as JSON ({e})."

    if not isinstance(data, dict):
        return False, "Error: predictions file must be a JSON object."

    zeroshot = data.get("zeroshot")
    if zeroshot is not None and not isinstance(zeroshot, dict):
        return False, "Error: 'zeroshot' in predictions file must be a JSON object."

    finetune = data.get("finetune")
    if finetune is not None and not isinstance(finetune, dict):
        return False, "Error: 'finetune' in predictions file must be a JSON object."

    hidden = data.get("hidden")
    valid_hidden, hidden_message = validate_hidden_predictions(
        hidden, EVAL_DATASETS_PATH
    )
    if not valid_hidden:
        return False, hidden_message

    if not any(
        isinstance(section, dict) and section
        for section in (zeroshot, finetune, hidden)
    ):
        return False, "Error: predictions file contains no predictions."

    fast_eval_results = data.get("fast_eval_results")
    if fast_eval_results is not None and not isinstance(fast_eval_results, list):
        return False, "Error: 'fast_eval_results' in predictions file must be a list."

    return True, "Upload successful."


def is_valid_multilingual_predictions(predictions: dict) -> tuple[bool, str]:
    """Validate multilingual track predictions.

    Expected format (produced by collate_results.py):
        {task_name: {subtask_or_lang: float_score, ...}, ...}

    Every value at the top level must be a dict, and every value inside that
    dict must be a number (the pre-computed accuracy).
    """
    if not isinstance(predictions, dict):
        return False, "Error: Submission must be a JSON object mapping task names to score dicts."

    for task_key, task_data in predictions.items():
        if task_key in SERVER_SCORED_TASKS:
            return False, (
                f"Error: '{task_key}' is scored server-side; remove it from "
                "the score file and upload its raw predictions instead."
            )
        if not isinstance(task_data, dict):
            return False, f"Error: '{task_key}' must map to a dictionary of {{subtask: score}} pairs, got {type(task_data).__name__}."
        if not task_data:
            return False, f"Error: '{task_key}' maps to an empty dict — at least one score is required."
        for subtask_key, value in task_data.items():
            if not isinstance(value, (int, float)):
                return False, f"Error: score for '{task_key}.{subtask_key}' must be a number, got {type(value).__name__}."

    return True, "Upload successful."


def is_valid_full_predictions(predictions: dict, track: str) -> tuple[bool, str]:
    out_msg = ""

    # Validate AoA if present — missing AoA is allowed (scored as 0)
    if "aoa" in predictions and predictions["aoa"] is not None and "results" in predictions["aoa"]:
        aoa_results = predictions["aoa"]
        target_num_items = NUM_CHECKPOINTS_10M if track == 'strict-small' else NUM_CHECKPOINTS_100M
        revisions_in_results = {r['step'] for r in aoa_results["results"]}
        if len(revisions_in_results) > target_num_items:
            out_msg = f"Error: AoA predictions contain results for {len(revisions_in_results)} checkpoints. Was expecting at most {target_num_items} for track {track}"
        else:
            for revision in revisions_in_results:
                revision_results = [r for r in aoa_results["results"] if r['step'] == revision]
                if len(revision_results) != AOA_SIZE:
                    out_msg = f"Error: AoA prediction for checkpoint {revision} contains {len(revision_results)} datapoints. Was expecting {AOA_SIZE}"

    if out_msg != "":
        return False, out_msg

    # Validate Global PIQA tasks. Unlike the per-subtask tasks these key their
    # predictions by `example_id` (one prediction each), so they are checked
    # separately and skipped in the generic per-subtask loop below.
    # Missing tasks are allowed — they are scored as 0.
    for task, num_expected_examples in GLOBAL_PIQA_SIZES.items():
        if task not in predictions or predictions[task] is None:
            continue
        task_predictions = predictions[task]
        if len(task_predictions) != num_expected_examples:
            out_msg = f"Error: {task} has {len(task_predictions)} examples, expected {num_expected_examples}."
            break
        for example_id, example in task_predictions.items():
            example_preds = example["predictions"]
            if len(example_preds) != 1:
                out_msg = f"Error: {task} example {example_id} has {len(example_preds)} predictions, expected 1."
                break
            if not isinstance(example_preds[0]["pred"], str):
                out_msg = f"Error: results for `{example_id}` (`{task}`) should be strings but aren't."
                break
        if out_msg != "":
            break

    if out_msg != "":
        return False, out_msg

    # Validate size and prediction type for each present task/subtask
    # Missing tasks or subtasks are allowed — they are scored as 0
    for task in predictions:
        if task in ["aoa", "fast_eval_results"]:
            continue
        if task in GLOBAL_PIQA_SIZES:
            continue
        if predictions[task] is None:
            continue

        for subtask in predictions[task]:
            if predictions[task][subtask] is None:
                continue
            if task not in NUM_EXPECTED_EXAMPLES or subtask not in NUM_EXPECTED_EXAMPLES[task]:
                continue

            num_expected_examples = NUM_EXPECTED_EXAMPLES[task][subtask]
            if len(predictions[task][subtask]["predictions"]) != num_expected_examples:
                out_msg = f"Error: {subtask} has the wrong number of examples."
                break

            if task == "glue":
                if not isinstance(predictions[task][subtask]["predictions"][0]["pred"], int):
                    out_msg = f"Error: results for `{subtask}` (`{task}`) should be integers but aren't."
                    break
            elif task == "reading":
                if not isinstance(predictions[task][subtask]["predictions"][0]["pred"], float):
                    out_msg = f"Error: results for `{subtask}` (`{task}`) should be floats but aren't."
                    break
            else:
                if not isinstance(predictions[task][subtask]["predictions"][0]["pred"], str):
                    out_msg = f"Error: results for `{subtask}` (`{task}`) should be strings but aren't."
                    break

        if out_msg != "":
            break

    if out_msg != "":
        return False, out_msg
    return True, "Upload successful."


def is_valid_fast_metrics(predictions: dict, track: str) -> tuple[bool, str]:
    # Missing fast eval results are allowed — they are scored as 0
    if "fast_eval_results" not in predictions or predictions["fast_eval_results"] is None:
        return True, "Upload successful."

    out_msg = ""
    results = predictions["fast_eval_results"]
    target_num_items = NUM_CHECKPOINTS_10M if track == 'strict-small' else NUM_CHECKPOINTS_100M
    checkpoint_word_counts = CHECKPOINT_WORD_COUNTS_10M if track == 'strict-small' else CHECKPOINT_WORD_COUNTS_100M
    for task in FAST_TASKS:
        # Missing fast tasks are allowed — scored as 0
        if task not in results:
            continue

        task_results = results[task]
        if type(task_results) is not list:
            out_msg = f"Error: {task} does not map to a list in the fast evaluation results"
            return False, out_msg

        if len(task_results) > target_num_items:
            out_msg = f"Error: {task} has more than {target_num_items} checkpoints in the fast evaluation results"
            return False, out_msg
        else:
            target_num_items = len(task_results)
            checkpoint_word_counts = checkpoint_word_counts[:target_num_items]

        for checkpoint_results, word_counts in zip(task_results, checkpoint_word_counts):
            # Missing checkpoint results are allowed — scored as 0
            if checkpoint_results is None:
                continue
            assert len(checkpoint_results) == len(FAST_TASKS[task]), f"Error: checkpoint results for {task} has different number of tasks"
            for subtask in FAST_TASKS[task]:
                if subtask not in checkpoint_results:
                    out_msg = f"Error: {subtask} not present under the {word_counts} word checkpoint for {task} in the fast evaluation results"
                    return False, out_msg

    return True, "Upload successful."
