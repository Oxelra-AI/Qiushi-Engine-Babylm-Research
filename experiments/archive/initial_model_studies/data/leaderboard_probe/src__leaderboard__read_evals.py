import glob
import json
import os
from dataclasses import dataclass

import dateutil
import numpy as np

from src.display.formatting import make_clickable_model
from src.display.utils import AutoEvalColumn, AutoEvalColumnMultilingual, Tasks
from src.about import (
    TasksMultilingual,
    MULTILINGUAL_ENG_ZEROSHOT_KEYS, MULTILINGUAL_NLD_ZEROSHOT_KEYS, MULTILINGUAL_ZHO_ZEROSHOT_KEYS,
    MULTILINGUAL_ENG_FINETUNE_KEYS, MULTILINGUAL_NLD_FINETUNE_KEYS, MULTILINGUAL_ZHO_FINETUNE_KEYS,
    MULTILINGUAL_ENG_ALL_KEYS, MULTILINGUAL_NLD_ALL_KEYS, MULTILINGUAL_ZHO_ALL_KEYS,
)
from src.submission.hidden_multilingual import SERVER_SCORED_RESULT_KEYS


@dataclass
class EvalResult:
    """Represents one full evaluation. Built from a combination of the result and request file for a given run.
    """
    eval_name: str  # org_model_track (uid)
    full_model: str  # org/model (name of model)
    repo_id: str  # org/model (path to model on HF)
    track: str
    org: str
    model: str
    revision: str  # commit hash, "" if main
    results: dict
    main_contributions: str
    model_type: str
    architecture: str
    optimizer: str
    learning_rate: float
    learning_rate_scheduler: str
    training_epochs: int
    batch_size: int
    tokenizer: str
    token_set_size: int
    seed: int
    num_layers: int
    num_heads: int
    max_seq_length: int
    num_parameters: int
    flops: int
    gpu_dev: int
    gpu_train: int
    training_data: str
    num_words_data: int
    data_genre: str
    data_preprocessing: str
    data_human_annotation: str
    data_syn_aug: str
    other_hyp: str
    model_description: str
    languages: str = ""
    num_words_en: int = 0
    num_words_nld: int = 0
    num_words_zho: int = 0
    teacher_models: str = ""
    date: str = ""  # submission date of request file

    @classmethod
    def init_from_json_file(self, json_filepath):
        """Inits the result from the specific model result file"""
        with open(json_filepath) as fp:
            data = json.load(fp)

        config = data.get("config")
        track = data.get("track")

        # Get model and org
        org_and_model = config.get("model_name", config.get("model_args", None))
        repo_id = config.get("hf_repo", config.get("hf_repo", None))
        main_contributions = config.get("main_contributions", None)
        model_type = config.get("model_type", None)
        architecture = config.get("architecture", None)
        optimizer = config.get("optimizer", None)
        learning_rate = config.get("learning_rate", None)
        learning_rate_scheduler = config.get("learning_rate_scheduler", None)
        training_epochs = config.get("training_epochs", None)
        batch_size = config.get("batch_size", None)
        tokenizer = config.get("tokenizer", None)
        token_set_size = config.get("token_set_size", None)
        seed = config.get("seed", None)
        num_layers = config.get("num_layers", None)
        num_heads = config.get("num_heads", None)
        max_seq_length = config.get("max_seq_length", None)
        num_parameters = config.get("num_parameters", None)
        if num_parameters is not None:
            num_parameters = round(num_parameters / 1e6)
        flops = config.get("flops", None)
        if flops is not None:
            flops = round(flops / 1e15, 2)
        gpu_dev = config.get("gpu_dev", None)
        gpu_train = config.get("gpu_train", None)
        training_data = config.get("training_data", None)
        num_words_data = config.get("num_words_data", None)
        if num_words_data is not None:
            num_words_data = round(num_words_data / 1e6)
        data_genre = config.get("data_genre", None)
        data_preprocessing = config.get("data_preprocessing", None)
        data_human_annotation = config.get("data_human_annotation", None)
        data_syn_aug = config.get("data_syn_aug", None)
        other_hyp = config.get("other_hyp", None)
        model_description = config.get("model_description", None)
        languages = config.get("languages", "")
        num_words_en = config.get("num_words_en", 0)
        num_words_nld = config.get("num_words_nld", 0)
        num_words_zho = config.get("num_words_zho", 0)
        teacher_models = config.get("teacher_models", "")
        org_and_model = org_and_model.split("/", 1)

        if len(org_and_model) == 1:
            if "/" in repo_id:
                org = repo_id.split("/")[0]
            else:
                org = repo_id
            model = org_and_model[0]
        else:
            org = org_and_model[0]
            model = org_and_model[1]
        full_model = "/".join(org_and_model)
        eval_name = "_".join(org_and_model) + f"_{track}"

        def _get_benchmark_score(benchmark):
            # Flatten a benchmark's subtask scores. None -> 0.0 (counts against the average);
            # non-numeric values are skipped so a legacy/raw structure (e.g. old submissions
            # that stored raw AoA surprisals under "aoa") maps to 0.0 rather than crashing the
            # whole leaderboard load.
            v = data["results"].get(benchmark)
            if not isinstance(v, dict):
                return 0.0
            scores = []
            for a in v.values():
                if a is None:
                    scores.append(0.0)
                elif isinstance(a, (int, float)) and not isinstance(a, bool):
                    scores.append(float(a))
            if not scores:
                return 0.0
            return float(np.mean(scores)) * 100.0

        # Extract results available in this file (some results are split in several files)
        results = {}
        if track.lower() == "multilingual":
            for task in TasksMultilingual:
                task_val = task.value
                if task_val.task_name is None:
                    # Zeroshot: average all values under the benchmark key; NaN if absent.
                    accs = np.array(
                        [list(v.values()) for k, v in data["results"].items() if task_val.benchmark == k]
                    ).flatten()
                    if accs.size == 0:
                        results[task_val.benchmark] = np.nan
                    else:
                        accs = np.array([a if a is not None else 0.0 for a in accs], dtype=float)
                        results[task_val.benchmark] = float(np.mean(accs)) * 100.0
                else:
                    # Finetune individual: stored with compound key "{benchmark}_{lang}".
                    compound_key = f"{task_val.benchmark}_{task_val.task_name}"
                    try:
                        value = data["results"][task_val.benchmark][task_val.task_name]
                        scale = 1.0 if task_val.metric == "delta logLik" else 100.0
                        results[compound_key] = value * scale
                    except Exception:
                        results[compound_key] = np.nan

            # GlobalPIQA per language: macro-average of the raw parallel/nonparallel scores,
            # stored under global_piqa_{lang} so it feeds that language's zeroshot average.
            def _ml_raw(key):
                accs = np.array(
                    [list(v.values()) for k, v in data["results"].items() if key == k]
                ).flatten()
                if accs.size == 0:
                    return np.nan
                accs = np.array([a if a is not None else 0.0 for a in accs], dtype=float)
                return float(np.mean(accs)) * 100.0

            for lang in ("en", "nl", "zh"):
                parts = [_ml_raw(f"global_piqa_parallel_{lang}"), _ml_raw(f"global_piqa_nonparallel_{lang}")]
                finite = [v for v in parts if not np.isnan(v)]
                results[f"global_piqa_{lang}"] = float(np.mean(finite)) if finite else np.nan
        else:
            for task in Tasks:
                task = task.value
                if task.task_name is None:
                    results[task.benchmark] = _get_benchmark_score(task.benchmark)
                else:
                    try:
                        results[task.task_name] = data["results"][task.benchmark][task.task_name] * 100
                    except Exception:
                        results[task.task_name] = 0.0

            # GlobalPIQA is shown as one column: the macro-average of its two raw
            # (parallel / nonparallel) subtask scores in the submission.
            results["global_piqa"] = (
                _get_benchmark_score("global_piqa_parallel")
                + _get_benchmark_score("global_piqa_nonparallel")
            ) / 2

            # AoA counts only when produced by the new pipeline, which uploads the raw
            # surprisals under 'aoa_surprisals'. Legacy results used 'aoa_raw' instead, so
            # results without the new key are zeroed to force a re-run under the new setup.
            if not data["results"].get("aoa_surprisals"):
                results["aoa"] = 0.0

            # Entity tracking counts only when scored from the "nothing"-filtered
            # predictions, marked by 'entity_tracking_filtered' in the stored results.
            # Legacy submissions predate that key, so their score is zeroed to force
            # re-collation under the new setup.
            if not data["results"].get("entity_tracking_filtered"):
                results["entity_tracking"] = 0.0

        return self(
            eval_name=eval_name,
            full_model=full_model,
            repo_id=repo_id,
            track=track,
            org=org,
            model=model,
            results=results,
            revision=config.get("model_sha", ""),
            main_contributions=main_contributions,
            model_type=model_type,
            architecture=architecture,
            optimizer=optimizer,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            training_epochs=training_epochs,
            batch_size=batch_size,
            tokenizer=tokenizer,
            token_set_size=token_set_size,
            seed=seed,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_length=max_seq_length,
            num_parameters=num_parameters,
            flops=flops,
            gpu_dev=gpu_dev,
            gpu_train=gpu_train,
            training_data=training_data,
            num_words_data=num_words_data,
            data_genre=data_genre,
            data_preprocessing=data_preprocessing,
            data_human_annotation=data_human_annotation,
            data_syn_aug=data_syn_aug,
            other_hyp=other_hyp,
            model_description=model_description,
            languages=languages,
            num_words_en=num_words_en,
            num_words_nld=num_words_nld,
            num_words_zho=num_words_zho,
            teacher_models=teacher_models,
        )

    def update_with_request_file(self, requests_path):
        """Finds the relevant request file for the current model and updates info with it"""
        request_file = get_request_file_for_model(requests_path, self.full_model, self.track, self.org)

        try:
            with open(request_file, "r") as f:
                request = json.load(f)
            self.date = request.get("submitted_time", "")
        except Exception:
            print(f"Could not find request file for {self.org}/{self.model}")

    def to_dict(self):
        """Converts the Eval Result to a dict compatible with our dataframe display"""
        if self.track.lower() == "multilingual":
            eval_column = AutoEvalColumnMultilingual
        else:
            eval_column = AutoEvalColumn

        if self.repo_id and self.repo_id != "Unknown":
            model_display_name = make_clickable_model(self.repo_id, self.full_model)
        else:
            model_display_name = self.full_model

        data_dict = {
            "eval_name": self.eval_name,  # not a column, just a save name,
            eval_column.model.name: model_display_name,
            eval_column.hf_repo.name: self.repo_id,
            eval_column.revision.name: self.revision,
            eval_column.main_contributions.name: self.main_contributions,
            eval_column.model_type.name: self.model_type,
            eval_column.architecture.name: self.architecture,
            eval_column.optimizer.name: self.optimizer,
            eval_column.learning_rate.name: self.learning_rate,
            eval_column.learning_rate_scheduler.name: self.learning_rate_scheduler,
            eval_column.training_epochs.name: self.training_epochs,
            eval_column.batch_size.name: self.batch_size,
            eval_column.tokenizer.name: self.tokenizer,
            eval_column.token_set_size.name: self.token_set_size,
            eval_column.seed.name: self.seed,
            eval_column.num_layers.name: self.num_layers,
            eval_column.num_heads.name: self.num_heads,
            eval_column.max_seq_length.name: self.max_seq_length,
            eval_column.num_parameters.name: self.num_parameters,
            eval_column.flops.name: self.flops,
            eval_column.gpu_dev.name: self.gpu_dev,
            eval_column.gpu_train.name: self.gpu_train,
            eval_column.training_data.name: self.training_data,
            eval_column.num_words_data.name: self.num_words_data,
            eval_column.data_genre.name: self.data_genre,
            eval_column.data_preprocessing.name: self.data_preprocessing,
            eval_column.data_human_annotation.name: self.data_human_annotation,
            eval_column.data_syn_aug.name: self.data_syn_aug,
            eval_column.other_hyp.name: self.other_hyp,
            eval_column.model_description.name: self.model_description,
        }

        if self.track.lower() == "multilingual":
            taskset = TasksMultilingual

            def _lang_avg(all_keys):
                """Fixed-denominator average; NaN if language was not evaluated at all."""
                vals = [self.results.get(k, np.nan) for k in all_keys]
                finite = [v for v in vals if not np.isnan(v)]
                if not finite:
                    return np.nan
                return round(sum(v if not np.isnan(v) else 0.0 for v in vals) / len(all_keys), 2)

            eng_zeroshot_avg = _lang_avg(MULTILINGUAL_ENG_ZEROSHOT_KEYS)
            nld_zeroshot_avg = _lang_avg(MULTILINGUAL_NLD_ZEROSHOT_KEYS)
            zho_zeroshot_avg = _lang_avg(MULTILINGUAL_ZHO_ZEROSHOT_KEYS)
            eng_finetune_avg = _lang_avg(MULTILINGUAL_ENG_FINETUNE_KEYS)
            nld_finetune_avg = _lang_avg(MULTILINGUAL_NLD_FINETUNE_KEYS)
            zho_finetune_avg = _lang_avg(MULTILINGUAL_ZHO_FINETUNE_KEYS)
            eng_avg = _lang_avg(MULTILINGUAL_ENG_ALL_KEYS)
            nld_avg = _lang_avg(MULTILINGUAL_NLD_ALL_KEYS)
            zho_avg = _lang_avg(MULTILINGUAL_ZHO_ALL_KEYS)
            lang_avgs = [eng_avg, nld_avg, zho_avg]
            multilingual_avg = round(sum(v if not np.isnan(v) else 0.0 for v in lang_avgs) / 3, 2)

            data_dict[eval_column.multilingual_average.name] = multilingual_avg
            data_dict[eval_column.eng_average.name] = eng_avg
            data_dict[eval_column.nld_average.name] = nld_avg
            data_dict[eval_column.zho_average.name] = zho_avg
            data_dict[eval_column.eng_zeroshot_average.name] = eng_zeroshot_avg
            data_dict[eval_column.nld_zeroshot_average.name] = nld_zeroshot_avg
            data_dict[eval_column.zho_zeroshot_average.name] = zho_zeroshot_avg
            data_dict[eval_column.eng_finetune_average.name] = eng_finetune_avg
            data_dict[eval_column.nld_finetune_average.name] = nld_finetune_avg
            data_dict[eval_column.zho_finetune_average.name] = zho_finetune_avg
            data_dict[eval_column.languages.name] = self.languages
            data_dict[eval_column.num_words_en.name] = self.num_words_en
            data_dict[eval_column.num_words_nld.name] = self.num_words_nld
            data_dict[eval_column.num_words_zho.name] = self.num_words_zho
            data_dict[eval_column.teacher_models.name] = self.teacher_models

            for task in taskset:
                if task.value.task_name is not None:
                    key = f"{task.value.benchmark}_{task.value.task_name}"
                else:
                    key = task.value.benchmark
                data_dict[task.value.col_name] = self.results.get(key, np.nan)
        else:
            taskset = Tasks
            # Fixed benchmark sets — missing tasks count as 0 so skipping never helps.
            # NLP tasks are measured directly; "human-like" tasks are measured against
            # human data/psychometrics. "Reading" is itself the mean of its spr/rt
            # subtasks, so it is one entry here (AoA gets 1/2 of the human-like weight).
            # GlobalPIQA is already the macro-average of its parallel/nonparallel subtasks.
            global_piqa = self.results.get("global_piqa") or 0.0
            NLP_BENCHMARK_KEYS = ("blimp", "blimp_supplement", "ewok", "entity_tracking", "comps", "glue")
            HUMAN_LIKE_BENCHMARK_KEYS = ("reading", "aoa")

            nlp_vals = [self.results.get(k) or 0.0 for k in NLP_BENCHMARK_KEYS] + [global_piqa]
            human_vals = [self.results.get(k) or 0.0 for k in HUMAN_LIKE_BENCHMARK_KEYS]
            overall_vals = nlp_vals + human_vals

            data_dict[eval_column.overall_average.name] = sum(overall_vals) / len(overall_vals)
            data_dict[eval_column.nlp_average.name] = sum(nlp_vals) / len(nlp_vals)
            data_dict[eval_column.human_like_average.name] = sum(human_vals) / len(human_vals)

            for task in taskset:
                key = task.value.task_name if task.value.task_name is not None else task.value.benchmark
                data_dict[task.value.col_name] = self.results.get(key, np.nan)

        return data_dict


def get_request_file_for_model(requests_path, model_name, track, org):
    """Selects the correct request file for a given model. Only keeps runs tagged as FINISHED"""
    request_files = os.path.join(
        requests_path,
        org,
        f"{model_name}_eval_request_*.json",
    )
    request_files = glob.glob(request_files)

    # Select correct request file (track)
    request_file = ""
    request_files = sorted(request_files, reverse=True)
    for tmp_request_file in request_files:
        with open(tmp_request_file, "r") as f:
            req_content = json.load(f)
            if (
                req_content["status"] in ["FINISHED"]
            ):
                request_file = tmp_request_file
    return request_file


def get_raw_eval_results(results_path: str, requests_path: str) -> list[EvalResult]:
    """From the path of the results folder root, extract all needed info for results"""
    model_result_filepaths = []

    for root, _, files in os.walk(results_path):
        # We should only have json files in model results
        if len(files) == 0 or any([not f.endswith(".json") for f in files]):
            continue

        # Sort the files by submission date. The filename is results_<ISO timestamp>.json,
        # which sorts chronologically as a plain string, so the newest file is processed
        # last (see the merge in get_raw_eval_results, which relies on this ordering).
        try:
            files.sort(key=lambda x: x.removesuffix(".json").removeprefix("results_"))
        except dateutil.parser._parser.ParserError:
            files = [files[-1]]

        for file in files:
            if not file.startswith("results_"):
                continue
            model_result_filepaths.append(os.path.join(root, file))

    eval_results = {}
    for model_result_filepath in model_result_filepaths:
        # Creation of result
        eval_result = EvalResult.init_from_json_file(model_result_filepath)
        eval_result.update_with_request_file(requests_path)

        # Store results of same eval together. Files are processed oldest -> newest, so a
        # resubmission under the same name (eval_name) refreshes the metadata while
        # accumulating results: for each task the latest non-null score wins, and tasks the
        # newer file leaves empty (null / 0.0 / NaN) keep their previous value. This lets
        # participants re-run a subset of tasks without redoing everything, while corrected
        # metadata from the latest submission is what gets shown.
        eval_name = eval_result.eval_name
        if eval_name in eval_results.keys():
            merged_results = dict(eval_results[eval_name].results)
            merged_results.update({
                k: v for k, v in eval_result.results.items()
                if (
                    v is not None
                    and (v != 0.0 or k in SERVER_SCORED_RESULT_KEYS)
                    and not (isinstance(v, float) and np.isnan(v))
                )
            })
            # Keep the newer entry (its metadata) and carry the merged results forward.
            eval_result.results = merged_results
            eval_results[eval_name] = eval_result
        else:
            eval_results[eval_name] = eval_result

    results = []
    for v in eval_results.values():
        try:
            v.to_dict()  # we test if the dict version is complete
            results.append(v)
        except KeyError:  # not all eval values present
            continue

    return results
