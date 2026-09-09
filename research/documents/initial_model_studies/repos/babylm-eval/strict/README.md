# Strict Evaluation BabyLM 2026

## Overview

This code provides the backend for evaluating models participating in the Strict and Strict-small track of BabyLM. The code for this has been adapted from last year's iteration. 

An addition from last year that we keep is that we have two evaluation types: **fast** evaluation uses a smaller set of evaluation samples, allows for quick testing of your models, and is what you will report performance on for the intermediate model checkpoints. The **full** evaluation should be run on your final model.

## Hidden Tasks

### GlobalPIQA

[GlobalPIQA](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-parallel) is a cultural commonsense-reasoning benchmark containing questions in a broad diversity of languages. For each language, GlobalPIQA contains a culture-specific "non-parallel" and shared, culture-agnostic "parallel" split. As the Strict and Strict-Small tracks only train on English data, we restrict our attention to the subset of English (`eng_latn`) questions. We use cloze-style evaluation, where completions are chosen using length-normalized conditional log-probabilties.

To set up and run the evaluation:

1. First download the dataset. From the `strict` directory, run the dedicated download script, which fetches the `eng_latn` parallel and nonparallel subsets into both `evaluation_data/full_eval` and `evaluation_data/fast_eval` (Global PIQA is not subsampled, so the fast subset is identical to the full data):
   ```bash
   python evaluation_pipeline/global_piqa/dl.py
   ```
2. Use the dedicated shell script under `scripts/` to evaluate both your final submission checkpoint and all intermediate (revision) checkpoints. The final checkpoint is scored against `eval_dir` (full_eval) and the intermediate checkpoints against `fast_eval_dir` (fast_eval):
   ```bash
   bash scripts/eval_zero_shot_global_piqa.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <track (strict/strict-small)> <eval_dir (optional, default:evaluation_data/full_eval)> <fast_eval_dir (optional, default:evaluation_data/fast_eval)>
   ```
   As with `eval_zero_shot_fast_all_revisions.sh`, edit the checkpoint for-loops in the script if your revision naming scheme differs from the `chck_<N>M` convention. The standard collation script should then be used to collate these new results.

## Tasks

- **Entity Tracking in Language Models** [(Kim & Schuster, 2023)](https://aclanthology.org/2023.acl-long.213/) - *Tests entity state tracking in LMs. Note: We have changed the evaluation of this task to evaluate LMs' ability to assign the highest probability to the correct continuation (akin to BLiMP and EWoK) rather than generate the correct completion itself as was originally done, to allow for simpler, zero-shot evaluation.*
- **Cloze probability, predictability ratings, and computational estimates for 205 English sentences, aligned with existing EEG and reading time data** [(De Varda et al., 2023)](https://link.springer.com/article/10.3758/s13428-023-02261-8) - *Connects LM predictions to human reading times, allowing us to assess to what extent LM processing is aligned with human language processing.*
- **COMPS: Conceptual Minimal Pair Sentences for testing Robust Property Knowledge and its Inheritance in Pre-trained Language Models** [(Misra et al., 2023)](https://aclanthology.org/2023.eacl-main.213/) (COMPS) [Misra et al. (2023)](https://aclanthology.org/2023.eacl-main.213/) introduce a task for testing the property knowledge of language models and whether they can infer that properties of superordinate concepts are inherited by subordinate concepts, each represented by nonce words. The dataset is composed of minimal pair sentences and models are evaluated by whether they assign higher probability to the correct sentence.
- **Age of Acquisition (AoA) Evaluation Benchmark** [(Chang & Bergen, 2022)](https://doi.org/10.1162/tacl_a_00444) This benchmark tracks word surprisal across training checkpoints to extract learning curves and compute ages of acquisition for vocabulary items. We compute surprisal scores as the negative log probability of target words given their contexts in a test set across training steps [c4-en-10k](https://huggingface.co/datasets/stas/c4-en-10k), and then fit sigmoid functions to the learning trajectory of each word. In the end, the benchmark enables direct comparison between language model and child language development by computing correlation scores between model-derived AoA and human AoA data from the MacArthur-Bates Communicative Development Inventory (CDI) [(Frank et al., 2017)](https://wordbank.stanford.edu).
- **BLiMP: The Benchmark of Linguistic Minimal Pairs** [(Warstadt et al., 2020)](https://aclanthology.org/2020.tacl-1.25/) - *A challenge set of 67 linguistic minimal pair paradigms covering morphology, syntax, and semantics. For each item a model must assign higher probability to the grammatical sentence over the ungrammatical one, providing a fine-grained probe of implicit grammatical knowledge.*
- **EWoK: Elements of World Knowledge** [(Ivanova et al., 2024)](https://aclanthology.org/2024.emnlp-main.903/) - *Tests whether LMs have acquired real-world knowledge about physical and social domains. Items are minimal pairs that differ in a single world-knowledge-relevant property (e.g. agent properties, spatial relations, social interactions), and models are evaluated by whether they assign higher probability to the contextually appropriate sentence.*
- **(Super)GLUE** [(Wang et al., 2018](https://aclanthology.org/D18-1269/)[; Wang et al., 2019)](https://papers.nips.cc/paper_files/paper/2019/hash/4496bf24afe7fab6f046bf4923da8de6-Abstract.html) - *A collection of diverse natural language understanding tasks (BoolQ, MNLI, MRPC, MultiRC, QQP, RTE, WSC) that require text classification or textual entailment. Models are fine-tuned on each task and evaluated on accuracy or F1, measuring how well LMs can transfer their learned representations to downstream NLU tasks.*


## Install

> [!Note]
> The package is currently not installable given that it is a first version. Instead we recommend installing the packages needed and using it as a python module, i.e. to run a part of the pipeline (for example finetuning) you would to do: `python -m evaluation_pipeline.finetune.run ...` from the root folder (the folder that contains the evaluation_pipeline folder).

To be able to use the pipeline you need to install the `requirements.txt` packages.

> [!Warning]
> These packages were installed using Python 3.13, in case some of the packages are not compatible with your Python version (either because the version is too recent or is not supported). In that case, you could either update your Python version or pip/conda install the following packages: `transformers`, `torch`, `scikit-learn`, `numpy`, `pandas`, `statsmodels`, `datasets`, `wandb`, and `nltk`.

## File Structure

```
evaluation_pipeline
├── __init__.py
├── ewok
│   ├── dl_and_filter.py
│   └── vocab.txt
├── finetune
│   ├── README.md
│   ├── __init__.py
│   ├── classifier_model.py
│   ├── dataset.py
│   ├── run.py
│   ├── trainer.py
│   └── utils.py
├── reading
│   ├── README.md
│   ├── __init__.py
│   ├── evaluation_functions.py
│   └── run.py
└── sentence_zero_shot
    ├── README.md
    ├── __init__.py
    ├── compute_results.py
    ├── dataset.py
    ├── read_files.py
    └── run.py
```

## Results File Structure

```
results
├── model_name1
│   ├── main
│   │   ├── finetune
│   │   │   ├── boolq
│   │   │   │   ├── predictions.jsonl
│   │   │   │   └── results.txt
│   │   │   ├── mnli
│   │   │   └── ...
│   │   └── zero_shot
│   │       ├── causal
│   │       │       ├── blimp
│   │       │       │   ├── blimp_filtered
│   │       │       │   │   ├── best_temperature_report.txt
│   │       │       │   │   └── predictions.jsonl
│   │       │       │   ├── supplement_filtered
│   │       │       │   ├── blimp_fast
│   │       │       │   └── ...
│   │       │       ├── ewok
│   │       │       └── ...
│   │       └── ...
│   ├── revision_name1
│   └── revision_name2
├── model_name2
│   ├── ...
└── ...
```

## Data

All evaluation datasets are stored on HuggingFace. To download them onto this repo, you should first log into your HuggingFace account in your terminal by running `huggingface-cli login` and then entering your login token.
After this, you can download a majority of the evaluation datasets by running

```
python -m scripts.download_evals
```

This will download all evaluation datasets except for the full EWoK data. For EWoK, first agree to the terms & conditions on the dataset's HuggingFace page:
- [EWoK](https://huggingface.co/datasets/ewok-core/ewok-core-1.0)
Afterwards, run `python -m evaluation_pipeline.ewok.dl_and_filter` from this folder (i.e., `strict`) to download the full EWoK dataset.

Finally, the EWoK fast dataset, which should be located under `evaluation_data/fast_eval/ewok_fast.zip` following the downloads, is password protected. The password to unzip the file is: BabyLM2025

## Evaluation 

### Zero-shot evaluation

Use the following shell script to evaluate on the full zero-shot evaluations:
```bash
./eval_zero_shot.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <eval_dir (optional, default:evaluation_data/full_eval)>
```

Use the following shell script to evaluate on the fast zero-shot evaluations:
```bash
./eval_zero_shot_fast.sh <path_to_model> <revision_name> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <eval_dir (optional, default:evaluation_data/fast_eval)>
```

> [!Note]
> The revision name indicates the checkpoint to use (for example in the gpt-bert baselines `chck_1M` is the model trained for about 1M words).

These will work out of the box if you use a HuggingFace-based model. In the case you are not, you can either go to the `hf_conversion_tutorial` folder to create a HF repository or adapt the code to work with a pure PyTorch implementation (it should not be too complicated). The implementation currently only supports three types of trained langauge modeling tasks: causal, mlm, and mntp (mlm shifted similarly to causal). If another objective (like diffusion for example) was used to train the models, you will need to edit the files.

In addition we have added a script called `eval_zero_shot_fast_all_revisions.sh` to evaluate all the checkpoints in a single call. To make sure this work with your naming scheme, make sure to edit the for-loops:
```bash
for i in {1..9}; do
    checkpoint="chck_${i}M"
```
To make sure they fit your checkpoint naming scheme. To run the script type the following:
```bash
./eval_zero_shot_fast_all_revisions.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <track> <eval_dir (optional, default:evaluation_data/fast_eval)>
```
> [!NOTE]
> The code assumes that you trained on the entire budget (100M words for strict-small and 1B words for strict). Please change this if this is not the case.

> [!Important]
> For the Encoder-Decoder backend, there exists two different evaluation styles for the sentence zero_shot evaluations. Those are either prefix `enc_dec_prefix` or bi-direction/fill-in-the-gap `enc_dec_mask`. The style of input to the encoder and decoder are the following:
> - **Encoder**: `CLS` text with a mask / prefix ending with a mask `EOS` (the code checks whether a `CLS` and `EOS` are defined, if not, they are not added).
> - **Decoder**: `BOS` / `MASK` text to predict in case of prefix (The choice of `BOS` or `MASK` depends on the architecture).

### AoA evaluation
Use the following shell script to evaluate on the fast zero-shot evaluations:
```bash
./eval_aoa.sh <path_to_model> <architecture (causal/mntp/mlm/enc_dec_mask/enc_dec_prefix)> <track> <eval_dir (optional, default:evaluation_data/full_eval/cdi_childes/cdi_childes.json)> <output_dir (optional, default:results)>
```

### Fine-tuning or low-rank adapter training

Like last year, we provide a script to support fine-tuning on all tasks:
```bash
./eval_finetune.sh <path_to_model> <learning_rate (optional, default: 3e-5)> <batch_size (optional, default: 32)> <max_epochs (optional, default: 10)> <seed (optional, default: 42)>
```
This will fine-tune your model on all (Super)GLUE tasks.

> [!Note]
> To make finetuning evaluations more efficient, this year we randomly subsampled MNLI and QQP to 10k training samples. We found that 10k training samples for these datasets is sufficient for minimizing variance due to randomness. We also removed CoLA, SST2, MNLI-mm, and QNLI because they are highly correlated with other datasets.

> [!Note]
> The hyperparameters are shared through all tasks, if you want to have different ones for every task, you will either need to edit the file or run the python command found in the file from the terminal.

> [!Note]
> There are more hyperparameters you can play with! Checkout the README in the finetune folder of the evaluation_pipeline for more information. In addition, you can edit also edit the classifier head.

> [!Important]
> By default the code only expects encoders or decoders. Add the flag `--enc_dec` to the python calls to use an encoder-decoder. For the fine-tuning, the default classification models does the following:
> - Passes the inputs to the Encoder
> - Initializes a 1 token sequence containing the start token (defined in the `config.json` of the model as `decoder_start_token_id`) to pass to the decoder.
> - Does the Decoder forward pass
> - Takes the encoding of the start token and passes it to the classification head.
> This is the way we decided to implement the finetuning, however, this can be adapted by yourselves.

> [!Note]
> The classification head and the way the model does classification can be adapted by you. Change the code found in the `evaluation_pipeline/finetune/classifier_model.py` file to do this.

## Submission Requirements

### Checkpoints

We require to provide the following checkpoints for all tracks:
- The model checkpoint every 1M words for the first 10M words (total count not unique) trained on (1M, 2M, ..., 10M)
- The model checkpoint every 10M words for the first 100M words (total count not unique) trained on (10M, 20M, ..., 100M)

For all the tracks except strict-small:
- The model checkpoint every 100M words for the first 1000M words (total count not unique) trained on (100M, 200M, ..., 1000M)

For the submiting the checkpoints we encourage creating multiple branches in a HuggingFace repository containing each checkpoint (a brach for the 7M checkpoint could be called chck_7M). Checkout [this repository](https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-mixed) for an example.

### Submission Evaluation

This year we require both the evaluation of the final model, on a set of full evaluation (which include the finetuning). And the evaluation of all the checkpoints mentioned above (or up until the one you trained, if for example you only train for 20M words then we require: 1M, 2M, ..., 10M, 20M) on a set of fast tasks, that do not include finetuning and are a subsampled set of the full evaluations.

### Submission Format
To create a submission file to the leaderboard or challenge use the following command:
```bash
bash collate_preds.sh NAME_OF_YOUR_MODEL BACKEND SUBMISSION_TRACK
```
The `--fast` flag in the script is to add the fast evaluation results of each checkpoint to the collation and make the submission valid for the BabyLM challenge.
> [!NOTE]
> Currently the code assumes that the checkpoint naming scheme is `chck_*M`. If you use a different naming scheme make sure to edit lines 16 and 17 of the `collate_preds.py` file. In addition the code assumes that the training is done on the maximum number of words possible, please edit this if it is not the case for you.

Make sure that all the evaluations have been run before collating them.


The submission is a JSON file where the first key represents the benchmark, the next the task of the benchmark, and the value either the label as a number (for GLUE tasks) or the predicted sentences for the zero-shot tasks. An example is:

```
{"glue": {"boolq": {"predictions": [{"id": "boolq_0", "pred": 0}, {"id": "boolq_1", "pred": 1}, ...]}}}
```

### Steps to submission
You should run the 'eval_zero_shot.sh' and 'eval_finetuning.sh' scripts on the final checkpoint. The 'eval_aoa.sh' and 'eval_zero_shot_fast_all_revisions.sh' scripts should likewise be run for all challenge tracks, but these conduct evaluations for all model checkpoints. If you are in the multimodal track, you should run 'eval_multimodal.sh' as well. Once you have run all scripts for evaluation, you should run 'collate_preds.sh' and submit to the [leaderboard](https://huggingface.co/spaces/BabyLM-community/babylm-leaderboard-2025-all-tasks).


----
## Visualizing Results

You can seamlessly visualize and analyze the results of your evaluation harness runs using Weights & Biases (W&B).

### Weights and Biases

To run your finetuning code with Weights and Biases, you need to pass the `--wandb` flag and set at minimum the `--wandb_entity` to your W&B user/project. You can also set the `--wandb_project` to specify which project the run should be logged to. By default this is *BabyLM Finetuning*. You can also set the name of the run with the `exp_name` flag, by default this is *model_name_task_seed*.

### Support

The best way to get support is to open an issue on this repo or join the [BabyLM slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-2gqgqaumu-5ebxxADuT561aT_ooKbT1Q). Join the `#evaluation-pipeline` channel, which is dedicated to support for use of this repository.
