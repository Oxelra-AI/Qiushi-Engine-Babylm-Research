# Sentence Zero Shot

This folder contains 4 python files used compare mutiple different sentences based on their sentence logit scores and choose the most likely sentence/ending.

## Compute Results

The `compute_results.py` files does all the computations of sentence logit scores as well as compare the sentences. It also organizes the predicitions and report metrics. If you add your own dataset, make sure to check the prediction output style is correct for you. It uses the "UID" field defined in the `read_files.py` file to assign the id of the predictions (in particular check the `compute_*_results.py` functions).

## Dataset

The `dataset.py` file defines the Dataset class used to iterate over the data. It processes the sentences in the causal, mlm, or mntp format. It handles the creation of the tokens, attention_mask, indices, and labels, as well as how to collate the datapoints to form a batch. In addition, the code only handle JSONL files for now.

## Read Files

The `read_files.py` file handles the load of the different tasks. If you want to include your own task make sure to add the loading strategy to this file. The required dictionary items for your dataset to function with the rest of the code are the following:
- `sentences`: A list of full sentences to read over (this means context plus option/completion/answer).
- `completions`: The parts of the sentences to sum calculate the sentence logits over (in case it is the whole sentence, just set it to the same value as `sentences`).
- `label`: The position of the correct completion (in the `completions` list).
- `UID`: The unique name of the task/sub-task to sum over for the average accuracy (the code does a macro average over the UIDs by default (`entity_tracking` does it differently, see the `process_results` function in `run.py` for more details)) and organize the prediction file.

## Run

The `run.py` file is the main file you will interact with. It does the entire zero shot evaluation and outputs the predictions and report files. Here is the list of all the flags available:

```python
parser.add_argument("--data_path", default="", type=pathlib.Path, help="Path to the data directory")
parser.add_argument("--output_dir", default="results", type=pathlib.Path, help="Path to the data directory")
parser.add_argument("--task", default="blimp", type=str, help="The task that is being evaluated.",
                    choices=["blimp", "ewok", "entity_tracking"])

parser.add_argument("--model_path_or_name", default="ltg/gpt-bert-babylm-small", type=str, help="Path to the model to evaluate.")
parser.add_argument("--backend", default="mlm", type=str, help="The evaluation backend strategy", choices=["mlm", "causal", "mntp", "enc_dec_mask", "enc_dec_prefix"])

parser.add_argument("--min_temperature", default=1.0, type=float, help="Minimum temperature to apply to the logits.")
parser.add_argument("--max_temperature", default=None, type=float, help="Maximum temperature to apply to the logits. If None, onlny the minimum temperature will be considered.")
parser.add_argument("--temperature_interval", default=0.05, type=float, help="Step size between temperatures applied to the logits.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size for evaluation")
parser.add_argument("--non_causal_batch_size", default=64, type=int, help="Mini-batch size to process each batch of inputs involving masked tokens")
parser.add_argument("--full_sentence_scores", action="store_true", help="Whether to use the entire sentence to calculate the sentence scores rather than just the completion. (Only implemented for EWoK)")
parser.add_argument("--save_predictions", action="store_true", help="Whether or not to save predictions.")
parser.add_argument("--revision_name", default=None, type=str, help="Name of the checkpoint/version of the model to test. (If None, the main will be used)")
```
