# Self-Paced Reading

Code is adapted and data is taken from https://github.com/Andrea-de-Varda/prediction-resource for the paper from de Verda et al. (2023)

This folder contains 2 python script to do the evaluation of the self-paced reading time as well as eye tracking prediction.

## Evaluation Functions

The `evaluation_functions.py` contains all the functions to evaluate the suprisal of the model for each word. The causal evaluation uses the regular log-prob, while the mlm/mntp uses the multi-mask ending method from Samuel (2024). By default we use a 3-mask ending.

## Run

The `run.py` file is the main file you will interact with. It does the entire zero shot evaluation and outputs the predictions and report files. Here is the list of all the flags available:

```python
parser.add_argument("--output_dir", default="results", type=pathlib.Path, help="The output directory where the results will be written.")
parser.add_argument("--data_path", default="reading/data/reading_data.csv", type=pathlib.Path, help="Path to file containing the lambada dataset, we expect it to be in a JSONL format.")
parser.add_argument("--model_path_or_name", default="ltg/gpt-bert-babylm-small", type=pathlib.Path, help="The path/name to/of the huggingface folder/repository.")
parser.add_argument("--backend", default="causal", type=str, help="The evaluation backend strategy.", choices=["mlm", "mntp", "causal", "enc_dec"])
parser.add_argument("--number_of_mask_tokens_to_append", default=3, type=int, help="When using either mlm or mntp, the number of mask tokens to append to approximate causal generation.")
parser.add_argument("--revision_name", default=None, type=str, help="Name of the checkpoint/version of the model to test. (If None, the main will be used)")
```

## Evaluation

The report outputs two different scores:
- The average increase in R^2 (as a percentage of the maximum increase in R^2 possible) of the Eye Tracking Variables with no spillover effect.
- The increase in R^2 (as a percentage of the maximum increase in R^2 possible) of the Self-paced Reading with 1-word spillover effect.

## Bibliography

de Varda, A. G., Marelli, M., & Amenta, S. (2023). Cloze probability, predictability ratings, and computational estimates for 205 English sentences, aligned with existing EEG and reading time data. Behavior Research Methods, 1-24.

Samuel, D. (2024). BERTs are generative in-context learners. Advances in Neural Information Processing Systems, 37, 2558-2589.
