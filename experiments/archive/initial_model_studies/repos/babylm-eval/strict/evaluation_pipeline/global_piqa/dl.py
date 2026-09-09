import json
import os
from datasets import load_dataset

# Download the "eng_latn" subset of the Global PIQA parallel and non-parallel
# datasets. Global PIQA is not subsampled for the fast evaluation, so the same
# data is written under both full_eval (used for the main checkpoint) and
# fast_eval (used for the intermediate checkpoints), giving it a dedicated
# fast_eval folder like every other fast task.
SUBSETS = {
    "mrlbenchmarks/global-piqa-parallel": "global_piqa_parallel",
    "mrlbenchmarks/global-piqa-nonparallel": "global_piqa_nonparallel",
}
BASE_DIRS = ["full_eval", "fast_eval"]

for repo_id, out_dir in SUBSETS.items():
    dataset = load_dataset(repo_id, "eng_latn", split="test")
    lines = [json.dumps(example) for example in dataset]

    for base_dir in BASE_DIRS:
        out_path = os.path.join("evaluation_data", base_dir, out_dir)
        os.makedirs(out_path, exist_ok=True)
        with open(os.path.join(out_path, "eng_latn.jsonl"), 'w') as outfile:
            for line in lines:
                outfile.write(line + "\n")
