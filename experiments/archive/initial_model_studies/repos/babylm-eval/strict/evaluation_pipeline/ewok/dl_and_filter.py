import json
import os
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from collections import defaultdict

vocab = set()
with open("evaluation_pipeline/ewok/vocab.txt", 'r') as vocabfile:
    for line in vocabfile:
        word = line.strip()
        vocab.add(word)

if not os.path.exists("evaluation_data/full_eval/ewok_filtered"):
    os.mkdir("evaluation_data/full_eval/ewok_filtered")

items_per_domain = defaultdict(list)
dataset = load_dataset("ewok-core/ewok-core-1.0", split="test")
for example in dataset:
    domain = example["Domain"]

    skip_sent = False
    for key in ("Context1", "Context2", "Target1", "Target2"):
        for word in word_tokenize(example[key].lower()):
            if word not in vocab:
                skip_sent = True
                break
    if skip_sent:
        continue

    # passed filter. add example to list
    items_per_domain[domain].append(example)

for domain in items_per_domain.keys():
    with open(f"evaluation_data/full_eval/ewok_filtered/{domain}.jsonl", 'w') as outfile:
        for item in items_per_domain[domain]:
            outfile.write(json.dumps(item)+"\n")
            swapped_item = item
            # Separate examples where context/target is flipped. Makes it easier to compute accuracies
            swapped_item["Context1"], swapped_item["Context2"] = swapped_item["Context2"], swapped_item["Context1"]
            swapped_item["Target1"], swapped_item["Target2"] = swapped_item["Target2"], swapped_item["Target1"]
            outfile.write(json.dumps(swapped_item)+"\n")
